/*
 * Copyright 2026, Sirius Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda_runtime_api.h>

#include <nixl.h>

#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
void check(nixl_status_t status)
{
  if (status != NIXL_SUCCESS) { throw std::runtime_error(nixlEnumStrings::statusStr(status)); }
}

void check(cudaError_t status)
{
  if (status != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(status)); }
}

struct device_buffer {
  explicit device_buffer(std::size_t bytes) { check(cudaMalloc(&data, bytes)); }
  ~device_buffer() { cudaFree(data); }
  void* data{};
};

struct transfer_request {
  explicit transfer_request(nixlAgent& owner) : agent(owner) {}
  ~transfer_request()
  {
    if (handle) { agent.releaseXferReq(handle); }
  }
  nixlAgent& agent;
  nixlXferReqH* handle{};
};
}  // namespace

int main()
{
  try {
    constexpr std::size_t bytes = 1U << 20;
    std::vector<std::uint8_t> expected(bytes), actual(bytes);
    for (std::size_t i = 0; i < bytes; ++i) {
      expected[i] = static_cast<std::uint8_t>(i * 17);
    }
    check(cudaSetDevice(0));
    device_buffer source(bytes), destination(bytes);
    check(cudaMemcpy(source.data, expected.data(), bytes, cudaMemcpyHostToDevice));
    check(cudaMemset(destination.data, 0, bytes));

    nixlAgentConfig config;
    config.useProgThread = true;
    config.syncMode      = nixl_thread_sync_t::NIXL_THREAD_SYNC_STRICT;
    nixlAgent sender("static-nixl-source", config), receiver("static-nixl-destination", config);
    nixlBackendH* backend{};
    check(sender.createBackend("UCX", {}, backend));
    check(receiver.createBackend("UCX", {}, backend));
    nixl_reg_dlist_t source_registration(VRAM_SEG), destination_registration(VRAM_SEG);
    source_registration.addDesc(
      nixlBlobDesc(reinterpret_cast<std::uintptr_t>(source.data), bytes, 0, ""));
    destination_registration.addDesc(
      nixlBlobDesc(reinterpret_cast<std::uintptr_t>(destination.data), bytes, 0, ""));
    check(sender.registerMem(source_registration));
    check(receiver.registerMem(destination_registration));
    std::string metadata, remote_name;
    check(sender.getLocalMD(metadata));
    check(receiver.loadRemoteMD(metadata, remote_name));
    check(receiver.makeConnection(remote_name));
    check(receiver.getLocalMD(metadata));
    check(sender.loadRemoteMD(metadata, remote_name));
    check(sender.makeConnection(remote_name));

    nixl_xfer_dlist_t local(VRAM_SEG), remote(VRAM_SEG);
    local.addDesc(nixlBasicDesc(reinterpret_cast<std::uintptr_t>(destination.data), bytes, 0));
    remote.addDesc(nixlBasicDesc(reinterpret_cast<std::uintptr_t>(source.data), bytes, 0));
    nixl_opt_args_t options;
    options.notif = "GPU read complete";
    transfer_request request(receiver);
    check(receiver.createXferReq(
      NIXL_READ, local, remote, "static-nixl-source", request.handle, &options));
    auto status         = receiver.postXferReq(request.handle);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
    while (status == NIXL_IN_PROG && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      status = receiver.getXferStatus(request.handle);
    }
    check(status);
    check(receiver.releaseXferReq(request.handle));
    request.handle = nullptr;
    check(cudaMemcpy(actual.data(), destination.data, bytes, cudaMemcpyDeviceToHost));
    if (actual != expected) { throw std::runtime_error("GPU payload differs"); }

    bool notified = false;
    while (!notified && std::chrono::steady_clock::now() < deadline) {
      nixl_notifs_t notifications;
      check(sender.getNotifs(notifications));
      for (const auto& message : notifications["static-nixl-destination"]) {
        notified = notified || message == "GPU read complete";
      }
      if (!notified) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); }
    }
    if (!notified) { throw std::runtime_error("completion notification timed out"); }
    check(receiver.deregisterMem(destination_registration));
    check(sender.deregisterMem(source_registration));
    std::cout << "Static NIXL UCX: 1 MiB GPU transfer and notification passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

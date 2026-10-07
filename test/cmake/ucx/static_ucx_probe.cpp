#include <cuda_runtime_api.h>

#include <ucp/api/ucp.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(ucs_status_t status)
{
  if (status != UCS_OK) { throw std::runtime_error(ucs_status_string(status)); }
}

void check(cudaError_t status)
{
  if (status != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(status)); }
}

void progress(void* request, ucp_worker_h sender, ucp_worker_h receiver)
{
  if (UCS_PTR_IS_ERR(request)) { check(UCS_PTR_STATUS(request)); }
  if (!request) { return; }
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
  ucs_status_t status;
  while ((status = ucp_request_check_status(request)) == UCS_INPROGRESS) {
    ucp_worker_progress(sender);
    ucp_worker_progress(receiver);
    if (std::chrono::steady_clock::now() > deadline) {
      throw std::runtime_error("UCX transfer timed out");
    }
  }
  ucp_request_free(request);
  check(status);
}

struct worker {
  ucp_context_h context;
  ucp_worker_h handle;
  ucp_mem_h memory;

  explicit worker(void* buffer, std::size_t size)
  {
    ucp_config_t* config;
    check(ucp_config_read(nullptr, nullptr, &config));
    ucp_params_t params{};
    params.field_mask = UCP_PARAM_FIELD_FEATURES;
    params.features   = UCP_FEATURE_RMA;
    check(ucp_init(&params, config, &context));
    ucp_config_release(config);

    ucp_context_attr_t attributes{};
    attributes.field_mask = UCP_ATTR_FIELD_MEMORY_TYPES;
    check(ucp_context_query(context, &attributes));
    if (!(attributes.memory_types & UCS_BIT(UCS_MEMORY_TYPE_CUDA))) {
      throw std::runtime_error("Static UCX did not register CUDA memory support");
    }

    ucp_worker_params_t worker_params{};
    worker_params.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;
    check(ucp_worker_create(context, &worker_params, &handle));

    ucp_mem_map_params_t map_params{};
    map_params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS | UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                            UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE;
    map_params.address     = buffer;
    map_params.length      = size;
    map_params.memory_type = UCS_MEMORY_TYPE_CUDA;
    check(ucp_mem_map(context, &map_params, &memory));
  }

  ~worker()
  {
    ucp_mem_unmap(context, memory);
    ucp_worker_destroy(handle);
    ucp_cleanup(context);
  }
};
}  // namespace

int main()
{
  try {
    constexpr std::size_t size = 1024 * 1024;
    std::vector<unsigned char> expected(size);
    for (std::size_t i = 0; i < size; ++i) {
      expected[i] = static_cast<unsigned char>(i * 37);
    }
    void* source;
    void* destination;
    check(cudaMalloc(&source, size));
    check(cudaMalloc(&destination, size));
    check(cudaMemcpy(source, expected.data(), size, cudaMemcpyHostToDevice));
    check(cudaMemset(destination, 0, size));

    {
      worker sender(source, size);
      worker receiver(destination, size);
      ucp_address_t* address;
      std::size_t address_size;
      check(ucp_worker_get_address(receiver.handle, &address, &address_size));
      ucp_ep_params_t endpoint_params{};
      endpoint_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
      endpoint_params.address    = address;
      ucp_ep_h endpoint;
      check(ucp_ep_create(sender.handle, &endpoint_params, &endpoint));
      ucp_worker_release_address(receiver.handle, address);

      void* key_buffer;
      std::size_t key_size;
      check(ucp_rkey_pack(receiver.context, receiver.memory, &key_buffer, &key_size));
      ucp_rkey_h key;
      check(ucp_ep_rkey_unpack(endpoint, key_buffer, &key));
      ucp_rkey_buffer_release(key_buffer);

      ucp_request_param_t params{};
      progress(
        ucp_put_nbx(
          endpoint, source, size, reinterpret_cast<std::uint64_t>(destination), key, &params),
        sender.handle,
        receiver.handle);
      progress(ucp_ep_flush_nbx(endpoint, &params), sender.handle, receiver.handle);
      check(cudaDeviceSynchronize());
      std::vector<unsigned char> actual(size);
      check(cudaMemcpy(actual.data(), destination, size, cudaMemcpyDeviceToHost));
      if (actual != expected) { throw std::runtime_error("GPU transfer data mismatch"); }

      ucp_rkey_destroy(key);
      progress(ucp_ep_close_nbx(endpoint, &params), sender.handle, receiver.handle);
    }
    check(cudaFree(destination));
    check(cudaFree(source));
    std::puts("Static UCX CUDA transfer passed (1048576 bytes)");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "Static UCX CUDA transfer failed: %s\n", error.what());
    return 1;
  }
}

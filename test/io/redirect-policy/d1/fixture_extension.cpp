#include "d1_observer.hpp"

#include <filesystem>

TEST_CASE("D1 real worker joint publication receipts", "[d1_joint]")
{
  for (auto const scenario : {"reuse",
                              "stale",
                              "precondition",
                              "off",
                              "disabled",
                              "weak",
                              "observer_failure",
                              "observer_overflow",
                              "observer_exception"}) {
    DYNAMIC_SECTION(scenario)
    {
      auto const payload = deterministic_payload(c3_object_size);
      range_http_server origin(
        payload, tagged_object_policy(), {}, scripted_list_mode::normal, true);
      range_http_server target(
        payload, tagged_object_policy(), {}, scripted_list_mode::normal, true);
      std::string name    = scenario;
      bool const disabled = name == "disabled";
      if (!disabled) c3_delegate(origin, target, "/same-publication-url");
      auto cfg = c3_config(origin, {authority(target.endpoint())}, name != "off");
      cfg.cluster_redirect.enabled = !disabled;
      cfg.max_connections          = 4;
      cfg.request_timeout_s        = 10;
      auto object                  = std::make_shared<redirect_object>("s3://bucket/d1",
                                                      "bucket",
                                                      "d1",
                                                      c3_object_size,
                                                      name == "weak" ? "W/\"weak\"" : redirect_tag);
      d1::session
        observation;  // declared before fixture: worker joined before recorder destruction
      c3_fixture fixture(origin.endpoint(), cfg);
      std::vector<std::string> terminals;
      auto run = [&](std::size_t offset, bool changed = false) {
        logged_read read(object, offset, 32);
        auto future = read.coordinator->get_future();
        fixture.reactor().enqueue(std::move(read.request));
        REQUIRE(std::move(future).wait(5s));
        bool actual_changed = false;
        std::size_t n       = 0;
        try {
          n = std::move(future).get();
        } catch (sirius::io::object_changed_error const&) {
          actual_changed = true;
        }
        REQUIRE(actual_changed == changed);
        REQUIRE(read.completions->load() == 1);
        if (!changed) {
          REQUIRE(n == 32);
          require_bytes_equal(*read.bytes,
                              std::span<std::uint8_t const>(payload).subspan(offset, 32));
        }
        terminals.push_back("{\"kind\":\"terminal\",\"id\":" + std::to_string(offset) +
                            ",\"completions\":1,\"bytes\":" + std::to_string(n) + ",\"result\":\"" +
                            (changed ? "object_changed" : "success") + "\"}");
      };
      if (name == "observer_failure") observation.data.fail_next.store(true);
      if (name == "observer_overflow") observation.data.capacity = 1;
      if (name == "observer_exception") observation.data.throw_next.store(true);
      run(0);  // real validated ranged GET publication (or deliberately ineligible)
      auto initial = object->route(fixture.context);
      if (name == "stale") {
        REQUIRE(initial != nullptr);
        auto gate = std::make_shared<response_gate>();
        scripted_http_response first, second;
        first.status  = 404;
        first.body    = "held";
        first.gate    = gate;
        second.status = 404;
        second.body   = "";
        target.script(
          "GET", {first, second, scripted_http_response{}, scripted_http_response{}}, false);
        logged_read stale(object, 64, 32);
        auto future = stale.coordinator->get_future();
        fixture.reactor().enqueue(std::move(stale.request));
        REQUIRE(gate->wait_until_entered());
        run(128);  // removes P1, refreshes, publishes P2 at exactly the SAME URL
        auto newer = object->route(fixture.context);
        REQUIRE(newer != nullptr);
        REQUIRE(newer.get() != initial.get());
        REQUIRE(*newer == *initial);
        gate->release();
        REQUIRE(std::move(future).wait(5s));
        REQUIRE(std::move(future).get() == 32);
        REQUIRE(stale.completions->load() == 1);
        require_bytes_equal(*stale.bytes, std::span<std::uint8_t const>(payload).subspan(64, 32));
        terminals.push_back(
          "{\"kind\":\"terminal\",\"id\":64,\"completions\":1,\"bytes\":32,\"result\":"
          "\"success\"}");
      } else if (name == "precondition") {
        scripted_http_response failure;
        failure.status = 412;
        failure.body   = "";
        target.script("GET", {failure}, false);
        run(64, true);
        REQUIRE(object->route(fixture.context).get() == initial.get());
      } else {
        run(64);
      }
      fixture.io->shutdown();
      auto const output = std::getenv("D1_OUT");
      REQUIRE(output != nullptr);
      std::ofstream out(std::filesystem::path(output) / (name + ".jsonl"));
      REQUIRE(out.good());
      d1::dump(observation.data, out, object.get(), fixture.context.get());
      for (auto const& row : terminals)
        out << row << '\n';
      // Independent server-side wire receipts, joined only by explicit unique Range.
      auto wire = [&](range_http_server const& server, char const* role) {
        for (auto const& req : server.requests()) {
          auto values = req.values("range");
          REQUIRE(req.method == "GET");
          REQUIRE(values.size() == 1);
          out << "{\"kind\":\"wire\",\"role\":\"" << role << "\",\"method\":\"" << req.method
              << "\",\"range\":\"" << values[0]
              << "\",\"if_match\":" << (!req.values("if-match").empty()) << "}\n";
        }
      };
      wire(origin, "origin");
      wire(target, "target");
      out.flush();
      REQUIRE(out.good());
      REQUIRE(observation.data.failed.load() == (name.rfind("observer_", 0) == 0));
    }
  }
}

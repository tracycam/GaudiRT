// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace gaudirt::runtime {

// High-level command access summary captured from framework/Bridge graph nodes.
// A command may contain compute only, collective only, or (as in production70
// command 5) a compute recipe coupled with a single trailing collective.
struct CapturedCommandDesc {
  size_t original_index = 0;
  bool compute = false;
  size_t collectives = 0;
  size_t external_events = 0;
  bool complete = true;
  uint64_t communicator = 0;
  std::vector<uint64_t> compute_reads;
  std::vector<uint64_t> compute_writes;
  std::vector<uint64_t> collective_reads;
  std::vector<uint64_t> collective_writes;
};

enum class SubCallKind : uint8_t {
  Compute = 0,
  Collective = 1,
};

struct LoweredSubCall {
  size_t subcall_index = 0;
  size_t original_command_index = 0;
  SubCallKind kind = SubCallKind::Compute;
  bool complete = true;
  size_t external_events = 0;
  uint64_t communicator = 0;
  std::vector<uint64_t> reads;
  std::vector<uint64_t> writes;
};

struct CoalescedRegion {
  uint64_t communicator = 0;
  size_t first_subcall = 0;
  size_t last_subcall = 0;
  size_t first_original_command = 0;
  size_t last_original_command = 0;
  size_t compute_subcalls = 0;
  size_t collective_subcalls = 0;
  std::vector<size_t> collective_indices;
};

struct LoweringReport {
  std::vector<LoweredSubCall> subcalls;
  std::vector<CoalescedRegion> regions;
  size_t original_commands = 0;
  size_t decomposed_mixed_commands = 0;
  size_t total_compute_subcalls = 0;
  size_t total_collective_subcalls = 0;
  size_t coalesced_collectives = 0;
};

class UnifiedRegionLowerer {
 public:
  // Decompose eligible mixed (compute + single collective) commands into two
  // ordered atomic sub-calls (Compute followed by Collective) when the
  // command is complete and free of external tensor events, then compute
  // maximal RAW-connected single-communicator serial regions.
  static LoweringReport lower(const std::vector<CapturedCommandDesc>& commands,
                              bool decompose_mixed = true) {
    LoweringReport report;
    report.original_commands = commands.size();
    report.subcalls.reserve(commands.size() + 8);

    for (size_t i = 0; i < commands.size(); ++i) {
      const auto& cmd = commands[i];
      const bool eligible_mixed =
          decompose_mixed && cmd.complete && cmd.external_events == 0 &&
          cmd.compute && cmd.collectives == 1 &&
          has_internal_raw(cmd.compute_writes, cmd.collective_reads);

      if (eligible_mixed) {
        ++report.decomposed_mixed_commands;
        LoweredSubCall comp;
        comp.subcall_index = report.subcalls.size();
        comp.original_command_index = i;
        comp.kind = SubCallKind::Compute;
        comp.complete = true;
        comp.external_events = 0;
        comp.communicator = 0;
        comp.reads = cmd.compute_reads;
        comp.writes = cmd.compute_writes;
        report.subcalls.push_back(std::move(comp));
        ++report.total_compute_subcalls;

        LoweredSubCall coll;
        coll.subcall_index = report.subcalls.size();
        coll.original_command_index = i;
        coll.kind = SubCallKind::Collective;
        coll.complete = true;
        coll.external_events = 0;
        coll.communicator = cmd.communicator;
        coll.reads = cmd.collective_reads;
        coll.writes = cmd.collective_writes;
        report.subcalls.push_back(std::move(coll));
        ++report.total_collective_subcalls;
      } else {
        LoweredSubCall sub;
        sub.subcall_index = report.subcalls.size();
        sub.original_command_index = i;
        sub.kind = (cmd.collectives > 0 && !cmd.compute)
                       ? SubCallKind::Collective
                       : SubCallKind::Compute;
        // If a mixed command was not decomposed, mark it as a barrier by
        // retaining external_events or incomplete status for region formation.
        sub.complete = cmd.complete && !(cmd.compute && cmd.collectives > 0);
        sub.external_events = cmd.external_events;
        sub.communicator = cmd.communicator;
        sub.reads = cmd.compute_reads;
        sub.reads.insert(sub.reads.end(), cmd.collective_reads.begin(),
                         cmd.collective_reads.end());
        sub.writes = cmd.compute_writes;
        sub.writes.insert(sub.writes.end(), cmd.collective_writes.begin(),
                          cmd.collective_writes.end());
        if (cmd.compute) ++report.total_compute_subcalls;
        report.total_collective_subcalls += cmd.collectives;
        report.subcalls.push_back(std::move(sub));
      }
    }

    build_regions(report);
    return report;
  }

 private:
  static bool has_internal_raw(const std::vector<uint64_t>& compute_writes,
                               const std::vector<uint64_t>& collective_reads) {
    for (uint64_t w : compute_writes) {
      if (w == 0) continue;
      if (std::find(collective_reads.begin(), collective_reads.end(), w) !=
          collective_reads.end()) {
        return true;
      }
    }
    return false;
  }

  static void build_regions(LoweringReport& report) {
    const auto& subs = report.subcalls;
    const size_t n = subs.size();
    if (n == 0) return;

    // Track last writer subcall index per allocation key for RAW edges.
    std::unordered_map<uint64_t, size_t> last_writer;
    std::vector<std::vector<size_t>> raw_producers(n);
    for (size_t i = 0; i < n; ++i) {
      for (uint64_t r : subs[i].reads) {
        if (r == 0) continue;
        auto it = last_writer.find(r);
        if (it != last_writer.end() && it->second != i) {
          raw_producers[i].push_back(it->second);
        }
      }
      for (uint64_t w : subs[i].writes) {
        if (w == 0) continue;
        last_writer[w] = i;
      }
    }

    auto has_raw = [&](size_t from, size_t to) {
      const auto& prods = raw_producers[to];
      return std::find(prods.begin(), prods.end(), from) != prods.end();
    };

    std::vector<bool> eligible_collective(n, false);
    for (size_t i = 1; i + 1 < n; ++i) {
      const auto& before = subs[i - 1];
      const auto& middle = subs[i];
      const auto& after = subs[i + 1];
      if (before.complete && middle.complete && after.complete &&
          before.kind == SubCallKind::Compute &&
          middle.kind == SubCallKind::Collective &&
          after.kind == SubCallKind::Compute &&
          before.external_events == 0 && middle.external_events == 0 &&
          after.external_events == 0 && has_raw(i - 1, i) &&
          has_raw(i, i + 1)) {
        eligible_collective[i] = true;
      }
    }

    CoalescedRegion current{};
    auto finish = [&]() {
      if (current.collective_indices.size() > 1) {
        report.regions.push_back(std::move(current));
      }
      current = {};
    };

    for (size_t i = 0; i < n; ++i) {
      const auto& sub = subs[i];
      if (!sub.complete || sub.external_events > 0) {
        finish();
        continue;
      }
      if (sub.kind != SubCallKind::Collective) {
        continue;
      }
      if (!eligible_collective[i] || sub.communicator == 0) {
        finish();
        continue;
      }
      if (current.communicator != 0 &&
          current.communicator != sub.communicator) {
        finish();
      }
      current.communicator = sub.communicator;
      current.collective_indices.push_back(i);
    }
    finish();

    auto pure_compute = [&](size_t idx) {
      const auto& s = subs[idx];
      return s.complete && s.kind == SubCallKind::Compute &&
             s.external_events == 0;
    };

    for (size_t r = 0; r < report.regions.size(); ++r) {
      const size_t min_first =
          (r == 0) ? 0 : (report.regions[r - 1].last_subcall + 1);
      const size_t max_last =
          (r + 1 < report.regions.size())
              ? (report.regions[r + 1].collective_indices.front() - 1)
              : (n - 1);
      size_t first = report.regions[r].collective_indices.front();
      size_t last = report.regions[r].collective_indices.back();
      while (first > min_first && pure_compute(first - 1) &&
             has_raw(first - 1, first)) {
        --first;
      }
      while (last < max_last && pure_compute(last + 1) &&
             has_raw(last, last + 1)) {
        ++last;
      }
      report.regions[r].first_subcall = first;
      report.regions[r].last_subcall = last;
      report.regions[r].first_original_command =
          subs[first].original_command_index;
      report.regions[r].last_original_command =
          subs[last].original_command_index;
      size_t comp_cnt = 0;
      size_t coll_cnt = 0;
      for (size_t k = first; k <= last; ++k) {
        if (subs[k].kind == SubCallKind::Compute)
          ++comp_cnt;
        else
          ++coll_cnt;
      }
      report.regions[r].compute_subcalls = comp_cnt;
      report.regions[r].collective_subcalls = coll_cnt;
      report.coalesced_collectives += coll_cnt;
    }
  }
};

}  // namespace gaudirt::runtime

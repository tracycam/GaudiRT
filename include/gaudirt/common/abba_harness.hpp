#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

namespace gaudirt::common {

struct SummaryStats {
  size_t count = 0;
  double min = 0.0;
  double p50 = 0.0;
  double p90 = 0.0;
  double max = 0.0;
  double mean = 0.0;
};

inline SummaryStats summarize(std::vector<double> samples) {
  SummaryStats s{};
  if (samples.empty()) return s;
  std::sort(samples.begin(), samples.end());
  s.count = samples.size();
  s.min = samples.front();
  s.max = samples.back();
  s.p50 = samples[samples.size() / 2];
  s.p90 = samples[(samples.size() * 90) / 100];
  double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
  s.mean = sum / static_cast<double>(samples.size());
  return s;
}

inline double now_us() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double, std::micro>(clock::now().time_since_epoch()).count();
}

} // namespace gaudirt::common

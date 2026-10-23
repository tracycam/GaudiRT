#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <synapse_api.h>

#define GAUDIRT_SYN_CHECK(expr)                                                \
  do {                                                                         \
    synStatus _st = (expr);                                                    \
    if (_st != synSuccess) {                                                   \
      std::fprintf(stderr, "[GaudiRT] Synapse error %d at %s:%d in %s\n",      \
                   static_cast<int>(_st), __FILE__, __LINE__, #expr);          \
      std::abort();                                                            \
    }                                                                          \
  } while (0)

namespace gaudirt::common {

class SynapseContext {
public:
  explicit SynapseContext(uint32_t module_id = 0) {
    GAUDIRT_SYN_CHECK(synInitialize());
    synModuleId mod = static_cast<synModuleId>(module_id);
    if (synDeviceAcquireByModuleId(&dev_, mod) != synSuccess) {
      GAUDIRT_SYN_CHECK(synDeviceAcquireByDeviceType(&dev_, synDeviceGaudi2));
    }
  }
  ~SynapseContext() {
    if (dev_ != SYN_INVALID_DEVICE_ID) {
      synDeviceRelease(dev_);
    }
    synDestroy();
  }
  SynapseContext(const SynapseContext&) = delete;
  SynapseContext& operator=(const SynapseContext&) = delete;

  synDeviceId device() const { return dev_; }

private:
  synDeviceId dev_ = SYN_INVALID_DEVICE_ID;
};

class DeviceBuffer {
public:
  DeviceBuffer(synDeviceId dev, uint64_t bytes) : dev_(dev), bytes_(bytes) {
    if (bytes_ > 0) {
      GAUDIRT_SYN_CHECK(synDeviceMalloc(dev_, bytes_, 0, 0, &addr_));
    }
  }
  ~DeviceBuffer() {
    if (addr_ != 0) {
      synDeviceFree(dev_, addr_, 0);
    }
  }
  DeviceBuffer(DeviceBuffer&& other) noexcept
      : dev_(other.dev_), bytes_(other.bytes_), addr_(other.addr_) {
    other.addr_ = 0;
    other.bytes_ = 0;
  }
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      if (addr_ != 0) {
        synDeviceFree(dev_, addr_, 0);
      }
      dev_ = other.dev_;
      bytes_ = other.bytes_;
      addr_ = other.addr_;
      other.addr_ = 0;
      other.bytes_ = 0;
    }
    return *this;
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  uint64_t addr() const { return addr_; }
  uint64_t bytes() const { return bytes_; }

private:
  synDeviceId dev_ = 0;
  uint64_t bytes_ = 0;
  uint64_t addr_ = 0;
};

class TimedEventPair {
public:
  explicit TimedEventPair(synDeviceId dev) : dev_(dev) {
    GAUDIRT_SYN_CHECK(synEventCreate(&start_, dev_, EVENT_COLLECT_TIME));
    GAUDIRT_SYN_CHECK(synEventCreate(&end_, dev_, EVENT_COLLECT_TIME));
  }
  ~TimedEventPair() {
    if (start_) synEventDestroy(start_);
    if (end_) synEventDestroy(end_);
  }
  TimedEventPair(const TimedEventPair&) = delete;
  TimedEventPair& operator=(const TimedEventPair&) = delete;

  void record_start(synStreamHandle stream) {
    GAUDIRT_SYN_CHECK(synEventRecord(start_, stream));
  }
  void record_end(synStreamHandle stream) {
    GAUDIRT_SYN_CHECK(synEventRecord(end_, stream));
  }
  double elapsed_us() const {
    GAUDIRT_SYN_CHECK(synEventSynchronize(end_));
    uint64_t ns = 0;
    GAUDIRT_SYN_CHECK(synEventElapsedTime(&ns, start_, end_));
    return static_cast<double>(ns) * 1e-3;
  }

private:
  synDeviceId dev_ = 0;
  synEventHandle start_ = nullptr;
  synEventHandle end_ = nullptr;
};

} // namespace gaudirt::common

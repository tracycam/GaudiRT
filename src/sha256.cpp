// SPDX-License-Identifier: Apache-2.0
#include "gaudirt/common/sha256.hpp"

#include <algorithm>
#include <array>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace gaudirt::common {

namespace {

constexpr std::array<uint32_t, 64> kRoundConstants = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

constexpr uint32_t rotate_right(uint32_t value, unsigned bits) {
  return (value >> bits) | (value << (32 - bits));
}

class Sha256 {
 public:
  void update(const uint8_t* data, size_t size) {
    if (size > UINT64_MAX - total_bytes_) {
      throw std::overflow_error("SHA256 input size overflow");
    }
    total_bytes_ += size;
    while (size != 0) {
      const size_t count = std::min(size, buffer_.size() - buffered_);
      std::copy_n(data, count, buffer_.begin() + buffered_);
      buffered_ += count;
      data += count;
      size -= count;
      if (buffered_ == buffer_.size()) {
        transform(buffer_.data());
        buffered_ = 0;
      }
    }
  }

  std::string finish() {
    const uint64_t bit_length = total_bytes_ * 8;
    buffer_[buffered_++] = 0x80;
    if (buffered_ > 56) {
      std::fill(buffer_.begin() + buffered_, buffer_.end(), 0);
      transform(buffer_.data());
      buffered_ = 0;
    }
    std::fill(buffer_.begin() + buffered_, buffer_.begin() + 56, 0);
    for (unsigned i = 0; i < 8; ++i) {
      buffer_[56 + i] = static_cast<uint8_t>(bit_length >> (56 - 8 * i));
    }
    transform(buffer_.data());

    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (const uint32_t word : state_) output << std::setw(8) << word;
    return output.str();
  }

 private:
  void transform(const uint8_t* block) {
    std::array<uint32_t, 64> words{};
    for (unsigned i = 0; i < 16; ++i) {
      words[i] = (static_cast<uint32_t>(block[4 * i]) << 24) |
                 (static_cast<uint32_t>(block[4 * i + 1]) << 16) |
                 (static_cast<uint32_t>(block[4 * i + 2]) << 8) |
                 static_cast<uint32_t>(block[4 * i + 3]);
    }
    for (unsigned i = 16; i < 64; ++i) {
      const uint32_t s0 = rotate_right(words[i - 15], 7) ^
                          rotate_right(words[i - 15], 18) ^ (words[i - 15] >> 3);
      const uint32_t s1 = rotate_right(words[i - 2], 17) ^
                          rotate_right(words[i - 2], 19) ^ (words[i - 2] >> 10);
      words[i] = words[i - 16] + s0 + words[i - 7] + s1;
    }

    uint32_t a = state_[0];
    uint32_t b = state_[1];
    uint32_t c = state_[2];
    uint32_t d = state_[3];
    uint32_t e = state_[4];
    uint32_t f = state_[5];
    uint32_t g = state_[6];
    uint32_t h = state_[7];
    for (unsigned i = 0; i < 64; ++i) {
      const uint32_t sum1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^
                            rotate_right(e, 25);
      const uint32_t choose = (e & f) ^ (~e & g);
      const uint32_t temp1 = h + sum1 + choose + kRoundConstants[i] + words[i];
      const uint32_t sum0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^
                            rotate_right(a, 22);
      const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const uint32_t temp2 = sum0 + majority;
      h = g;
      g = f;
      f = e;
      e = d + temp1;
      d = c;
      c = b;
      b = a;
      a = temp1 + temp2;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
  }

  std::array<uint32_t, 8> state_ = {0x6a09e667, 0xbb67ae85, 0x3c6ef372,
                                    0xa54ff53a, 0x510e527f, 0x9b05688c,
                                    0x1f83d9ab, 0x5be0cd19};
  std::array<uint8_t, 64> buffer_{};
  size_t buffered_ = 0;
  uint64_t total_bytes_ = 0;
};

}  // namespace

std::string sha256_bytes(const uint8_t* data, size_t size) {
  if (!data && size != 0) throw std::invalid_argument("null SHA256 input");
  Sha256 hash;
  hash.update(data, size);
  return hash.finish();
}

std::string sha256_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("cannot open file for SHA256: " + path.string());
  Sha256 hash;
  std::vector<uint8_t> buffer(1024 * 1024);
  while (input) {
    input.read(reinterpret_cast<char*>(buffer.data()), buffer.size());
    const std::streamsize count = input.gcount();
    if (count > 0) hash.update(buffer.data(), static_cast<size_t>(count));
  }
  if (!input.eof()) throw std::runtime_error("failed to read file for SHA256: " + path.string());
  return hash.finish();
}

}  // namespace gaudirt::common

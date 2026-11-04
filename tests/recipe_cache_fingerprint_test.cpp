// SPDX-License-Identifier: Apache-2.0
#include "gaudirt/runtime/recipe_registry.hpp"

#include <filesystem>
#include <fstream>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

// Run in separate processes, changing GC_KERNEL_PATH's provider contents in
// between. The sentinel is deliberately not a real recipe and is never loaded.
int main(int argc, char** argv) {
  try {
    if (argc != 3) throw std::invalid_argument("expected cache directory and seed/miss");
    gaudirt::runtime::RecipeCache cache(argv[1]);
    const auto sentinel = cache.directory() / "probe.recipe";
    const std::string mode(argv[2]);
    if (mode == "seed") {
      std::filesystem::create_directories(cache.directory());
      std::ofstream out(sentinel, std::ios::binary);
      out << "not a recipe";
      if (!out) throw std::runtime_error("cannot write sentinel");
      setenv("GC_KERNEL_PATH", "changed-provider", 1);
      bool rejected = false;
      try {
        gaudirt::runtime::RecipeCache invalid(argv[1]);
      } catch (const std::runtime_error&) {
        rejected = true;
      }
      if (!rejected) throw std::runtime_error("changed environment was accepted");
    } else if (mode == "miss") {
      if (std::filesystem::exists(sentinel) || cache.load("probe") != nullptr)
        throw std::runtime_error("changed provider reused previous cache");
    } else {
      throw std::invalid_argument("invalid test mode");
    }
    std::cout << cache.directory().filename().string() << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

#ifndef LLAMA_DART_GPU_POLICY_H_
#define LLAMA_DART_GPU_POLICY_H_

#include "llama_dart.h"

#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif

#include <cstdint>

namespace llama_dart_bridge_internal {

struct gpu_load_policy {
  uint32_t backend;
  int32_t n_gpu_layers;
  bool simulator_auto_cpu;
};

constexpr bool target_is_apple_simulator() {
#if defined(__APPLE__) && defined(TARGET_OS_SIMULATOR) && TARGET_OS_SIMULATOR
  return true;
#else
  return false;
#endif
}

constexpr gpu_load_policy resolve_gpu_load_policy(
    uint32_t requested_backend, int32_t requested_gpu_layers,
    bool is_apple_simulator = target_is_apple_simulator()) {
  if (is_apple_simulator &&
      requested_backend == LLAMA_DART_GPU_BACKEND_AUTO) {
    return {LLAMA_DART_GPU_BACKEND_CPU, 0, true};
  }
  return {requested_backend, requested_gpu_layers, false};
}

constexpr bool resolve_mmproj_use_gpu(bool requested,
                                      bool simulator_auto_cpu) {
  return requested && !simulator_auto_cpu;
}

} // namespace llama_dart_bridge_internal

#endif // LLAMA_DART_GPU_POLICY_H_

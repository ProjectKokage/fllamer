#pragma once

#include "llama.h"

namespace llama_dart_bridge_internal {

inline llama_load_mode resolve_model_load_mode(bool use_mmap,
                                               bool use_mlock) noexcept {
  if (use_mmap) {
    return use_mlock ? LLAMA_LOAD_MODE_MMAP_MLOCK : LLAMA_LOAD_MODE_MMAP;
  }
  return use_mlock ? LLAMA_LOAD_MODE_MLOCK : LLAMA_LOAD_MODE_NONE;
}

} // namespace llama_dart_bridge_internal

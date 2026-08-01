#pragma once

#include "llama_dart.h"

#include <cstdint>

namespace llama_dart_bridge_internal {

constexpr uint32_t resolve_quantized_v_flash_attention_mode(
    uint32_t requested_mode, bool quantized_value_cache) {
  return requested_mode == LLAMA_DART_FLASH_ATTENTION_AUTO &&
                 quantized_value_cache
             ? LLAMA_DART_FLASH_ATTENTION_ENABLED
             : requested_mode;
}

} // namespace llama_dart_bridge_internal

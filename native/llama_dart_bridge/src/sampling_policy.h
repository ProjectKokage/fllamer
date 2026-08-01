#pragma once

#include "llama.h"

#include <cmath>
#include <cstdint>
#include <vector>

namespace llama_dart_bridge_internal {

inline llama_sampler *init_suppress_tokens_sampler(
    int32_t vocabulary_size, const llama_token *tokens,
    int32_t token_count) {
  if (vocabulary_size <= 0 || tokens == nullptr || token_count <= 0) {
    return nullptr;
  }
  std::vector<llama_logit_bias> biases;
  biases.reserve(static_cast<size_t>(token_count));
  for (int32_t i = 0; i < token_count; ++i) {
    biases.push_back({tokens[i], -INFINITY});
  }
  return llama_sampler_init_logit_bias(vocabulary_size, biases.size(),
                                       biases.data());
}

} // namespace llama_dart_bridge_internal

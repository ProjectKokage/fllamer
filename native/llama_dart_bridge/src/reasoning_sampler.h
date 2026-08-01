#pragma once

#include "llama.h"

#include <vector>

namespace llama_dart_bridge_internal {

// Gates an optional lazy grammar with llama.cpp's reasoning-budget sampler.
// The returned sampler owns both inputs. generation_prefill is replayed only
// into the reasoning sampler so model-template state cannot be inferred from
// raw prompt text. Add this gate before the ordinary samplers in a
// llama_sampler_chain.
llama_sampler *init_reasoning_gate_sampler(
    llama_sampler *reasoning_budget, llama_sampler *lazy_grammar,
    std::vector<llama_token> generation_prefill);

} // namespace llama_dart_bridge_internal

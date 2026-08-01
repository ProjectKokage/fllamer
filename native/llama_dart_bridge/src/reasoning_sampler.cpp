#include "reasoning_sampler.h"

#include "llama-cpp.h"
#include "reasoning-budget.h"

#include <memory>
#include <utility>

namespace llama_dart_bridge_internal {
namespace {

struct reasoning_gate_sampler_context {
  llama_sampler_ptr reasoning_budget;
  llama_sampler_ptr lazy_grammar;
  std::vector<llama_token> generation_prefill;
};

bool grammar_should_apply(const reasoning_gate_sampler_context *context) {
  if (context->lazy_grammar == nullptr) {
    return false;
  }
  const common_reasoning_budget_state state =
      common_reasoning_budget_get_state(context->reasoning_budget.get());
  return state == REASONING_BUDGET_IDLE || state == REASONING_BUDGET_DONE;
}

void replay_generation_prefill(reasoning_gate_sampler_context *context) {
  for (const llama_token token : context->generation_prefill) {
    llama_sampler_accept(context->reasoning_budget.get(), token);
  }
}

const char *reasoning_gate_sampler_name(const llama_sampler *) {
  return "reasoning-gate";
}

void reasoning_gate_sampler_accept(llama_sampler *sampler, llama_token token) {
  auto *context = static_cast<reasoning_gate_sampler_context *>(sampler->ctx);
  // The token that completes a reasoning marker belongs to the state that
  // existed before it was accepted, matching common_sampler.
  const bool accept_grammar = grammar_should_apply(context);
  llama_sampler_accept(context->reasoning_budget.get(), token);
  const bool is_done = common_reasoning_budget_get_state(
                           context->reasoning_budget.get()) ==
                       REASONING_BUDGET_DONE;
  if (context->lazy_grammar != nullptr && !accept_grammar && is_done) {
    const llama_tokens *end_match = common_reasoning_budget_get_end_match(
        context->reasoning_budget.get());
    if (end_match != nullptr) {
      for (const llama_token end_token : *end_match) {
        llama_sampler_accept(context->lazy_grammar.get(), end_token);
      }
    }
  }
  if (accept_grammar) {
    llama_sampler_accept(context->lazy_grammar.get(), token);
  }
}

void reasoning_gate_sampler_apply(llama_sampler *sampler,
                                  llama_token_data_array *candidates) {
  auto *context = static_cast<reasoning_gate_sampler_context *>(sampler->ctx);
  llama_sampler_apply(context->reasoning_budget.get(), candidates);
  if (grammar_should_apply(context)) {
    llama_sampler_apply(context->lazy_grammar.get(), candidates);
  }
}

void reasoning_gate_sampler_reset(llama_sampler *sampler) {
  auto *context = static_cast<reasoning_gate_sampler_context *>(sampler->ctx);
  llama_sampler_reset(context->reasoning_budget.get());
  llama_sampler_reset(context->lazy_grammar.get());
  replay_generation_prefill(context);
}

void reasoning_gate_sampler_free(llama_sampler *sampler) {
  delete static_cast<reasoning_gate_sampler_context *>(sampler->ctx);
}

llama_sampler *reasoning_gate_sampler_clone(const llama_sampler *sampler);

llama_sampler_i reasoning_gate_sampler_interface = {
    /* .name              = */ reasoning_gate_sampler_name,
    /* .accept            = */ reasoning_gate_sampler_accept,
    /* .apply             = */ reasoning_gate_sampler_apply,
    /* .reset             = */ reasoning_gate_sampler_reset,
    /* .clone             = */ reasoning_gate_sampler_clone,
    /* .free              = */ reasoning_gate_sampler_free,
    /* .backend_init      = */ nullptr,
    /* .backend_accept    = */ nullptr,
    /* .backend_apply     = */ nullptr,
    /* .backend_set_input = */ nullptr,
};

llama_sampler *create_reasoning_gate_sampler(
    llama_sampler_ptr reasoning_budget, llama_sampler_ptr lazy_grammar,
    std::vector<llama_token> generation_prefill, bool replay_prefill) {
  if (reasoning_budget == nullptr) {
    return nullptr;
  }
  try {
    auto context = std::make_unique<reasoning_gate_sampler_context>(
        reasoning_gate_sampler_context{
            std::move(reasoning_budget),
            std::move(lazy_grammar),
            std::move(generation_prefill),
        });
    if (replay_prefill) {
      replay_generation_prefill(context.get());
    }
    reasoning_gate_sampler_context *raw_context = context.release();
    try {
      return llama_sampler_init(&reasoning_gate_sampler_interface,
                                raw_context);
    } catch (...) {
      delete raw_context;
      return nullptr;
    }
  } catch (...) {
    return nullptr;
  }
}

llama_sampler *reasoning_gate_sampler_clone(const llama_sampler *sampler) {
  const auto *context =
      static_cast<const reasoning_gate_sampler_context *>(sampler->ctx);
  try {
    llama_sampler_ptr reasoning_budget(
        llama_sampler_clone(context->reasoning_budget.get()));
    llama_sampler_ptr lazy_grammar(
        context->lazy_grammar == nullptr
            ? nullptr
            : llama_sampler_clone(context->lazy_grammar.get()));
    return create_reasoning_gate_sampler(
        std::move(reasoning_budget), std::move(lazy_grammar),
        context->generation_prefill,
        /* replay_prefill = */ false);
  } catch (...) {
    return nullptr;
  }
}

} // namespace

llama_sampler *init_reasoning_gate_sampler(
    llama_sampler *reasoning_budget, llama_sampler *lazy_grammar,
    std::vector<llama_token> generation_prefill) {
  return create_reasoning_gate_sampler(
      llama_sampler_ptr(reasoning_budget), llama_sampler_ptr(lazy_grammar),
      std::move(generation_prefill),
      /* replay_prefill = */ true);
}

} // namespace llama_dart_bridge_internal

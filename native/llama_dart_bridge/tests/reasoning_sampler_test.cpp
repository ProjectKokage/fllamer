#include "reasoning_sampler.h"
#include "sampling_policy.h"

#include "reasoning-budget.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>
#include <climits>
#include <cmath>
#include <utility>
#include <vector>

namespace {

struct mock_sampler_metrics {
  int apply_count = 0;
  int reset_count = 0;
  int free_count = 0;
  std::vector<llama_token> accepted;
};

struct mock_sampler_context {
  mock_sampler_metrics *metrics;
  std::vector<llama_token> rejected;
  size_t accepted_count = 0;
};

const char *mock_sampler_name(const llama_sampler *) {
  return "reasoning-test-mock";
}

void mock_sampler_accept(llama_sampler *sampler, llama_token token) {
  auto *context = static_cast<mock_sampler_context *>(sampler->ctx);
  context->metrics->accepted.push_back(token);
  context->accepted_count++;
}

void mock_sampler_apply(llama_sampler *sampler,
                        llama_token_data_array *candidates) {
  auto *context = static_cast<mock_sampler_context *>(sampler->ctx);
  context->metrics->apply_count++;
  for (size_t i = 0; i < candidates->size; ++i) {
    for (const llama_token rejected : context->rejected) {
      if (candidates->data[i].id == rejected) {
        candidates->data[i].logit = -INFINITY;
      }
    }
    // A tiny stateful behavior lets clone/reset tests verify that the wrapper
    // preserves child sampler state instead of merely cloning its type.
    if (context->accepted_count > 0 && candidates->data[i].id == 77) {
      candidates->data[i].logit = -INFINITY;
    }
  }
}

void mock_sampler_reset(llama_sampler *sampler) {
  auto *context = static_cast<mock_sampler_context *>(sampler->ctx);
  context->metrics->reset_count++;
  context->accepted_count = 0;
}

llama_sampler *mock_sampler_clone(const llama_sampler *sampler);

void mock_sampler_free(llama_sampler *sampler) {
  auto *context = static_cast<mock_sampler_context *>(sampler->ctx);
  context->metrics->free_count++;
  delete context;
}

llama_sampler_i mock_sampler_interface = {
    /* .name              = */ mock_sampler_name,
    /* .accept            = */ mock_sampler_accept,
    /* .apply             = */ mock_sampler_apply,
    /* .reset             = */ mock_sampler_reset,
    /* .clone             = */ mock_sampler_clone,
    /* .free              = */ mock_sampler_free,
    /* .backend_init      = */ nullptr,
    /* .backend_accept    = */ nullptr,
    /* .backend_apply     = */ nullptr,
    /* .backend_set_input = */ nullptr,
};

llama_sampler *init_mock_sampler(mock_sampler_metrics *metrics,
                                 std::vector<llama_token> rejected = {},
                                 size_t accepted_count = 0) {
  return llama_sampler_init(
      &mock_sampler_interface,
      new mock_sampler_context{
          metrics,
          std::move(rejected),
          accepted_count,
      });
}

llama_sampler *mock_sampler_clone(const llama_sampler *sampler) {
  const auto *context = static_cast<const mock_sampler_context *>(sampler->ctx);
  return init_mock_sampler(context->metrics, context->rejected,
                           context->accepted_count);
}

struct candidates {
  explicit candidates(std::initializer_list<llama_token> tokens) {
    for (const llama_token token : tokens) {
      data.push_back({token, 0.0f, 0.0f});
    }
    array = {data.data(), data.size(), -1, false};
  }

  float logit(llama_token token) const {
    for (const llama_token_data &candidate : data) {
      if (candidate.id == token) {
        return candidate.logit;
      }
    }
    assert(false);
    return -INFINITY;
  }

  std::vector<llama_token_data> data;
  llama_token_data_array array{};
};

llama_sampler *init_test_sampler(
    int32_t budget, const std::vector<llama_token> &prefill,
    mock_sampler_metrics *grammar_metrics,
    mock_sampler_metrics *sampling_metrics,
    llama_sampler **out_sampling = nullptr,
    common_reasoning_budget_state initial_state = REASONING_BUDGET_IDLE,
    bool include_lazy_grammar = true,
    std::vector<llama_tokens> end_sequences = {{12, 13}}) {
  const std::vector<llama_token> start{10, 11};
  assert(!end_sequences.empty());
  llama_sampler *reasoning = common_reasoning_budget_init(
      nullptr, {start}, end_sequences, end_sequences.front(), budget,
      initial_state);
  llama_sampler *grammar = include_lazy_grammar
                               ? init_mock_sampler(
                                     grammar_metrics,
                                     /* rejected = */ {12, 13})
                               : nullptr;
  llama_sampler *sampling = init_mock_sampler(sampling_metrics);
  if (out_sampling != nullptr) {
    *out_sampling = sampling;
  }
  llama_sampler *gate =
      llama_dart_bridge_internal::init_reasoning_gate_sampler(
          reasoning, grammar, prefill);
  if (gate == nullptr) {
    llama_sampler_free(sampling);
    return nullptr;
  }
  llama_sampler_chain_params params = llama_sampler_chain_default_params();
  params.no_perf = true;
  llama_sampler *sampler = llama_sampler_chain_init(params);
  if (sampler == nullptr) {
    llama_sampler_free(gate);
    llama_sampler_free(sampling);
    return nullptr;
  }
  llama_sampler_chain_add(sampler, gate);
  llama_sampler_chain_add(sampler, sampling);
  assert(llama_sampler_chain_n(sampler) == 2);
  return sampler;
}

void assert_forced(llama_sampler *sampler, llama_token token) {
  candidates current{12, 13, 20, 99};
  llama_sampler_apply(sampler, &current.array);
  assert(std::isfinite(current.logit(token)));
  for (const llama_token other : {12, 13, 20, 99}) {
    if (other != token) {
      assert(current.logit(other) == -INFINITY);
    }
  }
}

void test_model_suppress_tokens_sampler() {
  const llama_token suppressed[] = {13, 99};
  llama_sampler *sampler =
      llama_dart_bridge_internal::init_suppress_tokens_sampler(
          128, suppressed, 2);
  assert(sampler != nullptr);

  candidates current{12, 13, 20, 99};
  llama_sampler_apply(sampler, &current.array);
  assert(std::isfinite(current.logit(12)));
  assert(current.logit(13) == -INFINITY);
  assert(std::isfinite(current.logit(20)));
  assert(current.logit(99) == -INFINITY);

  llama_sampler_free(sampler);
  assert(llama_dart_bridge_internal::init_suppress_tokens_sampler(
             128, nullptr, 0) == nullptr);
}

void test_multitoken_forced_close_and_lazy_grammar() {
  mock_sampler_metrics grammar;
  mock_sampler_metrics sampling;
  llama_sampler *sampler = init_test_sampler(2, {10, 11}, &grammar, &sampling);
  assert(sampler != nullptr);

  candidates private_candidates{12, 13, 20};
  llama_sampler_apply(sampler, &private_candidates.array);
  assert(grammar.apply_count == 0);
  assert(sampling.apply_count == 1);

  llama_sampler_accept(sampler, 20);
  llama_sampler_accept(sampler, 21);
  assert(grammar.accepted.empty());
  assert((sampling.accepted == std::vector<llama_token>{20, 21}));

  assert_forced(sampler, 12);
  assert(grammar.apply_count == 0);
  llama_sampler_accept(sampler, 12);
  assert_forced(sampler, 13);
  llama_sampler_accept(sampler, 13);
  assert((grammar.accepted == std::vector<llama_token>{12, 13}));

  candidates public_candidates{20, 99};
  llama_sampler_apply(sampler, &public_candidates.array);
  assert(grammar.apply_count == 1);
  llama_sampler_accept(sampler, 99);
  assert((grammar.accepted == std::vector<llama_token>{12, 13, 99}));

  llama_sampler_free(sampler);
  assert(grammar.free_count == 1);
  assert(sampling.free_count == 1);
}

void test_natural_multitoken_close() {
  mock_sampler_metrics grammar;
  mock_sampler_metrics sampling;
  llama_sampler *sampler = init_test_sampler(5, {10, 11}, &grammar, &sampling);
  assert(sampler != nullptr);

  llama_sampler_accept(sampler, 20);
  llama_sampler_accept(sampler, 12);
  candidates partial_close_candidates{13, 20};
  llama_sampler_apply(sampler, &partial_close_candidates.array);
  assert(grammar.apply_count == 0);
  llama_sampler_accept(sampler, 13);
  assert((grammar.accepted == std::vector<llama_token>{12, 13}));

  llama_sampler_accept(sampler, 99);
  assert((grammar.accepted == std::vector<llama_token>{12, 13, 99}));
  assert((sampling.accepted == std::vector<llama_token>{20, 12, 13, 99}));

  llama_sampler_free(sampler);
}

void test_alternative_natural_close() {
  mock_sampler_metrics grammar;
  mock_sampler_metrics sampling;
  llama_sampler *sampler = init_test_sampler(
      5, {10, 11}, &grammar, &sampling, nullptr, REASONING_BUDGET_IDLE,
      /* include_lazy_grammar = */ true,
      /* end_sequences = */ {{12, 13}, {30, 31}});
  assert(sampler != nullptr);

  llama_sampler_accept(sampler, 20);
  llama_sampler_accept(sampler, 30);
  llama_sampler_accept(sampler, 31);
  assert((grammar.accepted == std::vector<llama_token>{30, 31}));

  llama_sampler_accept(sampler, 99);
  assert((grammar.accepted == std::vector<llama_token>{30, 31, 99}));

  llama_sampler_free(sampler);
}

void test_generation_prefill_consumes_budget() {
  mock_sampler_metrics grammar;
  mock_sampler_metrics sampling;
  llama_sampler *sampler =
      init_test_sampler(2, {7, 10, 11, 20}, &grammar, &sampling);
  assert(sampler != nullptr);
  assert(grammar.accepted.empty());
  assert(sampling.accepted.empty());

  llama_sampler_accept(sampler, 21);
  assert_forced(sampler, 12);
  assert(grammar.apply_count == 0);

  llama_sampler_free(sampler);
}

void test_generated_zero_budget_start_and_second_block() {
  mock_sampler_metrics grammar;
  mock_sampler_metrics sampling;
  llama_sampler *sampler = init_test_sampler(0, {}, &grammar, &sampling);
  assert(sampler != nullptr);

  llama_sampler_accept(sampler, 10);
  llama_sampler_accept(sampler, 11);
  assert((grammar.accepted == std::vector<llama_token>{10, 11}));
  assert_forced(sampler, 12);
  llama_sampler_accept(sampler, 12);
  assert_forced(sampler, 13);
  llama_sampler_accept(sampler, 13);

  // A zero-budget close must not truncate the public response that follows.
  candidates public_candidates{20, 99};
  llama_sampler_apply(sampler, &public_candidates.array);
  assert(std::isfinite(public_candidates.logit(20)));
  assert(std::isfinite(public_candidates.logit(99)));
  llama_sampler_accept(sampler, 99);
  assert((sampling.accepted ==
          std::vector<llama_token>{10, 11, 12, 13, 99}));

  // llama.cpp re-arms the budget when a model emits a later reasoning block.
  llama_sampler_accept(sampler, 10);
  llama_sampler_accept(sampler, 11);
  assert((grammar.accepted ==
          std::vector<llama_token>{10, 11, 12, 13, 99, 10, 11}));
  assert_forced(sampler, 12);

  llama_sampler_free(sampler);
}

void test_reasoning_without_lazy_grammar() {
  mock_sampler_metrics unused_grammar;
  mock_sampler_metrics sampling;
  llama_sampler *sampler = init_test_sampler(
      1, {10, 11}, &unused_grammar, &sampling, nullptr,
      REASONING_BUDGET_IDLE,
      /* include_lazy_grammar = */ false);
  assert(sampler != nullptr);

  llama_sampler_accept(sampler, 20);
  assert_forced(sampler, 12);
  assert(unused_grammar.apply_count == 0);
  assert(unused_grammar.accepted.empty());
  assert(unused_grammar.free_count == 0);

  llama_sampler_free(sampler);
  assert(sampling.free_count == 1);
}

void test_prompt_history_does_not_prime_reasoning() {
  mock_sampler_metrics grammar;
  mock_sampler_metrics sampling;
  llama_sampler *sampling_child = nullptr;
  llama_sampler *sampler =
      init_test_sampler(2, {7}, &grammar, &sampling, &sampling_child);
  assert(sampler != nullptr);

  // Prompt/history tokens are accepted by the ordinary child, not by the
  // reasoning wrapper. Marker-shaped user content must not activate a budget.
  llama_sampler_accept(sampling_child, 10);
  llama_sampler_accept(sampling_child, 11);
  candidates public_candidates{20, 99};
  llama_sampler_apply(sampler, &public_candidates.array);
  assert(grammar.apply_count == 1);

  llama_sampler_free(sampler);
}

void test_reset_and_clone_preserve_prefill_state() {
  mock_sampler_metrics grammar;
  mock_sampler_metrics sampling;
  llama_sampler *sampler = init_test_sampler(2, {10, 11}, &grammar, &sampling);
  assert(sampler != nullptr);

  llama_sampler_accept(sampler, 20);
  llama_sampler *clone = llama_sampler_clone(sampler);
  assert(clone != nullptr);
  candidates cloned_child_state{20, 77};
  llama_sampler_apply(clone, &cloned_child_state.array);
  assert(cloned_child_state.logit(77) == -INFINITY);

  llama_sampler_accept(sampler, 21);
  assert_forced(sampler, 12);
  candidates clone_before_exhaustion{12, 13, 20};
  llama_sampler_apply(clone, &clone_before_exhaustion.array);
  assert(std::isfinite(clone_before_exhaustion.logit(20)));
  llama_sampler_accept(clone, 22);
  assert_forced(clone, 12);

  llama_sampler_accept(sampler, 12);
  llama_sampler *forcing_clone = llama_sampler_clone(sampler);
  assert(forcing_clone != nullptr);
  assert_forced(sampler, 13);
  assert_forced(forcing_clone, 13);

  llama_sampler_reset(sampler);
  assert(grammar.reset_count == 1);
  assert(sampling.reset_count == 1);
  candidates after_reset{12, 13, 20};
  llama_sampler_apply(sampler, &after_reset.array);
  assert(std::isfinite(after_reset.logit(20)));
  assert(grammar.apply_count == 0);
  candidates reset_child_state{20, 77};
  llama_sampler_apply(sampler, &reset_child_state.array);
  assert(std::isfinite(reset_child_state.logit(77)));
  llama_sampler_accept(sampler, 23);
  llama_sampler_accept(sampler, 24);
  assert_forced(sampler, 12);

  llama_sampler_free(forcing_clone);
  llama_sampler_free(clone);
  llama_sampler_free(sampler);
  assert(grammar.free_count == 3);
  assert(sampling.free_count == 3);
}

void test_waiting_utf8_and_unlimited_budget_suppress_grammar() {
  mock_sampler_metrics waiting_grammar;
  mock_sampler_metrics waiting_sampling;
  llama_sampler *waiting =
      init_test_sampler(1, {}, &waiting_grammar, &waiting_sampling, nullptr,
                        REASONING_BUDGET_WAITING_UTF8);
  assert(waiting != nullptr);

  candidates waiting_candidates{12, 13, 20};
  llama_sampler_apply(waiting, &waiting_candidates.array);
  assert(waiting_grammar.apply_count == 0);
  llama_sampler_accept(waiting, 20);
  assert_forced(waiting, 12);
  llama_sampler_free(waiting);

  mock_sampler_metrics unlimited_grammar;
  mock_sampler_metrics unlimited_sampling;
  llama_sampler *unlimited = init_test_sampler(
      INT_MAX, {10, 11}, &unlimited_grammar, &unlimited_sampling);
  assert(unlimited != nullptr);
  candidates unlimited_candidates{12, 13, 20};
  llama_sampler_apply(unlimited, &unlimited_candidates.array);
  assert(unlimited_grammar.apply_count == 0);
  llama_sampler_accept(unlimited, 20);
  assert(unlimited_grammar.accepted.empty());
  llama_sampler_free(unlimited);
}

} // namespace

int main() {
  test_model_suppress_tokens_sampler();
  test_multitoken_forced_close_and_lazy_grammar();
  test_natural_multitoken_close();
  test_alternative_natural_close();
  test_generation_prefill_consumes_budget();
  test_generated_zero_budget_start_and_second_block();
  test_reasoning_without_lazy_grammar();
  test_prompt_history_does_not_prime_reasoning();
  test_reset_and_clone_preserve_prefill_state();
  test_waiting_utf8_and_unlimited_budget_suppress_grammar();
  return 0;
}

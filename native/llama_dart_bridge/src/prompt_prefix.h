#ifndef LLAMA_DART_BRIDGE_PROMPT_PREFIX_H_
#define LLAMA_DART_BRIDGE_PROMPT_PREFIX_H_

#include <algorithm>
#include <cstddef>
#include <vector>

namespace llama_dart_bridge_internal {

struct prompt_prefix_reuse_plan {
  bool exact_prefix;
  size_t suffix_start;
};

template <typename Token>
prompt_prefix_reuse_plan resolve_prompt_prefix_reuse(
    const std::vector<Token> &committed,
    const std::vector<Token> &prompt) {
  if (committed.size() > prompt.size() ||
      !std::equal(committed.begin(), committed.end(), prompt.begin())) {
    return {false, 0};
  }
  return {true, committed.size()};
}

// Where a chat prompt's generation prompt begins: the number of prompt tokens
// before `generation`, or 0 when the prompt does not end with those tokens.
template <typename Token>
size_t generation_prompt_boundary(const std::vector<Token> &prompt,
                                  const std::vector<Token> &generation) {
  if (generation.empty() || generation.size() >= prompt.size() ||
      !std::equal(generation.begin(), generation.end(),
                  prompt.end() - static_cast<std::ptrdiff_t>(
                                     generation.size()))) {
    return 0;
  }
  return prompt.size() - generation.size();
}

// Whether a prompt can continue from a kept boundary: the context still holds
// the boundary's tokens at its start, and the prompt starts with them and goes
// on past them.
template <typename Token>
bool prompt_continues_boundary(const std::vector<Token> &committed,
                               const std::vector<Token> &boundary,
                               const std::vector<Token> &prompt) {
  return !boundary.empty() && boundary.size() <= committed.size() &&
         boundary.size() < prompt.size() &&
         std::equal(boundary.begin(), boundary.end(), committed.begin()) &&
         std::equal(boundary.begin(), boundary.end(), prompt.begin());
}

}  // namespace llama_dart_bridge_internal

#endif  // LLAMA_DART_BRIDGE_PROMPT_PREFIX_H_

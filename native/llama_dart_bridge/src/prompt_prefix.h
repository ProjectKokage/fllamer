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

}  // namespace llama_dart_bridge_internal

#endif  // LLAMA_DART_BRIDGE_PROMPT_PREFIX_H_

#ifndef LLAMA_DART_BRIDGE_CHAT_PROMPT_RUNS_H_
#define LLAMA_DART_BRIDGE_CHAT_PROMPT_RUNS_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace llama_dart_bridge_internal {

// A range of bytes of a prompt, end exclusive.
struct byte_span {
  size_t begin;
  size_t end;
};

// A range of a prompt that is tokenized in one call, and whether that call
// parses special tokens.
struct prompt_run {
  size_t begin;
  size_t end;
  bool parse_special;
};

// The texts of a vocabulary's control and unknown tokens: the tokens the
// tokenizer produces from text only when it parses special tokens.
class control_token_texts {
public:
  void add(std::string text) {
    if (text.empty()) {
      return;
    }
    longest_ = std::max(longest_, text.size());
    by_first_byte_[static_cast<unsigned char>(text.front())].push_back(
        std::move(text));
  }

  // Whether `prompt` holds the text of one of the tokens at a place that
  // shares at least one byte with `span`.
  bool spelled_over(std::string_view prompt, byte_span span) const {
    if (longest_ == 0 || span.begin >= span.end) {
      return false;
    }
    const size_t first =
        span.begin >= longest_ - 1 ? span.begin - (longest_ - 1) : 0;
    for (size_t position = first; position < span.end; ++position) {
      const std::vector<std::string> &candidates =
          by_first_byte_[static_cast<unsigned char>(prompt[position])];
      for (const std::string &text : candidates) {
        if (position + text.size() > span.begin &&
            prompt.compare(position, text.size(), text) == 0) {
          return true;
        }
      }
    }
    return false;
  }

private:
  std::array<std::vector<std::string>, 256> by_first_byte_;
  size_t longest_ = 0;
};

// Whether `spans` are non-empty ranges of a prompt of `prompt_size` bytes, in
// order and apart from each other.
inline bool valid_message_text_spans(const std::vector<byte_span> &spans,
                                     size_t prompt_size) {
  size_t previous_end = 0;
  for (const byte_span &span : spans) {
    if (span.begin < previous_end || span.begin >= span.end ||
        span.end > prompt_size) {
      return false;
    }
    previous_end = span.end;
  }
  return true;
}

// How a chat prompt is tokenized when the template's own text may hold
// special tokens and the text of its messages may not.
//
// `message_text_spans` are the ranges of `prompt` that came from the chat's
// messages. A range over which the prompt spells a control token becomes a
// run of its own that is tokenized without special-token parsing, so that
// text stays text. Everything else, the template's text and the message text
// that spells no such token, stays in runs that are tokenized with parsing on,
// exactly as the whole prompt would be. A prompt in which no message spells a
// control token is therefore one run, and its tokens are those of the joined
// prompt.
inline std::vector<prompt_run> chat_prompt_runs(
    const control_token_texts &control_tokens, std::string_view prompt,
    const std::vector<byte_span> &message_text_spans) {
  std::vector<prompt_run> runs;
  size_t cursor = 0;
  for (const byte_span &span : message_text_spans) {
    if (!control_tokens.spelled_over(prompt, span)) {
      continue;
    }
    if (!runs.empty() && !runs.back().parse_special &&
        runs.back().end == span.begin) {
      runs.back().end = span.end;
    } else {
      if (span.begin > cursor) {
        runs.push_back({cursor, span.begin, true});
      }
      runs.push_back({span.begin, span.end, false});
    }
    cursor = span.end;
  }
  if (cursor < prompt.size() || runs.empty()) {
    runs.push_back({cursor, prompt.size(), true});
  }
  return runs;
}

}  // namespace llama_dart_bridge_internal

#endif  // LLAMA_DART_BRIDGE_CHAT_PROMPT_RUNS_H_

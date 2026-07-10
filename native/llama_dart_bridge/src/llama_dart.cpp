#include "llama_dart.h"

#include "ggml-backend.h"
#include "llama.h"
#include "chat.h"
#include "common.h"
#include "json-schema-to-grammar.h"
#include "log.h"
#include "mtmd-helper.h"
#include "mtmd.h"
#include "speculative.h"
#include "state_snapshot.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <vector>

#ifndef LLAMA_DART_UPSTREAM_COMMIT
#define LLAMA_DART_UPSTREAM_COMMIT "unknown"
#endif

#ifndef LLAMA_DART_BUILD_FLAGS
#define LLAMA_DART_BUILD_FLAGS "unknown"
#endif

#ifndef LLAMA_DART_HAS_METAL
#define LLAMA_DART_HAS_METAL 0
#endif

#ifndef LLAMA_DART_HAS_VULKAN
#define LLAMA_DART_HAS_VULKAN 0
#endif

struct llama_dart_lora_adapter;

struct llama_dart_model {
  llama_model *model = nullptr;
  size_t active_contexts = 0;
  size_t active_lora_adapters = 0;
  uint32_t gpu_backend = LLAMA_DART_GPU_BACKEND_CPU;
  int32_t n_gpu_layers = 0;
  bool use_mmap = true;
  bool use_mlock = false;
  bool check_tensors = true;
  bool vocab_only = false;
};

struct llama_dart_context {
  llama_context *context = nullptr;
  mtmd_context *multimodal = nullptr;
  llama_dart_model *model = nullptr;
  common_init_speculative_result_ptr speculative_init;
  common_speculative_ptr speculative;
  common_params_speculative speculative_params;
  llama_context *speculative_context = nullptr;
  uint32_t speculative_type = LLAMA_DART_SPECULATIVE_NONE;
  uint32_t speculative_draft_max = 0;
  bool speculative_needs_warmup = false;
  int32_t position = 0;
  std::vector<llama_token> token_history;
  uint32_t speculative_ngram_n = 0;
  uint32_t speculative_ngram_m = 0;
  uint32_t kv_cache_key_type = LLAMA_DART_KV_CACHE_DEFAULT;
  uint32_t kv_cache_value_type = LLAMA_DART_KV_CACHE_DEFAULT;
  uint32_t flash_attention = LLAMA_DART_FLASH_ATTENTION_AUTO;
  bool kv_cache_offload = true;
  bool swa_full = true;
  bool kv_unified = false;
  size_t active_generations = 0;
  std::vector<llama_dart_lora_adapter *> lora_adapters;
  std::atomic_bool cancel_requested{false};
};

struct llama_dart_generation {
  llama_dart_context *context = nullptr;
  const llama_vocab *vocab = nullptr;
  llama_sampler *sampler = nullptr;
  uint32_t max_tokens = 0;
  uint32_t prompt_tokens = 0;
  uint32_t generated_tokens = 0;
  uint32_t speculative_draft_tokens = 0;
  uint32_t speculative_accepted_tokens = 0;
  double speculative_draft_ms = 0.0;
  double speculative_verify_ms = 0.0;
  double prompt_eval_ms = 0.0;
  double time_to_first_token_ms = 0.0;
  std::chrono::steady_clock::time_point total_start;
  std::chrono::steady_clock::time_point decode_start;
  std::vector<std::string> stop_sequences;
  std::vector<llama_token> stop_tokens;
  std::string generated;
  size_t emitted_size = 0;
  size_t stop_holdback = 0;
  bool done = false;
};

struct llama_dart_lora_adapter {
  llama_adapter_lora *adapter = nullptr;
  llama_dart_model *model = nullptr;
  size_t active_contexts = 0;
};

namespace {
using llama_dart_bridge_internal::decode_state_snapshot;
using llama_dart_bridge_internal::finalize_state_snapshot;
using llama_dart_bridge_internal::kStateSnapshotHeaderSize;
using llama_dart_bridge_internal::read_u32_le;
using llama_dart_bridge_internal::state_snapshot_decode_result;
using llama_dart_bridge_internal::state_snapshot_layout;
using llama_dart_bridge_internal::state_snapshot_size;
using llama_dart_bridge_internal::state_snapshot_view;
using llama_dart_bridge_internal::write_state_snapshot_header;
using llama_dart_bridge_internal::write_u32_le;
using steady_clock = std::chrono::steady_clock;

static_assert(sizeof(llama_token) == sizeof(int32_t));

struct last_error_storage {
  static constexpr size_t capacity = 4096;

  void clear() noexcept { data[0] = '\0'; }

  void assign_parts(const char *first, const char *second = nullptr,
                    const char *third = nullptr,
                    const char *fourth = nullptr) noexcept {
    size_t written = 0;
    const char *parts[] = {first, second, third, fourth};
    for (const char *part : parts) {
      if (part == nullptr) {
        continue;
      }
      const size_t source_length = std::strlen(part);
      const size_t remaining = capacity - 1 - written;
      size_t length = std::min(source_length, remaining);
      if (length < source_length) {
        while (length > 0 &&
               (static_cast<unsigned char>(part[length]) & 0xc0u) == 0x80u) {
          length -= 1;
        }
      }
      std::memcpy(data.data() + written, part, length);
      written += length;
      if (length < source_length) {
        break;
      }
    }
    data[written] = '\0';
  }

  last_error_storage &operator=(const char *message) noexcept {
    assign_parts(message);
    return *this;
  }

  last_error_storage &operator=(const std::string &message) noexcept {
    return *this = message.c_str();
  }

  const char *c_str() const noexcept { return data.data(); }

  std::array<char, capacity> data{};
};

thread_local last_error_storage last_error;
std::mutex backend_mutex;
size_t backend_references = 0;
struct captured_log_record {
  uint32_t level;
  std::string message;
};
std::mutex log_mutex;
std::deque<captured_log_record> captured_logs;
std::atomic<uint32_t> minimum_log_level{LLAMA_DART_LOG_DISABLED};
size_t captured_log_bytes = 0;
uint64_t dropped_log_records = 0;
uint32_t continuation_log_level = LLAMA_DART_LOG_INFO;
constexpr size_t kMaxMediaBytes = 64u * 1024u * 1024u;
constexpr size_t kMaxMediaInputs = 64u;
constexpr size_t kMaxStopTokens = 1024u;
constexpr size_t kMaxCapturedLogBytes = 1024u * 1024u;
constexpr size_t kMaxCapturedLogRecords = 4096u;
constexpr int32_t kMaxModelMetadataScalarSize = 4096;
constexpr uint32_t kChatPlanVersion = 1u;

struct parsed_chat_plan {
  std::string prompt;
  std::string grammar;
  bool grammar_lazy = false;
  std::string generation_prompt;
  std::vector<common_grammar_trigger> grammar_triggers;
  std::vector<std::string> additional_stops;
  std::string parser;
  common_chat_format format = COMMON_CHAT_FORMAT_CONTENT_ONLY;
};

void capture_log_callback(ggml_log_level level, const char *text, void *) {
  if (text == nullptr || text[0] == '\0') {
    return;
  }
  try {
    std::lock_guard<std::mutex> lock(log_mutex);
    uint32_t effective_level = LLAMA_DART_LOG_INFO;
    switch (level) {
    case GGML_LOG_LEVEL_DEBUG:
      effective_level = LLAMA_DART_LOG_DEBUG;
      break;
    case GGML_LOG_LEVEL_WARN:
      effective_level = LLAMA_DART_LOG_WARNING;
      break;
    case GGML_LOG_LEVEL_ERROR:
      effective_level = LLAMA_DART_LOG_ERROR;
      break;
    case GGML_LOG_LEVEL_CONT:
      effective_level = continuation_log_level;
      break;
    case GGML_LOG_LEVEL_NONE:
    case GGML_LOG_LEVEL_INFO:
    default:
      effective_level = LLAMA_DART_LOG_INFO;
      break;
    }
    if (level != GGML_LOG_LEVEL_CONT) {
      continuation_log_level = effective_level;
    }
    const uint32_t minimum =
        minimum_log_level.load(std::memory_order_relaxed);
    if (minimum == LLAMA_DART_LOG_DISABLED || effective_level < minimum) {
      return;
    }

    const size_t text_size =
        std::min(std::strlen(text), kMaxCapturedLogBytes);
    captured_log_record record{
        effective_level, std::string(text, text_size)};
    while (!captured_logs.empty() &&
           (captured_logs.size() >= kMaxCapturedLogRecords ||
            captured_log_bytes + record.message.size() >
                kMaxCapturedLogBytes)) {
      captured_log_bytes -= captured_logs.front().message.size();
      captured_logs.pop_front();
      dropped_log_records += 1;
    }
    captured_log_bytes += record.message.size();
    captured_logs.push_back(std::move(record));
  } catch (...) {
    // Logging must never unwind through an upstream native callback.
  }
}

llama_dart_result fail(llama_dart_result result, const char *message) noexcept {
  last_error = message == nullptr ? "native bridge error" : message;
  return result;
}

llama_dart_result fail(llama_dart_result result,
                       const std::string &message) noexcept {
  return fail(result, message.c_str());
}

llama_dart_result fail_parts(llama_dart_result result, const char *first,
                             const char *second = nullptr,
                             const char *third = nullptr,
                             const char *fourth = nullptr) noexcept {
  last_error.assign_parts(first, second, third, fourth);
  return result;
}

llama_dart_result backend_retain() noexcept {
  try {
    std::lock_guard<std::mutex> lock(backend_mutex);
    if (backend_references == 0) {
      llama_log_set(capture_log_callback, nullptr);
      mtmd_helper_log_set(capture_log_callback, nullptr);
      common_log_set_verbosity_thold(-1);
      llama_backend_init();
    }
    backend_references += 1;
    last_error.clear();
    return LLAMA_DART_SUCCESS;
  } catch (const std::exception &error) {
    return fail(LLAMA_DART_ERROR_INTERNAL, error.what());
  } catch (...) {
    return fail(LLAMA_DART_ERROR_INTERNAL, "unknown backend init failure");
  }
}

llama_dart_result backend_release() noexcept {
  try {
    std::lock_guard<std::mutex> lock(backend_mutex);
    if (backend_references == 0) {
      return LLAMA_DART_SUCCESS;
    }
    backend_references -= 1;
    if (backend_references == 0) {
      llama_backend_free();
    }
    return LLAMA_DART_SUCCESS;
  } catch (const std::exception &error) {
    return fail(LLAMA_DART_ERROR_INTERNAL, error.what());
  } catch (...) {
    return fail(LLAMA_DART_ERROR_INTERNAL, "unknown backend free failure");
  }
}

llama_dart_result validate_struct(uint32_t actual, size_t expected,
                                  const char *name) {
  if (actual < expected) {
    return fail_parts(LLAMA_DART_ERROR_INVALID_ARGUMENT, name,
                      " struct_size is too small");
  }
  return LLAMA_DART_SUCCESS;
}

bool read_model_metadata(const llama_model *model, const std::string &key,
                         std::string *out) {
  const int32_t required =
      llama_model_meta_val_str(model, key.c_str(), nullptr, 0);
  if (required < 0 || required > kMaxModelMetadataScalarSize) {
    return false;
  }
  std::vector<char> buffer(static_cast<size_t>(required) + 1);
  const int32_t copied = llama_model_meta_val_str(
      model, key.c_str(), buffer.data(), buffer.size());
  if (copied != required) {
    return false;
  }
  out->assign(buffer.data(), static_cast<size_t>(copied));
  return true;
}

bool read_model_metadata_int32(const llama_model *model,
                               const std::string &key, int32_t *out) {
  std::string value;
  if (!read_model_metadata(model, key, &value) || value.empty()) {
    return false;
  }
  int32_t parsed = 0;
  const std::from_chars_result result =
      std::from_chars(value.data(), value.data() + value.size(), parsed);
  if (result.ec != std::errc() || result.ptr != value.data() + value.size() ||
      parsed < 0) {
    return false;
  }
  *out = parsed;
  return true;
}

void release_context_lora_adapters(llama_dart_context *context) {
  for (llama_dart_lora_adapter *adapter : context->lora_adapters) {
    if (adapter != nullptr && adapter->active_contexts > 0) {
      adapter->active_contexts -= 1;
    }
  }
  context->lora_adapters.clear();
}

bool fits_int32(size_t value) {
  return value <= static_cast<size_t>(std::numeric_limits<int32_t>::max());
}

bool contains_nul(const uint8_t *data, size_t size) {
  return size > 0 &&
         std::memchr(data, '\0', size) != nullptr;
}

bool contains_line_break(const uint8_t *data, size_t size) {
  return size > 0 &&
         (std::memchr(data, '\n', size) != nullptr ||
          std::memchr(data, '\r', size) != nullptr);
}

bool contains_ascii_whitespace(const uint8_t *data, size_t size) {
  if (data == nullptr || size == 0) {
    return false;
  }
  for (size_t i = 0; i < size; ++i) {
    switch (data[i]) {
    case ' ':
    case '\t':
    case '\n':
    case '\r':
    case '\f':
    case '\v':
      return true;
    default:
      break;
    }
  }
  return false;
}

bool is_ascii_blank(const uint8_t *data, size_t size) {
  if (data == nullptr || size == 0) {
    return true;
  }
  for (size_t i = 0; i < size; ++i) {
    switch (data[i]) {
    case ' ':
    case '\t':
    case '\n':
    case '\r':
    case '\f':
    case '\v':
      break;
    default:
      return false;
    }
  }
  return true;
}

void clear_char_buffer(char *buffer, size_t size) {
  if (buffer != nullptr && size > 0) {
    buffer[0] = '\0';
  }
}

void clear_byte_buffer(uint8_t *buffer, size_t size) {
  if (buffer != nullptr && size > 0) {
    buffer[0] = 0;
  }
}

void clear_token_buffer(int32_t *buffer, size_t size) {
  if (buffer != nullptr && size > 0) {
    buffer[0] = 0;
  }
}

bool valid_bool(uint8_t value) { return value == 0 || value == 1; }

bool string_contains_nul(const std::string &value) {
  return value.find('\0') != std::string::npos;
}

llama_dart_result parse_json_object(const uint8_t *data, size_t size,
                                    const char *name,
                                    nlohmann::ordered_json *out) {
  if (data == nullptr || size == 0) {
    return fail_parts(LLAMA_DART_ERROR_INVALID_ARGUMENT, name,
                      " must not be empty");
  }
  if (!fits_int32(size)) {
    return fail_parts(LLAMA_DART_ERROR_INVALID_ARGUMENT, name,
                      " is too large");
  }
  if (contains_nul(data, size) || is_ascii_blank(data, size)) {
    return fail_parts(LLAMA_DART_ERROR_INVALID_ARGUMENT, name,
                      " must contain non-blank JSON without NUL");
  }
  try {
    const char *begin = reinterpret_cast<const char *>(data);
    *out = nlohmann::ordered_json::parse(begin, begin + size);
    if (!out->is_object()) {
      return fail_parts(LLAMA_DART_ERROR_INVALID_ARGUMENT, name,
                        " must be a JSON object");
    }
    return LLAMA_DART_SUCCESS;
  } catch (const nlohmann::json::parse_error &error) {
    return fail_parts(LLAMA_DART_ERROR_INVALID_ARGUMENT, "invalid ", name,
                      ": ", error.what());
  } catch (const std::bad_alloc &) {
    return fail(LLAMA_DART_ERROR_INTERNAL, "native allocation failed");
  } catch (const std::exception &error) {
    return fail_parts(LLAMA_DART_ERROR_INTERNAL, "failed to parse ", name,
                      ": ", error.what());
  } catch (...) {
    return fail_parts(LLAMA_DART_ERROR_INTERNAL, "unknown ", name,
                      " parsing failure");
  }
}

nlohmann::ordered_json serialize_chat_plan(
    const common_chat_params &params) {
  nlohmann::ordered_json triggers = nlohmann::ordered_json::array();
  for (const common_grammar_trigger &trigger : params.grammar_triggers) {
    triggers.push_back({
        {"type", static_cast<int>(trigger.type)},
        {"value", trigger.value},
        {"token", trigger.token},
    });
  }
  return {
      {"version", kChatPlanVersion},
      {"prompt", params.prompt},
      {"grammar", params.grammar},
      {"grammar_lazy", params.grammar_lazy},
      {"generation_prompt", params.generation_prompt},
      {"grammar_triggers", std::move(triggers)},
      {"additional_stops", params.additional_stops},
      {"parser", params.parser},
      {"format", static_cast<int>(params.format)},
  };
}

llama_dart_result parse_chat_plan(const uint8_t *data, size_t size,
                                  parsed_chat_plan *out) {
  nlohmann::ordered_json plan;
  const llama_dart_result parsed =
      parse_json_object(data, size, "chat plan", &plan);
  if (parsed != LLAMA_DART_SUCCESS) {
    return parsed;
  }
  try {
    if (plan.at("version").get<uint32_t>() != kChatPlanVersion) {
      return fail(LLAMA_DART_ERROR_UNSUPPORTED,
                  "chat plan version is not supported");
    }
    parsed_chat_plan value;
    value.prompt = plan.at("prompt").get<std::string>();
    value.grammar = plan.at("grammar").get<std::string>();
    value.grammar_lazy = plan.at("grammar_lazy").get<bool>();
    value.generation_prompt =
        plan.at("generation_prompt").get<std::string>();
    value.additional_stops =
        plan.at("additional_stops").get<std::vector<std::string>>();
    value.parser = plan.at("parser").get<std::string>();
    const int format = plan.at("format").get<int>();
    if (format < 0 || format >= COMMON_CHAT_FORMAT_COUNT) {
      return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                  "chat plan format is invalid");
    }
    value.format = static_cast<common_chat_format>(format);

    const nlohmann::ordered_json &triggers = plan.at("grammar_triggers");
    if (!triggers.is_array()) {
      return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                  "chat plan grammar_triggers must be an array");
    }
    value.grammar_triggers.reserve(triggers.size());
    for (const nlohmann::ordered_json &item : triggers) {
      const int type = item.at("type").get<int>();
      if (type < COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN ||
          type > COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL) {
        return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                    "chat plan grammar trigger type is invalid");
      }
      common_grammar_trigger trigger;
      trigger.type = static_cast<common_grammar_trigger_type>(type);
      trigger.value = item.at("value").get<std::string>();
      trigger.token = item.at("token").get<llama_token>();
      if (string_contains_nul(trigger.value)) {
        return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                    "chat plan grammar trigger must not contain NUL");
      }
      if (trigger.type == COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN &&
          trigger.token == LLAMA_TOKEN_NULL) {
        return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                    "chat plan grammar token trigger is invalid");
      }
      value.grammar_triggers.push_back(std::move(trigger));
    }

    if (value.prompt.empty() || string_contains_nul(value.prompt) ||
        string_contains_nul(value.grammar) ||
        string_contains_nul(value.generation_prompt) ||
        string_contains_nul(value.parser)) {
      return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                  "chat plan contains invalid text");
    }
    if (value.grammar_lazy &&
        (value.grammar.empty() || value.grammar_triggers.empty())) {
      return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                  "lazy chat grammar requires grammar and triggers");
    }
    for (const std::string &stop : value.additional_stops) {
      if (stop.empty() || string_contains_nul(stop)) {
        return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                    "chat plan stop sequence is invalid");
      }
    }
    *out = std::move(value);
    last_error.clear();
    return LLAMA_DART_SUCCESS;
  } catch (const std::bad_alloc &) {
    return fail(LLAMA_DART_ERROR_INTERNAL, "native allocation failed");
  } catch (const nlohmann::json::exception &error) {
    return fail_parts(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                      "invalid chat plan: ", error.what());
  } catch (const std::exception &error) {
    return fail_parts(LLAMA_DART_ERROR_INTERNAL,
                      "failed to decode chat plan: ", error.what());
  } catch (...) {
    return fail(LLAMA_DART_ERROR_INTERNAL,
                "unknown chat plan decoding failure");
  }
}

bool truncate_at_stop_sequence(std::string *text,
                               const std::vector<std::string> &stops) {
  size_t end = std::string::npos;
  for (const std::string &stop : stops) {
    const size_t index = text->find(stop);
    if (index != std::string::npos && (end == std::string::npos || index < end)) {
      end = index;
    }
  }
  if (end == std::string::npos) {
    return false;
  }
  text->resize(end);
  return true;
}

double elapsed_ms(steady_clock::time_point start,
                  steady_clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

bool append_token_piece(const llama_vocab *vocab, llama_token token,
                        std::string *out);

bool valid_pooling_type(int32_t value) {
  switch (value) {
  case LLAMA_POOLING_TYPE_UNSPECIFIED:
  case LLAMA_POOLING_TYPE_NONE:
  case LLAMA_POOLING_TYPE_MEAN:
  case LLAMA_POOLING_TYPE_CLS:
  case LLAMA_POOLING_TYPE_LAST:
  case LLAMA_POOLING_TYPE_RANK:
    return true;
  default:
    return false;
  }
}

bool valid_attention_type(int32_t value) {
  switch (value) {
  case LLAMA_ATTENTION_TYPE_UNSPECIFIED:
  case LLAMA_ATTENTION_TYPE_CAUSAL:
  case LLAMA_ATTENTION_TYPE_NON_CAUSAL:
    return true;
  default:
    return false;
  }
}

bool valid_kv_cache_type(uint32_t value) {
  switch (value) {
  case LLAMA_DART_KV_CACHE_DEFAULT:
  case LLAMA_DART_KV_CACHE_F32:
  case LLAMA_DART_KV_CACHE_F16:
  case LLAMA_DART_KV_CACHE_BF16:
  case LLAMA_DART_KV_CACHE_Q8_0:
  case LLAMA_DART_KV_CACHE_Q4_0:
  case LLAMA_DART_KV_CACHE_Q4_1:
  case LLAMA_DART_KV_CACHE_IQ4_NL:
  case LLAMA_DART_KV_CACHE_Q5_0:
  case LLAMA_DART_KV_CACHE_Q5_1:
    return true;
  default:
    return false;
  }
}

ggml_type kv_cache_type_for(uint32_t value) {
  switch (value) {
  case LLAMA_DART_KV_CACHE_F32:
    return GGML_TYPE_F32;
  case LLAMA_DART_KV_CACHE_DEFAULT:
  case LLAMA_DART_KV_CACHE_F16:
    return GGML_TYPE_F16;
  case LLAMA_DART_KV_CACHE_BF16:
    return GGML_TYPE_BF16;
  case LLAMA_DART_KV_CACHE_Q8_0:
    return GGML_TYPE_Q8_0;
  case LLAMA_DART_KV_CACHE_Q4_0:
    return GGML_TYPE_Q4_0;
  case LLAMA_DART_KV_CACHE_Q4_1:
    return GGML_TYPE_Q4_1;
  case LLAMA_DART_KV_CACHE_IQ4_NL:
    return GGML_TYPE_IQ4_NL;
  case LLAMA_DART_KV_CACHE_Q5_0:
    return GGML_TYPE_Q5_0;
  case LLAMA_DART_KV_CACHE_Q5_1:
    return GGML_TYPE_Q5_1;
  default:
    return GGML_TYPE_F16;
  }
}

bool valid_flash_attention(uint32_t value) {
  return value == LLAMA_DART_FLASH_ATTENTION_AUTO ||
         value == LLAMA_DART_FLASH_ATTENTION_DISABLED ||
         value == LLAMA_DART_FLASH_ATTENTION_ENABLED;
}

llama_flash_attn_type flash_attention_for(uint32_t value) {
  switch (value) {
  case LLAMA_DART_FLASH_ATTENTION_DISABLED:
    return LLAMA_FLASH_ATTN_TYPE_DISABLED;
  case LLAMA_DART_FLASH_ATTENTION_ENABLED:
    return LLAMA_FLASH_ATTN_TYPE_ENABLED;
  case LLAMA_DART_FLASH_ATTENTION_AUTO:
  default:
    return LLAMA_FLASH_ATTN_TYPE_AUTO;
  }
}

bool valid_media_type(uint32_t value) {
  return value == LLAMA_DART_MEDIA_IMAGE || value == LLAMA_DART_MEDIA_AUDIO;
}

bool valid_gpu_backend(uint32_t value) {
  return value == LLAMA_DART_GPU_BACKEND_AUTO ||
         value == LLAMA_DART_GPU_BACKEND_CPU ||
         value == LLAMA_DART_GPU_BACKEND_METAL ||
         value == LLAMA_DART_GPU_BACKEND_VULKAN;
}

bool valid_speculative_type(uint32_t value) {
  return value == LLAMA_DART_SPECULATIVE_NONE ||
         value == LLAMA_DART_SPECULATIVE_NGRAM_SIMPLE ||
         value == LLAMA_DART_SPECULATIVE_NGRAM_MAP_K ||
         value == LLAMA_DART_SPECULATIVE_NGRAM_MAP_K4V ||
         value == LLAMA_DART_SPECULATIVE_NGRAM_MOD ||
         value == LLAMA_DART_SPECULATIVE_NGRAM_CACHE ||
         value == LLAMA_DART_SPECULATIVE_DRAFT_MODEL ||
         value == LLAMA_DART_SPECULATIVE_EAGLE3 ||
         value == LLAMA_DART_SPECULATIVE_DFLASH ||
         value == LLAMA_DART_SPECULATIVE_MTP;
}

bool is_ngram_speculation(uint32_t value) {
  return value == LLAMA_DART_SPECULATIVE_NGRAM_SIMPLE ||
         value == LLAMA_DART_SPECULATIVE_NGRAM_MAP_K ||
         value == LLAMA_DART_SPECULATIVE_NGRAM_MAP_K4V ||
         value == LLAMA_DART_SPECULATIVE_NGRAM_MOD ||
         value == LLAMA_DART_SPECULATIVE_NGRAM_CACHE;
}

bool is_ngram_map_speculation(uint32_t value) {
  return value == LLAMA_DART_SPECULATIVE_NGRAM_SIMPLE ||
         value == LLAMA_DART_SPECULATIVE_NGRAM_MAP_K ||
         value == LLAMA_DART_SPECULATIVE_NGRAM_MAP_K4V;
}

bool is_model_backed_speculation(uint32_t value) {
  return value == LLAMA_DART_SPECULATIVE_DRAFT_MODEL ||
         value == LLAMA_DART_SPECULATIVE_EAGLE3 ||
         value == LLAMA_DART_SPECULATIVE_DFLASH ||
         value == LLAMA_DART_SPECULATIVE_MTP;
}

bool speculative_target_requires_all_outputs(uint32_t value) {
  return value == LLAMA_DART_SPECULATIVE_EAGLE3 ||
         value == LLAMA_DART_SPECULATIVE_DFLASH ||
         value == LLAMA_DART_SPECULATIVE_MTP;
}

common_speculative_type common_speculative_type_for(uint32_t value) {
  switch (value) {
  case LLAMA_DART_SPECULATIVE_DRAFT_MODEL:
    return COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE;
  case LLAMA_DART_SPECULATIVE_EAGLE3:
    return COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3;
  case LLAMA_DART_SPECULATIVE_DFLASH:
    return COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH;
  case LLAMA_DART_SPECULATIVE_MTP:
    return COMMON_SPECULATIVE_TYPE_DRAFT_MTP;
  case LLAMA_DART_SPECULATIVE_NGRAM_SIMPLE:
    return COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE;
  case LLAMA_DART_SPECULATIVE_NGRAM_MAP_K:
    return COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K;
  case LLAMA_DART_SPECULATIVE_NGRAM_MAP_K4V:
    return COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V;
  case LLAMA_DART_SPECULATIVE_NGRAM_MOD:
    return COMMON_SPECULATIVE_TYPE_NGRAM_MOD;
  case LLAMA_DART_SPECULATIVE_NGRAM_CACHE:
    return COMMON_SPECULATIVE_TYPE_NGRAM_CACHE;
  default:
    return COMMON_SPECULATIVE_TYPE_NONE;
  }
}

const char *gpu_backend_registry_name(uint32_t backend) {
  switch (backend) {
  case LLAMA_DART_GPU_BACKEND_METAL:
    return "MTL";
  case LLAMA_DART_GPU_BACKEND_VULKAN:
    return "Vulkan";
  default:
    return nullptr;
  }
}

bool backend_has_devices(const char *name) {
  ggml_backend_reg_t registry = ggml_backend_reg_by_name(name);
  return registry != nullptr && ggml_backend_reg_dev_count(registry) > 0;
}

uint32_t auto_gpu_backend(int32_t n_gpu_layers) {
  if (n_gpu_layers == 0 || !llama_supports_gpu_offload()) {
    return LLAMA_DART_GPU_BACKEND_CPU;
  }
  const bool has_metal = backend_has_devices("MTL");
  const bool has_vulkan = backend_has_devices("Vulkan");
  if (has_metal && !has_vulkan) {
    return LLAMA_DART_GPU_BACKEND_METAL;
  }
  if (has_vulkan && !has_metal) {
    return LLAMA_DART_GPU_BACKEND_VULKAN;
  }
  return LLAMA_DART_GPU_BACKEND_AUTO;
}

llama_dart_result select_gpu_devices(
    uint32_t backend, int32_t n_gpu_layers,
    std::vector<ggml_backend_dev_t> *devices, uint32_t *effective_backend) {
  if (backend == LLAMA_DART_GPU_BACKEND_CPU || n_gpu_layers == 0) {
    ggml_backend_dev_t cpu =
        ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu == nullptr) {
      return fail(LLAMA_DART_ERROR_INTERNAL,
                  "CPU backend device is not available");
    }
    devices->push_back(cpu);
    devices->push_back(nullptr);
    *effective_backend = LLAMA_DART_GPU_BACKEND_CPU;
    return LLAMA_DART_SUCCESS;
  }
  if (backend == LLAMA_DART_GPU_BACKEND_AUTO) {
    *effective_backend = auto_gpu_backend(n_gpu_layers);
    return LLAMA_DART_SUCCESS;
  }

  const char *registry_name = gpu_backend_registry_name(backend);
  ggml_backend_reg_t registry = ggml_backend_reg_by_name(registry_name);
  if (registry == nullptr || ggml_backend_reg_dev_count(registry) == 0) {
    return fail(LLAMA_DART_ERROR_UNSUPPORTED,
                backend == LLAMA_DART_GPU_BACKEND_METAL
                    ? "Metal backend is not available in this native build"
                    : "Vulkan backend is not available in this native build");
  }
  const size_t count = ggml_backend_reg_dev_count(registry);
  devices->reserve(count + 1);
  for (size_t i = 0; i < count; ++i) {
    devices->push_back(ggml_backend_reg_dev_get(registry, i));
  }
  devices->push_back(nullptr);
  *effective_backend = backend;
  return LLAMA_DART_SUCCESS;
}

llama_dart_result media_file_exceeds_limit(const uint8_t *data, size_t size,
                                           bool *out) noexcept {
  *out = false;
  try {
    const std::string path(reinterpret_cast<const char *>(data), size);
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
      return LLAMA_DART_SUCCESS;
    }
    const std::streamoff length = file.tellg();
    *out = length >= 0 &&
           static_cast<uint64_t>(length) >
               static_cast<uint64_t>(kMaxMediaBytes);
    return LLAMA_DART_SUCCESS;
  } catch (const std::bad_alloc &) {
    return fail(LLAMA_DART_ERROR_INTERNAL, "native allocation failed");
  } catch (const std::exception &error) {
    return fail_parts(LLAMA_DART_ERROR_INTERNAL,
                      "failed to inspect media file: ", error.what());
  } catch (...) {
    return fail(LLAMA_DART_ERROR_INTERNAL,
                "unknown media file inspection failure");
  }
}

bool valid_chat_role(const uint8_t *data, size_t size) {
  const auto equals = [data, size](const char *role) {
    const size_t role_size = std::strlen(role);
    return size == role_size && std::memcmp(data, role, role_size) == 0;
  };
  return equals("system") || equals("user") || equals("assistant") ||
         equals("tool");
}

bool context_abort_callback(void *data) {
  llama_dart_context *context = static_cast<llama_dart_context *>(data);
  return context != nullptr &&
         context->cancel_requested.load(std::memory_order_relaxed);
}

bool is_cancelled(const llama_dart_context *context) {
  return context != nullptr &&
         context->cancel_requested.load(std::memory_order_relaxed);
}

llama_dart_result fail_cancelled(llama_dart_context *context) {
  if (context != nullptr) {
    context->cancel_requested.store(false, std::memory_order_relaxed);
  }
  return fail(LLAMA_DART_ERROR_CANCELLED, "generation cancelled");
}

llama_dart_result decode_tokens_at(llama_dart_context *context,
                                   const llama_token *tokens,
                                   int32_t token_count,
                                   llama_pos start_position,
                                   bool logits_all = false) {
  if (token_count <= 0) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "token_count must be positive");
  }
  if (start_position < 0 ||
      start_position + token_count > llama_n_ctx(context->context)) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "decode would exceed context size");
  }
  llama_batch batch = llama_batch_init(token_count, 0, 1);
  if (batch.token == nullptr || batch.pos == nullptr ||
      batch.n_seq_id == nullptr || batch.seq_id == nullptr ||
      batch.logits == nullptr) {
    llama_batch_free(batch);
    return fail(LLAMA_DART_ERROR_INTERNAL, "native allocation failed");
  }

  batch.n_tokens = token_count;
  const bool target_outputs_all =
      logits_all ||
      speculative_target_requires_all_outputs(context->speculative_type);
  for (int32_t i = 0; i < token_count; ++i) {
    batch.token[i] = tokens[i];
    batch.pos[i] = start_position + i;
    batch.n_seq_id[i] = 1;
    batch.seq_id[i][0] = 0;
    batch.logits[i] = target_outputs_all || i == token_count - 1 ? 1 : 0;
  }

  const int32_t decoded = llama_decode(context->context, batch);
  bool speculative_processed = true;
  if (decoded == 0) {
    if (is_model_backed_speculation(context->speculative_type)) {
      std::fill(batch.logits, batch.logits + token_count, 0);
      batch.logits[token_count - 1] = 1;
    }
    speculative_processed =
        common_speculative_process(context->speculative.get(), batch);
  }
  llama_batch_free(batch);
  if (decoded == 2 || is_cancelled(context)) {
    return fail_cancelled(context);
  }
  if (decoded != 0) {
    return fail(LLAMA_DART_ERROR_GENERATION, "failed to decode tokens");
  }
  if (!speculative_processed) {
    return fail(LLAMA_DART_ERROR_GENERATION,
                "failed to update speculative decoding state");
  }
  last_error.clear();
  return LLAMA_DART_SUCCESS;
}

llama_dart_result remove_speculative_sequence_tail(
    llama_dart_context *context, llama_pos start_position) {
  if (context->speculative_context == nullptr) {
    return LLAMA_DART_SUCCESS;
  }
  llama_memory_t memory = llama_get_memory(context->speculative_context);
  if (memory == nullptr ||
      !llama_memory_seq_rm(memory, 0, start_position, -1)) {
    return fail(LLAMA_DART_ERROR_UNSUPPORTED,
                "speculative context cannot remove rejected draft tokens");
  }
  last_error.clear();
  return LLAMA_DART_SUCCESS;
}

llama_dart_result capture_context_state(llama_context *context,
                                        const char *name,
                                        std::vector<uint8_t> *out) {
  const size_t size = llama_state_get_size(context);
  if (size == 0) {
    return fail_parts(LLAMA_DART_ERROR_INTERNAL, name,
                      " context state is empty");
  }
  out->resize(size);
  if (llama_state_get_data(context, out->data(), out->size()) != size) {
    out->clear();
    return fail_parts(LLAMA_DART_ERROR_INTERNAL, "failed to capture ", name,
                      " context state");
  }
  return LLAMA_DART_SUCCESS;
}

bool restore_context_state(llama_context *context, const uint8_t *data,
                           size_t size) {
  return data != nullptr && size > 0 &&
         llama_state_set_data(context, data, size) == size;
}

void restore_context_state_best_effort(
    llama_context *context, const std::vector<uint8_t> &state) {
  if (!state.empty()) {
    llama_state_set_data(context, state.data(), state.size());
  }
}

llama_dart_result remove_sequence_tail(llama_dart_context *context,
                                       llama_pos start_position) {
  llama_memory_t memory = llama_get_memory(context->context);
  if (memory == nullptr) {
    return fail(LLAMA_DART_ERROR_UNSUPPORTED,
                "context has no removable memory");
  }
  if (!llama_memory_seq_rm(memory, 0, start_position, -1)) {
    return fail(LLAMA_DART_ERROR_UNSUPPORTED,
                "context memory cannot remove rejected draft tokens");
  }
  last_error.clear();
  return LLAMA_DART_SUCCESS;
}

void remember_tokens(llama_dart_context *context, const llama_token *tokens,
                     size_t token_count) {
  if (context == nullptr || tokens == nullptr || token_count == 0) {
    return;
  }
  context->token_history.insert(context->token_history.end(), tokens,
                                tokens + token_count);
  size_t max_history = static_cast<size_t>(llama_n_ctx(context->context));
  if (max_history == 0) {
    max_history = 1;
  }
  if (context->token_history.size() > max_history) {
    context->token_history.erase(
        context->token_history.begin(),
        context->token_history.end() -
            static_cast<std::ptrdiff_t>(max_history));
  }
}

llama_dart_result decode_prompt_tokens(llama_dart_context *context,
                                       const std::vector<llama_token> &tokens) {
  const int32_t max_batch = llama_n_batch(context->context);
  if (max_batch <= 0) {
    return fail(LLAMA_DART_ERROR_GENERATION,
                "context reported invalid batch size");
  }
  size_t offset = 0;
  while (offset < tokens.size()) {
    const size_t remaining = tokens.size() - offset;
    const int32_t count = static_cast<int32_t>(
        std::min<size_t>(remaining, static_cast<size_t>(max_batch)));
    const llama_dart_result decoded = decode_tokens_at(
        context, tokens.data() + offset, count, context->position);
    if (decoded != LLAMA_DART_SUCCESS) {
      return decoded;
    }
    remember_tokens(context, tokens.data() + offset,
                    static_cast<size_t>(count));
    context->position += count;
    offset += static_cast<size_t>(count);
  }
  last_error.clear();
  return LLAMA_DART_SUCCESS;
}

bool completion_adds_special(const llama_dart_context *context,
                             const llama_dart_completion_config *config) {
  return config->add_special == LLAMA_DART_ADD_SPECIAL_ALWAYS ||
         (config->add_special == LLAMA_DART_ADD_SPECIAL_IF_CONTEXT_EMPTY &&
          context->position == 0);
}

llama_dart_result decode_multimodal_prompt(
    llama_dart_context *context, const llama_dart_completion_config *config,
    std::vector<llama_token> *sampler_tokens, uint32_t *prompt_token_count) {
  using bitmap_ptr =
      std::unique_ptr<mtmd_bitmap, decltype(&mtmd_bitmap_free)>;
  using chunks_ptr =
      std::unique_ptr<mtmd_input_chunks, decltype(&mtmd_input_chunks_free)>;

  std::vector<bitmap_ptr> owned_bitmaps;
  std::vector<const mtmd_bitmap *> bitmaps;
  owned_bitmaps.reserve(config->media_input_count);
  bitmaps.reserve(config->media_input_count);
  for (size_t i = 0; i < config->media_input_count; ++i) {
    const llama_dart_media_input &media = config->media_inputs[i];
    mtmd_helper_bitmap_wrapper wrapper{};
    if (media.path_size > 0) {
      const std::string path(
          reinterpret_cast<const char *>(media.path_data), media.path_size);
      wrapper = mtmd_helper_bitmap_init_from_file(context->multimodal,
                                                  path.c_str(), false);
    } else {
      wrapper = mtmd_helper_bitmap_init_from_buf(
          context->multimodal, media.content_data, media.content_size, false);
    }
    if (wrapper.video_ctx != nullptr) {
      mtmd_helper_video_free(wrapper.video_ctx);
      if (wrapper.bitmap != nullptr) {
        mtmd_bitmap_free(wrapper.bitmap);
      }
      return fail(LLAMA_DART_ERROR_UNSUPPORTED,
                  "video input is not available in mobile builds");
    }
    if (wrapper.bitmap == nullptr) {
      return fail(LLAMA_DART_ERROR_GENERATION,
                  "failed to decode media input");
    }
    const bool is_audio = mtmd_bitmap_is_audio(wrapper.bitmap);
    if ((media.type == LLAMA_DART_MEDIA_AUDIO) != is_audio) {
      mtmd_bitmap_free(wrapper.bitmap);
      return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                  media.type == LLAMA_DART_MEDIA_AUDIO
                      ? "audio input did not contain supported audio data"
                      : "image input did not contain supported image data");
    }
    owned_bitmaps.emplace_back(wrapper.bitmap, mtmd_bitmap_free);
    bitmaps.push_back(wrapper.bitmap);
  }

  chunks_ptr chunks(mtmd_input_chunks_init(), mtmd_input_chunks_free);
  if (chunks == nullptr) {
    return fail(LLAMA_DART_ERROR_INTERNAL, "native allocation failed");
  }
  const std::string prompt(
      reinterpret_cast<const char *>(config->prompt_data),
      config->prompt_size);
  mtmd_input_text text{
      prompt.c_str(),
      completion_adds_special(context, config),
      config->parse_special != 0,
  };
  const int32_t tokenized =
      mtmd_tokenize(context->multimodal, chunks.get(), &text, bitmaps.data(),
                    bitmaps.size());
  if (tokenized == 1) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "prompt media markers do not match media inputs");
  }
  if (tokenized != 0) {
    return fail(LLAMA_DART_ERROR_GENERATION,
                "failed to preprocess multimodal prompt");
  }

  const size_t n_tokens = mtmd_helper_get_n_tokens(chunks.get());
  const llama_pos n_positions = mtmd_helper_get_n_pos(chunks.get());
  if (n_tokens > std::numeric_limits<uint32_t>::max() ||
      n_positions < 0 ||
      static_cast<uint64_t>(context->position) +
              static_cast<uint64_t>(n_positions) >
          static_cast<uint64_t>(llama_n_ctx(context->context))) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "multimodal prompt exceeds context size");
  }

  sampler_tokens->clear();
  for (size_t i = 0; i < mtmd_input_chunks_size(chunks.get()); ++i) {
    const mtmd_input_chunk *chunk =
        mtmd_input_chunks_get(chunks.get(), i);
    if (chunk == nullptr ||
        mtmd_input_chunk_get_type(chunk) != MTMD_INPUT_CHUNK_TYPE_TEXT) {
      continue;
    }
    size_t text_token_count = 0;
    const llama_token *text_tokens =
        mtmd_input_chunk_get_tokens_text(chunk, &text_token_count);
    if (text_tokens != nullptr && text_token_count > 0) {
      sampler_tokens->insert(sampler_tokens->end(), text_tokens,
                             text_tokens + text_token_count);
    }
  }

  if (is_cancelled(context)) {
    return fail_cancelled(context);
  }
  llama_pos new_position = context->position;
  const int32_t evaluated = mtmd_helper_eval_chunks(
      context->multimodal, context->context, chunks.get(), context->position,
      0, static_cast<int32_t>(llama_n_batch(context->context)), true,
      &new_position);
  if (evaluated != 0 || is_cancelled(context)) {
    if (is_cancelled(context)) {
      return fail_cancelled(context);
    }
    return fail(LLAMA_DART_ERROR_GENERATION,
                "failed to evaluate multimodal prompt");
  }

  context->position = new_position;
  context->token_history.clear();
  if (!sampler_tokens->empty()) {
    remember_tokens(context, sampler_tokens->data(), sampler_tokens->size());
  }
  *prompt_token_count = static_cast<uint32_t>(n_tokens);
  last_error.clear();
  return LLAMA_DART_SUCCESS;
}

llama_dart_result decode_completion_prompt(
    llama_dart_context *context, const llama_vocab *vocab,
    const llama_dart_completion_config *config,
    std::vector<llama_token> *sampler_tokens, uint32_t *prompt_token_count) {
  sampler_tokens->clear();
  *prompt_token_count = 0;
  if (config->prompt_size == 0) {
    last_error.clear();
    return LLAMA_DART_SUCCESS;
  }
  if (config->media_input_count > 0) {
    return decode_multimodal_prompt(context, config, sampler_tokens,
                                    prompt_token_count);
  }

  int32_t required = llama_tokenize(
      vocab, reinterpret_cast<const char *>(config->prompt_data),
      static_cast<int32_t>(config->prompt_size), nullptr, 0,
      completion_adds_special(context, config), config->parse_special != 0);
  if (required == std::numeric_limits<int32_t>::min()) {
    return fail(LLAMA_DART_ERROR_INTERNAL, "tokenization overflowed");
  }
  if (required < 0) {
    required = -required;
  }
  sampler_tokens->resize(static_cast<size_t>(required));
  const int32_t actual = llama_tokenize(
      vocab, reinterpret_cast<const char *>(config->prompt_data),
      static_cast<int32_t>(config->prompt_size), sampler_tokens->data(),
      required, completion_adds_special(context, config),
      config->parse_special != 0);
  if (actual < 0) {
    return fail(LLAMA_DART_ERROR_GENERATION,
                "failed to tokenize completion prompt");
  }
  if (actual == 0) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "completion prompt produced no tokens");
  }
  sampler_tokens->resize(static_cast<size_t>(actual));
  *prompt_token_count = static_cast<uint32_t>(sampler_tokens->size());
  return decode_prompt_tokens(context, *sampler_tokens);
}

llama_dart_result add_sampler(llama_sampler *chain, llama_sampler *sampler) {
  if (sampler == nullptr) {
    return fail(LLAMA_DART_ERROR_INTERNAL, "failed to create sampler");
  }
  llama_sampler_chain_add(chain, sampler);
  return LLAMA_DART_SUCCESS;
}

void accept_sampler_tokens(llama_sampler *sampler,
                           const std::vector<llama_token> &tokens) {
  for (const llama_token token : tokens) {
    llama_sampler_accept(sampler, token);
  }
}

llama_dart_result create_completion_sampler(
    llama_dart_context *context, const llama_vocab *vocab,
    const llama_dart_completion_config *config, const std::string &grammar,
    const std::string &grammar_root, const parsed_chat_plan *chat_plan,
    llama_sampler **out_sampler) {
  *out_sampler = nullptr;
  llama_sampler_chain_params sampler_params =
      llama_sampler_chain_default_params();
  sampler_params.no_perf = true;
  llama_sampler *sampler = llama_sampler_chain_init(sampler_params);
  if (sampler == nullptr) {
    return fail(LLAMA_DART_ERROR_INTERNAL, "failed to create sampler");
  }

  llama_dart_result added = LLAMA_DART_SUCCESS;
  if (!grammar.empty()) {
    llama_sampler *grammar_sampler = nullptr;
    if (chat_plan != nullptr && chat_plan->grammar_lazy) {
      std::vector<std::string> trigger_patterns;
      std::vector<llama_token> trigger_tokens;
      for (const common_grammar_trigger &trigger :
           chat_plan->grammar_triggers) {
        switch (trigger.type) {
        case COMMON_GRAMMAR_TRIGGER_TYPE_WORD:
          trigger_patterns.push_back(regex_escape(trigger.value));
          break;
        case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN:
          trigger_patterns.push_back(trigger.value);
          break;
        case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL: {
          const std::string &pattern = trigger.value;
          std::string anchored = "^$";
          if (!pattern.empty()) {
            anchored = (pattern.front() == '^' ? "" : "^") + pattern +
                       (pattern.back() == '$' ? "" : "$");
          }
          trigger_patterns.push_back(std::move(anchored));
          break;
        }
        case COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN:
          trigger_tokens.push_back(trigger.token);
          break;
        default:
          llama_sampler_free(sampler);
          return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                      "chat plan grammar trigger type is invalid");
        }
      }
      std::vector<const char *> pattern_data;
      pattern_data.reserve(trigger_patterns.size());
      for (const std::string &pattern : trigger_patterns) {
        pattern_data.push_back(pattern.c_str());
      }
      grammar_sampler = llama_sampler_init_grammar_lazy_patterns(
          vocab, grammar.c_str(), grammar_root.c_str(), pattern_data.data(),
          pattern_data.size(), trigger_tokens.data(), trigger_tokens.size());
    } else {
      grammar_sampler = llama_sampler_init_grammar(
          vocab, grammar.c_str(), grammar_root.c_str());
    }
    if (grammar_sampler != nullptr && chat_plan != nullptr &&
        !chat_plan->generation_prompt.empty()) {
      const llama_tokens prefill = common_tokenize(
          vocab, chat_plan->generation_prompt, false, true);
      for (size_t i = 0; i < prefill.size(); ++i) {
        const std::string piece =
            common_token_to_piece(vocab, prefill[i], true);
        if (i == 0 && !piece.empty() &&
            std::isspace(static_cast<unsigned char>(piece.front())) &&
            !std::isspace(static_cast<unsigned char>(
                chat_plan->generation_prompt.front()))) {
          continue;
        }
        llama_sampler_accept(grammar_sampler, prefill[i]);
      }
    }
    added = add_sampler(sampler, grammar_sampler);
    if (added != LLAMA_DART_SUCCESS) {
      llama_sampler_free(sampler);
      return fail(LLAMA_DART_ERROR_GENERATION,
                  "failed to parse completion grammar");
    }
  }
  if (config->mirostat == 1) {
    added = add_sampler(sampler, llama_sampler_init_temp(config->temperature));
    if (added == LLAMA_DART_SUCCESS) {
      added = add_sampler(
          sampler, llama_sampler_init_mirostat(llama_vocab_n_tokens(vocab),
                                               config->seed,
                                               config->mirostat_tau,
                                               config->mirostat_eta, 100));
    }
  } else if (config->mirostat == 2) {
    added = add_sampler(sampler, llama_sampler_init_temp(config->temperature));
    if (added == LLAMA_DART_SUCCESS) {
      added = add_sampler(sampler, llama_sampler_init_mirostat_v2(
                                       config->seed, config->mirostat_tau,
                                       config->mirostat_eta));
    }
  } else if (config->temperature <= 0.0f) {
    added = add_sampler(sampler, llama_sampler_init_greedy());
  } else {
    if (config->top_k > 0) {
      added = add_sampler(sampler, llama_sampler_init_top_k(config->top_k));
    }
    if (added == LLAMA_DART_SUCCESS && config->top_p > 0.0f &&
        config->top_p < 1.0f) {
      added =
          add_sampler(sampler, llama_sampler_init_top_p(config->top_p, 1));
    }
    if (added == LLAMA_DART_SUCCESS && config->min_p > 0.0f) {
      added =
          add_sampler(sampler, llama_sampler_init_min_p(config->min_p, 1));
    }
    if (added == LLAMA_DART_SUCCESS && config->typical_p > 0.0f &&
        config->typical_p < 1.0f) {
      added = add_sampler(sampler,
                          llama_sampler_init_typical(config->typical_p, 1));
    }
    if (added == LLAMA_DART_SUCCESS && config->penalty_last_n > 0 &&
        (config->repeat_penalty != 1.0f ||
         config->frequency_penalty != 0.0f ||
         config->presence_penalty != 0.0f)) {
      added = add_sampler(sampler,
                          llama_sampler_init_penalties(
                              config->penalty_last_n, config->repeat_penalty,
                              config->frequency_penalty,
                              config->presence_penalty));
    }
    if (added == LLAMA_DART_SUCCESS) {
      added = add_sampler(sampler, llama_sampler_init_temp(config->temperature));
    }
    if (added == LLAMA_DART_SUCCESS) {
      added = add_sampler(sampler, llama_sampler_init_dist(config->seed));
    }
  }
  if (added != LLAMA_DART_SUCCESS) {
    llama_sampler_free(sampler);
    return added;
  }

  if (grammar.empty()) {
    accept_sampler_tokens(sampler, context->token_history);
  }

  *out_sampler = sampler;
  last_error.clear();
  return LLAMA_DART_SUCCESS;
}

llama_dart_result convert_json_schema_to_grammar(
    const uint8_t *schema_data, size_t schema_size, std::string *out) {
  if (schema_data == nullptr || schema_size == 0) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "JSON schema must not be empty");
  }
  if (!fits_int32(schema_size)) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "JSON schema is too large");
  }
  if (contains_nul(schema_data, schema_size)) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "JSON schema must not contain NUL");
  }
  if (is_ascii_blank(schema_data, schema_size)) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "JSON schema must not be blank");
  }
  try {
    const char *begin = reinterpret_cast<const char *>(schema_data);
    nlohmann::ordered_json schema =
        nlohmann::ordered_json::parse(begin, begin + schema_size);
    *out = json_schema_to_grammar(schema);
    if (out->empty()) {
      return fail(LLAMA_DART_ERROR_UNSUPPORTED,
                  "JSON schema produced an empty grammar");
    }
    last_error.clear();
    return LLAMA_DART_SUCCESS;
  } catch (const nlohmann::json::parse_error &error) {
    return fail_parts(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                      "invalid JSON schema: ", error.what());
  } catch (const std::bad_alloc &) {
    return fail(LLAMA_DART_ERROR_INTERNAL, "native allocation failed");
  } catch (const std::exception &error) {
    return fail_parts(LLAMA_DART_ERROR_UNSUPPORTED,
                      "unsupported JSON schema: ", error.what());
  } catch (...) {
    return fail(LLAMA_DART_ERROR_INTERNAL,
                "unknown JSON schema conversion failure");
  }
}

llama_dart_result completion_grammar(
    const llama_dart_completion_config *config, std::string *grammar,
    std::string *grammar_root, const parsed_chat_plan *chat_plan) {
  grammar->clear();
  *grammar_root = "root";
  if (chat_plan != nullptr) {
    *grammar = chat_plan->grammar;
    return LLAMA_DART_SUCCESS;
  }
  if (config->json_schema_size > 0) {
    return convert_json_schema_to_grammar(config->json_schema_data,
                                          config->json_schema_size, grammar);
  }
  if (config->grammar_size > 0) {
    grammar->assign(reinterpret_cast<const char *>(config->grammar_data),
                    config->grammar_size);
    grammar_root->assign(
        reinterpret_cast<const char *>(config->grammar_root_data),
        config->grammar_root_size);
  }
  return LLAMA_DART_SUCCESS;
}

llama_dart_result completion_chat_plan(
    const llama_dart_completion_config *config, parsed_chat_plan *out,
    bool *has_plan) {
  *has_plan = false;
  if (config->chat_plan_size == 0) {
    return LLAMA_DART_SUCCESS;
  }
  const llama_dart_result parsed =
      parse_chat_plan(config->chat_plan_data, config->chat_plan_size, out);
  if (parsed != LLAMA_DART_SUCCESS) {
    return parsed;
  }
  if (config->prompt_size != out->prompt.size() ||
      std::memcmp(config->prompt_data, out->prompt.data(),
                  out->prompt.size()) != 0) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "chat plan prompt does not match completion prompt");
  }
  *has_plan = true;
  last_error.clear();
  return LLAMA_DART_SUCCESS;
}

void append_unique_stops(const std::vector<std::string> &source,
                         std::vector<std::string> *destination) {
  for (const std::string &stop : source) {
    if (std::find(destination->begin(), destination->end(), stop) ==
        destination->end()) {
      destination->push_back(stop);
    }
  }
}

llama_token sample_and_accept_completion_token(llama_sampler *sampler,
                                               llama_context *context,
                                               int32_t logits_index) {
  return llama_sampler_sample(sampler, context, logits_index);
}

llama_dart_result commit_sampled_completion_token(
    llama_dart_context *context, const llama_vocab *vocab, llama_token token,
    const std::vector<std::string> &stop_sequences,
    const std::vector<llama_token> &stop_tokens, std::string *generated,
    uint32_t *produced_tokens, bool *done) {
  *produced_tokens = 0;
  *done = false;
  if (llama_vocab_is_eog(vocab, token) ||
      std::find(stop_tokens.begin(), stop_tokens.end(), token) !=
          stop_tokens.end()) {
    *done = true;
    last_error.clear();
    return LLAMA_DART_SUCCESS;
  }
  if (!append_token_piece(vocab, token, generated)) {
    return fail(LLAMA_DART_ERROR_GENERATION,
                "failed to convert token to text");
  }
  const bool stopped =
      truncate_at_stop_sequence(generated, stop_sequences);
  const llama_dart_result decoded =
      decode_tokens_at(context, &token, 1, context->position);
  if (decoded != LLAMA_DART_SUCCESS) {
    return decoded;
  }
  remember_tokens(context, &token, 1);
  context->position += 1;
  *produced_tokens = 1;
  *done = stopped;
  last_error.clear();
  return LLAMA_DART_SUCCESS;
}

llama_dart_result decode_single_completion_token(
    llama_dart_context *context, const llama_vocab *vocab,
    llama_sampler *sampler, const std::vector<std::string> &stop_sequences,
    const std::vector<llama_token> &stop_tokens, std::string *generated,
    uint32_t *produced_tokens, bool *done) {
  if (is_cancelled(context)) {
    return fail_cancelled(context);
  }
  const llama_token token =
      sample_and_accept_completion_token(sampler, context->context, -1);
  return commit_sampled_completion_token(
      context, vocab, token, stop_sequences, stop_tokens, generated,
      produced_tokens, done);
}

llama_dart_result decode_ngram_completion_tokens(
    llama_dart_context *context, const llama_vocab *vocab,
    llama_sampler *sampler, uint32_t max_tokens, std::string *generated,
    uint32_t *produced_tokens, bool *done, uint32_t *draft_tokens,
    uint32_t *accepted_draft_tokens, double *draft_ms, double *verify_ms) {
  *produced_tokens = 0;
  *done = false;
  *draft_tokens = 0;
  *accepted_draft_tokens = 0;
  *draft_ms = 0.0;
  *verify_ms = 0.0;
  if (is_cancelled(context)) {
    return fail_cancelled(context);
  }
  if (max_tokens == 0) {
    *done = true;
    last_error.clear();
    return LLAMA_DART_SUCCESS;
  }

  const llama_token token =
      sample_and_accept_completion_token(sampler, context->context, -1);
  if (llama_vocab_is_eog(vocab, token)) {
    *done = true;
    last_error.clear();
    return LLAMA_DART_SUCCESS;
  }

  const int32_t batch_size = llama_n_batch(context->context);
  const int32_t context_size = llama_n_ctx(context->context);
  if (batch_size <= 1 || context_size <= context->position + 1 ||
      max_tokens <= 1) {
    if (!append_token_piece(vocab, token, generated)) {
      return fail(LLAMA_DART_ERROR_GENERATION,
                  "failed to convert token to text");
    }
    const llama_dart_result decoded =
        decode_tokens_at(context, &token, 1, context->position);
    if (decoded != LLAMA_DART_SUCCESS) {
      return decoded;
    }
    remember_tokens(context, &token, 1);
    context->position += 1;
    *produced_tokens = 1;
    last_error.clear();
    return LLAMA_DART_SUCCESS;
  }

  const int32_t configured_draft_max =
      common_speculative_n_max(&context->speculative_params);
  const uint32_t max_draft = std::min<uint32_t>(
      configured_draft_max <= 0
          ? 0
          : static_cast<uint32_t>(configured_draft_max),
      std::min<uint32_t>(
          max_tokens - 1,
          static_cast<uint32_t>(
              std::min<int32_t>(batch_size - 1,
                                context_size - context->position - 1))));
  const steady_clock::time_point draft_start = steady_clock::now();
  std::vector<llama_token> draft;
  common_speculative_get_draft_params(context->speculative.get(), 0) = {
      /* .drafting = */ true,
      /* .n_max    = */ static_cast<int32_t>(max_draft),
      /* .n_past   = */ context->position,
      /* .id_last  = */ token,
      /* .prompt   = */ &context->token_history,
      /* .result   = */ &draft,
  };
  common_speculative_draft(context->speculative.get());
  *draft_ms = elapsed_ms(draft_start, steady_clock::now());
  if (draft.empty()) {
    if (!append_token_piece(vocab, token, generated)) {
      return fail(LLAMA_DART_ERROR_GENERATION,
                  "failed to convert token to text");
    }
    const llama_dart_result decoded =
        decode_tokens_at(context, &token, 1, context->position);
    if (decoded != LLAMA_DART_SUCCESS) {
      return decoded;
    }
    remember_tokens(context, &token, 1);
    context->position += 1;
    *produced_tokens = 1;
    last_error.clear();
    return LLAMA_DART_SUCCESS;
  }

  std::vector<llama_token> batch_tokens;
  batch_tokens.reserve(draft.size() + 1);
  batch_tokens.push_back(token);
  batch_tokens.insert(batch_tokens.end(), draft.begin(), draft.end());
  const steady_clock::time_point verify_start = steady_clock::now();
  const llama_dart_result decoded = decode_tokens_at(
      context, batch_tokens.data(), static_cast<int32_t>(batch_tokens.size()),
      context->position, true);
  if (decoded != LLAMA_DART_SUCCESS) {
    return decoded;
  }

  *draft_tokens = static_cast<uint32_t>(draft.size());
  size_t accepted = 0;
  bool sampled_eog = false;
  bool sampled_replacement = false;
  llama_token replacement = LLAMA_TOKEN_NULL;
  for (size_t i = 0; i < draft.size(); ++i) {
    const llama_token sampled = sample_and_accept_completion_token(
        sampler, context->context, static_cast<int32_t>(i));
    if (llama_vocab_is_eog(vocab, sampled)) {
      sampled_eog = true;
      break;
    }
    if (sampled != draft[i]) {
      sampled_replacement = true;
      replacement = sampled;
      break;
    }
    accepted += 1;
  }

  std::vector<llama_token> output;
  output.reserve(2 + accepted);
  output.push_back(token);
  output.insert(output.end(), draft.begin(),
                draft.begin() + static_cast<std::ptrdiff_t>(accepted));
  common_speculative_accept(context->speculative.get(), 0,
                            static_cast<uint16_t>(accepted));
  *accepted_draft_tokens = static_cast<uint32_t>(accepted);

  if (sampled_eog || sampled_replacement) {
    const llama_pos remove_from =
        context->position + static_cast<llama_pos>(output.size());
    const llama_dart_result removed = remove_sequence_tail(context, remove_from);
    if (removed != LLAMA_DART_SUCCESS) {
      return removed;
    }
  }

  if (sampled_replacement) {
    const llama_dart_result decoded_replacement =
        decode_tokens_at(context, &replacement, 1,
                         context->position +
                             static_cast<llama_pos>(output.size()));
    if (decoded_replacement != LLAMA_DART_SUCCESS) {
      return decoded_replacement;
    }
    output.push_back(replacement);
  }
  *verify_ms = elapsed_ms(verify_start, steady_clock::now());

  for (const llama_token out_token : output) {
    if (!append_token_piece(vocab, out_token, generated)) {
      return fail(LLAMA_DART_ERROR_GENERATION,
                  "failed to convert token to text");
    }
  }
  remember_tokens(context, output.data(), output.size());
  context->position += static_cast<int32_t>(output.size());
  *produced_tokens = static_cast<uint32_t>(output.size());
  *done = sampled_eog;
  last_error.clear();
  return LLAMA_DART_SUCCESS;
}

llama_dart_result decode_model_backed_completion_tokens(
    llama_dart_context *context, const llama_vocab *vocab,
    llama_sampler *sampler, uint32_t max_tokens, std::string *generated,
    uint32_t *produced_tokens, bool *done, uint32_t *draft_tokens,
    uint32_t *accepted_draft_tokens, double *draft_ms, double *verify_ms) {
  *produced_tokens = 0;
  *done = false;
  *draft_tokens = 0;
  *accepted_draft_tokens = 0;
  *draft_ms = 0.0;
  *verify_ms = 0.0;
  if (is_cancelled(context)) {
    return fail_cancelled(context);
  }
  if (max_tokens == 0) {
    *done = true;
    last_error.clear();
    return LLAMA_DART_SUCCESS;
  }

  const llama_token token =
      sample_and_accept_completion_token(sampler, context->context, -1);
  if (llama_vocab_is_eog(vocab, token)) {
    *done = true;
    last_error.clear();
    return LLAMA_DART_SUCCESS;
  }

  const int32_t batch_size = llama_n_batch(context->context);
  const int32_t context_size = llama_n_ctx(context->context);
  const uint32_t max_draft = std::min<uint32_t>(
      context->speculative_draft_max,
      std::min<uint32_t>(
          max_tokens - 1,
          static_cast<uint32_t>(std::max<int32_t>(
              0, std::min<int32_t>(batch_size - 1,
                                   context_size - context->position - 1)))));
  if (max_draft == 0) {
    return commit_sampled_completion_token(context, vocab, token, {}, {},
                                           generated, produced_tokens, done);
  }

  llama_memory_t speculative_memory =
      llama_get_memory(context->speculative_context);
  if (speculative_memory == nullptr) {
    return fail(LLAMA_DART_ERROR_UNSUPPORTED,
                "speculative context has no removable memory");
  }
  const llama_pos speculative_remove_from =
      llama_memory_seq_pos_max(speculative_memory, 0) + 1;
  std::vector<llama_token> draft;
  const steady_clock::time_point draft_start = steady_clock::now();
  common_speculative_get_draft_params(context->speculative.get(), 0) = {
      /* .drafting = */ true,
      /* .n_max    = */ static_cast<int32_t>(max_draft),
      /* .n_past   = */ context->position,
      /* .id_last  = */ token,
      /* .prompt   = */ &context->token_history,
      /* .result   = */ &draft,
  };
  common_speculative_draft(context->speculative.get());

  const llama_dart_result draft_trimmed =
      remove_speculative_sequence_tail(context, speculative_remove_from);
  *draft_ms = elapsed_ms(draft_start, steady_clock::now());
  if (draft_trimmed != LLAMA_DART_SUCCESS) {
    return draft_trimmed;
  }
  if (draft.empty()) {
    return commit_sampled_completion_token(context, vocab, token, {}, {},
                                           generated, produced_tokens, done);
  }

  std::vector<llama_token> batch_tokens;
  batch_tokens.reserve(draft.size() + 1);
  batch_tokens.push_back(token);
  batch_tokens.insert(batch_tokens.end(), draft.begin(), draft.end());
  const steady_clock::time_point verify_start = steady_clock::now();
  const llama_dart_result decoded = decode_tokens_at(
      context, batch_tokens.data(), static_cast<int32_t>(batch_tokens.size()),
      context->position, true);
  if (decoded != LLAMA_DART_SUCCESS) {
    return decoded;
  }

  *draft_tokens = static_cast<uint32_t>(draft.size());
  std::vector<llama_token> output;
  output.reserve(draft.size() + 1);
  output.push_back(token);

  bool sampled_eog = false;
  bool sampled_replacement = false;
  llama_token replacement = LLAMA_TOKEN_NULL;
  size_t accepted = 0;
  for (size_t i = 0; i < draft.size(); ++i) {
    const llama_token sampled = sample_and_accept_completion_token(
        sampler, context->context, static_cast<int32_t>(i));
    if (llama_vocab_is_eog(vocab, sampled)) {
      sampled_eog = true;
      break;
    }
    if (sampled != draft[i]) {
      sampled_replacement = true;
      replacement = sampled;
      break;
    }
    output.push_back(sampled);
    accepted += 1;
  }

  common_speculative_accept(context->speculative.get(), 0,
                            static_cast<uint16_t>(accepted));
  *accepted_draft_tokens = static_cast<uint32_t>(accepted);

  if (accepted < draft.size()) {
    const llama_pos remove_from =
        context->position + static_cast<llama_pos>(output.size());
    const llama_dart_result target_trimmed =
        remove_sequence_tail(context, remove_from);
    if (target_trimmed != LLAMA_DART_SUCCESS) {
      return target_trimmed;
    }
    const llama_dart_result speculative_trimmed =
        remove_speculative_sequence_tail(context, remove_from);
    if (speculative_trimmed != LLAMA_DART_SUCCESS) {
      return speculative_trimmed;
    }
  }

  if (sampled_replacement) {
    const llama_pos replacement_position =
        context->position + static_cast<llama_pos>(output.size());
    const llama_dart_result replacement_decoded =
        decode_tokens_at(context, &replacement, 1, replacement_position);
    if (replacement_decoded != LLAMA_DART_SUCCESS) {
      return replacement_decoded;
    }
    output.push_back(replacement);
  }
  *verify_ms = elapsed_ms(verify_start, steady_clock::now());

  for (const llama_token out_token : output) {
    if (!append_token_piece(vocab, out_token, generated)) {
      return fail(LLAMA_DART_ERROR_GENERATION,
                  "failed to convert token to text");
    }
  }
  remember_tokens(context, output.data(), output.size());
  context->position += static_cast<int32_t>(output.size());
  *produced_tokens = static_cast<uint32_t>(output.size());
  *done = sampled_eog;
  last_error.clear();
  return LLAMA_DART_SUCCESS;
}

llama_dart_result decode_completion_token(
    llama_dart_context *context, const llama_vocab *vocab,
    llama_sampler *sampler, const std::vector<std::string> &stop_sequences,
    const std::vector<llama_token> &stop_tokens, uint32_t max_tokens,
    std::string *generated, uint32_t *produced_tokens, bool *done,
    uint32_t *draft_tokens, uint32_t *accepted_draft_tokens, double *draft_ms,
    double *verify_ms) {
  *draft_tokens = 0;
  *accepted_draft_tokens = 0;
  *draft_ms = 0.0;
  *verify_ms = 0.0;
  const bool model_backed =
      is_model_backed_speculation(context->speculative_type);
  if (model_backed && context->speculative != nullptr &&
      context->speculative_needs_warmup && stop_sequences.empty() &&
      stop_tokens.empty()) {
    const llama_dart_result result = decode_single_completion_token(
        context, vocab, sampler, stop_sequences, stop_tokens, generated,
        produced_tokens, done);
    if (result == LLAMA_DART_SUCCESS && *produced_tokens > 0) {
      context->speculative_needs_warmup = false;
    }
    return result;
  }
  if (model_backed && context->speculative != nullptr &&
      stop_sequences.empty() && stop_tokens.empty()) {
    return decode_model_backed_completion_tokens(
        context, vocab, sampler, max_tokens, generated, produced_tokens, done,
        draft_tokens, accepted_draft_tokens, draft_ms, verify_ms);
  }
  if (is_ngram_speculation(context->speculative_type) &&
      context->speculative != nullptr && stop_sequences.empty() &&
      stop_tokens.empty()) {
    return decode_ngram_completion_tokens(
        context, vocab, sampler, max_tokens, generated, produced_tokens, done,
        draft_tokens, accepted_draft_tokens, draft_ms, verify_ms);
  }
  return decode_single_completion_token(context, vocab, sampler, stop_sequences,
                                        stop_tokens, generated,
                                        produced_tokens, done);
}

llama_dart_result validate_completion_request(
    llama_dart_context *context, const llama_dart_completion_config *config,
    const llama_vocab **out_vocab) {
  if (config == nullptr) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT, "config must not be null");
  }
  const llama_dart_result validation =
      validate_struct(config->struct_size, sizeof(llama_dart_completion_config),
                      "llama_dart_completion_config");
  if (validation != LLAMA_DART_SUCCESS) {
    return validation;
  }
  if (config->prompt_data == nullptr && config->prompt_size > 0) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "prompt_data must not be null when prompt_size is positive");
  }
  if (config->add_special > LLAMA_DART_ADD_SPECIAL_IF_CONTEXT_EMPTY ||
      !valid_bool(config->parse_special)) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "completion tokenization modes are invalid");
  }
  if (!fits_int32(config->prompt_size) || !fits_int32(config->max_tokens) ||
      !fits_int32(config->grammar_size) ||
      !fits_int32(config->grammar_root_size) ||
      !fits_int32(config->json_schema_size) ||
      !fits_int32(config->chat_plan_size) ||
      !fits_int32(config->stop_token_count) ||
      !fits_int32(config->stop_sequence_count) ||
      !fits_int32(config->media_input_count)) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "completion input is too large");
  }
  if (config->media_input_count > kMaxMediaInputs) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "completion supports at most 64 media inputs");
  }
  if (config->media_input_count > 0 && config->media_inputs == nullptr) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "media_inputs must not be null when media_input_count is "
                "positive");
  }
  if (config->media_input_count == 0 && config->media_inputs != nullptr) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "media_inputs must be null when media_input_count is zero");
  }
  if (config->media_input_count > 0 && config->prompt_size == 0) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "media inputs require a prompt");
  }
  if (contains_nul(config->prompt_data, config->prompt_size)) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "prompt must not contain NUL bytes");
  }
  if (config->prompt_size > 0 &&
      is_ascii_blank(config->prompt_data, config->prompt_size)) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "prompt must not be blank");
  }
  if (config->grammar_data == nullptr && config->grammar_size > 0) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "grammar_data must not be null when grammar_size is positive");
  }
  if (config->grammar_root_data == nullptr && config->grammar_root_size > 0) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "grammar_root_data must not be null when grammar_root_size is "
                "positive");
  }
  if (config->max_tokens == 0 && config->prompt_size == 0) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "prompt-only prefill requires a prompt");
  }
  if (config->top_k < 0) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "top_k must be non-negative");
  }
  if (config->penalty_last_n < 0) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "penalty_last_n must be non-negative");
  }
  if (config->repeat_penalty < 0.0f) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "repeat_penalty must be non-negative");
  }
  if (!std::isfinite(config->temperature) ||
      !std::isfinite(config->top_p) || !std::isfinite(config->min_p) ||
      !std::isfinite(config->typical_p) ||
      !std::isfinite(config->repeat_penalty) ||
      !std::isfinite(config->frequency_penalty) ||
      !std::isfinite(config->presence_penalty)) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "sampler floats must be finite");
  }
  if (config->mirostat > 2) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "mirostat must be 0, 1, or 2");
  }
  if (config->mirostat != 0 &&
      (!std::isfinite(config->mirostat_tau) ||
       !std::isfinite(config->mirostat_eta) ||
       config->mirostat_tau <= 0.0f || config->mirostat_eta <= 0.0f)) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "mirostat tau and eta must be positive finite values");
  }
  if (config->temperature < 0.0f) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "temperature must be non-negative");
  }
  if (config->top_p < 0.0f || config->top_p > 1.0f ||
      config->min_p < 0.0f || config->min_p > 1.0f ||
      config->typical_p < 0.0f || config->typical_p > 1.0f) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "sampler probabilities must be in [0, 1]");
  }
  if (contains_nul(config->grammar_data, config->grammar_size) ||
      contains_nul(config->grammar_root_data, config->grammar_root_size)) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "grammar strings must not contain NUL bytes");
  }
  if (config->grammar_size > 0 &&
      is_ascii_blank(config->grammar_data, config->grammar_size)) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "grammar must not be blank");
  }
  if (config->grammar_root_size > 0 &&
      is_ascii_blank(config->grammar_root_data, config->grammar_root_size)) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "grammar_root must not be blank");
  }
  if (contains_line_break(config->grammar_root_data,
                          config->grammar_root_size)) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "grammar_root must not contain line breaks");
  }
  if (contains_ascii_whitespace(config->grammar_root_data,
                                config->grammar_root_size)) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "grammar_root must not contain whitespace");
  }
  if (config->grammar_size > 0 && config->grammar_root_size == 0) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "grammar_root must not be empty when grammar is provided");
  }
  if (config->grammar_size == 0 && config->grammar_root_size > 0) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "grammar_root must not be provided without grammar");
  }
  if (config->json_schema_data == nullptr &&
      config->json_schema_size > 0) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "json_schema_data must not be null when json_schema_size is "
                "positive");
  }
  if (config->json_schema_data != nullptr &&
      config->json_schema_size == 0) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "JSON schema must not be empty");
  }
  if (config->json_schema_size > 0 && config->grammar_size > 0) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "JSON schema and grammar are mutually exclusive");
  }
  if (config->chat_plan_data == nullptr && config->chat_plan_size > 0) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "chat_plan_data must not be null when chat_plan_size is "
                "positive");
  }
  if (config->chat_plan_data != nullptr && config->chat_plan_size == 0) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "chat plan must not be empty");
  }
  if (config->chat_plan_size > 0 &&
      (config->grammar_size > 0 || config->json_schema_size > 0)) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "chat plan cannot be combined with grammar or JSON schema");
  }
  if (config->chat_plan_size > 0 &&
      (contains_nul(config->chat_plan_data, config->chat_plan_size) ||
       is_ascii_blank(config->chat_plan_data, config->chat_plan_size))) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "chat plan must contain non-blank JSON without NUL");
  }
  if (config->json_schema_size > 0 &&
      (contains_nul(config->json_schema_data, config->json_schema_size) ||
       is_ascii_blank(config->json_schema_data, config->json_schema_size))) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "JSON schema must contain non-blank JSON without NUL");
  }
  if (config->stop_sequence_count > 0 && config->stop_sequences == nullptr) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "stop_sequences must not be null when stop_sequence_count is "
                "positive");
  }
  if (config->stop_token_count > kMaxStopTokens) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "completion supports at most 1024 stop tokens");
  }
  if (config->stop_token_count > 0 && config->stop_tokens == nullptr) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "stop_tokens must not be null when stop_token_count is "
                "positive");
  }
  if (config->stop_token_count == 0 && config->stop_tokens != nullptr) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "stop_tokens must be null when stop_token_count is zero");
  }
  for (size_t i = 0; i < config->stop_sequence_count; ++i) {
    const llama_dart_string_view stop = config->stop_sequences[i];
    if (stop.data == nullptr || stop.size == 0) {
      return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                  "stop sequence must not be empty");
    }
    if (!fits_int32(stop.size)) {
      return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                  "completion input is too large");
    }
    if (contains_nul(stop.data, stop.size)) {
      return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                  "stop sequence must not contain NUL bytes");
    }
  }
  if (context == nullptr || context->context == nullptr ||
      context->model == nullptr || context->model->model == nullptr) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT, "context must not be null");
  }
  if (context->active_generations > 0) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "context already has an active generation");
  }
  if (context->speculative_type != LLAMA_DART_SPECULATIVE_NONE &&
      context->speculative == nullptr) {
    return fail(LLAMA_DART_ERROR_INTERNAL,
                "speculative decoding state is unavailable");
  }

  const llama_vocab *vocab = llama_model_get_vocab(context->model->model);
  if (vocab == nullptr) {
    return fail(LLAMA_DART_ERROR_UNSUPPORTED, "model has no vocabulary");
  }
  const int32_t vocab_size = llama_vocab_n_tokens(vocab);
  for (size_t i = 0; i < config->stop_token_count; ++i) {
    const llama_token token = config->stop_tokens[i];
    if (token < 0 || token >= vocab_size) {
      return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                  "stop token id is outside the model vocabulary");
    }
    for (size_t j = 0; j < i; ++j) {
      if (config->stop_tokens[j] == token) {
        return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                    "stop token ids must be unique");
      }
    }
  }
  if (is_cancelled(context)) {
    return fail_cancelled(context);
  }
  if (config->media_input_count > 0 && context->multimodal == nullptr) {
    return fail(LLAMA_DART_ERROR_UNSUPPORTED,
                "media inputs require an mmproj-backed context");
  }
  if (config->media_input_count > 0 &&
      (context->speculative_ngram_n > 0 ||
       context->speculative != nullptr)) {
    return fail(LLAMA_DART_ERROR_UNSUPPORTED,
                "media inputs cannot use speculative decoding");
  }
  for (size_t i = 0; i < config->media_input_count; ++i) {
    const llama_dart_media_input &media = config->media_inputs[i];
    const llama_dart_result media_validation =
        validate_struct(media.struct_size, sizeof(llama_dart_media_input),
                        "llama_dart_media_input");
    if (media_validation != LLAMA_DART_SUCCESS) {
      return media_validation;
    }
    if (!valid_media_type(media.type)) {
      return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                  "media input type is invalid");
    }
    const bool has_path = media.path_data != nullptr && media.path_size > 0;
    const bool has_content =
        media.content_data != nullptr && media.content_size > 0;
    if (media.path_data != nullptr && media.path_size == 0) {
      return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                  "media path must not be empty");
    }
    if (media.content_data != nullptr && media.content_size == 0) {
      return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                  "media content must not be empty");
    }
    if (has_path == has_content) {
      return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                  "media input must contain exactly one path or content "
                  "buffer");
    }
    if (!fits_int32(media.path_size) || !fits_int32(media.content_size)) {
      return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                  "media input is too large");
    }
    if (has_path &&
        is_ascii_blank(media.path_data, media.path_size)) {
      return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                  "media path must not be blank");
    }
    if (has_path && contains_nul(media.path_data, media.path_size)) {
      return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                  "media path must not contain NUL");
    }
    if (has_path &&
        contains_line_break(media.path_data, media.path_size)) {
      return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                  "media path must not contain line breaks");
    }
    bool file_exceeds_limit = false;
    if (has_path) {
      const llama_dart_result inspected = media_file_exceeds_limit(
          media.path_data, media.path_size, &file_exceeds_limit);
      if (inspected != LLAMA_DART_SUCCESS) {
        return inspected;
      }
    }
    if (media.content_size > kMaxMediaBytes || file_exceeds_limit) {
      return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                  "media input exceeds the 64 MiB limit");
    }
    if (media.type == LLAMA_DART_MEDIA_IMAGE &&
        !mtmd_support_vision(context->multimodal)) {
      return fail(LLAMA_DART_ERROR_UNSUPPORTED,
                  "the loaded mmproj does not support image input");
    }
    if (media.type == LLAMA_DART_MEDIA_AUDIO &&
        !mtmd_support_audio(context->multimodal)) {
      return fail(LLAMA_DART_ERROR_UNSUPPORTED,
                  "the loaded mmproj does not support audio input");
    }
  }

  *out_vocab = vocab;
  last_error.clear();
  return LLAMA_DART_SUCCESS;
}

void replace_all(std::string *text, const std::string &from,
                 const std::string &to) {
  size_t pos = 0;
  while ((pos = text->find(from, pos)) != std::string::npos) {
    text->replace(pos, from.size(), to);
    pos += to.size();
  }
}

llama_dart_result tokenize_bytes(const llama_vocab *vocab,
                                 const uint8_t *data, size_t size,
                                 bool add_special, bool parse_special,
                                 const char *empty_message,
                                 std::vector<llama_token> *out_tokens) {
  int32_t required = llama_tokenize(
      vocab, reinterpret_cast<const char *>(data), static_cast<int32_t>(size),
      nullptr, 0, add_special, parse_special);
  if (required == std::numeric_limits<int32_t>::min()) {
    return fail(LLAMA_DART_ERROR_INTERNAL, "tokenization overflowed");
  }
  if (required < 0) {
    required = -required;
  }
  if (required == 0) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT, empty_message);
  }
  out_tokens->resize(static_cast<size_t>(required));
  const int32_t actual = llama_tokenize(
      vocab, reinterpret_cast<const char *>(data), static_cast<int32_t>(size),
      out_tokens->data(), required, add_special, parse_special);
  if (actual < 0) {
    return fail(LLAMA_DART_ERROR_RERANKING, "failed to tokenize rerank text");
  }
  if (actual == 0) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT, empty_message);
  }
  out_tokens->resize(static_cast<size_t>(actual));
  return LLAMA_DART_SUCCESS;
}

bool append_token_piece(const llama_vocab *vocab, llama_token token,
                        std::string *out) {
  char small[256];
  int32_t written = llama_token_to_piece(vocab, token, small, sizeof(small), 0,
                                         true);
  if (written >= 0) {
    out->append(small, static_cast<size_t>(written));
    return true;
  }

  const int32_t needed = -written;
  if (needed <= 0) {
    return false;
  }
  std::vector<char> buffer(static_cast<size_t>(needed));
  written = llama_token_to_piece(vocab, token, buffer.data(),
                                 static_cast<int32_t>(buffer.size()), 0, true);
  if (written < 0) {
    return false;
  }
  out->append(buffer.data(), static_cast<size_t>(written));
  return true;
}

llama_dart_result copy_to_buffer(const std::string &text,
                                 llama_dart_buffer *out_text) {
  out_text->data = nullptr;
  out_text->size = 0;
  if (text.empty()) {
    return LLAMA_DART_SUCCESS;
  }

  void *data = std::malloc(text.size());
  if (data == nullptr) {
    return fail(LLAMA_DART_ERROR_INTERNAL, "native allocation failed");
  }
  std::memcpy(data, text.data(), text.size());
  out_text->data = static_cast<uint8_t *>(data);
  out_text->size = text.size();
  return LLAMA_DART_SUCCESS;
}

llama_dart_result copy_to_float_buffer(const float *values, size_t length,
                                       llama_dart_float_buffer *out_values) {
  out_values->data = nullptr;
  out_values->length = 0;
  if (length == 0) {
    return LLAMA_DART_SUCCESS;
  }
  if (values == nullptr) {
    return fail(LLAMA_DART_ERROR_EMBEDDING, "embedding output was null");
  }
  if (length > std::numeric_limits<size_t>::max() / sizeof(float)) {
    return fail(LLAMA_DART_ERROR_INTERNAL, "embedding output is too large");
  }
  for (size_t i = 0; i < length; ++i) {
    if (!std::isfinite(values[i])) {
      return fail(LLAMA_DART_ERROR_EMBEDDING,
                  "embedding output contained a non-finite value");
    }
  }

  void *data = std::malloc(length * sizeof(float));
  if (data == nullptr) {
    return fail(LLAMA_DART_ERROR_INTERNAL, "native allocation failed");
  }
  std::memcpy(data, values, length * sizeof(float));
  out_values->data = static_cast<float *>(data);
  out_values->length = length;
  return LLAMA_DART_SUCCESS;
}

llama_dart_result warm_up_native_context(llama_dart_context *owner,
                                         llama_context *context) {
  const llama_model *model = llama_get_model(context);
  const llama_vocab *vocab = model == nullptr ? nullptr
                                               : llama_model_get_vocab(model);
  if (model == nullptr || vocab == nullptr || llama_vocab_n_tokens(vocab) <= 0) {
    return fail(LLAMA_DART_ERROR_UNSUPPORTED,
                "warm-up requires a model vocabulary");
  }

  std::vector<llama_token> tokens;
  const llama_token bos = llama_vocab_bos(vocab);
  const llama_token eos = llama_vocab_eos(vocab);
  if (bos != LLAMA_TOKEN_NULL) {
    tokens.push_back(bos);
  }
  if (eos != LLAMA_TOKEN_NULL) {
    tokens.push_back(eos);
  }
  if (tokens.empty()) {
    tokens.push_back(0);
  }

  if (llama_model_has_encoder(model)) {
    const int32_t encoded =
        llama_encode(context, llama_batch_get_one(
                                  tokens.data(),
                                  static_cast<int32_t>(tokens.size())));
    if (is_cancelled(owner)) {
      return fail_cancelled(owner);
    }
    if (encoded != 0) {
      return fail(LLAMA_DART_ERROR_GENERATION,
                  "failed to encode warm-up tokens");
    }
    llama_token decoder_start = llama_model_decoder_start_token(model);
    if (decoder_start == LLAMA_TOKEN_NULL) {
      decoder_start = bos == LLAMA_TOKEN_NULL ? 0 : bos;
    }
    tokens.assign(1, decoder_start);
  }

  if (llama_model_has_decoder(model)) {
    const size_t count =
        std::min(tokens.size(), static_cast<size_t>(llama_n_batch(context)));
    if (count == 0) {
      return fail(LLAMA_DART_ERROR_UNSUPPORTED,
                  "warm-up requires a positive batch size");
    }
    const int32_t decoded = llama_decode(
        context, llama_batch_get_one(tokens.data(), static_cast<int32_t>(count)));
    if (decoded == 2 || is_cancelled(owner)) {
      return fail_cancelled(owner);
    }
    if (decoded != 0) {
      return fail(LLAMA_DART_ERROR_GENERATION,
                  "failed to decode warm-up tokens");
    }
  }

  llama_synchronize(context);
  if (is_cancelled(owner)) {
    return fail_cancelled(owner);
  }
  llama_perf_context_reset(context);
  last_error.clear();
  return LLAMA_DART_SUCCESS;
}

bool context_can_shift(const llama_dart_context *context) {
  if (context == nullptr || context->context == nullptr ||
      context->multimodal != nullptr || context->position < 0 ||
      context->token_history.size() !=
          static_cast<size_t>(context->position)) {
    return false;
  }
  llama_memory_t target_memory = llama_get_memory(context->context);
  if (target_memory == nullptr || !llama_memory_can_shift(target_memory)) {
    return false;
  }
  if (context->speculative_context != nullptr) {
    llama_memory_t draft_memory =
        llama_get_memory(context->speculative_context);
    if (draft_memory == nullptr || !llama_memory_can_shift(draft_memory)) {
      return false;
    }
  }
  return true;
}
} // namespace

uint32_t llama_dart_abi_version(void) { return LLAMA_DART_ABI_VERSION; }

const char *llama_dart_upstream_commit(void) {
  return LLAMA_DART_UPSTREAM_COMMIT;
}

const char *llama_dart_build_flags(void) {
  return LLAMA_DART_BUILD_FLAGS;
}

const char *llama_dart_model_file_type_name(int32_t ftype) {
  last_error.clear();
  return llama_ftype_name(static_cast<llama_ftype>(ftype));
}

const char *llama_dart_multimodal_marker(void) {
  last_error.clear();
  return mtmd_default_marker();
}

llama_dart_result llama_dart_log_set_level(uint32_t minimum_level) {
  if (minimum_level > LLAMA_DART_LOG_ERROR) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "native log level is invalid");
  }
  try {
    std::lock_guard<std::mutex> lock(log_mutex);
    captured_logs.clear();
    captured_log_bytes = 0;
    dropped_log_records = 0;
    continuation_log_level = LLAMA_DART_LOG_INFO;
    minimum_log_level.store(minimum_level, std::memory_order_relaxed);
    last_error.clear();
    return LLAMA_DART_SUCCESS;
  } catch (const std::exception &error) {
    return fail(LLAMA_DART_ERROR_INTERNAL, error.what());
  } catch (...) {
    return fail(LLAMA_DART_ERROR_INTERNAL,
                "unknown native log configuration failure");
  }
}

llama_dart_result llama_dart_log_next(uint32_t *out_level,
                                      llama_dart_buffer *out_message) {
  if (out_message != nullptr) {
    out_message->data = nullptr;
    out_message->size = 0;
  }
  if (out_level == nullptr || out_message == nullptr) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "log outputs must not be null");
  }
  *out_level = LLAMA_DART_LOG_DISABLED;
  try {
    std::lock_guard<std::mutex> lock(log_mutex);
    const bool report_dropped = dropped_log_records > 0;
    if (!report_dropped && captured_logs.empty()) {
      last_error.clear();
      return LLAMA_DART_SUCCESS;
    }

    std::string dropped_message;
    const captured_log_record *record = nullptr;
    uint32_t level = LLAMA_DART_LOG_WARNING;
    if (report_dropped) {
      dropped_message = std::to_string(dropped_log_records) +
                        " native log messages were dropped because the "
                        "capture buffer was full.";
    } else {
      record = &captured_logs.front();
      level = record->level;
    }
    const std::string &message =
        report_dropped ? dropped_message : record->message;
    void *data = std::malloc(message.size());
    if (data == nullptr) {
      return fail(LLAMA_DART_ERROR_INTERNAL, "native allocation failed");
    }
    std::memcpy(data, message.data(), message.size());
    out_message->data = static_cast<uint8_t *>(data);
    out_message->size = message.size();
    *out_level = level;
    if (report_dropped) {
      dropped_log_records = 0;
    } else {
      captured_log_bytes -= captured_logs.front().message.size();
      captured_logs.pop_front();
    }
    last_error.clear();
    return LLAMA_DART_SUCCESS;
  } catch (const std::bad_alloc &) {
    return fail(LLAMA_DART_ERROR_INTERNAL, "native allocation failed");
  } catch (const std::exception &error) {
    return fail(LLAMA_DART_ERROR_INTERNAL, error.what());
  } catch (...) {
    return fail(LLAMA_DART_ERROR_INTERNAL, "unknown log drain failure");
  }
}

llama_dart_result llama_dart_backend_init(void) {
  try {
    return backend_retain();
  } catch (const std::exception &error) {
    return fail(LLAMA_DART_ERROR_INTERNAL, error.what());
  } catch (...) {
    return fail(LLAMA_DART_ERROR_INTERNAL, "unknown backend init failure");
  }
}

llama_dart_result llama_dart_backend_free(void) {
  const llama_dart_result released = backend_release();
  if (released == LLAMA_DART_SUCCESS) {
    last_error.clear();
  }
  return released;
}

llama_dart_result
llama_dart_get_capabilities(llama_dart_capabilities *out_capabilities) {
  if (out_capabilities == nullptr) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "out_capabilities must not be null");
  }

  const llama_dart_result validation =
      validate_struct(out_capabilities->struct_size,
                      sizeof(llama_dart_capabilities),
                      "llama_dart_capabilities");
  if (validation != LLAMA_DART_SUCCESS) {
    return validation;
  }

  std::memset(out_capabilities, 0, sizeof(llama_dart_capabilities));
  out_capabilities->struct_size = sizeof(llama_dart_capabilities);
  out_capabilities->abi_version = LLAMA_DART_ABI_VERSION;
  out_capabilities->flags =
      LLAMA_DART_CAP_MODEL_LOADING | LLAMA_DART_CAP_TOKENIZATION |
      LLAMA_DART_CAP_STRUCTURED_OUTPUT | LLAMA_DART_CAP_TEXT_GENERATION |
      LLAMA_DART_CAP_EMBEDDINGS | LLAMA_DART_CAP_RERANKING |
      LLAMA_DART_CAP_MULTIMODAL | LLAMA_DART_CAP_LORA |
      LLAMA_DART_CAP_SPECULATIVE_DECODING | LLAMA_DART_CAP_MTP |
      LLAMA_DART_CAP_TOOL_CALLING | LLAMA_DART_CAP_LOGGING |
      LLAMA_DART_CAP_PREFILL;
#if LLAMA_DART_HAS_METAL
  out_capabilities->flags |= LLAMA_DART_CAP_METAL;
#endif
#if LLAMA_DART_HAS_VULKAN
  out_capabilities->flags |= LLAMA_DART_CAP_VULKAN;
#endif
  last_error.clear();
  return LLAMA_DART_SUCCESS;
}

llama_dart_result llama_dart_model_load(
    const llama_dart_model_load_config *config, llama_dart_model **out_model) {
  if (out_model == nullptr) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "out_model must not be null");
  }
  *out_model = nullptr;
  if (config == nullptr) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT, "config must not be null");
  }

  const llama_dart_result validation =
      validate_struct(config->struct_size, sizeof(llama_dart_model_load_config),
                      "llama_dart_model_load_config");
  if (validation != LLAMA_DART_SUCCESS) {
    return validation;
  }
  if (config->model_path_data == nullptr || config->model_path_size == 0) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "model_path_data must not be empty");
  }
  if (!valid_bool(config->vocab_only) || !valid_bool(config->use_mmap) ||
      !valid_bool(config->use_mlock) || !valid_bool(config->check_tensors)) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "model load boolean fields must be 0 or 1");
  }
  if (!valid_gpu_backend(config->gpu_backend)) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "gpu_backend is invalid");
  }
  if (config->n_gpu_layers < -1) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "n_gpu_layers must be -1 or non-negative");
  }
  if (config->gpu_backend == LLAMA_DART_GPU_BACKEND_CPU &&
      config->n_gpu_layers != 0) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "CPU backend requires n_gpu_layers to be zero");
  }
  if ((config->gpu_backend == LLAMA_DART_GPU_BACKEND_METAL ||
       config->gpu_backend == LLAMA_DART_GPU_BACKEND_VULKAN) &&
      config->n_gpu_layers == 0) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "explicit GPU backends require at least one GPU layer");
  }
  if (!fits_int32(config->model_path_size)) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "model path is too large");
  }
  if (is_ascii_blank(config->model_path_data, config->model_path_size)) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "model path must not be blank");
  }
  if (contains_nul(config->model_path_data, config->model_path_size)) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "model path must not contain NUL");
  }
  if (contains_line_break(config->model_path_data, config->model_path_size)) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "model path must not contain line breaks");
  }

  try {
    const llama_dart_result retained = backend_retain();
    if (retained != LLAMA_DART_SUCCESS) {
      return retained;
    }

    const std::string path(
        reinterpret_cast<const char *>(config->model_path_data),
        config->model_path_size);

    std::unique_ptr<llama_dart_model> handle =
        std::make_unique<llama_dart_model>();
    llama_model_params params = llama_model_default_params();
    std::vector<ggml_backend_dev_t> devices;
    uint32_t effective_backend = LLAMA_DART_GPU_BACKEND_CPU;
    const llama_dart_result selected =
        select_gpu_devices(config->gpu_backend, config->n_gpu_layers, &devices,
                           &effective_backend);
    if (selected != LLAMA_DART_SUCCESS) {
      backend_release();
      return selected;
    }
    params.n_gpu_layers = config->n_gpu_layers;
    if (!devices.empty()) {
      params.devices = devices.data();
    }
    params.vocab_only = config->vocab_only != 0;
    params.use_mmap = config->use_mmap != 0;
    params.use_mlock = config->use_mlock != 0;
    params.check_tensors = config->check_tensors != 0;

    llama_model *loaded = llama_model_load_from_file(path.c_str(), params);
    if (loaded == nullptr) {
      backend_release();
      return fail(LLAMA_DART_ERROR_MODEL_LOAD,
                  "llama_model_load_from_file returned null");
    }

    handle->model = loaded;
    handle->gpu_backend = effective_backend;
    handle->n_gpu_layers = params.n_gpu_layers;
    handle->use_mmap = params.use_mmap;
    handle->use_mlock = params.use_mlock;
    handle->check_tensors = params.check_tensors;
    handle->vocab_only = params.vocab_only;
    *out_model = handle.release();
    last_error.clear();
    return LLAMA_DART_SUCCESS;
  } catch (const std::bad_alloc &) {
    backend_release();
    return fail(LLAMA_DART_ERROR_INTERNAL, "native allocation failed");
  } catch (const std::exception &error) {
    backend_release();
    return fail(LLAMA_DART_ERROR_INTERNAL, error.what());
  } catch (...) {
    backend_release();
    return fail(LLAMA_DART_ERROR_INTERNAL, "unknown model load failure");
  }
}

void llama_dart_model_free(llama_dart_model *model) {
  if (model == nullptr) {
    return;
  }
  if (model->active_contexts > 0) {
    fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
         "model still has active contexts");
    return;
  }
  if (model->active_lora_adapters > 0) {
    fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
         "model still has active LoRA adapters");
    return;
  }
  if (model->model != nullptr) {
    llama_model_free(model->model);
  }
  delete model;
  if (backend_release() == LLAMA_DART_SUCCESS) {
    last_error.clear();
  }
}

llama_dart_result
llama_dart_model_get_info(const llama_dart_model *model,
                          llama_dart_model_info *out_info) {
  if (out_info == nullptr) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT, "out_info must not be null");
  }
  const llama_dart_result validation =
      validate_struct(out_info->struct_size, sizeof(llama_dart_model_info),
                      "llama_dart_model_info");
  if (validation != LLAMA_DART_SUCCESS) {
    return validation;
  }

  std::memset(out_info, 0, sizeof(llama_dart_model_info));
  out_info->struct_size = sizeof(llama_dart_model_info);
  if (model == nullptr || model->model == nullptr) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT, "model must not be null");
  }

  const llama_vocab *vocab = llama_model_get_vocab(model->model);
  out_info->vocab_type =
      vocab == nullptr ? 0 : static_cast<int32_t>(llama_vocab_type(vocab));
  out_info->n_vocab = vocab == nullptr ? 0 : llama_vocab_n_tokens(vocab);
  out_info->token_bos = vocab == nullptr ? LLAMA_TOKEN_NULL
                                         : llama_vocab_bos(vocab);
  out_info->token_eos = vocab == nullptr ? LLAMA_TOKEN_NULL
                                         : llama_vocab_eos(vocab);
  out_info->token_eot = vocab == nullptr ? LLAMA_TOKEN_NULL
                                         : llama_vocab_eot(vocab);
  out_info->token_sep = vocab == nullptr ? LLAMA_TOKEN_NULL
                                         : llama_vocab_sep(vocab);
  out_info->token_nl = vocab == nullptr ? LLAMA_TOKEN_NULL
                                        : llama_vocab_nl(vocab);
  out_info->token_pad = vocab == nullptr ? LLAMA_TOKEN_NULL
                                         : llama_vocab_pad(vocab);
  out_info->token_mask = vocab == nullptr ? LLAMA_TOKEN_NULL
                                          : llama_vocab_mask(vocab);
  out_info->add_bos =
      vocab != nullptr && llama_vocab_get_add_bos(vocab) ? 1 : 0;
  out_info->add_eos =
      vocab != nullptr && llama_vocab_get_add_eos(vocab) ? 1 : 0;
  out_info->add_sep =
      vocab != nullptr && llama_vocab_get_add_sep(vocab) ? 1 : 0;
  out_info->ftype = static_cast<int32_t>(llama_model_ftype(model->model));
  out_info->n_ctx_train = llama_model_n_ctx_train(model->model);
  out_info->n_embd = llama_model_n_embd(model->model);
  out_info->n_embd_inp = llama_model_n_embd_inp(model->model);
  out_info->n_embd_out = llama_model_n_embd_out(model->model);
  out_info->n_layer = llama_model_n_layer(model->model);
  out_info->n_layer_nextn = llama_model_n_layer_nextn(model->model);
  if (out_info->n_layer > 0) {
    out_info->n_head = llama_model_n_head(model->model);
    out_info->n_head_kv = llama_model_n_head_kv(model->model);
  }
  if (model->vocab_only) {
    try {
      std::string architecture;
      if (read_model_metadata(model->model, "general.architecture",
                              &architecture) &&
          !architecture.empty()) {
        const std::string prefix = architecture + ".";
        read_model_metadata_int32(model->model,
                                  prefix + "nextn_predict_layers",
                                  &out_info->n_layer_nextn);
        read_model_metadata_int32(model->model, "general.file_type",
                                  &out_info->ftype);
        read_model_metadata_int32(model->model, prefix + "context_length",
                                  &out_info->n_ctx_train);
        read_model_metadata_int32(model->model, prefix + "embedding_length",
                                  &out_info->n_embd);
        out_info->n_embd_inp = out_info->n_embd;
        out_info->n_embd_out = out_info->n_embd;
        read_model_metadata_int32(model->model,
                                  prefix + "embedding_length_out",
                                  &out_info->n_embd_out);
        int32_t total_layers = 0;
        if (read_model_metadata_int32(model->model, prefix + "block_count",
                                      &total_layers)) {
          out_info->n_layer =
              std::max(0, total_layers - out_info->n_layer_nextn);
        }
        read_model_metadata_int32(model->model,
                                  prefix + "attention.head_count",
                                  &out_info->n_head);
        if (!read_model_metadata_int32(model->model,
                                       prefix + "attention.head_count_kv",
                                       &out_info->n_head_kv)) {
          out_info->n_head_kv = out_info->n_head;
        }
      }
    } catch (const std::bad_alloc &) {
      return fail(LLAMA_DART_ERROR_INTERNAL, "native allocation failed");
    } catch (const std::exception &error) {
      return fail(LLAMA_DART_ERROR_INTERNAL, error.what());
    } catch (...) {
      return fail(LLAMA_DART_ERROR_INTERNAL,
                  "unknown model metadata inspection failure");
    }
  }
  out_info->size_bytes = llama_model_size(model->model);
  out_info->n_params = llama_model_n_params(model->model);
  out_info->has_encoder = llama_model_has_encoder(model->model) ? 1 : 0;
  out_info->has_decoder = llama_model_has_decoder(model->model) ? 1 : 0;
  out_info->is_recurrent = llama_model_is_recurrent(model->model) ? 1 : 0;
  out_info->is_hybrid = llama_model_is_hybrid(model->model) ? 1 : 0;
  out_info->is_diffusion = llama_model_is_diffusion(model->model) ? 1 : 0;
  last_error.clear();
  return LLAMA_DART_SUCCESS;
}

llama_dart_result llama_dart_model_get_description(
    const llama_dart_model *model, char *buffer, size_t buffer_size,
    size_t *out_size) {
  clear_char_buffer(buffer, buffer_size);
  if (out_size == nullptr) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT, "out_size must not be null");
  }
  *out_size = 0;
  if (model == nullptr || model->model == nullptr) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT, "model must not be null");
  }
  if (buffer == nullptr && buffer_size > 0) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "description buffer must not be null when capacity is positive");
  }
  if (model->vocab_only) {
    int32_t description_size = llama_model_meta_val_str(
        model->model, "general.name", nullptr, 0);
    const char fallback[] = "vocab-only GGUF";
    if (description_size < 0) {
      description_size = static_cast<int32_t>(sizeof(fallback) - 1);
    } else if (description_size > kMaxModelMetadataScalarSize) {
      return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                  "model description is too large");
    }
    *out_size = static_cast<size_t>(description_size);
    if (buffer == nullptr || buffer_size <= *out_size) {
      return fail(LLAMA_DART_ERROR_BUFFER_TOO_SMALL,
                  "description buffer is too small");
    }
    const int32_t copied = llama_model_meta_val_str(
        model->model, "general.name", buffer, buffer_size);
    if (copied < 0) {
      std::memcpy(buffer, fallback, sizeof(fallback));
    } else if (copied != description_size) {
      clear_char_buffer(buffer, buffer_size);
      return fail(LLAMA_DART_ERROR_INTERNAL,
                  "failed to copy model description");
    }
    last_error.clear();
    return LLAMA_DART_SUCCESS;
  }
  const int32_t size = llama_model_desc(model->model, buffer, buffer_size);
  if (size < 0) {
    return fail(LLAMA_DART_ERROR_INTERNAL, "llama_model_desc failed");
  }
  *out_size = static_cast<size_t>(size);
  if (buffer == nullptr || buffer_size <= static_cast<size_t>(size)) {
    return fail(LLAMA_DART_ERROR_BUFFER_TOO_SMALL,
                "description buffer is too small");
  }
  last_error.clear();
  return LLAMA_DART_SUCCESS;
}

llama_dart_result llama_dart_model_metadata_count(
    const llama_dart_model *model, size_t *out_count) {
  if (out_count == nullptr) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT, "out_count must not be null");
  }
  *out_count = 0;
  if (model == nullptr || model->model == nullptr) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT, "model must not be null");
  }
  const int32_t count = llama_model_meta_count(model->model);
  if (count < 0) {
    return fail(LLAMA_DART_ERROR_INTERNAL, "failed to read model metadata");
  }
  *out_count = static_cast<size_t>(count);
  last_error.clear();
  return LLAMA_DART_SUCCESS;
}

llama_dart_result llama_dart_model_metadata_get(
    const llama_dart_model *model, size_t index, char *key_buffer,
    size_t key_buffer_size, size_t *out_key_size, char *value_buffer,
    size_t value_buffer_size, size_t *out_value_size) {
  clear_char_buffer(key_buffer, key_buffer_size);
  clear_char_buffer(value_buffer, value_buffer_size);
  if (out_key_size == nullptr || out_value_size == nullptr) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "metadata output sizes must not be null");
  }
  *out_key_size = 0;
  *out_value_size = 0;
  if (model == nullptr || model->model == nullptr) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT, "model must not be null");
  }
  if ((key_buffer == nullptr && key_buffer_size > 0) ||
      (value_buffer == nullptr && value_buffer_size > 0)) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "metadata buffers must not be null when capacity is positive");
  }
  if (!fits_int32(index)) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "metadata index is too large");
  }

  const int32_t key_size = llama_model_meta_key_by_index(
      model->model, static_cast<int32_t>(index), nullptr, 0);
  const int32_t value_size = llama_model_meta_val_str_by_index(
      model->model, static_cast<int32_t>(index), nullptr, 0);
  if (key_size < 0 || value_size < 0) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "metadata index is out of range");
  }

  *out_key_size = static_cast<size_t>(key_size);
  *out_value_size = static_cast<size_t>(value_size);
  if (key_buffer == nullptr || value_buffer == nullptr ||
      key_buffer_size <= static_cast<size_t>(key_size) ||
      value_buffer_size <= static_cast<size_t>(value_size)) {
    return fail(LLAMA_DART_ERROR_BUFFER_TOO_SMALL,
                "metadata buffers are too small");
  }

  const int32_t copied_key = llama_model_meta_key_by_index(
      model->model, static_cast<int32_t>(index), key_buffer, key_buffer_size);
  const int32_t copied_value = llama_model_meta_val_str_by_index(
      model->model, static_cast<int32_t>(index), value_buffer,
      value_buffer_size);
  if (copied_key != key_size || copied_value != value_size) {
    return fail(LLAMA_DART_ERROR_INTERNAL,
                "failed to copy model metadata");
  }
  last_error.clear();
  return LLAMA_DART_SUCCESS;
}

llama_dart_result llama_dart_model_tokenize(
    const llama_dart_model *model, const uint8_t *text_data, size_t text_size,
    int32_t *tokens, size_t tokens_capacity, size_t *out_token_count,
    uint8_t add_special, uint8_t parse_special) {
  clear_token_buffer(tokens, tokens_capacity);
  if (out_token_count == nullptr) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "out_token_count must not be null");
  }
  *out_token_count = 0;
  if (model == nullptr || model->model == nullptr) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT, "model must not be null");
  }
  if (text_data == nullptr && text_size > 0) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "text_data must not be null when text_size is positive");
  }
  if (tokens == nullptr && tokens_capacity > 0) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "tokens must not be null when tokens_capacity is positive");
  }
  if (!fits_int32(text_size) || !fits_int32(tokens_capacity)) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "tokenization input is too large");
  }
  if (!valid_bool(add_special) || !valid_bool(parse_special)) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "tokenization boolean flags must be 0 or 1");
  }
  if (contains_nul(text_data, text_size)) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "tokenization text must not contain NUL bytes");
  }

  const llama_vocab *vocab = llama_model_get_vocab(model->model);
  if (vocab == nullptr) {
    return fail(LLAMA_DART_ERROR_UNSUPPORTED, "model has no vocabulary");
  }

  const int32_t result = llama_tokenize(
      vocab, reinterpret_cast<const char *>(text_data),
      static_cast<int32_t>(text_size), reinterpret_cast<llama_token *>(tokens),
      static_cast<int32_t>(tokens_capacity), add_special != 0,
      parse_special != 0);
  if (result == std::numeric_limits<int32_t>::min()) {
    return fail(LLAMA_DART_ERROR_INTERNAL, "tokenization overflowed");
  }
  if (result < 0) {
    *out_token_count = static_cast<size_t>(-result);
    clear_token_buffer(tokens, tokens_capacity);
    return fail(LLAMA_DART_ERROR_BUFFER_TOO_SMALL,
                "token buffer is too small");
  }

  *out_token_count = static_cast<size_t>(result);
  last_error.clear();
  return LLAMA_DART_SUCCESS;
}

llama_dart_result llama_dart_model_detokenize(
    const llama_dart_model *model, const int32_t *tokens, size_t token_count,
    uint8_t *text_data, size_t text_capacity, size_t *out_text_size,
    uint8_t remove_special, uint8_t unparse_special) {
  clear_byte_buffer(text_data, text_capacity);
  if (out_text_size == nullptr) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "out_text_size must not be null");
  }
  *out_text_size = 0;
  if (model == nullptr || model->model == nullptr) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT, "model must not be null");
  }
  if (tokens == nullptr && token_count > 0) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "tokens must not be null when token_count is positive");
  }
  if (text_data == nullptr && text_capacity > 0) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "text_data must not be null when text_capacity is positive");
  }
  if (!fits_int32(token_count) || !fits_int32(text_capacity)) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "detokenization input is too large");
  }
  if (!valid_bool(remove_special) || !valid_bool(unparse_special)) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "detokenization boolean flags must be 0 or 1");
  }

  const llama_vocab *vocab = llama_model_get_vocab(model->model);
  if (vocab == nullptr) {
    return fail(LLAMA_DART_ERROR_UNSUPPORTED, "model has no vocabulary");
  }
  const int32_t vocab_size = llama_vocab_n_tokens(vocab);
  if (vocab_size <= 0) {
    return fail(LLAMA_DART_ERROR_UNSUPPORTED, "model vocabulary is empty");
  }
  for (size_t i = 0; i < token_count; ++i) {
    if (tokens[i] < 0 || tokens[i] >= vocab_size) {
      return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                  "token id is outside the model vocabulary");
    }
  }

  const int32_t result = llama_detokenize(
      vocab, reinterpret_cast<const llama_token *>(tokens),
      static_cast<int32_t>(token_count), reinterpret_cast<char *>(text_data),
      static_cast<int32_t>(text_capacity), remove_special != 0,
      unparse_special != 0);
  if (result < 0) {
    *out_text_size = static_cast<size_t>(-result);
    clear_byte_buffer(text_data, text_capacity);
    return fail(LLAMA_DART_ERROR_BUFFER_TOO_SMALL,
                "text buffer is too small");
  }

  *out_text_size = static_cast<size_t>(result);
  last_error.clear();
  return LLAMA_DART_SUCCESS;
}

llama_dart_result llama_dart_model_apply_chat_template(
    const llama_dart_model *model, const llama_dart_chat_message *messages,
    size_t message_count, uint8_t add_assistant_prompt,
    llama_dart_buffer *out_prompt) {
  if (out_prompt == nullptr) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "out_prompt must not be null");
  }
  out_prompt->data = nullptr;
  out_prompt->size = 0;
  if (model == nullptr || model->model == nullptr) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT, "model must not be null");
  }
  if (messages == nullptr && message_count > 0) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "messages must not be null when message_count is positive");
  }
  if (message_count == 0) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "chat messages must not be empty");
  }
  if (!fits_int32(message_count)) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "too many chat messages");
  }
  if (!valid_bool(add_assistant_prompt)) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "add_assistant_prompt must be 0 or 1");
  }

  try {
    std::vector<std::string> roles;
    std::vector<std::string> contents;
    std::vector<llama_chat_message> chat;
    roles.reserve(message_count);
    contents.reserve(message_count);
    chat.reserve(message_count);

    for (size_t i = 0; i < message_count; ++i) {
      const llama_dart_chat_message &message = messages[i];
      const llama_dart_result validation =
          validate_struct(message.struct_size, sizeof(llama_dart_chat_message),
                          "llama_dart_chat_message");
      if (validation != LLAMA_DART_SUCCESS) {
        return validation;
      }
      if (message.role_data == nullptr || message.role_size == 0) {
        return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                    "chat message role must not be empty");
      }
      if (is_ascii_blank(message.role_data, message.role_size)) {
        return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                    "chat message role must not be blank");
      }
      if (message.content_data == nullptr && message.content_size > 0) {
        return fail(
            LLAMA_DART_ERROR_INVALID_ARGUMENT,
            "chat message content must not be null when content_size is "
            "positive");
      }
      if (!fits_int32(message.role_size) || !fits_int32(message.content_size)) {
        return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                    "chat message text is too large");
      }
      if (contains_nul(message.role_data, message.role_size) ||
          contains_nul(message.content_data, message.content_size)) {
        return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                    "chat messages must not contain NUL");
      }
      if (is_ascii_blank(message.content_data, message.content_size)) {
        return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                    "chat message content must not be blank");
      }
      if (contains_line_break(message.role_data, message.role_size)) {
        return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                    "chat message role must not contain line breaks");
      }
      if (!valid_chat_role(message.role_data, message.role_size)) {
        return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                    "chat message role is not supported");
      }

      roles.emplace_back(reinterpret_cast<const char *>(message.role_data),
                         message.role_size);
      contents.emplace_back(reinterpret_cast<const char *>(message.content_data),
                            message.content_size);
      chat.push_back(llama_chat_message{roles.back().c_str(),
                                        contents.back().c_str()});
    }

    const char *tmpl = llama_model_chat_template(model->model, nullptr);
    int32_t size = llama_chat_apply_template(
        tmpl, chat.data(), chat.size(), add_assistant_prompt != 0, nullptr, 0);
    if (size < 0) {
      return fail(LLAMA_DART_ERROR_UNSUPPORTED,
                  "model chat template is not supported by llama.cpp");
    }

    std::string prompt(static_cast<size_t>(size), '\0');
    size = llama_chat_apply_template(tmpl, chat.data(), chat.size(),
                                     add_assistant_prompt != 0, prompt.data(),
                                     size);
    if (size < 0) {
      return fail(LLAMA_DART_ERROR_UNSUPPORTED,
                  "model chat template is not supported by llama.cpp");
    }
    prompt.resize(static_cast<size_t>(size));

    const llama_dart_result copied = copy_to_buffer(prompt, out_prompt);
    if (copied != LLAMA_DART_SUCCESS) {
      return copied;
    }
    last_error.clear();
    return LLAMA_DART_SUCCESS;
  } catch (const std::bad_alloc &) {
    return fail(LLAMA_DART_ERROR_INTERNAL, "native allocation failed");
  } catch (const std::exception &error) {
    return fail(LLAMA_DART_ERROR_INTERNAL, error.what());
  } catch (...) {
    return fail(LLAMA_DART_ERROR_INTERNAL, "unknown chat template failure");
  }
}

llama_dart_result llama_dart_model_get_chat_template_capabilities(
    const llama_dart_model *model,
    llama_dart_chat_template_capabilities *out_capabilities) {
  if (out_capabilities == nullptr) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "out_capabilities must not be null");
  }
  const llama_dart_result validation = validate_struct(
      out_capabilities->struct_size,
      sizeof(llama_dart_chat_template_capabilities),
      "llama_dart_chat_template_capabilities");
  if (validation != LLAMA_DART_SUCCESS) {
    return validation;
  }
  std::memset(out_capabilities, 0,
              sizeof(llama_dart_chat_template_capabilities));
  out_capabilities->struct_size =
      sizeof(llama_dart_chat_template_capabilities);
  if (model == nullptr || model->model == nullptr) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT, "model must not be null");
  }

  try {
    common_chat_templates_ptr templates =
        common_chat_templates_init(model->model, "");
    if (templates == nullptr) {
      return fail(LLAMA_DART_ERROR_UNSUPPORTED,
                  "model chat template is not available");
    }
    const std::map<std::string, bool> caps =
        common_chat_templates_get_caps(templates.get());
    out_capabilities->supports_tools = caps.at("supports_tools") ? 1 : 0;
    out_capabilities->supports_tool_calls =
        caps.at("supports_tool_calls") ? 1 : 0;
    out_capabilities->supports_parallel_tool_calls =
        caps.at("supports_parallel_tool_calls") ? 1 : 0;
    last_error.clear();
    return LLAMA_DART_SUCCESS;
  } catch (const std::bad_alloc &) {
    return fail(LLAMA_DART_ERROR_INTERNAL, "native allocation failed");
  } catch (const std::exception &error) {
    return fail(LLAMA_DART_ERROR_UNSUPPORTED, error.what());
  } catch (...) {
    return fail(LLAMA_DART_ERROR_INTERNAL,
                "unknown chat template capability failure");
  }
}

llama_dart_result llama_dart_model_create_chat_plan(
    const llama_dart_model *model, const uint8_t *request_data,
    size_t request_size, llama_dart_buffer *out_plan) {
  if (out_plan == nullptr) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "out_plan must not be null");
  }
  out_plan->data = nullptr;
  out_plan->size = 0;
  if (model == nullptr || model->model == nullptr) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT, "model must not be null");
  }

  nlohmann::ordered_json request;
  const llama_dart_result parsed = parse_json_object(
      request_data, request_size, "chat request", &request);
  if (parsed != LLAMA_DART_SUCCESS) {
    return parsed;
  }

  common_chat_templates_inputs inputs;
  try {
    inputs.messages =
        common_chat_msgs_parse_oaicompat(request.at("messages"));
    if (inputs.messages.empty()) {
      return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                  "chat messages must not be empty");
    }
    const nlohmann::ordered_json tools =
        request.value("tools", nlohmann::ordered_json::array());
    inputs.tools = common_chat_tools_parse_oaicompat(tools);
    const std::string tool_choice = request.value("tool_choice", "auto");
    inputs.tool_choice = common_chat_tool_choice_parse_oaicompat(tool_choice);
    inputs.parallel_tool_calls =
        request.value("parallel_tool_calls", false);
    inputs.add_generation_prompt =
        request.value("add_generation_prompt", true);
    inputs.use_jinja = true;
    inputs.grammar = request.value("grammar", "");
    if (request.contains("json_schema") &&
        !request.at("json_schema").is_null()) {
      if (!request.at("json_schema").is_object()) {
        return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                    "json_schema must be an object");
      }
      inputs.json_schema = request.at("json_schema").dump();
    }
    if (!inputs.grammar.empty() && !inputs.json_schema.empty()) {
      return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                  "JSON schema and grammar are mutually exclusive");
    }
    if (!inputs.tools.empty() &&
        inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE &&
        (!inputs.grammar.empty() || !inputs.json_schema.empty())) {
      return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                  "active tools cannot be combined with structured output");
    }
    if (inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED &&
        inputs.tools.empty()) {
      return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                  "required tool choice needs at least one tool");
    }
  } catch (const std::bad_alloc &) {
    return fail(LLAMA_DART_ERROR_INTERNAL, "native allocation failed");
  } catch (const nlohmann::json::exception &error) {
    return fail_parts(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                      "invalid chat request: ", error.what());
  } catch (const std::exception &error) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT, error.what());
  } catch (...) {
    return fail(LLAMA_DART_ERROR_INTERNAL,
                "unknown chat request validation failure");
  }

  try {
    common_chat_templates_ptr templates =
        common_chat_templates_init(model->model, "");
    if (templates == nullptr) {
      return fail(LLAMA_DART_ERROR_UNSUPPORTED,
                  "model chat template is not available");
    }
    const std::map<std::string, bool> caps =
        common_chat_templates_get_caps(templates.get());
    const bool active_tools =
        !inputs.tools.empty() &&
        inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE;
    if (active_tools &&
        (!caps.at("supports_tools") || !caps.at("supports_tool_calls"))) {
      return fail(LLAMA_DART_ERROR_UNSUPPORTED,
                  "model chat template does not support tools");
    }
    const bool has_tool_history = std::any_of(
        inputs.messages.begin(), inputs.messages.end(),
        [](const common_chat_msg &message) {
          return !message.tool_calls.empty() || message.role == "tool" ||
                 !message.tool_call_id.empty();
        });
    if (has_tool_history && !caps.at("supports_tool_calls")) {
      return fail(LLAMA_DART_ERROR_UNSUPPORTED,
                  "model chat template does not support tool-call history");
    }
    if (inputs.parallel_tool_calls &&
        !caps.at("supports_parallel_tool_calls")) {
      inputs.parallel_tool_calls = false;
    }
    const common_chat_params params =
        common_chat_templates_apply(templates.get(), inputs);
    const std::string plan = serialize_chat_plan(params).dump();
    const llama_dart_result copied = copy_to_buffer(plan, out_plan);
    if (copied != LLAMA_DART_SUCCESS) {
      return copied;
    }
    last_error.clear();
    return LLAMA_DART_SUCCESS;
  } catch (const std::bad_alloc &) {
    return fail(LLAMA_DART_ERROR_INTERNAL, "native allocation failed");
  } catch (const std::exception &error) {
    return fail_parts(LLAMA_DART_ERROR_UNSUPPORTED,
                      "chat template could not render tools: ", error.what());
  } catch (...) {
    return fail(LLAMA_DART_ERROR_INTERNAL,
                "unknown chat plan creation failure");
  }
}

llama_dart_result llama_dart_chat_parse_output(
    const uint8_t *plan_data, size_t plan_size, const uint8_t *output_data,
    size_t output_size, llama_dart_buffer *out_message_json) {
  if (out_message_json == nullptr) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "out_message_json must not be null");
  }
  out_message_json->data = nullptr;
  out_message_json->size = 0;
  if (output_data == nullptr && output_size > 0) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "output_data must not be null when output_size is positive");
  }
  if (!fits_int32(output_size) || contains_nul(output_data, output_size)) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "chat output is too large or contains NUL");
  }

  parsed_chat_plan plan;
  const llama_dart_result parsed =
      parse_chat_plan(plan_data, plan_size, &plan);
  if (parsed != LLAMA_DART_SUCCESS) {
    return parsed;
  }

  try {
    common_chat_parser_params parser_params;
    parser_params.format = plan.format;
    parser_params.generation_prompt = plan.generation_prompt;
    parser_params.parse_tool_calls = true;
    if (!plan.parser.empty()) {
      parser_params.parser.load(plan.parser);
    }
    const std::string output(
        reinterpret_cast<const char *>(output_data), output_size);
    const common_chat_msg message =
        common_chat_parse(output, false, parser_params);
    const std::string message_json =
        message.to_json_oaicompat().dump();
    const llama_dart_result copied =
        copy_to_buffer(message_json, out_message_json);
    if (copied != LLAMA_DART_SUCCESS) {
      return copied;
    }
    last_error.clear();
    return LLAMA_DART_SUCCESS;
  } catch (const std::bad_alloc &) {
    return fail(LLAMA_DART_ERROR_INTERNAL, "native allocation failed");
  } catch (const std::exception &error) {
    return fail_parts(LLAMA_DART_ERROR_GENERATION,
                      "failed to parse chat output: ", error.what());
  } catch (...) {
    return fail(LLAMA_DART_ERROR_INTERNAL,
                "unknown chat output parsing failure");
  }
}

llama_dart_result llama_dart_json_schema_to_grammar(
    const uint8_t *schema_data, size_t schema_size,
    llama_dart_buffer *out_grammar) {
  if (out_grammar == nullptr) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "out_grammar must not be null");
  }
  out_grammar->data = nullptr;
  out_grammar->size = 0;
  try {
    std::string grammar;
    const llama_dart_result converted =
        convert_json_schema_to_grammar(schema_data, schema_size, &grammar);
    if (converted != LLAMA_DART_SUCCESS) {
      return converted;
    }
    const llama_dart_result copied = copy_to_buffer(grammar, out_grammar);
    if (copied != LLAMA_DART_SUCCESS) {
      return copied;
    }
    last_error.clear();
    return LLAMA_DART_SUCCESS;
  } catch (const std::bad_alloc &) {
    return fail(LLAMA_DART_ERROR_INTERNAL, "native allocation failed");
  } catch (const std::exception &error) {
    return fail(LLAMA_DART_ERROR_INTERNAL, error.what());
  } catch (...) {
    return fail(LLAMA_DART_ERROR_INTERNAL,
                "unknown JSON schema conversion failure");
  }
}

llama_dart_result llama_dart_context_create(
    llama_dart_model *model, const llama_dart_context_config *config,
    llama_dart_context **out_context) {
  if (out_context == nullptr) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "out_context must not be null");
  }
  *out_context = nullptr;
  if (model == nullptr || model->model == nullptr) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT, "model must not be null");
  }
  if (config == nullptr) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT, "config must not be null");
  }

  const llama_dart_result validation =
      validate_struct(config->struct_size, sizeof(llama_dart_context_config),
                      "llama_dart_context_config");
  if (validation != LLAMA_DART_SUCCESS) {
    return validation;
  }
  if (config->context_size == 0) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "context_size must be positive");
  }
  if (config->batch_size == 0) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "batch_size must be positive");
  }
  if (config->ubatch_size == 0) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "ubatch_size must be positive");
  }
  if (config->ubatch_size > config->batch_size) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "ubatch_size must not exceed batch_size");
  }
  if (config->threads < 0) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "threads must be non-negative");
  }
  if (config->batch_threads < 0) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "batch_threads must be non-negative");
  }
  if (!valid_bool(config->embeddings)) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "embeddings must be 0 or 1");
  }
  if (!valid_bool(config->mmproj_use_gpu)) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "mmproj_use_gpu must be 0 or 1");
  }
  if (config->mmproj_path_data == nullptr &&
      config->mmproj_path_size > 0) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "mmproj_path_data must not be null when mmproj_path_size is "
                "positive");
  }
  if (config->mmproj_path_data != nullptr &&
      config->mmproj_path_size == 0) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "mmproj path must not be empty");
  }
  if (!fits_int32(config->mmproj_path_size)) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "mmproj path is too large");
  }
  if (config->mmproj_path_size > 0 &&
      is_ascii_blank(config->mmproj_path_data, config->mmproj_path_size)) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "mmproj path must not be blank");
  }
  if (contains_nul(config->mmproj_path_data, config->mmproj_path_size)) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "mmproj path must not contain NUL");
  }
  if (contains_line_break(config->mmproj_path_data,
                          config->mmproj_path_size)) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "mmproj path must not contain line breaks");
  }
  if (config->mmproj_path_size == 0 && config->mmproj_use_gpu != 0) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "mmproj_use_gpu requires an mmproj path");
  }
  if (config->embeddings != 0 && config->mmproj_path_size > 0) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "embedding contexts cannot load an mmproj");
  }
  if (config->mmproj_path_size > 0 &&
      config->batch_size >
          static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "multimodal batch_size must fit int32");
  }
  if (!valid_pooling_type(config->pooling_type)) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "pooling_type is invalid");
  }
  if (!valid_attention_type(config->attention_type)) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "attention_type is invalid");
  }
  if (!valid_kv_cache_type(config->kv_cache_key_type)) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "kv_cache_key_type is invalid");
  }
  if (!valid_kv_cache_type(config->kv_cache_value_type)) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "kv_cache_value_type is invalid");
  }
  if (!valid_flash_attention(config->flash_attention)) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "flash_attention is invalid");
  }
  if (!valid_bool(config->kv_cache_offload)) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "kv_cache_offload must be 0 or 1");
  }
  if (!valid_bool(config->swa_full)) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "swa_full must be 0 or 1");
  }
  if (!valid_bool(config->kv_unified)) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "kv_unified must be 0 or 1");
  }
  if (config->flash_attention == LLAMA_DART_FLASH_ATTENTION_DISABLED &&
      ggml_is_quantized(kv_cache_type_for(config->kv_cache_value_type))) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "quantized V cache types require Flash Attention");
  }
  if (!valid_speculative_type(config->speculative_type)) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "speculative_type is invalid");
  }
  if (config->speculative_model_path_data == nullptr &&
      config->speculative_model_path_size > 0) {
    return fail(
        LLAMA_DART_ERROR_INVALID_ARGUMENT,
        "speculative_model_path_data must not be null when its size is "
        "positive");
  }
  if (config->speculative_model_path_data != nullptr &&
      config->speculative_model_path_size == 0) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "speculative model path must not be empty");
  }
  if (!fits_int32(config->speculative_model_path_size)) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "speculative model path is too large");
  }
  if (config->speculative_model_path_size > 0 &&
      is_ascii_blank(config->speculative_model_path_data,
                     config->speculative_model_path_size)) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "speculative model path must not be blank");
  }
  if (contains_nul(config->speculative_model_path_data,
                   config->speculative_model_path_size)) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "speculative model path must not contain NUL");
  }
  if (contains_line_break(config->speculative_model_path_data,
                          config->speculative_model_path_size)) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "speculative model path must not contain line breaks");
  }
  if (config->speculative_ngram_n > 1024 ||
      config->speculative_ngram_m > 1024 ||
      config->speculative_ngram_min_draft > 1024) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "speculative ngram values must be <= 1024");
  }
  const bool model_backed_speculation =
      is_model_backed_speculation(config->speculative_type);
  const bool ngram_speculation =
      is_ngram_speculation(config->speculative_type);
  const bool ngram_map_speculation =
      is_ngram_map_speculation(config->speculative_type);
  if (model_backed_speculation &&
      (!fits_int32(config->context_size) ||
       !fits_int32(config->batch_size) ||
       !fits_int32(config->ubatch_size))) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "model-backed context and batch sizes must fit int32");
  }
  if (ngram_map_speculation) {
    if (config->speculative_ngram_n == 0 ||
        config->speculative_ngram_m < config->speculative_ngram_n ||
        config->speculative_ngram_min_draft != 0) {
      return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                  "ngram map speculation requires positive n <= m and no "
                  "minimum draft length");
    }
  } else if (config->speculative_type ==
             LLAMA_DART_SPECULATIVE_NGRAM_MOD) {
    if (config->speculative_ngram_n == 0 ||
        config->speculative_ngram_m == 0 ||
        config->speculative_ngram_min_draft == 0 ||
        config->speculative_ngram_min_draft >
            config->speculative_ngram_m) {
      return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                  "ngram-mod requires positive match/min/max values with "
                  "minimum <= maximum");
    }
  } else if (config->speculative_type ==
             LLAMA_DART_SPECULATIVE_NGRAM_CACHE) {
    if (config->speculative_ngram_n != 0 ||
        config->speculative_ngram_m != 0 ||
        config->speculative_ngram_min_draft != 0) {
      return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                  "ngram-cache uses fixed upstream sizing");
    }
  } else if (config->speculative_ngram_n > 0 ||
             config->speculative_ngram_m > 0 ||
             config->speculative_ngram_min_draft > 0) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "ngram values require an ngram speculative strategy");
  }
  if (ngram_speculation &&
      (config->speculative_model_path_size > 0 ||
       config->speculative_draft_max != 0)) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "ngram speculation does not use model-backed settings");
  }
  if (model_backed_speculation) {
    if (config->speculative_draft_max == 0 ||
        config->speculative_draft_max > 1024) {
      return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                  "speculative_draft_max must be between 1 and 1024");
    }
    if (config->speculative_type != LLAMA_DART_SPECULATIVE_MTP &&
        config->speculative_model_path_size == 0) {
      return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                  "draft-model, EAGLE-3, and DFlash speculation require a "
                  "model path");
    }
    if (config->mmproj_path_size > 0) {
      return fail(LLAMA_DART_ERROR_UNSUPPORTED,
                  "model-backed speculation cannot use an mmproj context");
    }
  } else if (config->speculative_model_path_size > 0 ||
             config->speculative_draft_max > 0) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "speculative model settings require model-backed speculation");
  }
  if (config->embeddings != 0 &&
      config->speculative_type != LLAMA_DART_SPECULATIVE_NONE) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "embedding contexts cannot use speculative decoding");
  }
  if (ngram_speculation &&
      (llama_model_is_recurrent(model->model) ||
       llama_model_is_hybrid(model->model))) {
    return fail(LLAMA_DART_ERROR_UNSUPPORTED,
                "ngram speculative decoding is not supported for recurrent or "
                "hybrid models");
  }
  if (model->vocab_only) {
    return fail(LLAMA_DART_ERROR_CONTEXT_CREATE,
                "cannot create context from vocab-only model");
  }

  try {
    llama_context_params params = llama_context_default_params();
    params.n_ctx = config->context_size;
    params.n_batch = config->batch_size;
    params.n_ubatch = config->ubatch_size;
    params.n_threads = config->threads;
    params.n_threads_batch = config->batch_threads;
    params.embeddings = config->embeddings != 0;
    params.pooling_type =
        static_cast<enum llama_pooling_type>(config->pooling_type);
    params.attention_type =
        static_cast<enum llama_attention_type>(config->attention_type);
    params.type_k = kv_cache_type_for(config->kv_cache_key_type);
    params.type_v = kv_cache_type_for(config->kv_cache_value_type);
    params.flash_attn_type = flash_attention_for(config->flash_attention);
    const bool context_gpu_enabled =
        model->gpu_backend != LLAMA_DART_GPU_BACKEND_CPU;
    params.offload_kqv =
        context_gpu_enabled && config->kv_cache_offload != 0;
    params.op_offload = context_gpu_enabled;
    params.swa_full = config->swa_full != 0;
    params.kv_unified = config->kv_unified != 0;
    if (config->speculative_type == LLAMA_DART_SPECULATIVE_EAGLE3 ||
        config->speculative_type == LLAMA_DART_SPECULATIVE_DFLASH ||
        config->speculative_type == LLAMA_DART_SPECULATIVE_MTP) {
      params.n_rs_seq = config->speculative_draft_max;
    }
    if (params.embeddings &&
        params.attention_type != LLAMA_ATTENTION_TYPE_CAUSAL) {
      params.n_ubatch = params.n_batch;
    }

    llama_context_ptr target_owner;
    std::unique_ptr<llama_dart_context> handle(new llama_dart_context);
    handle->model = model;
    handle->speculative_type = config->speculative_type;
    handle->speculative_draft_max = config->speculative_draft_max;
    handle->speculative_ngram_n = config->speculative_ngram_n;
    handle->speculative_ngram_m = config->speculative_ngram_m;
    handle->kv_cache_key_type = config->kv_cache_key_type;
    handle->kv_cache_value_type = config->kv_cache_value_type;
    handle->flash_attention = config->flash_attention;
    handle->kv_cache_offload = params.offload_kqv;
    handle->swa_full = config->swa_full != 0;
    handle->kv_unified = config->kv_unified != 0;
    params.abort_callback = context_abort_callback;
    params.abort_callback_data = handle.get();
    const std::string mmproj_path =
        config->mmproj_path_size == 0
            ? std::string()
            : std::string(
                  reinterpret_cast<const char *>(config->mmproj_path_data),
                  config->mmproj_path_size);

    llama_context *created = llama_init_from_model(model->model, params);
    if (created == nullptr) {
      return fail(LLAMA_DART_ERROR_CONTEXT_CREATE,
                  "llama_init_from_model returned null");
    }

    target_owner.reset(created);
    handle->context = created;
    if (ngram_speculation) {
      const common_context_seq_rm_type target_removal =
          common_context_can_seq_rm(created);
      if (target_removal == COMMON_CONTEXT_SEQ_RM_TYPE_NO ||
          target_removal == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
        return fail(LLAMA_DART_ERROR_UNSUPPORTED,
                    "ngram speculation requires removable context memory");
      }

      handle->speculative_params.types = {
          common_speculative_type_for(config->speculative_type)};
      common_params_speculative_ngram_map ngram_params;
      ngram_params.size_n =
          static_cast<uint16_t>(config->speculative_ngram_n);
      ngram_params.size_m =
          static_cast<uint16_t>(config->speculative_ngram_m);
      switch (config->speculative_type) {
      case LLAMA_DART_SPECULATIVE_NGRAM_SIMPLE:
        handle->speculative_params.ngram_simple = ngram_params;
        break;
      case LLAMA_DART_SPECULATIVE_NGRAM_MAP_K:
        handle->speculative_params.ngram_map_k = ngram_params;
        break;
      case LLAMA_DART_SPECULATIVE_NGRAM_MAP_K4V:
        handle->speculative_params.ngram_map_k4v = ngram_params;
        break;
      case LLAMA_DART_SPECULATIVE_NGRAM_MOD:
        handle->speculative_params.ngram_mod.n_match =
            static_cast<int32_t>(config->speculative_ngram_n);
        handle->speculative_params.ngram_mod.n_min =
            static_cast<int32_t>(config->speculative_ngram_min_draft);
        handle->speculative_params.ngram_mod.n_max =
            static_cast<int32_t>(config->speculative_ngram_m);
        break;
      case LLAMA_DART_SPECULATIVE_NGRAM_CACHE:
        break;
      default:
        return fail(LLAMA_DART_ERROR_INTERNAL,
                    "invalid ngram speculation type");
      }
      try {
        handle->speculative.reset(
            common_speculative_init(handle->speculative_params, 1));
      } catch (const std::bad_alloc &) {
        throw;
      } catch (const std::exception &error) {
        return fail(LLAMA_DART_ERROR_UNSUPPORTED, error.what());
      }
      if (handle->speculative == nullptr) {
        return fail(LLAMA_DART_ERROR_UNSUPPORTED,
                    "failed to initialize ngram speculative decoding");
      }
    }
    if (model_backed_speculation) {
      common_params speculative_params;
      speculative_params.n_ctx = static_cast<int32_t>(config->context_size);
      speculative_params.n_batch = static_cast<int32_t>(config->batch_size);
      speculative_params.n_ubatch = static_cast<int32_t>(config->ubatch_size);
      speculative_params.n_parallel = 1;
      speculative_params.cpuparams.n_threads = config->threads;
      speculative_params.cpuparams_batch.n_threads = config->batch_threads;
      speculative_params.cache_type_k = params.type_k;
      speculative_params.cache_type_v = params.type_v;
      speculative_params.flash_attn_type = params.flash_attn_type;
      speculative_params.no_kv_offload = !params.offload_kqv;
      speculative_params.no_op_offload = !params.op_offload;
      speculative_params.swa_full = params.swa_full;
      speculative_params.kv_unified = params.kv_unified;
      speculative_params.speculative.types = {
          common_speculative_type_for(config->speculative_type)};
      speculative_params.speculative.draft.n_max =
          static_cast<int32_t>(config->speculative_draft_max);
      speculative_params.speculative.draft.cache_type_k = params.type_k;
      speculative_params.speculative.draft.cache_type_v = params.type_v;
      speculative_params.speculative.draft.ctx_tgt = created;

      const std::string speculative_model_path =
          config->speculative_model_path_size == 0
              ? std::string()
              : std::string(reinterpret_cast<const char *>(
                                config->speculative_model_path_data),
                            config->speculative_model_path_size);
      if (!speculative_model_path.empty()) {
        speculative_params.model.path = speculative_model_path;
        speculative_params.speculative.draft.mparams.path =
            speculative_model_path;
      }

      const int32_t draft_gpu_layers = model->n_gpu_layers;
      speculative_params.n_gpu_layers = draft_gpu_layers;
      speculative_params.speculative.draft.n_gpu_layers = draft_gpu_layers;
      speculative_params.use_mmap = model->use_mmap;
      speculative_params.use_mlock = model->use_mlock;
      speculative_params.check_tensors = model->check_tensors;
      uint32_t effective_backend = LLAMA_DART_GPU_BACKEND_AUTO;
      const llama_dart_result devices_selected = select_gpu_devices(
          model->gpu_backend, draft_gpu_layers, &speculative_params.devices,
          &effective_backend);
      if (devices_selected != LLAMA_DART_SUCCESS) {
        return devices_selected;
      }
      speculative_params.speculative.draft.devices =
          speculative_params.devices;

      speculative_params =
          common_base_params_to_speculative(speculative_params);

      common_init_speculative_result_ptr speculative_init =
          common_init_speculative_from_params(speculative_params,
                                              model->model, created);
      if (speculative_init == nullptr) {
        return fail(LLAMA_DART_ERROR_INTERNAL,
                    "upstream speculative initialization returned null");
      }
      if (!speculative_model_path.empty() &&
          speculative_init->model() == nullptr) {
        return fail(LLAMA_DART_ERROR_MODEL_LOAD,
                    "failed to load the speculative model");
      }
      if (speculative_init->context() == nullptr) {
        return fail(speculative_model_path.empty()
                        ? LLAMA_DART_ERROR_UNSUPPORTED
                        : LLAMA_DART_ERROR_CONTEXT_CREATE,
                    speculative_model_path.empty()
                        ? "target model does not expose usable MTP heads"
                        : "failed to initialize the speculative context");
      }

      llama_context *draft_context = speculative_init->context();
      const common_context_seq_rm_type target_removal =
          common_context_can_seq_rm(created);
      const common_context_seq_rm_type draft_removal =
          common_context_can_seq_rm(draft_context);
      if (target_removal == COMMON_CONTEXT_SEQ_RM_TYPE_NO ||
          target_removal == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
          draft_removal == COMMON_CONTEXT_SEQ_RM_TYPE_NO ||
          draft_removal == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
        return fail(LLAMA_DART_ERROR_UNSUPPORTED,
                    "model-backed speculation requires removable context "
                    "memory");
      }
      llama_set_abort_callback(draft_context, context_abort_callback,
                               handle.get());
      speculative_params.speculative.draft.ctx_dft = draft_context;
      handle->speculative_context = draft_context;
      handle->speculative_params = speculative_params.speculative;
      try {
        handle->speculative.reset(
            common_speculative_init(handle->speculative_params, 1));
      } catch (const std::bad_alloc &) {
        throw;
      } catch (const std::exception &error) {
        return fail(LLAMA_DART_ERROR_UNSUPPORTED, error.what());
      }
      if (handle->speculative == nullptr) {
        return fail(LLAMA_DART_ERROR_UNSUPPORTED,
                    "failed to initialize speculative decoding strategy");
      }
      handle->speculative_init = std::move(speculative_init);
    }
    if (config->mmproj_path_size > 0) {
      mtmd_context_params mmproj_params = mtmd_context_params_default();
      mmproj_params.use_gpu = config->mmproj_use_gpu != 0;
      mmproj_params.print_timings = false;
      if (config->threads > 0) {
        mmproj_params.n_threads = config->threads;
      }
      mmproj_params.batch_max_tokens =
          static_cast<int32_t>(config->batch_size);
      handle->multimodal = mtmd_init_from_file(
          mmproj_path.c_str(), model->model, mmproj_params);
      if (handle->multimodal == nullptr) {
        return fail(LLAMA_DART_ERROR_CONTEXT_CREATE,
                    "mtmd_init_from_file returned null");
      }
    }
    model->active_contexts += 1;
    target_owner.release();
    *out_context = handle.release();
    last_error.clear();
    return LLAMA_DART_SUCCESS;
  } catch (const std::bad_alloc &) {
    return fail(LLAMA_DART_ERROR_INTERNAL, "native allocation failed");
  } catch (const std::exception &error) {
    return fail(LLAMA_DART_ERROR_INTERNAL, error.what());
  } catch (...) {
    return fail(LLAMA_DART_ERROR_INTERNAL, "unknown context create failure");
  }
}

void llama_dart_context_free(llama_dart_context *context) {
  if (context == nullptr) {
    return;
  }
  if (context->active_generations > 0) {
    fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
         "context still has active generations");
    return;
  }
  release_context_lora_adapters(context);
  context->speculative.reset();
  context->speculative_init.reset();
  context->speculative_context = nullptr;
  if (context->multimodal != nullptr) {
    mtmd_free(context->multimodal);
  }
  if (context->context != nullptr) {
    llama_free(context->context);
  }
  if (context->model != nullptr && context->model->active_contexts > 0) {
    context->model->active_contexts -= 1;
  }
  delete context;
  last_error.clear();
}

llama_dart_result llama_dart_context_reset(llama_dart_context *context) {
  if (context == nullptr || context->context == nullptr) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT, "context must not be null");
  }
  if (context->active_generations > 0) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "context still has active generations");
  }
  try {
    context->cancel_requested.store(false, std::memory_order_relaxed);
    const bool has_speculation =
        context->speculative_type != LLAMA_DART_SPECULATIVE_NONE;
    if (has_speculation) {
      context->speculative.reset();
    }
    if (context->speculative_context != nullptr) {
      llama_memory_t speculative_memory =
          llama_get_memory(context->speculative_context);
      if (speculative_memory == nullptr) {
        return fail(LLAMA_DART_ERROR_UNSUPPORTED,
                    "speculative context has no clearable memory");
      }
      llama_memory_clear(speculative_memory, true);
    }
    llama_memory_t memory = llama_get_memory(context->context);
    if (memory == nullptr) {
      return fail(LLAMA_DART_ERROR_UNSUPPORTED,
                  "context has no clearable memory");
    }
    llama_memory_clear(memory, true);
    if (has_speculation) {
      context->speculative.reset(
          common_speculative_init(context->speculative_params, 1));
      if (context->speculative == nullptr) {
        return fail(LLAMA_DART_ERROR_INTERNAL,
                    "failed to reset speculative decoding state");
      }
    }
    context->position = 0;
    context->token_history.clear();
    context->speculative_needs_warmup = false;
    last_error.clear();
    return LLAMA_DART_SUCCESS;
  } catch (const std::bad_alloc &) {
    return fail(LLAMA_DART_ERROR_INTERNAL, "native allocation failed");
  } catch (const std::exception &error) {
    return fail(LLAMA_DART_ERROR_INTERNAL, error.what());
  } catch (...) {
    return fail(LLAMA_DART_ERROR_INTERNAL,
                "unknown speculative reset failure");
  }
}

llama_dart_result llama_dart_context_warm_up(llama_dart_context *context) {
  if (context == nullptr || context->context == nullptr ||
      context->model == nullptr || context->model->model == nullptr) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT, "context must not be null");
  }
  if (context->active_generations > 0) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "context still has active generations");
  }
  if (context->position != 0 || !context->token_history.empty()) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "warm-up requires an empty context; call reset first");
  }
  if (is_cancelled(context)) {
    return fail_cancelled(context);
  }

  try {
    const llama_dart_result prepared = llama_dart_context_reset(context);
    if (prepared != LLAMA_DART_SUCCESS) {
      return prepared;
    }
    llama_dart_result result =
        warm_up_native_context(context, context->context);
    if (result == LLAMA_DART_SUCCESS &&
        context->speculative_context != nullptr) {
      result = warm_up_native_context(context, context->speculative_context);
    }
    const last_error_storage warm_up_error = last_error;
    const llama_dart_result reset = llama_dart_context_reset(context);
    if (reset != LLAMA_DART_SUCCESS) {
      return reset;
    }
    if (result != LLAMA_DART_SUCCESS) {
      last_error = warm_up_error;
      return result;
    }
    last_error.clear();
    return LLAMA_DART_SUCCESS;
  } catch (const std::bad_alloc &) {
    llama_dart_context_reset(context);
    return fail(LLAMA_DART_ERROR_INTERNAL, "native allocation failed");
  } catch (const std::exception &error) {
    llama_dart_context_reset(context);
    return fail(LLAMA_DART_ERROR_INTERNAL, error.what());
  } catch (...) {
    llama_dart_context_reset(context);
    return fail(LLAMA_DART_ERROR_INTERNAL, "unknown warm-up failure");
  }
}

llama_dart_result llama_dart_context_cancel(llama_dart_context *context) {
  if (context == nullptr || context->context == nullptr) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT, "context must not be null");
  }
  context->cancel_requested.store(true, std::memory_order_relaxed);
  last_error.clear();
  return LLAMA_DART_SUCCESS;
}

llama_dart_result llama_dart_context_get_info(
    const llama_dart_context *context, llama_dart_context_info *out_info) {
  if (out_info == nullptr) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT, "out_info must not be null");
  }
  const llama_dart_result validation =
      validate_struct(out_info->struct_size, sizeof(llama_dart_context_info),
                      "llama_dart_context_info");
  if (validation != LLAMA_DART_SUCCESS) {
    return validation;
  }

  std::memset(out_info, 0, sizeof(llama_dart_context_info));
  out_info->struct_size = sizeof(llama_dart_context_info);
  if (context == nullptr || context->context == nullptr) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT, "context must not be null");
  }
  out_info->context_size = llama_n_ctx(context->context);
  out_info->sequence_context_size = llama_n_ctx_seq(context->context);
  out_info->batch_size = llama_n_batch(context->context);
  out_info->ubatch_size = llama_n_ubatch(context->context);
  out_info->max_sequences = llama_n_seq_max(context->context);
  out_info->supports_vision =
      context->multimodal != nullptr &&
              mtmd_support_vision(context->multimodal)
          ? 1
          : 0;
  out_info->supports_audio =
      context->multimodal != nullptr &&
              mtmd_support_audio(context->multimodal)
          ? 1
          : 0;
  out_info->gpu_backend =
      context->model == nullptr ? LLAMA_DART_GPU_BACKEND_CPU
                                : context->model->gpu_backend;
  out_info->supports_context_shift = context_can_shift(context) ? 1 : 0;
  out_info->used_tokens = static_cast<uint32_t>(context->position);
  out_info->kv_cache_key_type = context->kv_cache_key_type;
  out_info->kv_cache_value_type = context->kv_cache_value_type;
  out_info->flash_attention = context->flash_attention;
  out_info->kv_cache_offload = context->kv_cache_offload ? 1 : 0;
  out_info->swa_full = context->swa_full ? 1 : 0;
  out_info->kv_unified = context->kv_unified ? 1 : 0;
  last_error.clear();
  return LLAMA_DART_SUCCESS;
}

llama_dart_result llama_dart_context_state_get(
    llama_dart_context *context, llama_dart_buffer *out_state) {
  if (out_state == nullptr) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT, "out_state must not be null");
  }
  out_state->data = nullptr;
  out_state->size = 0;
  if (context == nullptr || context->context == nullptr) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT, "context must not be null");
  }
  if (context->active_generations > 0) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "context still has active generations");
  }
  try {
    const size_t target_size = llama_state_get_size(context->context);
    if (target_size == 0) {
      return fail(LLAMA_DART_ERROR_INTERNAL,
                  "target context state is empty");
    }
    const bool has_model_backed_speculation =
        is_model_backed_speculation(context->speculative_type);
    const size_t draft_size =
        has_model_backed_speculation && context->speculative_context != nullptr
            ? llama_state_get_size(context->speculative_context)
            : 0;
    if (has_model_backed_speculation && draft_size == 0) {
      return fail(LLAMA_DART_ERROR_INTERNAL,
                  "speculative context state is empty");
    }

    std::vector<uint8_t> speculative_state;
    if (context->speculative != nullptr &&
        !common_speculative_get_state(context->speculative.get(), 0,
                                      speculative_state)) {
      speculative_state.clear();
    }

    const state_snapshot_layout layout{
        /* .speculative_type = */ context->speculative_type,
        /* .target_size = */ static_cast<uint64_t>(target_size),
        /* .draft_size = */ static_cast<uint64_t>(draft_size),
        /* .speculative_size = */
        static_cast<uint64_t>(speculative_state.size()),
        /* .token_count = */
        static_cast<uint64_t>(context->token_history.size()),
        /* .position = */ context->position,
    };
    size_t snapshot_size = 0;
    if (!state_snapshot_size(layout, &snapshot_size)) {
      return fail(LLAMA_DART_ERROR_INTERNAL,
                  "context state snapshot is too large");
    }
    uint8_t *data = static_cast<uint8_t *>(std::malloc(snapshot_size));
    if (data == nullptr) {
      return fail(LLAMA_DART_ERROR_INTERNAL, "native allocation failed");
    }
    write_state_snapshot_header(data, layout);
    size_t offset = kStateSnapshotHeaderSize;
    if (llama_state_get_data(context->context, data + offset, target_size) !=
        target_size) {
      std::free(data);
      return fail(LLAMA_DART_ERROR_INTERNAL,
                  "failed to copy target context state");
    }
    offset += target_size;
    if (draft_size > 0) {
      if (llama_state_get_data(context->speculative_context, data + offset,
                               draft_size) != draft_size) {
        std::free(data);
        return fail(LLAMA_DART_ERROR_INTERNAL,
                    "failed to copy speculative context state");
      }
      offset += draft_size;
    }
    if (!speculative_state.empty()) {
      std::memcpy(data + offset, speculative_state.data(),
                  speculative_state.size());
      offset += speculative_state.size();
    }
    for (const llama_token token : context->token_history) {
      uint32_t encoded = 0;
      std::memcpy(&encoded, &token, sizeof(token));
      write_u32_le(data + offset, encoded);
      offset += sizeof(uint32_t);
    }
    finalize_state_snapshot(data, snapshot_size);
    out_state->data = data;
    out_state->size = snapshot_size;
    last_error.clear();
    return LLAMA_DART_SUCCESS;
  } catch (const std::bad_alloc &) {
    return fail(LLAMA_DART_ERROR_INTERNAL, "native allocation failed");
  } catch (const std::exception &error) {
    return fail(LLAMA_DART_ERROR_INTERNAL, error.what());
  } catch (...) {
    return fail(LLAMA_DART_ERROR_INTERNAL,
                "unknown context state capture failure");
  }
}

llama_dart_result llama_dart_context_state_set(
    llama_dart_context *context, const uint8_t *state_data, size_t state_size) {
  if (context == nullptr || context->context == nullptr) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT, "context must not be null");
  }
  if (state_data == nullptr || state_size == 0) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "state_data must not be empty");
  }
  if (context->active_generations > 0) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "context still has active generations");
  }
  try {
    state_snapshot_view snapshot;
    std::string snapshot_error;
    const state_snapshot_decode_result decoded = decode_state_snapshot(
        state_data, state_size, &snapshot, &snapshot_error);
    const bool has_model_backed_speculation =
        is_model_backed_speculation(context->speculative_type);
    if (decoded == state_snapshot_decode_result::not_snapshot) {
      if (has_model_backed_speculation) {
        return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                    "legacy target-only state cannot restore a model-backed "
                    "speculative context");
      }
      // Legacy ABI 26 and earlier snapshots contain only upstream target state.
      const size_t read =
          llama_state_set_data(context->context, state_data, state_size);
      if (read != state_size) {
        return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                    "failed to restore legacy context state");
      }
      llama_memory_t memory = llama_get_memory(context->context);
      const llama_pos max_pos =
          memory == nullptr ? -1 : llama_memory_seq_pos_max(memory, 0);
      context->position = max_pos < 0 ? 0 : max_pos + 1;
      context->token_history.clear();
      context->speculative_needs_warmup = false;
      context->cancel_requested.store(false, std::memory_order_relaxed);
      last_error.clear();
      return LLAMA_DART_SUCCESS;
    }
    if (decoded == state_snapshot_decode_result::invalid) {
      return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT, snapshot_error);
    }
    if (snapshot.layout.speculative_type != context->speculative_type) {
      return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                  "snapshot speculative strategy does not match context");
    }
    if (has_model_backed_speculation !=
            (snapshot.layout.draft_size > 0) ||
        (!has_model_backed_speculation &&
         snapshot.layout.speculative_size > 0) ||
        (has_model_backed_speculation &&
         context->speculative_context == nullptr)) {
      return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                  "snapshot sections do not match context capabilities");
    }
    if (snapshot.layout.position > llama_n_ctx(context->context) ||
        snapshot.layout.token_count > llama_n_ctx(context->context)) {
      return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                  "snapshot exceeds the context size");
    }

    const llama_vocab *vocab = llama_model_get_vocab(context->model->model);
    const int32_t vocab_size = llama_vocab_n_tokens(vocab);
    std::vector<llama_token> restored_tokens(
        static_cast<size_t>(snapshot.layout.token_count));
    for (size_t i = 0; i < restored_tokens.size(); ++i) {
      const uint32_t encoded =
          read_u32_le(snapshot.tokens + i * sizeof(uint32_t));
      llama_token token = 0;
      std::memcpy(&token, &encoded, sizeof(token));
      if (token < 0 || token >= vocab_size) {
        return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                    "snapshot token history contains an invalid token id");
      }
      restored_tokens[i] = token;
    }

    std::vector<uint8_t> target_backup;
    llama_dart_result captured = capture_context_state(
        context->context, "target", &target_backup);
    if (captured != LLAMA_DART_SUCCESS) {
      return captured;
    }
    std::vector<uint8_t> draft_backup;
    if (has_model_backed_speculation) {
      captured = capture_context_state(context->speculative_context,
                                       "speculative", &draft_backup);
      if (captured != LLAMA_DART_SUCCESS) {
        return captured;
      }
    }
    std::vector<uint8_t> old_speculative_state;
    const bool had_old_speculative_state =
        context->speculative != nullptr &&
        common_speculative_get_state(context->speculative.get(), 0,
                                     old_speculative_state);
    const bool old_needs_warmup = context->speculative_needs_warmup;
    auto rollback_contexts = [&]() {
      restore_context_state_best_effort(context->context, target_backup);
      if (has_model_backed_speculation) {
        restore_context_state_best_effort(context->speculative_context,
                                          draft_backup);
      }
    };

    if (!restore_context_state(
            context->context, snapshot.target,
            static_cast<size_t>(snapshot.layout.target_size))) {
      rollback_contexts();
      return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                  "failed to restore target context state");
    }
    llama_memory_t target_memory = llama_get_memory(context->context);
    const llama_pos target_max =
        target_memory == nullptr
            ? -1
            : llama_memory_seq_pos_max(target_memory, 0);
    const int64_t restored_position = target_max < 0 ? 0 : target_max + 1;
    if (restored_position != snapshot.layout.position) {
      rollback_contexts();
      return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                  "snapshot position does not match target context state");
    }
    if (has_model_backed_speculation &&
        !restore_context_state(
            context->speculative_context, snapshot.draft,
            static_cast<size_t>(snapshot.layout.draft_size))) {
      rollback_contexts();
      return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                  "failed to restore speculative context state");
    }

    const bool has_speculation =
        context->speculative_type != LLAMA_DART_SPECULATIVE_NONE;
    if (has_speculation) {
      context->speculative.reset();
      try {
        context->speculative.reset(
            common_speculative_init(context->speculative_params, 1));
      } catch (...) {
        context->speculative.reset();
      }
      if (context->speculative == nullptr) {
        rollback_contexts();
        try {
          context->speculative.reset(
              common_speculative_init(context->speculative_params, 1));
          if (had_old_speculative_state && context->speculative != nullptr) {
            common_speculative_set_state(context->speculative.get(), 0,
                                         old_speculative_state);
          }
        } catch (...) {
          context->speculative.reset();
        }
        context->speculative_needs_warmup =
            has_model_backed_speculation &&
            (old_needs_warmup ||
             (context->speculative_type !=
                  LLAMA_DART_SPECULATIVE_DRAFT_MODEL &&
              !had_old_speculative_state));
        return fail(LLAMA_DART_ERROR_INTERNAL,
                    "failed to recreate speculative decoding state");
      }
      if (has_model_backed_speculation &&
          snapshot.layout.speculative_size > 0) {
        std::vector<uint8_t> speculative_state(
            snapshot.speculative,
            snapshot.speculative +
                static_cast<size_t>(snapshot.layout.speculative_size));
        common_speculative_set_state(context->speculative.get(), 0,
                                     speculative_state);
      }
      context->speculative_needs_warmup =
          has_model_backed_speculation &&
          context->speculative_type !=
              LLAMA_DART_SPECULATIVE_DRAFT_MODEL &&
          snapshot.layout.speculative_size == 0;
    } else {
      context->speculative_needs_warmup = false;
    }

    context->position = static_cast<int32_t>(snapshot.layout.position);
    context->token_history = std::move(restored_tokens);
    context->cancel_requested.store(false, std::memory_order_relaxed);
    last_error.clear();
    return LLAMA_DART_SUCCESS;
  } catch (const std::bad_alloc &) {
    return fail(LLAMA_DART_ERROR_INTERNAL, "native allocation failed");
  } catch (const std::exception &error) {
    return fail(LLAMA_DART_ERROR_INTERNAL, error.what());
  } catch (...) {
    return fail(LLAMA_DART_ERROR_INTERNAL,
                "unknown context state restore failure");
  }
}

llama_dart_result llama_dart_context_shift(
    llama_dart_context *context, uint32_t keep_tokens,
    uint32_t discard_tokens, uint32_t *out_discarded_tokens) {
  if (out_discarded_tokens == nullptr) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "out_discarded_tokens must not be null");
  }
  *out_discarded_tokens = 0;
  if (context == nullptr || context->context == nullptr) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT, "context must not be null");
  }
  if (context->active_generations > 0) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "context still has active generations");
  }
  if (!context_can_shift(context)) {
    return fail(LLAMA_DART_ERROR_UNSUPPORTED,
                context->multimodal != nullptr
                    ? "context shift is unsupported for multimodal contexts"
                    : "context memory or token history cannot be shifted");
  }
  if (is_cancelled(context)) {
    return fail_cancelled(context);
  }
  if (context->position < 2 ||
      keep_tokens >= static_cast<uint32_t>(context->position - 1)) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "context shift must retain at least one tail token");
  }
  const uint32_t removable =
      static_cast<uint32_t>(context->position) - keep_tokens;
  const uint32_t discarded =
      discard_tokens == 0 ? removable / 2 : discard_tokens;
  if (discarded == 0 || discarded >= removable) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "discard_tokens must leave at least one tail token");
  }

  std::vector<llama_token> next_history;
  std::vector<uint8_t> target_backup;
  std::vector<uint8_t> draft_backup;
  common_speculative_ptr shifted_speculative;
  bool mutated = false;
  auto restore_backups = [&]() {
    bool restored = restore_context_state(
        context->context, target_backup.data(), target_backup.size());
    if (context->speculative_context != nullptr) {
      restored =
          restore_context_state(context->speculative_context,
                                draft_backup.data(), draft_backup.size()) &&
          restored;
    }
    return restored;
  };
  auto fail_after_mutation = [&](llama_dart_result result,
                                 const char *message) {
    if (!restore_backups()) {
      return fail(LLAMA_DART_ERROR_INTERNAL,
                  "context shift failed and rollback was unsuccessful");
    }
    return fail(result, message);
  };

  try {
    next_history = context->token_history;
    next_history.erase(
        next_history.begin() + static_cast<std::ptrdiff_t>(keep_tokens),
        next_history.begin() +
            static_cast<std::ptrdiff_t>(keep_tokens + discarded));
    llama_dart_result captured = capture_context_state(
        context->context, "target", &target_backup);
    if (captured != LLAMA_DART_SUCCESS) {
      return captured;
    }
    if (context->speculative_context != nullptr) {
      captured = capture_context_state(context->speculative_context,
                                       "speculative", &draft_backup);
      if (captured != LLAMA_DART_SUCCESS) {
        return captured;
      }
    }

    llama_memory_t target_memory = llama_get_memory(context->context);
    if (!llama_memory_seq_rm(target_memory, 0,
                             static_cast<llama_pos>(keep_tokens),
                             static_cast<llama_pos>(keep_tokens + discarded))) {
      return fail(LLAMA_DART_ERROR_UNSUPPORTED,
                  "target context could not remove the requested token range");
    }
    mutated = true;
    llama_memory_seq_add(target_memory, 0,
                         static_cast<llama_pos>(keep_tokens + discarded),
                         context->position, -static_cast<llama_pos>(discarded));

    if (context->speculative_context != nullptr) {
      llama_memory_t draft_memory =
          llama_get_memory(context->speculative_context);
      if (!llama_memory_seq_rm(
              draft_memory, 0, static_cast<llama_pos>(keep_tokens),
              static_cast<llama_pos>(keep_tokens + discarded))) {
        return fail_after_mutation(
            LLAMA_DART_ERROR_UNSUPPORTED,
            "speculative context could not remove the requested token range");
      }
      llama_memory_seq_add(
          draft_memory, 0,
          static_cast<llama_pos>(keep_tokens + discarded), context->position,
          -static_cast<llama_pos>(discarded));
    }

    const llama_pos expected_position =
        context->position - static_cast<llama_pos>(discarded);
    if (llama_memory_seq_pos_max(target_memory, 0) + 1 != expected_position) {
      return fail_after_mutation(
          LLAMA_DART_ERROR_INTERNAL,
          "target context position is inconsistent after shifting");
    }
    if (context->speculative_context != nullptr) {
      llama_memory_t draft_memory =
          llama_get_memory(context->speculative_context);
      if (llama_memory_seq_pos_max(draft_memory, 0) + 1 !=
          expected_position) {
        return fail_after_mutation(
            LLAMA_DART_ERROR_INTERNAL,
            "speculative context position is inconsistent after shifting");
      }
    }
    if (is_cancelled(context)) {
      if (!restore_backups()) {
        return fail(LLAMA_DART_ERROR_INTERNAL,
                    "context shift cancellation rollback was unsuccessful");
      }
      return fail_cancelled(context);
    }

    if (is_ngram_speculation(context->speculative_type)) {
      shifted_speculative.reset(
          common_speculative_init(context->speculative_params, 1));
      if (shifted_speculative == nullptr) {
        return fail_after_mutation(
            LLAMA_DART_ERROR_INTERNAL,
            "failed to rebuild ngram state after context shift");
      }
    }

    context->position = expected_position;
    context->token_history = std::move(next_history);
    if (shifted_speculative != nullptr) {
      context->speculative = std::move(shifted_speculative);
    }
    *out_discarded_tokens = discarded;
    last_error.clear();
    return LLAMA_DART_SUCCESS;
  } catch (const std::bad_alloc &) {
    if (mutated && !restore_backups()) {
      return fail(LLAMA_DART_ERROR_INTERNAL,
                  "context shift allocation failed and rollback was "
                  "unsuccessful");
    }
    return fail(LLAMA_DART_ERROR_INTERNAL, "native allocation failed");
  } catch (const std::exception &error) {
    if (mutated && !restore_backups()) {
      return fail(LLAMA_DART_ERROR_INTERNAL,
                  "context shift failed and rollback was unsuccessful");
    }
    return fail(LLAMA_DART_ERROR_INTERNAL, error.what());
  } catch (...) {
    if (mutated && !restore_backups()) {
      return fail(LLAMA_DART_ERROR_INTERNAL,
                  "context shift failed and rollback was unsuccessful");
    }
    return fail(LLAMA_DART_ERROR_INTERNAL, "unknown context shift failure");
  }
}

llama_dart_result llama_dart_context_complete(
    llama_dart_context *context, const llama_dart_completion_config *config,
    llama_dart_buffer *out_text, llama_dart_completion_stats *out_stats) {
  if (out_text == nullptr) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT, "out_text must not be null");
  }
  out_text->data = nullptr;
  out_text->size = 0;
  if (out_stats != nullptr) {
    const llama_dart_result stats_validation = validate_struct(
        out_stats->struct_size, sizeof(llama_dart_completion_stats),
        "llama_dart_completion_stats");
    if (stats_validation != LLAMA_DART_SUCCESS) {
      return stats_validation;
    }
    std::memset(out_stats, 0, sizeof(llama_dart_completion_stats));
    out_stats->struct_size = sizeof(llama_dart_completion_stats);
  }
  if (config == nullptr) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT, "config must not be null");
  }

  const llama_vocab *vocab = nullptr;
  const llama_dart_result validation =
      validate_completion_request(context, config, &vocab);
  if (validation != LLAMA_DART_SUCCESS) {
    return validation;
  }

  try {
    const steady_clock::time_point total_start = steady_clock::now();
    double prompt_eval_ms = 0.0;
    double decode_ms = 0.0;
    double time_to_first_token_ms = 0.0;
    uint32_t generated_tokens = 0;
    uint32_t speculative_draft_tokens = 0;
    uint32_t speculative_accepted_tokens = 0;
    double speculative_draft_ms = 0.0;
    double speculative_verify_ms = 0.0;
    parsed_chat_plan chat_plan;
    bool has_chat_plan = false;
    const llama_dart_result plan_parsed =
        completion_chat_plan(config, &chat_plan, &has_chat_plan);
    if (plan_parsed != LLAMA_DART_SUCCESS) {
      return plan_parsed;
    }
    std::string grammar;
    std::string grammar_root;
    const llama_dart_result grammar_built =
        completion_grammar(config, &grammar, &grammar_root,
                           has_chat_plan ? &chat_plan : nullptr);
    if (grammar_built != LLAMA_DART_SUCCESS) {
      return grammar_built;
    }
    std::vector<std::string> stop_sequences;
    stop_sequences.reserve(
        config->stop_sequence_count +
        (has_chat_plan ? chat_plan.additional_stops.size() : 0));
    for (size_t i = 0; i < config->stop_sequence_count; ++i) {
      stop_sequences.emplace_back(
          reinterpret_cast<const char *>(config->stop_sequences[i].data),
          config->stop_sequences[i].size);
    }
    if (has_chat_plan) {
      append_unique_stops(chat_plan.additional_stops, &stop_sequences);
    }
    std::vector<llama_token> stop_tokens;
    if (config->stop_token_count > 0) {
      stop_tokens.assign(config->stop_tokens,
                         config->stop_tokens + config->stop_token_count);
    }

    std::vector<llama_token> prompt_tokens;
    uint32_t prompt_token_count = 0;

    llama_sampler *sampler = nullptr;
    const llama_dart_result sampler_created = create_completion_sampler(
        context, vocab, config, grammar, grammar_root,
        has_chat_plan ? &chat_plan : nullptr, &sampler);
    if (sampler_created != LLAMA_DART_SUCCESS) {
      return sampler_created;
    }
    if (config->prompt_size > 0) {
      const steady_clock::time_point prompt_start = steady_clock::now();
      const llama_dart_result decoded = decode_completion_prompt(
          context, vocab, config, &prompt_tokens, &prompt_token_count);
      const steady_clock::time_point prompt_end = steady_clock::now();
      prompt_eval_ms = elapsed_ms(prompt_start, prompt_end);
      if (decoded != LLAMA_DART_SUCCESS) {
        llama_sampler_free(sampler);
        return decoded;
      }
      if (grammar.empty()) {
        accept_sampler_tokens(sampler, prompt_tokens);
      }
      if (prompt_token_count > 0) {
        context->speculative_needs_warmup = false;
      }
    } else if (context->position == 0) {
      llama_sampler_free(sampler);
      return fail(LLAMA_DART_ERROR_GENERATION,
                  "prompt must not be empty for first completion");
    }
    common_speculative_begin(context->speculative.get(), 0,
                             context->token_history);

    std::string generated;
    const steady_clock::time_point decode_start = steady_clock::now();
    while (generated_tokens < config->max_tokens) {
      uint32_t produced = 0;
      bool done = false;
      uint32_t drafted = 0;
      uint32_t accepted = 0;
      double draft_ms = 0.0;
      double verify_ms = 0.0;
      const llama_dart_result decoded = decode_completion_token(
          context, vocab, sampler, stop_sequences, stop_tokens,
          config->max_tokens - generated_tokens, &generated, &produced, &done,
          &drafted, &accepted, &draft_ms, &verify_ms);
      if (decoded != LLAMA_DART_SUCCESS) {
        llama_sampler_free(sampler);
        return decoded;
      }
      speculative_draft_tokens += drafted;
      speculative_accepted_tokens += accepted;
      speculative_draft_ms += draft_ms;
      speculative_verify_ms += verify_ms;
      if (produced > 0) {
        if (generated_tokens == 0) {
          time_to_first_token_ms = elapsed_ms(total_start, steady_clock::now());
        }
        generated_tokens += produced;
      }
      if (done) {
        break;
      }
    }
    const steady_clock::time_point decode_end = steady_clock::now();
    decode_ms = elapsed_ms(decode_start, decode_end);

    llama_sampler_free(sampler);
    const llama_dart_result copied = copy_to_buffer(generated, out_text);
    if (copied != LLAMA_DART_SUCCESS) {
      return copied;
    }
    if (out_stats != nullptr) {
      out_stats->prompt_tokens = prompt_token_count;
      out_stats->generated_tokens = generated_tokens;
      out_stats->prompt_eval_ms = prompt_eval_ms;
      out_stats->decode_ms = decode_ms;
      out_stats->total_ms = elapsed_ms(total_start, steady_clock::now());
      out_stats->time_to_first_token_ms = time_to_first_token_ms;
      out_stats->speculative_draft_tokens = speculative_draft_tokens;
      out_stats->speculative_accepted_tokens = speculative_accepted_tokens;
      out_stats->speculative_draft_ms = speculative_draft_ms;
      out_stats->speculative_verify_ms = speculative_verify_ms;
    }
    context->cancel_requested.store(false, std::memory_order_relaxed);
    last_error.clear();
    return LLAMA_DART_SUCCESS;
  } catch (const std::bad_alloc &) {
    return fail(LLAMA_DART_ERROR_INTERNAL, "native allocation failed");
  } catch (const std::exception &error) {
    return fail(LLAMA_DART_ERROR_INTERNAL, error.what());
  } catch (...) {
    return fail(LLAMA_DART_ERROR_INTERNAL, "unknown generation failure");
  }
}

llama_dart_result llama_dart_generation_start(
    llama_dart_context *context, const llama_dart_completion_config *config,
    llama_dart_generation **out_generation) {
  if (out_generation == nullptr) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "out_generation must not be null");
  }
  *out_generation = nullptr;

  const llama_vocab *vocab = nullptr;
  const llama_dart_result validation =
      validate_completion_request(context, config, &vocab);
  if (validation != LLAMA_DART_SUCCESS) {
    return validation;
  }

  try {
    std::unique_ptr<llama_dart_generation> generation(
        new llama_dart_generation);
    generation->context = context;
    generation->vocab = vocab;
    generation->max_tokens = config->max_tokens;
    generation->total_start = steady_clock::now();

    parsed_chat_plan chat_plan;
    bool has_chat_plan = false;
    const llama_dart_result plan_parsed =
        completion_chat_plan(config, &chat_plan, &has_chat_plan);
    if (plan_parsed != LLAMA_DART_SUCCESS) {
      return plan_parsed;
    }
    std::string grammar;
    std::string grammar_root;
    const llama_dart_result grammar_built =
        completion_grammar(config, &grammar, &grammar_root,
                           has_chat_plan ? &chat_plan : nullptr);
    if (grammar_built != LLAMA_DART_SUCCESS) {
      return grammar_built;
    }
    generation->stop_sequences.reserve(
        config->stop_sequence_count +
        (has_chat_plan ? chat_plan.additional_stops.size() : 0));
    for (size_t i = 0; i < config->stop_sequence_count; ++i) {
      generation->stop_sequences.emplace_back(
          reinterpret_cast<const char *>(config->stop_sequences[i].data),
          config->stop_sequences[i].size);
    }
    if (has_chat_plan) {
      append_unique_stops(chat_plan.additional_stops,
                          &generation->stop_sequences);
    }
    if (config->stop_token_count > 0) {
      generation->stop_tokens.assign(
          config->stop_tokens,
          config->stop_tokens + config->stop_token_count);
    }
    for (const std::string &stop : generation->stop_sequences) {
      const size_t holdback = stop.empty() ? 0 : stop.size() - 1;
      if (holdback > generation->stop_holdback) {
        generation->stop_holdback = holdback;
      }
    }

    std::vector<llama_token> prompt_tokens;
    uint32_t prompt_token_count = 0;

    llama_sampler *sampler = nullptr;
    const llama_dart_result sampler_created = create_completion_sampler(
        context, vocab, config, grammar, grammar_root,
        has_chat_plan ? &chat_plan : nullptr, &sampler);
    if (sampler_created != LLAMA_DART_SUCCESS) {
      return sampler_created;
    }
    if (config->prompt_size > 0) {
      const steady_clock::time_point prompt_start = steady_clock::now();
      const llama_dart_result decoded = decode_completion_prompt(
          context, vocab, config, &prompt_tokens, &prompt_token_count);
      const steady_clock::time_point prompt_end = steady_clock::now();
      generation->prompt_eval_ms = elapsed_ms(prompt_start, prompt_end);
      if (decoded != LLAMA_DART_SUCCESS) {
        llama_sampler_free(sampler);
        return decoded;
      }
      if (grammar.empty()) {
        accept_sampler_tokens(sampler, prompt_tokens);
      }
      if (prompt_token_count > 0) {
        context->speculative_needs_warmup = false;
      }
      generation->prompt_tokens = prompt_token_count;
    } else if (context->position == 0) {
      llama_sampler_free(sampler);
      return fail(LLAMA_DART_ERROR_GENERATION,
                  "prompt must not be empty for first completion");
    }
    common_speculative_begin(context->speculative.get(), 0,
                             context->token_history);

    generation->sampler = sampler;
    generation->decode_start = steady_clock::now();
    context->active_generations += 1;
    *out_generation = generation.release();
    last_error.clear();
    return LLAMA_DART_SUCCESS;
  } catch (const std::bad_alloc &) {
    return fail(LLAMA_DART_ERROR_INTERNAL, "native allocation failed");
  } catch (const std::exception &error) {
    return fail(LLAMA_DART_ERROR_INTERNAL, error.what());
  } catch (...) {
    return fail(LLAMA_DART_ERROR_INTERNAL, "unknown generation start failure");
  }
}

llama_dart_result llama_dart_generation_next(
    llama_dart_generation *generation, llama_dart_buffer *out_text,
    llama_dart_completion_stats *out_stats, uint8_t *out_done) {
  if (out_text == nullptr) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT, "out_text must not be null");
  }
  out_text->data = nullptr;
  out_text->size = 0;
  if (out_stats != nullptr) {
    const llama_dart_result stats_validation = validate_struct(
        out_stats->struct_size, sizeof(llama_dart_completion_stats),
        "llama_dart_completion_stats");
    if (stats_validation != LLAMA_DART_SUCCESS) {
      return stats_validation;
    }
    std::memset(out_stats, 0, sizeof(llama_dart_completion_stats));
    out_stats->struct_size = sizeof(llama_dart_completion_stats);
  }
  if (out_done == nullptr) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT, "out_done must not be null");
  }
  *out_done = 0;
  if (generation == nullptr || generation->context == nullptr ||
      generation->sampler == nullptr || generation->vocab == nullptr) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "generation must not be null");
  }

  try {
    if (!generation->done &&
        generation->generated_tokens < generation->max_tokens) {
      uint32_t produced = 0;
      bool done = false;
      uint32_t drafted = 0;
      uint32_t accepted = 0;
      double draft_ms = 0.0;
      double verify_ms = 0.0;
      const llama_dart_result decoded = decode_completion_token(
          generation->context, generation->vocab, generation->sampler,
          generation->stop_sequences, generation->stop_tokens,
          generation->max_tokens - generation->generated_tokens,
          &generation->generated, &produced, &done, &drafted, &accepted,
          &draft_ms, &verify_ms);
      if (decoded != LLAMA_DART_SUCCESS) {
        return decoded;
      }
      generation->speculative_draft_tokens += drafted;
      generation->speculative_accepted_tokens += accepted;
      generation->speculative_draft_ms += draft_ms;
      generation->speculative_verify_ms += verify_ms;
      if (produced > 0) {
        if (generation->generated_tokens == 0) {
          generation->time_to_first_token_ms =
              elapsed_ms(generation->total_start, steady_clock::now());
        }
        generation->generated_tokens += produced;
      }
      generation->done =
          done || generation->generated_tokens >= generation->max_tokens;
    } else {
      generation->done = true;
    }

    size_t safe_size = generation->generated.size();
    if (!generation->done) {
      safe_size = safe_size > generation->stop_holdback
                      ? safe_size - generation->stop_holdback
                      : 0;
    }
    if (generation->emitted_size > safe_size) {
      generation->emitted_size = safe_size;
    }
    std::string chunk;
    if (safe_size > generation->emitted_size) {
      chunk = generation->generated.substr(
          generation->emitted_size, safe_size - generation->emitted_size);
      generation->emitted_size = safe_size;
    }

    const llama_dart_result copied = copy_to_buffer(chunk, out_text);
    if (copied != LLAMA_DART_SUCCESS) {
      return copied;
    }
    *out_done = generation->done ? 1 : 0;
    if (out_stats != nullptr) {
      out_stats->prompt_tokens = generation->prompt_tokens;
      out_stats->generated_tokens = generation->generated_tokens;
      out_stats->prompt_eval_ms = generation->prompt_eval_ms;
      out_stats->decode_ms =
          elapsed_ms(generation->decode_start, steady_clock::now());
      out_stats->total_ms =
          elapsed_ms(generation->total_start, steady_clock::now());
      out_stats->time_to_first_token_ms =
          generation->time_to_first_token_ms;
      out_stats->speculative_draft_tokens =
          generation->speculative_draft_tokens;
      out_stats->speculative_accepted_tokens =
          generation->speculative_accepted_tokens;
      out_stats->speculative_draft_ms = generation->speculative_draft_ms;
      out_stats->speculative_verify_ms = generation->speculative_verify_ms;
    }
    last_error.clear();
    return LLAMA_DART_SUCCESS;
  } catch (const std::bad_alloc &) {
    return fail(LLAMA_DART_ERROR_INTERNAL, "native allocation failed");
  } catch (const std::exception &error) {
    return fail(LLAMA_DART_ERROR_INTERNAL, error.what());
  } catch (...) {
    return fail(LLAMA_DART_ERROR_INTERNAL, "unknown generation next failure");
  }
}

void llama_dart_generation_free(llama_dart_generation *generation) {
  if (generation == nullptr) {
    return;
  }
  if (generation->sampler != nullptr) {
    llama_sampler_free(generation->sampler);
  }
  if (generation->context != nullptr &&
      generation->context->active_generations > 0) {
    generation->context->cancel_requested.store(false,
                                                std::memory_order_relaxed);
    generation->context->active_generations -= 1;
  }
  delete generation;
  last_error.clear();
}

llama_dart_result llama_dart_context_embed(
    llama_dart_context *context, const llama_dart_embedding_config *config,
    llama_dart_float_buffer *out_embedding) {
  if (out_embedding == nullptr) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "out_embedding must not be null");
  }
  out_embedding->data = nullptr;
  out_embedding->length = 0;
  if (config == nullptr) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT, "config must not be null");
  }

  const llama_dart_result validation =
      validate_struct(config->struct_size, sizeof(llama_dart_embedding_config),
                      "llama_dart_embedding_config");
  if (validation != LLAMA_DART_SUCCESS) {
    return validation;
  }
  if (config->text_data == nullptr && config->text_size > 0) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "text_data must not be null when text_size is positive");
  }
  if (!valid_bool(config->add_special) || !valid_bool(config->parse_special)) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "embedding boolean fields must be 0 or 1");
  }
  if (config->text_size == 0) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "embedding text must not be empty");
  }
  if (!fits_int32(config->text_size)) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "embedding input is too large");
  }
  if (is_ascii_blank(config->text_data, config->text_size)) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "embedding text must not be blank");
  }
  if (contains_nul(config->text_data, config->text_size)) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "embedding text must not contain NUL bytes");
  }
  if (context == nullptr || context->context == nullptr ||
      context->model == nullptr || context->model->model == nullptr) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT, "context must not be null");
  }
  if (context->active_generations > 0) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "context still has active generations");
  }
  if (is_cancelled(context)) {
    return fail_cancelled(context);
  }

  llama_model *model = context->model->model;
  if (llama_model_has_encoder(model) && llama_model_has_decoder(model)) {
    return fail(LLAMA_DART_ERROR_UNSUPPORTED,
                "encoder-decoder embeddings are not supported");
  }

  const enum llama_pooling_type pooling = llama_pooling_type(context->context);
  if (pooling == LLAMA_POOLING_TYPE_NONE) {
    return fail(LLAMA_DART_ERROR_UNSUPPORTED,
                "model does not expose pooled sequence embeddings");
  }
  if (pooling == LLAMA_POOLING_TYPE_RANK) {
    return fail(LLAMA_DART_ERROR_UNSUPPORTED,
                "reranking models are not exposed through embedText");
  }

  const llama_vocab *vocab = llama_model_get_vocab(model);
  if (vocab == nullptr) {
    return fail(LLAMA_DART_ERROR_UNSUPPORTED, "model has no vocabulary");
  }

  try {
    int32_t required =
        llama_tokenize(vocab, reinterpret_cast<const char *>(config->text_data),
                       static_cast<int32_t>(config->text_size), nullptr, 0,
                       config->add_special != 0, config->parse_special != 0);
    if (required == std::numeric_limits<int32_t>::min()) {
      return fail(LLAMA_DART_ERROR_INTERNAL, "tokenization overflowed");
    }
    if (required < 0) {
      required = -required;
    }
    if (required == 0) {
      return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                  "embedding text produced no tokens");
    }
    if (required > static_cast<int32_t>(llama_n_batch(context->context))) {
      return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                  "embedding input exceeds context batch size");
    }

    std::vector<llama_token> tokens(static_cast<size_t>(required));
    const int32_t actual = llama_tokenize(
        vocab, reinterpret_cast<const char *>(config->text_data),
        static_cast<int32_t>(config->text_size), tokens.data(), required,
        config->add_special != 0, config->parse_special != 0);
    if (actual < 0) {
      return fail(LLAMA_DART_ERROR_EMBEDDING,
                  "failed to tokenize embedding text");
    }
    if (actual == 0) {
      return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                  "embedding text produced no tokens");
    }
    tokens.resize(static_cast<size_t>(actual));

    llama_memory_t memory = llama_get_memory(context->context);
    if (memory != nullptr) {
      llama_memory_clear(memory, true);
    }

    llama_batch batch = llama_batch_init(actual, 0, 1);
    if (batch.token == nullptr || batch.pos == nullptr ||
        batch.n_seq_id == nullptr || batch.seq_id == nullptr ||
        batch.logits == nullptr) {
      llama_batch_free(batch);
      return fail(LLAMA_DART_ERROR_INTERNAL, "native allocation failed");
    }

    batch.n_tokens = actual;
    for (int32_t i = 0; i < actual; ++i) {
      batch.token[i] = tokens[static_cast<size_t>(i)];
      batch.pos[i] = i;
      batch.n_seq_id[i] = 1;
      batch.seq_id[i][0] = 0;
      batch.logits[i] = 1;
    }

    const int32_t decoded = llama_decode(context->context, batch);
    llama_batch_free(batch);
    if (decoded == 2 || is_cancelled(context)) {
      return fail_cancelled(context);
    }
    if (decoded != 0) {
      return fail(LLAMA_DART_ERROR_EMBEDDING,
                  "failed to decode embedding input");
    }

    const float *embedding = llama_get_embeddings_seq(context->context, 0);
    const int32_t length = llama_model_n_embd_out(model);
    if (length <= 0) {
      return fail(LLAMA_DART_ERROR_EMBEDDING,
                  "model reported an invalid embedding size");
    }

    const llama_dart_result copied = copy_to_float_buffer(
        embedding, static_cast<size_t>(length), out_embedding);
    if (copied != LLAMA_DART_SUCCESS) {
      return copied;
    }
    last_error.clear();
    return LLAMA_DART_SUCCESS;
  } catch (const std::bad_alloc &) {
    return fail(LLAMA_DART_ERROR_INTERNAL, "native allocation failed");
  } catch (const std::exception &error) {
    return fail(LLAMA_DART_ERROR_INTERNAL, error.what());
  } catch (...) {
    return fail(LLAMA_DART_ERROR_INTERNAL, "unknown embedding failure");
  }
}

llama_dart_result llama_dart_context_rerank(
    llama_dart_context *context, const llama_dart_rerank_config *config,
    float *out_score) {
  if (out_score == nullptr) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT, "out_score must not be null");
  }
  *out_score = 0.0f;
  if (config == nullptr) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT, "config must not be null");
  }

  const llama_dart_result validation =
      validate_struct(config->struct_size, sizeof(llama_dart_rerank_config),
                      "llama_dart_rerank_config");
  if (validation != LLAMA_DART_SUCCESS) {
    return validation;
  }
  if (config->query_data == nullptr && config->query_size > 0) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "query_data must not be null when query_size is positive");
  }
  if (config->document_data == nullptr && config->document_size > 0) {
    return fail(
        LLAMA_DART_ERROR_INVALID_ARGUMENT,
        "document_data must not be null when document_size is positive");
  }
  if (!valid_bool(config->add_special) || !valid_bool(config->parse_special)) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "rerank boolean fields must be 0 or 1");
  }
  if (config->query_size == 0) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "rerank query must not be empty");
  }
  if (config->document_size == 0) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "rerank document must not be empty");
  }
  if (!fits_int32(config->query_size) || !fits_int32(config->document_size)) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT, "rerank input is too large");
  }
  if (is_ascii_blank(config->query_data, config->query_size)) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "rerank query must not be blank");
  }
  if (is_ascii_blank(config->document_data, config->document_size)) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "rerank document must not be blank");
  }
  if (contains_nul(config->query_data, config->query_size) ||
      contains_nul(config->document_data, config->document_size)) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "rerank text must not contain NUL bytes");
  }
  if (context == nullptr || context->context == nullptr ||
      context->model == nullptr || context->model->model == nullptr) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT, "context must not be null");
  }
  if (context->active_generations > 0) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "context still has active generations");
  }
  if (is_cancelled(context)) {
    return fail_cancelled(context);
  }

  llama_model *model = context->model->model;
  if (llama_model_has_encoder(model) && llama_model_has_decoder(model)) {
    return fail(LLAMA_DART_ERROR_UNSUPPORTED,
                "encoder-decoder reranking is not supported");
  }
  if (llama_pooling_type(context->context) != LLAMA_POOLING_TYPE_RANK) {
    return fail(LLAMA_DART_ERROR_UNSUPPORTED,
                "context is not configured for rank pooling");
  }

  const llama_vocab *vocab = llama_model_get_vocab(model);
  if (vocab == nullptr) {
    return fail(LLAMA_DART_ERROR_UNSUPPORTED, "model has no vocabulary");
  }

  try {
    std::vector<llama_token> tokens;
    const char *rerank_prompt = llama_model_chat_template(model, "rerank");
    if (rerank_prompt != nullptr) {
      std::string prompt = rerank_prompt;
      const std::string query(
          reinterpret_cast<const char *>(config->query_data),
          config->query_size);
      const std::string document(
          reinterpret_cast<const char *>(config->document_data),
          config->document_size);
      replace_all(&prompt, "{query}", query);
      replace_all(&prompt, "{document}", document);
      const llama_dart_result tokenized = tokenize_bytes(
          vocab, reinterpret_cast<const uint8_t *>(prompt.data()),
          prompt.size(), config->add_special != 0,
          config->parse_special != 0, "rerank prompt produced no tokens",
          &tokens);
      if (tokenized != LLAMA_DART_SUCCESS) {
        return tokenized;
      }
    } else {
      std::vector<llama_token> query_tokens;
      llama_dart_result tokenized = tokenize_bytes(
          vocab, config->query_data, config->query_size, false, false,
          "rerank query produced no tokens", &query_tokens);
      if (tokenized != LLAMA_DART_SUCCESS) {
        return tokenized;
      }
      std::vector<llama_token> document_tokens;
      tokenized = tokenize_bytes(vocab, config->document_data,
                                 config->document_size, false, false,
                                 "rerank document produced no tokens",
                                 &document_tokens);
      if (tokenized != LLAMA_DART_SUCCESS) {
        return tokenized;
      }

      llama_token eos_token = llama_vocab_eos(vocab);
      if (eos_token == LLAMA_TOKEN_NULL) {
        eos_token = llama_vocab_sep(vocab);
      }
      if (eos_token == LLAMA_TOKEN_NULL && llama_vocab_get_add_eos(vocab)) {
        return fail(LLAMA_DART_ERROR_UNSUPPORTED,
                    "model has no EOS or SEP token for reranking");
      }

      if (llama_vocab_get_add_bos(vocab)) {
        const llama_token bos = llama_vocab_bos(vocab);
        if (bos != LLAMA_TOKEN_NULL) {
          tokens.push_back(bos);
        }
      }
      tokens.insert(tokens.end(), query_tokens.begin(), query_tokens.end());
      if (llama_vocab_get_add_eos(vocab)) {
        tokens.push_back(eos_token);
      }
      if (llama_vocab_get_add_sep(vocab)) {
        const llama_token sep = llama_vocab_sep(vocab);
        if (sep != LLAMA_TOKEN_NULL) {
          tokens.push_back(sep);
        }
      }
      tokens.insert(tokens.end(), document_tokens.begin(),
                    document_tokens.end());
      if (llama_vocab_get_add_eos(vocab)) {
        tokens.push_back(eos_token);
      }
    }

    if (tokens.empty()) {
      return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                  "rerank input produced no tokens");
    }
    if (tokens.size() > static_cast<size_t>(llama_n_batch(context->context))) {
      return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                  "rerank input exceeds context batch size");
    }

    llama_memory_t memory = llama_get_memory(context->context);
    if (memory != nullptr) {
      llama_memory_clear(memory, true);
    }

    llama_batch batch =
        llama_batch_init(static_cast<int32_t>(tokens.size()), 0, 1);
    if (batch.token == nullptr || batch.pos == nullptr ||
        batch.n_seq_id == nullptr || batch.seq_id == nullptr ||
        batch.logits == nullptr) {
      llama_batch_free(batch);
      return fail(LLAMA_DART_ERROR_INTERNAL, "native allocation failed");
    }

    batch.n_tokens = static_cast<int32_t>(tokens.size());
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
      batch.token[i] = tokens[static_cast<size_t>(i)];
      batch.pos[i] = i;
      batch.n_seq_id[i] = 1;
      batch.seq_id[i][0] = 0;
      batch.logits[i] = i == batch.n_tokens - 1 ? 1 : 0;
    }

    const int32_t decoded = llama_decode(context->context, batch);
    llama_batch_free(batch);
    if (decoded == 2 || is_cancelled(context)) {
      return fail_cancelled(context);
    }
    if (decoded != 0) {
      return fail(LLAMA_DART_ERROR_RERANKING,
                  "failed to decode rerank input");
    }

    const float *score = llama_get_embeddings_seq(context->context, 0);
    if (score == nullptr) {
      return fail(LLAMA_DART_ERROR_RERANKING,
                  "model did not return a rerank score");
    }
    if (!std::isfinite(score[0])) {
      return fail(LLAMA_DART_ERROR_RERANKING,
                  "model returned a non-finite rerank score");
    }
    *out_score = score[0];
    last_error.clear();
    return LLAMA_DART_SUCCESS;
  } catch (const std::bad_alloc &) {
    return fail(LLAMA_DART_ERROR_INTERNAL, "native allocation failed");
  } catch (const std::exception &error) {
    return fail(LLAMA_DART_ERROR_INTERNAL, error.what());
  } catch (...) {
    return fail(LLAMA_DART_ERROR_INTERNAL, "unknown rerank failure");
  }
}

llama_dart_result llama_dart_lora_load(
    llama_dart_model *model, const llama_dart_lora_load_config *config,
    llama_dart_lora_adapter **out_adapter) {
  if (out_adapter == nullptr) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "out_adapter must not be null");
  }
  *out_adapter = nullptr;
  if (model == nullptr || model->model == nullptr) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT, "model must not be null");
  }
  if (config == nullptr) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT, "config must not be null");
  }

  const llama_dart_result validation =
      validate_struct(config->struct_size, sizeof(llama_dart_lora_load_config),
                      "llama_dart_lora_load_config");
  if (validation != LLAMA_DART_SUCCESS) {
    return validation;
  }
  if (config->path_data == nullptr || config->path_size == 0) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "lora path must not be empty");
  }
  if (!fits_int32(config->path_size)) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "lora path is too large");
  }
  if (is_ascii_blank(config->path_data, config->path_size)) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "lora path must not be blank");
  }
  if (contains_nul(config->path_data, config->path_size)) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "lora path must not contain NUL bytes");
  }
  if (contains_line_break(config->path_data, config->path_size)) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "lora path must not contain line breaks");
  }

  try {
    const std::string path(reinterpret_cast<const char *>(config->path_data),
                           config->path_size);
    std::unique_ptr<llama_adapter_lora, decltype(&llama_adapter_lora_free)>
        adapter(llama_adapter_lora_init(model->model, path.c_str()),
                llama_adapter_lora_free);
    if (adapter == nullptr) {
      return fail(LLAMA_DART_ERROR_LORA, "failed to load LoRA adapter");
    }

    std::unique_ptr<llama_dart_lora_adapter> handle =
        std::make_unique<llama_dart_lora_adapter>();
    handle->adapter = adapter.release();
    handle->model = model;
    model->active_lora_adapters += 1;
    *out_adapter = handle.release();
    last_error.clear();
    return LLAMA_DART_SUCCESS;
  } catch (const std::bad_alloc &) {
    return fail(LLAMA_DART_ERROR_INTERNAL, "native allocation failed");
  } catch (const std::exception &error) {
    return fail(LLAMA_DART_ERROR_LORA, error.what());
  } catch (...) {
    return fail(LLAMA_DART_ERROR_LORA, "unknown LoRA load failure");
  }
}

void llama_dart_lora_free(llama_dart_lora_adapter *adapter) {
  if (adapter == nullptr) {
    return;
  }
  if (adapter->active_contexts > 0) {
    fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
         "LoRA adapter is still active on contexts");
    return;
  }
  if (adapter->adapter != nullptr) {
    llama_adapter_lora_free(adapter->adapter);
  }
  if (adapter->model != nullptr && adapter->model->active_lora_adapters > 0) {
    adapter->model->active_lora_adapters -= 1;
  }
  delete adapter;
  last_error.clear();
}

llama_dart_result llama_dart_context_set_lora_adapters(
    llama_dart_context *context, llama_dart_lora_adapter **adapters,
    const float *scales, size_t adapter_count) {
  if (context == nullptr || context->context == nullptr) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT, "context must not be null");
  }
  if (context->active_generations > 0) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "context still has active generations");
  }
  if (adapter_count > 0 && adapters == nullptr) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "adapters must not be null when adapter_count is positive");
  }
  if (adapter_count > 0 && scales == nullptr) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "scales must not be null when adapter_count is positive");
  }
  if (!fits_int32(adapter_count)) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "too many LoRA adapters");
  }

  try {
    std::vector<llama_adapter_lora *> native_adapters;
    native_adapters.reserve(adapter_count);
    std::vector<llama_dart_lora_adapter *> next_adapters;
    next_adapters.reserve(adapter_count);
    std::vector<float> native_scales;
    native_scales.reserve(adapter_count);
    for (size_t i = 0; i < adapter_count; ++i) {
      if (adapters[i] == nullptr || adapters[i]->adapter == nullptr) {
        return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                    "adapter must not be null");
      }
      if (adapters[i]->model != context->model) {
        return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                    "adapter does not belong to this model");
      }
      if (!std::isfinite(scales[i])) {
        return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                    "LoRA scales must be finite");
      }
      if (scales[i] < 0.0f) {
        return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                    "LoRA scales must be non-negative");
      }
      if (scales[i] != 0.0f &&
          std::find(next_adapters.begin(), next_adapters.end(), adapters[i]) !=
              next_adapters.end()) {
        return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                    "duplicate LoRA adapter");
      }
      native_adapters.push_back(adapters[i]->adapter);
      native_scales.push_back(scales[i]);
      if (scales[i] != 0.0f) {
        next_adapters.push_back(adapters[i]);
      }
    }

    const int32_t result = llama_set_adapters_lora(
        context->context, native_adapters.empty() ? nullptr
                                                  : native_adapters.data(),
        adapter_count,
        native_scales.empty() ? nullptr : native_scales.data());
    if (result != 0) {
      return fail(LLAMA_DART_ERROR_LORA, "failed to set LoRA adapters");
    }
    release_context_lora_adapters(context);
    context->lora_adapters = std::move(next_adapters);
    for (llama_dart_lora_adapter *adapter : context->lora_adapters) {
      adapter->active_contexts += 1;
    }
    last_error.clear();
    return LLAMA_DART_SUCCESS;
  } catch (const std::bad_alloc &) {
    return fail(LLAMA_DART_ERROR_INTERNAL, "native allocation failed");
  } catch (const std::exception &error) {
    return fail(LLAMA_DART_ERROR_LORA, error.what());
  } catch (...) {
    return fail(LLAMA_DART_ERROR_LORA, "unknown LoRA apply failure");
  }
}

void llama_dart_buffer_free(uint8_t *data) {
  if (data == nullptr) {
    return;
  }
  std::free(data);
  last_error.clear();
}

void llama_dart_float_buffer_free(float *data) {
  if (data == nullptr) {
    return;
  }
  std::free(data);
  last_error.clear();
}

const char *llama_dart_last_error_message(void) { return last_error.c_str(); }

void llama_dart_clear_last_error(void) { last_error.clear(); }

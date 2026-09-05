#include "llama_dart.h"
#include "gpu_policy.h"
#include "kv_cache_policy.h"
#include "load_policy.h"
#include "prompt_prefix.h"
#include "speculative.h"
#include "state_snapshot.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

// Check cached metadata against actual emitted bytes for every vocabulary ID.
static void verify_token_piece_bound(const llama_dart_model *model) {
  llama_dart_model_info info{};
  info.struct_size = sizeof(info);
  assert(llama_dart_model_get_info(model, &info) == LLAMA_DART_SUCCESS);
  assert(info.maximum_token_piece_bytes > 0);
  std::vector<uint8_t> piece(info.maximum_token_piece_bytes);
  size_t largest = 0;
  for (int32_t token = 0; token < info.n_vocab; ++token) {
    size_t size = 0;
    assert(llama_dart_model_detokenize(model, &token, 1, piece.data(),
               piece.size(), &size, 0, 1) == LLAMA_DART_SUCCESS);
    assert(size <= info.maximum_token_piece_bytes);
    largest = std::max(largest, size);
  }
  assert(largest == info.maximum_token_piece_bytes);
  llama_dart_model_info again{};
  again.struct_size = sizeof(again);
  assert(llama_dart_model_get_info(model, &again) == LLAMA_DART_SUCCESS);
  assert(again.maximum_token_piece_bytes == info.maximum_token_piece_bytes);
}

int main() {
  using namespace llama_dart_bridge_internal;

  const std::vector<int32_t> empty_prompt_history;
  const std::vector<int32_t> first_prompt_tokens{1, 2, 3};
  const prompt_prefix_reuse_plan empty_prefix =
      resolve_prompt_prefix_reuse(empty_prompt_history, first_prompt_tokens);
  assert(empty_prefix.exact_prefix);
  assert(empty_prefix.suffix_start == 0);
  const prompt_prefix_reuse_plan appended_prefix =
      resolve_prompt_prefix_reuse(first_prompt_tokens, {1, 2, 3, 4, 5});
  assert(appended_prefix.exact_prefix);
  assert(appended_prefix.suffix_start == 3);
  const prompt_prefix_reuse_plan identical_prefix =
      resolve_prompt_prefix_reuse(first_prompt_tokens, first_prompt_tokens);
  assert(identical_prefix.exact_prefix);
  assert(identical_prefix.suffix_start == 3);
  const prompt_prefix_reuse_plan changed_prefix =
      resolve_prompt_prefix_reuse(first_prompt_tokens, {1, 9, 3, 4});
  assert(!changed_prefix.exact_prefix);
  assert(changed_prefix.suffix_start == 0);
  const prompt_prefix_reuse_plan trimmed_prefix =
      resolve_prompt_prefix_reuse(first_prompt_tokens, {2, 3, 4});
  assert(!trimmed_prefix.exact_prefix);
  assert(trimmed_prefix.suffix_start == 0);

  assert(resolve_model_load_mode(false, false) == LLAMA_LOAD_MODE_NONE);
  assert(resolve_model_load_mode(true, false) == LLAMA_LOAD_MODE_MMAP);
  assert(resolve_model_load_mode(false, true) == LLAMA_LOAD_MODE_MLOCK);
  assert(resolve_model_load_mode(true, true) ==
         LLAMA_LOAD_MODE_MMAP_MLOCK);
  assert(resolve_quantized_v_flash_attention_mode(
             LLAMA_DART_FLASH_ATTENTION_AUTO, false) ==
         LLAMA_DART_FLASH_ATTENTION_AUTO);
  assert(resolve_quantized_v_flash_attention_mode(
             LLAMA_DART_FLASH_ATTENTION_AUTO, true) ==
         LLAMA_DART_FLASH_ATTENTION_ENABLED);
  assert(resolve_quantized_v_flash_attention_mode(
             LLAMA_DART_FLASH_ATTENTION_ENABLED, true) ==
         LLAMA_DART_FLASH_ATTENTION_ENABLED);

  const gpu_load_policy simulator_auto = resolve_gpu_load_policy(
      LLAMA_DART_GPU_BACKEND_AUTO, -1, true);
  assert(simulator_auto.backend == LLAMA_DART_GPU_BACKEND_CPU);
  assert(simulator_auto.n_gpu_layers == 0);
  assert(simulator_auto.simulator_auto_cpu);
  assert(!resolve_mmproj_use_gpu(true, simulator_auto.simulator_auto_cpu));
  const gpu_load_policy simulator_bounded_auto = resolve_gpu_load_policy(
      LLAMA_DART_GPU_BACKEND_AUTO, 4, true);
  assert(simulator_bounded_auto.backend == LLAMA_DART_GPU_BACKEND_CPU);
  assert(simulator_bounded_auto.n_gpu_layers == 0);
  assert(simulator_bounded_auto.simulator_auto_cpu);
  const gpu_load_policy simulator_explicit_metal = resolve_gpu_load_policy(
      LLAMA_DART_GPU_BACKEND_METAL, -1, true);
  assert(simulator_explicit_metal.backend == LLAMA_DART_GPU_BACKEND_METAL);
  assert(simulator_explicit_metal.n_gpu_layers == -1);
  assert(!simulator_explicit_metal.simulator_auto_cpu);
  const gpu_load_policy simulator_explicit_cpu = resolve_gpu_load_policy(
      LLAMA_DART_GPU_BACKEND_CPU, 0, true);
  assert(simulator_explicit_cpu.backend == LLAMA_DART_GPU_BACKEND_CPU);
  assert(simulator_explicit_cpu.n_gpu_layers == 0);
  assert(!simulator_explicit_cpu.simulator_auto_cpu);
  assert(resolve_mmproj_use_gpu(
      true, simulator_explicit_cpu.simulator_auto_cpu));
  assert(!resolve_mmproj_use_gpu(
      false, simulator_explicit_cpu.simulator_auto_cpu));
  const gpu_load_policy device_auto = resolve_gpu_load_policy(
      LLAMA_DART_GPU_BACKEND_AUTO, -1, false);
  assert(device_auto.backend == LLAMA_DART_GPU_BACKEND_AUTO);
  assert(device_auto.n_gpu_layers == -1);
  assert(!device_auto.simulator_auto_cpu);

  const gpu_load_policy target_auto =
      resolve_gpu_load_policy(LLAMA_DART_GPU_BACKEND_AUTO, -1);
  if (target_is_apple_simulator()) {
    assert(target_auto.backend == LLAMA_DART_GPU_BACKEND_CPU);
    assert(target_auto.n_gpu_layers == 0);
    assert(target_auto.simulator_auto_cpu);
  } else {
    assert(target_auto.backend == LLAMA_DART_GPU_BACKEND_AUTO);
    assert(target_auto.n_gpu_layers == -1);
    assert(!target_auto.simulator_auto_cpu);
  }

  const state_snapshot_layout snapshot_layout{
      /* .speculative_type = */ 4,
      /* .target_size = */ 3,
      /* .draft_size = */ 2,
      /* .speculative_size = */ 1,
      /* .token_count = */ 2,
      /* .position = */ 5,
  };
  size_t snapshot_size = 0;
  assert(state_snapshot_size(snapshot_layout, &snapshot_size));
  assert(snapshot_size == kStateSnapshotHeaderSize + 3 + 2 + 1 + 8);
  std::vector<uint8_t> snapshot(snapshot_size, 0);
  write_state_snapshot_header(snapshot.data(), snapshot_layout);
  size_t snapshot_offset = kStateSnapshotHeaderSize;
  snapshot[snapshot_offset++] = 1;
  snapshot[snapshot_offset++] = 2;
  snapshot[snapshot_offset++] = 3;
  snapshot[snapshot_offset++] = 4;
  snapshot[snapshot_offset++] = 5;
  snapshot[snapshot_offset++] = 6;
  write_u32_le(snapshot.data() + snapshot_offset, 7);
  snapshot_offset += 4;
  write_u32_le(snapshot.data() + snapshot_offset, 8);
  finalize_state_snapshot(snapshot.data(), snapshot.size());

  state_snapshot_view snapshot_view;
  std::string snapshot_error;
  assert(decode_state_snapshot(snapshot.data(), snapshot.size(),
                               &snapshot_view, &snapshot_error) ==
         state_snapshot_decode_result::success);
  assert(snapshot_view.layout.speculative_type == 4);
  assert(snapshot_view.layout.position == 5);
  assert(snapshot_view.target[2] == 3);
  assert(snapshot_view.draft[1] == 5);
  assert(snapshot_view.speculative[0] == 6);
  assert(read_u32_le(snapshot_view.tokens) == 7);
  assert(read_u32_le(snapshot_view.tokens + 4) == 8);

  std::vector<uint8_t> corrupt_snapshot = snapshot;
  corrupt_snapshot.back() ^= 1;
  assert(decode_state_snapshot(corrupt_snapshot.data(),
                               corrupt_snapshot.size(), &snapshot_view,
                               &snapshot_error) ==
         state_snapshot_decode_result::invalid);
  assert(snapshot_error.find("checksum") != std::string::npos);
  assert(decode_state_snapshot(snapshot.data(), snapshot.size() - 1,
                               &snapshot_view, &snapshot_error) ==
         state_snapshot_decode_result::invalid);
  std::vector<uint8_t> bad_header = snapshot;
  write_u32_le(bad_header.data() + 8, 2);
  assert(decode_state_snapshot(bad_header.data(), bad_header.size(),
                               &snapshot_view, &snapshot_error) ==
         state_snapshot_decode_result::invalid);
  bad_header = snapshot;
  write_u32_le(bad_header.data() + 12, 0);
  assert(decode_state_snapshot(bad_header.data(), bad_header.size(),
                               &snapshot_view, &snapshot_error) ==
         state_snapshot_decode_result::invalid);
  const uint8_t legacy_state[] = {1, 2, 3};
  assert(decode_state_snapshot(legacy_state, sizeof(legacy_state),
                               &snapshot_view, &snapshot_error) ==
         state_snapshot_decode_result::not_snapshot);
  state_snapshot_layout oversized_layout = snapshot_layout;
  oversized_layout.token_count = std::numeric_limits<uint64_t>::max();
  assert(!state_snapshot_size(oversized_layout, &snapshot_size));

  const llama_tokens ngram_history{99, 1, 2, 10, 11, 12,
                                   1,  2, 10, 11, 12, 1};
  for (const common_speculative_type type : {
           COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE,
           COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K,
           COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V}) {
    common_params_speculative speculative_params;
    speculative_params.types = {type};
    common_params_speculative_ngram_map ngram_params;
    ngram_params.size_n = 2;
    ngram_params.size_m = 3;
    ngram_params.min_hits = 1;
    if (type == COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE) {
      speculative_params.ngram_simple = ngram_params;
    } else if (type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K) {
      speculative_params.ngram_map_k = ngram_params;
    } else {
      speculative_params.ngram_map_k4v = ngram_params;
    }
    common_speculative_ptr spec(
        common_speculative_init(speculative_params, 1));
    assert(spec != nullptr);
    common_speculative_begin(spec.get(), 0, ngram_history);
    llama_tokens draft;
    common_speculative_get_draft_params(spec.get(), 0) = {
        /* .drafting = */ true,
        /* .n_max    = */ 3,
        /* .n_past   = */ static_cast<llama_pos>(ngram_history.size()),
        /* .id_last  = */ 2,
        /* .prompt   = */ &ngram_history,
        /* .result   = */ &draft,
    };
    common_speculative_draft(spec.get());
    assert((draft == llama_tokens{10, 11, 12}));
    common_speculative_accept(spec.get(), 0,
                              static_cast<uint16_t>(draft.size()));
  }
  for (const common_speculative_type type : {
           COMMON_SPECULATIVE_TYPE_NGRAM_MOD,
           COMMON_SPECULATIVE_TYPE_NGRAM_CACHE}) {
    common_params_speculative speculative_params;
    speculative_params.types = {type};
    if (type == COMMON_SPECULATIVE_TYPE_NGRAM_MOD) {
      speculative_params.ngram_mod.n_match = 2;
      speculative_params.ngram_mod.n_min = 1;
      speculative_params.ngram_mod.n_max = 3;
    }
    common_speculative_ptr spec(
        common_speculative_init(speculative_params, 1));
    assert(spec != nullptr);
    common_speculative_begin(spec.get(), 0, ngram_history);
    llama_tokens draft;
    common_speculative_get_draft_params(spec.get(), 0) = {
        /* .drafting = */ true,
        /* .n_max    = */ 3,
        /* .n_past   = */ static_cast<llama_pos>(ngram_history.size()),
        /* .id_last  = */ 2,
        /* .prompt   = */ &ngram_history,
        /* .result   = */ &draft,
    };
    common_speculative_draft(spec.get());
    assert(!draft.empty());
    common_speculative_accept(spec.get(), 0,
                              static_cast<uint16_t>(draft.size()));
  }

  assert(llama_dart_abi_version() == LLAMA_DART_ABI_VERSION);
  assert(std::strlen(llama_dart_upstream_commit()) > 0);
  assert(std::strstr(llama_dart_build_flags(), "LLAMA_BUILD_COMMON=ON") !=
         nullptr);
  assert(std::strstr(llama_dart_build_flags(), "LLAMA_BUILD_MTMD=ON") !=
         nullptr);
  assert(std::strstr(llama_dart_build_flags(), "MTMD_VIDEO=OFF") != nullptr);
#if defined(__ANDROID__)
  if (std::strstr(llama_dart_build_flags(), "GGML_VULKAN=ON") != nullptr) {
    assert(std::strstr(llama_dart_build_flags(),
                       "GGML_VULKAN_ANDROID_SAFE_QUANT=1") != nullptr);
    assert(std::strstr(llama_dart_build_flags(),
                       "GGML_VULKAN_ANDROID_SAFE_K_QUANT=1") != nullptr);
  } else {
    assert(std::strstr(llama_dart_build_flags(),
                       "GGML_VULKAN_ANDROID_SAFE_QUANT=0") != nullptr);
    assert(std::strstr(llama_dart_build_flags(),
                       "GGML_VULKAN_ANDROID_SAFE_K_QUANT=0") != nullptr);
  }
#else
  assert(std::strstr(llama_dart_build_flags(),
                     "GGML_VULKAN_ANDROID_SAFE_QUANT=0") != nullptr);
  assert(std::strstr(llama_dart_build_flags(),
                     "GGML_VULKAN_ANDROID_SAFE_K_QUANT=0") != nullptr);
#endif
  assert(std::strlen(llama_dart_multimodal_marker()) > 0);

  assert(llama_dart_backend_init() == LLAMA_DART_SUCCESS);
  assert(llama_dart_backend_free() == LLAMA_DART_SUCCESS);

  llama_dart_capabilities capabilities{};
  capabilities.struct_size = sizeof(capabilities);
  assert(llama_dart_get_capabilities(&capabilities) == LLAMA_DART_SUCCESS);
  assert(capabilities.struct_size == sizeof(capabilities));
  assert(capabilities.abi_version == LLAMA_DART_ABI_VERSION);
  assert((capabilities.flags & LLAMA_DART_CAP_MODEL_LOADING) != 0);
  assert((capabilities.flags & LLAMA_DART_CAP_TOKENIZATION) != 0);
  assert((capabilities.flags & LLAMA_DART_CAP_TEXT_GENERATION) != 0);
  assert((capabilities.flags & LLAMA_DART_CAP_EMBEDDINGS) != 0);
  assert((capabilities.flags & LLAMA_DART_CAP_RERANKING) != 0);
  assert((capabilities.flags & LLAMA_DART_CAP_LORA) != 0);
  assert((capabilities.flags & LLAMA_DART_CAP_SPECULATIVE_DECODING) != 0);
  assert((capabilities.flags & LLAMA_DART_CAP_MULTIMODAL) != 0);
  assert((capabilities.flags & LLAMA_DART_CAP_MTP) != 0);
  assert((capabilities.flags & LLAMA_DART_CAP_TOOL_CALLING) != 0);
  assert((capabilities.flags & LLAMA_DART_CAP_LOGGING) != 0);
  assert((capabilities.flags & LLAMA_DART_CAP_PREFILL) != 0);
  if (std::strstr(llama_dart_build_flags(), "GGML_METAL=ON") != nullptr) {
    assert((capabilities.flags & LLAMA_DART_CAP_METAL) != 0);
  } else {
    assert((capabilities.flags & LLAMA_DART_CAP_METAL) == 0);
  }
  if (std::strstr(llama_dart_build_flags(), "GGML_VULKAN=ON") != nullptr) {
    assert((capabilities.flags & LLAMA_DART_CAP_VULKAN) != 0);
  } else {
    assert((capabilities.flags & LLAMA_DART_CAP_VULKAN) == 0);
  }
  assert(std::strlen(llama_dart_last_error_message()) == 0);

  llama_dart_buffer log_message{};
  uint32_t log_level = 123;
  assert(llama_dart_log_next(nullptr, &log_message) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(log_message.data == nullptr);
  assert(log_message.size == 0);
  assert(llama_dart_log_next(&log_level, nullptr) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(llama_dart_log_set_level(99) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(llama_dart_log_set_level(LLAMA_DART_LOG_DEBUG) ==
         LLAMA_DART_SUCCESS);

  const char json_schema[] =
      R"({"type":"object","properties":{"name":{"type":"string"}},"required":["name"],"additionalProperties":false})";
  llama_dart_buffer schema_grammar{};
  assert(llama_dart_json_schema_to_grammar(
             reinterpret_cast<const uint8_t *>(json_schema),
             std::strlen(json_schema), &schema_grammar) == LLAMA_DART_SUCCESS);
  assert(schema_grammar.data != nullptr);
  assert(schema_grammar.size > 0);
  const std::string schema_grammar_text(
      reinterpret_cast<const char *>(schema_grammar.data),
      schema_grammar.size);
  assert(schema_grammar_text.find("root ::=") != std::string::npos);
  llama_dart_buffer_free(schema_grammar.data);
  const char referenced_json_schema[] =
      R"({"$defs":{"code":{"type":"string","pattern":"^[A-Z]{2}$"}},"$ref":"#/$defs/code"})";
  assert(llama_dart_json_schema_to_grammar(
             reinterpret_cast<const uint8_t *>(referenced_json_schema),
             std::strlen(referenced_json_schema), &schema_grammar) ==
         LLAMA_DART_SUCCESS);
  assert(schema_grammar.data != nullptr);
  assert(schema_grammar.size > 0);
  llama_dart_buffer_free(schema_grammar.data);
  const char invalid_json_schema[] = "{";
  assert(llama_dart_json_schema_to_grammar(
             reinterpret_cast<const uint8_t *>(invalid_json_schema),
             std::strlen(invalid_json_schema), &schema_grammar) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(schema_grammar.data == nullptr);
  assert(schema_grammar.size == 0);
  assert(llama_dart_json_schema_to_grammar(nullptr, 0, &schema_grammar) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(llama_dart_json_schema_to_grammar(
             reinterpret_cast<const uint8_t *>(json_schema),
             std::strlen(json_schema), nullptr) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);

  assert(llama_dart_get_capabilities(nullptr) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(std::strlen(llama_dart_last_error_message()) > 0);

  const char path[] = "missing-model.gguf";
  llama_dart_model_load_config config{};
  config.struct_size = sizeof(config);
  config.model_path_data = reinterpret_cast<const uint8_t *>(path);
  config.model_path_size = std::strlen(path);
  config.n_gpu_layers = 0;
  config.vocab_only = 1;
  config.use_mmap = 1;
  config.use_mlock = 0;
  config.check_tensors = 1;

  llama_dart_model *model = nullptr;
  assert(llama_dart_model_load(&config, &model) ==
         LLAMA_DART_ERROR_MODEL_LOAD);
  assert(model == nullptr);
  assert(std::strlen(llama_dart_last_error_message()) > 0);

  const uint8_t path_with_nul[] = {'b', 'a', 'd', '\0', 'p', 'a', 't', 'h'};
  llama_dart_model_load_config invalid_model_config = config;
  invalid_model_config.model_path_data = path_with_nul;
  invalid_model_config.model_path_size = sizeof(path_with_nul);
  assert(llama_dart_model_load(&invalid_model_config, &model) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(model == nullptr);
  const uint8_t blank_path[] = {' ', '\t'};
  invalid_model_config = config;
  invalid_model_config.model_path_data = blank_path;
  invalid_model_config.model_path_size = sizeof(blank_path);
  assert(llama_dart_model_load(&invalid_model_config, &model) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(model == nullptr);
  const uint8_t path_with_line_break[] = {'b', 'a', 'd', '\n', 'p', 'a',
                                          't', 'h'};
  invalid_model_config = config;
  invalid_model_config.model_path_data = path_with_line_break;
  invalid_model_config.model_path_size = sizeof(path_with_line_break);
  assert(llama_dart_model_load(&invalid_model_config, &model) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(model == nullptr);
  invalid_model_config = config;
  invalid_model_config.model_path_size = std::numeric_limits<size_t>::max();
  assert(llama_dart_model_load(&invalid_model_config, &model) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(model == nullptr);
  const uint8_t chat_template_byte[] = {'x'};
  invalid_model_config = config;
  invalid_model_config.chat_template_size = sizeof(chat_template_byte);
  assert(llama_dart_model_load(&invalid_model_config, &model) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(model == nullptr);
  invalid_model_config = config;
  invalid_model_config.chat_template_data = chat_template_byte;
  assert(llama_dart_model_load(&invalid_model_config, &model) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(model == nullptr);
  const uint8_t blank_chat_template[] = {' ', '\t', '\n'};
  invalid_model_config = config;
  invalid_model_config.chat_template_data = blank_chat_template;
  invalid_model_config.chat_template_size = sizeof(blank_chat_template);
  assert(llama_dart_model_load(&invalid_model_config, &model) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(model == nullptr);
  const uint8_t nul_chat_template[] = {'x', '\0', 'y'};
  invalid_model_config = config;
  invalid_model_config.chat_template_data = nul_chat_template;
  invalid_model_config.chat_template_size = sizeof(nul_chat_template);
  assert(llama_dart_model_load(&invalid_model_config, &model) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(model == nullptr);
  const uint8_t invalid_utf8_chat_template[] = {0xf0, 0x28, 0x8c, 0x28};
  invalid_model_config = config;
  invalid_model_config.chat_template_data = invalid_utf8_chat_template;
  invalid_model_config.chat_template_size =
      sizeof(invalid_utf8_chat_template);
  assert(llama_dart_model_load(&invalid_model_config, &model) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(model == nullptr);
  invalid_model_config = config;
  invalid_model_config.chat_template_data = chat_template_byte;
  invalid_model_config.chat_template_size =
      std::numeric_limits<size_t>::max();
  assert(llama_dart_model_load(&invalid_model_config, &model) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(model == nullptr);
  invalid_model_config = config;
  invalid_model_config.use_mmap = 2;
  assert(llama_dart_model_load(&invalid_model_config, &model) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(model == nullptr);
  invalid_model_config = config;
  invalid_model_config.load_mtp = 2;
  assert(llama_dart_model_load(&invalid_model_config, &model) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(model == nullptr);
  invalid_model_config = config;
  invalid_model_config.n_gpu_layers = -2;
  assert(llama_dart_model_load(&invalid_model_config, &model) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(model == nullptr);
  invalid_model_config = config;
  invalid_model_config.gpu_backend = 99;
  assert(llama_dart_model_load(&invalid_model_config, &model) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(model == nullptr);
  invalid_model_config = config;
  invalid_model_config.gpu_backend = LLAMA_DART_GPU_BACKEND_CPU;
  invalid_model_config.n_gpu_layers = -1;
  assert(llama_dart_model_load(&invalid_model_config, &model) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(model == nullptr);
  invalid_model_config = config;
  invalid_model_config.gpu_backend = LLAMA_DART_GPU_BACKEND_METAL;
  invalid_model_config.n_gpu_layers = 0;
  assert(llama_dart_model_load(&invalid_model_config, &model) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(model == nullptr);
  invalid_model_config.n_gpu_layers = -1;
  assert(llama_dart_model_load(&invalid_model_config, &model) ==
         ((capabilities.flags & LLAMA_DART_CAP_METAL) != 0
              ? LLAMA_DART_ERROR_MODEL_LOAD
              : LLAMA_DART_ERROR_UNSUPPORTED));
  assert(model == nullptr);
  invalid_model_config = config;
  invalid_model_config.gpu_backend = LLAMA_DART_GPU_BACKEND_VULKAN;
  invalid_model_config.n_gpu_layers = -1;
  assert(llama_dart_model_load(&invalid_model_config, &model) ==
         ((capabilities.flags & LLAMA_DART_CAP_VULKAN) != 0
              ? LLAMA_DART_ERROR_MODEL_LOAD
              : LLAMA_DART_ERROR_UNSUPPORTED));
  assert(model == nullptr);
  invalid_model_config = config;
  invalid_model_config.n_gpu_layers = -1;
  assert(llama_dart_model_load(&invalid_model_config, &model) ==
         LLAMA_DART_ERROR_MODEL_LOAD);
  assert(model == nullptr);

  llama_dart_model_info stale_model_info{};
  stale_model_info.struct_size = sizeof(stale_model_info);
  stale_model_info.n_vocab = 123;
  stale_model_info.maximum_token_piece_bytes = 123;
  assert(llama_dart_model_get_info(nullptr, &stale_model_info) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(stale_model_info.struct_size == sizeof(stale_model_info));
  assert(stale_model_info.n_vocab == 0);
  assert(stale_model_info.maximum_token_piece_bytes == 0);

  llama_dart_model_free(nullptr);
  llama_dart_lora_free(nullptr);
  llama_dart_buffer_free(nullptr);
  llama_dart_float_buffer_free(nullptr);

  assert(llama_dart_context_reset(nullptr) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(std::strlen(llama_dart_last_error_message()) > 0);
  uint8_t *allocated_buffer = static_cast<uint8_t *>(std::malloc(1));
  assert(allocated_buffer != nullptr);
  llama_dart_buffer_free(allocated_buffer);
  assert(std::strlen(llama_dart_last_error_message()) == 0);
  assert(llama_dart_context_reset(nullptr) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(std::strlen(llama_dart_last_error_message()) > 0);
  float *allocated_float_buffer =
      static_cast<float *>(std::malloc(sizeof(float)));
  assert(allocated_float_buffer != nullptr);
  llama_dart_float_buffer_free(allocated_float_buffer);
  assert(std::strlen(llama_dart_last_error_message()) == 0);

  size_t stale_size = 123;
  char stale_description[] = "stale";
  assert(llama_dart_model_get_description(nullptr, stale_description,
                                          sizeof(stale_description),
                                          &stale_size) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(stale_size == 0);
  assert(stale_description[0] == '\0');
  stale_size = 123;
  assert(llama_dart_model_metadata_count(nullptr, &stale_size) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(stale_size == 0);
  size_t stale_key_size = 123;
  size_t stale_value_size = 456;
  char stale_key[] = "key";
  char stale_value[] = "value";
  assert(llama_dart_model_metadata_get(nullptr, 0, stale_key,
                                       sizeof(stale_key), &stale_key_size,
                                       stale_value, sizeof(stale_value),
                                       &stale_value_size) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(stale_key_size == 0);
  assert(stale_value_size == 0);
  assert(stale_key[0] == '\0');
  assert(stale_value[0] == '\0');
  stale_size = 123;
  int32_t stale_token_buffer[] = {123};
  assert(llama_dart_model_tokenize(nullptr, nullptr, 0, stale_token_buffer, 1,
                                   &stale_size, 0, 0) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(stale_size == 0);
  assert(stale_token_buffer[0] == 0);
  stale_size = 123;
  uint8_t stale_text_buffer[] = {'s', 't', 'a', 'l', 'e'};
  assert(llama_dart_model_detokenize(nullptr, nullptr, 0, stale_text_buffer,
                                     sizeof(stale_text_buffer),
                                     &stale_size, 0, 0) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(stale_size == 0);
  assert(stale_text_buffer[0] == 0);
  llama_dart_model *stale_model =
      reinterpret_cast<llama_dart_model *>(static_cast<uintptr_t>(1));
  assert(llama_dart_model_load(nullptr, &stale_model) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(stale_model == nullptr);
  llama_dart_context *stale_context =
      reinterpret_cast<llama_dart_context *>(static_cast<uintptr_t>(1));
  assert(llama_dart_context_create(nullptr, nullptr, &stale_context) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(stale_context == nullptr);
  llama_dart_context_info stale_context_info{};
  stale_context_info.struct_size = sizeof(stale_context_info);
  stale_context_info.context_size = 123;
  assert(llama_dart_context_get_info(nullptr, &stale_context_info) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(stale_context_info.struct_size == sizeof(stale_context_info));
  assert(stale_context_info.context_size == 0);
  uint32_t discarded_tokens = 123;
  assert(llama_dart_context_shift(nullptr, 0, 0, &discarded_tokens) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(discarded_tokens == 0);
  assert(llama_dart_context_shift(nullptr, 0, 0, nullptr) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  llama_dart_lora_adapter *stale_adapter =
      reinterpret_cast<llama_dart_lora_adapter *>(static_cast<uintptr_t>(1));
  assert(llama_dart_lora_load(nullptr, nullptr, &stale_adapter) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(stale_adapter == nullptr);
  llama_dart_buffer stale_buffer{
      reinterpret_cast<uint8_t *>(static_cast<uintptr_t>(1)), 123};
  assert(llama_dart_model_get_chat_template(nullptr, &stale_buffer) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(stale_buffer.data == nullptr);
  assert(stale_buffer.size == 0);
  assert(llama_dart_model_get_chat_template(nullptr, nullptr) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  stale_buffer.data =
      reinterpret_cast<uint8_t *>(static_cast<uintptr_t>(1));
  stale_buffer.size = 123;
  assert(llama_dart_model_apply_chat_template(nullptr, nullptr, 0, 0,
                                              &stale_buffer) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(stale_buffer.data == nullptr);
  assert(stale_buffer.size == 0);
  stale_buffer.data = reinterpret_cast<uint8_t *>(static_cast<uintptr_t>(1));
  stale_buffer.size = 123;
  assert(llama_dart_model_create_chat_plan(nullptr, nullptr, 0,
                                           &stale_buffer) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(stale_buffer.data == nullptr);
  assert(stale_buffer.size == 0);
  stale_buffer.data = reinterpret_cast<uint8_t *>(static_cast<uintptr_t>(1));
  stale_buffer.size = 123;
  const uint8_t invalid_chat_plan[] = {'{', '}'};
  assert(llama_dart_chat_parse_output(
             invalid_chat_plan, sizeof(invalid_chat_plan), nullptr, 0,
             &stale_buffer) == LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(stale_buffer.data == nullptr);
  assert(stale_buffer.size == 0);
  llama_dart_chat_template_capabilities stale_chat_capabilities{};
  stale_chat_capabilities.struct_size =
      sizeof(stale_chat_capabilities);
  stale_chat_capabilities.supports_tools = 1;
  assert(llama_dart_model_get_chat_template_capabilities(
             nullptr, &stale_chat_capabilities) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(stale_chat_capabilities.struct_size ==
         sizeof(stale_chat_capabilities));
  assert(stale_chat_capabilities.supports_tools == 0);
  assert(llama_dart_model_get_chat_template_capabilities(nullptr, nullptr) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(llama_dart_chat_parse_output(nullptr, 0, nullptr, 0, nullptr) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  stale_buffer.data = reinterpret_cast<uint8_t *>(static_cast<uintptr_t>(1));
  stale_buffer.size = 123;
  assert(llama_dart_context_state_get(nullptr, &stale_buffer) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(stale_buffer.data == nullptr);
  assert(stale_buffer.size == 0);
  llama_dart_float_buffer stale_float_buffer{
      reinterpret_cast<float *>(static_cast<uintptr_t>(1)), 123};
  assert(llama_dart_context_embed(nullptr, nullptr, &stale_float_buffer) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(stale_float_buffer.data == nullptr);
  assert(stale_float_buffer.length == 0);
  float stale_score = 123.0f;
  assert(llama_dart_context_rerank(nullptr, nullptr, &stale_score) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(stale_score == 0.0f);

  llama_dart_completion_config completion_config{};
  completion_config.struct_size = sizeof(completion_config);
  completion_config.max_tokens = 1;
  const uint8_t oversized_text[] = {'x'};
  llama_dart_buffer completion{};
  llama_dart_completion_stats invalid_stats{};
  llama_dart_completion_stats stale_stats{};
  stale_stats.struct_size = sizeof(stale_stats);
  stale_stats.generated_tokens = 123;
  completion.data = reinterpret_cast<uint8_t *>(static_cast<uintptr_t>(1));
  completion.size = 123;
  assert(llama_dart_context_complete(nullptr, nullptr, &completion,
                                     &stale_stats) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(completion.data == nullptr);
  assert(completion.size == 0);
  assert(stale_stats.struct_size == sizeof(stale_stats));
  assert(stale_stats.generated_tokens == 0);
  assert(llama_dart_context_complete(nullptr, &completion_config, &completion,
                                     &invalid_stats) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(llama_dart_context_complete(nullptr, &completion_config,
                                     &completion, nullptr) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(completion.data == nullptr);
  assert(completion.size == 0);
  llama_dart_generation *generation = nullptr;
  assert(llama_dart_generation_start(nullptr, &completion_config,
                                     &generation) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(generation == nullptr);
  assert(llama_dart_generation_start(nullptr, &completion_config, nullptr) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  uint8_t generation_done = 7;
  stale_stats.generated_tokens = 123;
  completion.data = reinterpret_cast<uint8_t *>(static_cast<uintptr_t>(1));
  completion.size = 123;
  assert(llama_dart_generation_next(nullptr, &completion, &stale_stats,
                                    &generation_done) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(completion.data == nullptr);
  assert(completion.size == 0);
  assert(generation_done == 0);
  assert(stale_stats.struct_size == sizeof(stale_stats));
  assert(stale_stats.generated_tokens == 0);
  stale_stats.generated_tokens = 123;
  completion.data = reinterpret_cast<uint8_t *>(static_cast<uintptr_t>(1));
  completion.size = 123;
  assert(llama_dart_generation_next(nullptr, &completion, &stale_stats,
                                    nullptr) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(completion.data == nullptr);
  assert(completion.size == 0);
  assert(stale_stats.struct_size == sizeof(stale_stats));
  assert(stale_stats.generated_tokens == 0);
  llama_dart_generation_free(nullptr);
  const uint8_t prompt_with_nul[] = {'b', 'a', 'd', '\0', 'p', 'r', 'o', 'm',
                                     'p', 't'};
  completion_config.prompt_data = prompt_with_nul;
  completion_config.prompt_size = sizeof(prompt_with_nul);
  assert(llama_dart_context_complete(nullptr, &completion_config,
                                     &completion, nullptr) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(completion.data == nullptr);
  assert(completion.size == 0);
  const uint8_t blank_prompt[] = {' ', '\t'};
  completion_config.prompt_data = blank_prompt;
  completion_config.prompt_size = sizeof(blank_prompt);
  assert(llama_dart_context_complete(nullptr, &completion_config,
                                     &completion, nullptr) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(std::strstr(llama_dart_last_error_message(), "blank") != nullptr);
  assert(completion.data == nullptr);
  assert(completion.size == 0);
  completion_config.prompt_data = oversized_text;
  completion_config.prompt_size = std::numeric_limits<size_t>::max();
  assert(llama_dart_context_complete(nullptr, &completion_config,
                                     &completion, nullptr) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(completion.data == nullptr);
  assert(completion.size == 0);
  completion_config.prompt_data = oversized_text;
  completion_config.prompt_size = sizeof(oversized_text);
  completion_config.add_special = 3;
  assert(llama_dart_context_complete(nullptr, &completion_config,
                                     &completion, nullptr) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  completion_config.add_special = LLAMA_DART_ADD_SPECIAL_IF_CONTEXT_EMPTY;
  completion_config.reuse_prompt_prefix = 2;
  assert(llama_dart_context_complete(nullptr, &completion_config,
                                     &completion, nullptr) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  completion_config.reuse_prompt_prefix = 0;
  completion_config.prompt_data = nullptr;
  completion_config.prompt_size = 0;
  completion_config.stop_sequence_count = 1;
  assert(llama_dart_context_complete(nullptr, &completion_config,
                                     &completion, nullptr) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  llama_dart_string_view stop_sequence{};
  completion_config.stop_sequences = &stop_sequence;
  assert(llama_dart_context_complete(nullptr, &completion_config,
                                     &completion, nullptr) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  stop_sequence.data = oversized_text;
  stop_sequence.size = std::numeric_limits<size_t>::max();
  assert(llama_dart_context_complete(nullptr, &completion_config,
                                     &completion, nullptr) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  const uint8_t stop_with_nul[] = {'s', 't', 'o', 'p', '\0'};
  stop_sequence.data = stop_with_nul;
  stop_sequence.size = sizeof(stop_with_nul);
  assert(llama_dart_context_complete(nullptr, &completion_config,
                                     &completion, nullptr) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  completion_config.stop_sequences = nullptr;
  completion_config.stop_sequence_count = 0;
  const uint8_t blank_grammar[] = {' ', '\t', '\n'};
  const uint8_t grammar_root[] = {'r', 'o', 'o', 't'};
  completion_config.grammar_data = blank_grammar;
  completion_config.grammar_size = sizeof(blank_grammar);
  completion_config.grammar_root_data = grammar_root;
  completion_config.grammar_root_size = sizeof(grammar_root);
  assert(llama_dart_context_complete(nullptr, &completion_config,
                                     &completion, nullptr) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(completion.data == nullptr);
  assert(completion.size == 0);
  const uint8_t grammar[] = {'r', 'o', 'o', 't', ' ', ':', ':', '=', ' ',
                             '"', 'x', '"'};
  const uint8_t blank_grammar_root[] = {' ', '\t'};
  completion_config.grammar_data = grammar;
  completion_config.grammar_size = sizeof(grammar);
  completion_config.grammar_root_data = blank_grammar_root;
  completion_config.grammar_root_size = sizeof(blank_grammar_root);
  assert(llama_dart_context_complete(nullptr, &completion_config,
                                     &completion, nullptr) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(completion.data == nullptr);
  assert(completion.size == 0);
  const uint8_t whitespace_grammar_root[] = {'r', 'o', 'o', 't', ' ', 'x'};
  completion_config.grammar_root_data = whitespace_grammar_root;
  completion_config.grammar_root_size = sizeof(whitespace_grammar_root);
  assert(llama_dart_context_complete(nullptr, &completion_config,
                                     &completion, nullptr) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(std::strstr(llama_dart_last_error_message(), "whitespace") != nullptr);
  assert(completion.data == nullptr);
  assert(completion.size == 0);
  completion_config.grammar_data = nullptr;
  completion_config.grammar_size = 0;
  completion_config.grammar_root_data = nullptr;
  completion_config.grammar_root_size = 0;
  completion_config.json_schema_data = blank_grammar;
  completion_config.json_schema_size = sizeof(blank_grammar);
  assert(llama_dart_context_complete(nullptr, &completion_config,
                                     &completion, nullptr) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(std::strstr(llama_dart_last_error_message(), "non-blank") != nullptr);
  completion_config.json_schema_data =
      reinterpret_cast<const uint8_t *>(json_schema);
  completion_config.json_schema_size = std::strlen(json_schema);
  completion_config.grammar_data = grammar;
  completion_config.grammar_size = sizeof(grammar);
  completion_config.grammar_root_data = grammar_root;
  completion_config.grammar_root_size = sizeof(grammar_root);
  assert(llama_dart_context_complete(nullptr, &completion_config,
                                     &completion, nullptr) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(std::strstr(llama_dart_last_error_message(), "mutually exclusive") !=
         nullptr);
  completion_config.grammar_data = nullptr;
  completion_config.grammar_size = 0;
  completion_config.grammar_root_data = nullptr;
  completion_config.grammar_root_size = 0;
  assert(llama_dart_context_complete(nullptr, &completion_config,
                                     &completion, nullptr) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(std::strstr(llama_dart_last_error_message(), "context") != nullptr);
  completion_config.json_schema_data = nullptr;
  completion_config.json_schema_size = 0;
  completion_config.chat_plan_size = 1;
  assert(llama_dart_context_complete(nullptr, &completion_config,
                                     &completion, nullptr) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  completion_config.chat_plan_data = oversized_text;
  completion_config.chat_plan_size = 0;
  assert(llama_dart_context_complete(nullptr, &completion_config,
                                     &completion, nullptr) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  completion_config.chat_plan_size = sizeof(oversized_text);
  completion_config.grammar_data = grammar;
  completion_config.grammar_size = sizeof(grammar);
  completion_config.grammar_root_data = grammar_root;
  completion_config.grammar_root_size = sizeof(grammar_root);
  assert(llama_dart_context_complete(nullptr, &completion_config,
                                     &completion, nullptr) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(std::strstr(llama_dart_last_error_message(), "context") != nullptr);
  assert(std::strstr(llama_dart_last_error_message(), "cannot be combined") ==
         nullptr);
  completion_config.grammar_data = nullptr;
  completion_config.grammar_size = 0;
  completion_config.grammar_root_data = nullptr;
  completion_config.grammar_root_size = 0;
  completion_config.json_schema_data =
      reinterpret_cast<const uint8_t *>(json_schema);
  completion_config.json_schema_size = std::strlen(json_schema);
  assert(llama_dart_context_complete(nullptr, &completion_config,
                                     &completion, nullptr) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(std::strstr(llama_dart_last_error_message(), "context") != nullptr);
  assert(std::strstr(llama_dart_last_error_message(), "cannot be combined") ==
         nullptr);
  completion_config.chat_plan_data = nullptr;
  completion_config.chat_plan_size = 0;
  completion_config.json_schema_data = nullptr;
  completion_config.json_schema_size = 0;
  int32_t stop_token = 1;
  completion_config.stop_token_count = 1;
  assert(llama_dart_context_complete(nullptr, &completion_config,
                                     &completion, nullptr) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  completion_config.stop_tokens = &stop_token;
  completion_config.stop_token_count = 0;
  assert(llama_dart_context_complete(nullptr, &completion_config,
                                     &completion, nullptr) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  completion_config.stop_token_count = 1025;
  assert(llama_dart_context_complete(nullptr, &completion_config,
                                     &completion, nullptr) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  completion_config.stop_tokens = nullptr;
  completion_config.stop_token_count = 0;
  completion_config.prompt_data = oversized_text;
  completion_config.prompt_size = sizeof(oversized_text);
  completion_config.media_input_count = 1;
  assert(llama_dart_context_complete(nullptr, &completion_config,
                                     &completion, nullptr) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  completion_config.media_input_count = 65;
  assert(llama_dart_context_complete(nullptr, &completion_config,
                                     &completion, nullptr) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  llama_dart_media_input media_input{};
  media_input.struct_size = sizeof(media_input);
  media_input.type = LLAMA_DART_MEDIA_IMAGE;
  media_input.content_data = oversized_text;
  media_input.content_size = sizeof(oversized_text);
  completion_config.media_inputs = &media_input;
  completion_config.media_input_count = 0;
  assert(llama_dart_context_complete(nullptr, &completion_config,
                                     &completion, nullptr) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  completion_config.media_inputs = &media_input;
  completion_config.media_input_count = 1;
  completion_config.prompt_data = nullptr;
  completion_config.prompt_size = 0;
  assert(llama_dart_context_complete(nullptr, &completion_config,
                                     &completion, nullptr) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  completion_config.media_inputs = nullptr;
  completion_config.media_input_count = 0;

  const char *fixture_model_path = std::getenv("LLAMA_DART_TEST_VOCAB_MODEL");
  if (fixture_model_path == nullptr || std::strlen(fixture_model_path) == 0) {
    fixture_model_path = LLAMA_DART_TEST_VOCAB_MODEL;
  }
  std::ifstream fixture_model(fixture_model_path, std::ios::binary);
  if (!fixture_model.good()) {
    llama_dart_clear_last_error();
    assert(std::strlen(llama_dart_last_error_message()) == 0);
    return 0;
  }

  llama_dart_model_load_config fixture_config{};
  fixture_config.struct_size = sizeof(fixture_config);
  fixture_config.model_path_data =
      reinterpret_cast<const uint8_t *>(fixture_model_path);
  fixture_config.model_path_size = std::strlen(fixture_model_path);
  fixture_config.n_gpu_layers = 0;
  fixture_config.vocab_only = 1;
  fixture_config.use_mmap = 1;
  fixture_config.use_mlock = 0;
  fixture_config.check_tensors = 1;

  llama_dart_model *template_less_model = nullptr;
  assert(llama_dart_model_load(&fixture_config, &template_less_model) ==
         LLAMA_DART_SUCCESS);
  assert(template_less_model != nullptr);
  llama_dart_buffer missing_template{};
  assert(llama_dart_model_get_chat_template(template_less_model,
                                            &missing_template) ==
         LLAMA_DART_ERROR_UNSUPPORTED);
  assert(missing_template.data == nullptr);
  assert(missing_template.size == 0);
  const uint8_t missing_template_text[] = {'h', 'e', 'l', 'l', 'o'};
  const uint8_t missing_template_role[] = {'u', 's', 'e', 'r'};
  llama_dart_chat_message missing_template_message{};
  missing_template_message.struct_size = sizeof(missing_template_message);
  missing_template_message.role_data = missing_template_role;
  missing_template_message.role_size = sizeof(missing_template_role);
  missing_template_message.content_data = missing_template_text;
  missing_template_message.content_size = sizeof(missing_template_text);
  assert(llama_dart_model_apply_chat_template(
             template_less_model, &missing_template_message, 1, 1,
             &missing_template) == LLAMA_DART_ERROR_UNSUPPORTED);
  assert(missing_template.data == nullptr);
  assert(missing_template.size == 0);
  llama_dart_chat_template_capabilities missing_template_capabilities{};
  missing_template_capabilities.struct_size =
      sizeof(missing_template_capabilities);
  assert(llama_dart_model_get_chat_template_capabilities(
             template_less_model, &missing_template_capabilities) ==
         LLAMA_DART_ERROR_UNSUPPORTED);
  assert(missing_template_capabilities.supports_tools == 0);
  assert(missing_template_capabilities.supports_tool_calls == 0);
  assert(missing_template_capabilities.supports_parallel_tool_calls == 0);
  const char missing_template_request[] =
      R"({"messages":[{"role":"user","content":"hello"}],"tools":[],"tool_choice":"auto","parallel_tool_calls":false,"add_generation_prompt":true})";
  assert(llama_dart_model_create_chat_plan(
             template_less_model,
             reinterpret_cast<const uint8_t *>(missing_template_request),
             std::strlen(missing_template_request), &missing_template) ==
         LLAMA_DART_ERROR_UNSUPPORTED);
  assert(missing_template.data == nullptr);
  assert(missing_template.size == 0);
  llama_dart_model_free(template_less_model);

  const uint8_t explicit_chat_template[] = {'c', 'h', 'a', 't', 'm', 'l'};
  fixture_config.chat_template_data = explicit_chat_template;
  fixture_config.chat_template_size = sizeof(explicit_chat_template);
  assert(llama_dart_model_load(&fixture_config, &model) == LLAMA_DART_SUCCESS);
  assert(model != nullptr);
  llama_dart_buffer effective_template{};
  assert(llama_dart_model_get_chat_template(model, &effective_template) ==
         LLAMA_DART_SUCCESS);
  assert(effective_template.size == sizeof(explicit_chat_template));
  assert(std::memcmp(effective_template.data, explicit_chat_template,
                     sizeof(explicit_chat_template)) == 0);
  llama_dart_buffer_free(effective_template.data);

  llama_dart_model_load_config qwen_fixture_config = fixture_config;
  const char *qwen_fixture_path = LLAMA_DART_TEST_QWEN_VOCAB_MODEL;
  qwen_fixture_config.model_path_data = reinterpret_cast<const uint8_t *>(qwen_fixture_path);
  qwen_fixture_config.model_path_size = std::strlen(qwen_fixture_path);
  llama_dart_model *qwen_model = nullptr;
  assert(llama_dart_model_load(&qwen_fixture_config, &qwen_model) == LLAMA_DART_SUCCESS);
  verify_token_piece_bound(qwen_model);
  llama_dart_model_free(qwen_model);

  llama_dart_model_info info{};
  info.struct_size = sizeof(info);
  assert(llama_dart_model_get_info(model, &info) == LLAMA_DART_SUCCESS);
  assert(info.n_vocab > 0);
  verify_token_piece_bound(model);
  assert(info.n_ctx_train > 0);
  assert(info.n_embd > 0);
  assert(info.n_embd_inp > 0);
  assert(info.n_embd_out > 0);
  assert(info.n_layer > 0);
  assert(info.n_head > 0);
  assert(info.n_head_kv > 0);
  assert(info.token_bos >= -1);
  assert(info.token_eos >= -1);
  assert(info.token_nl >= -1);
  assert(std::strlen(llama_dart_model_file_type_name(info.ftype)) > 0);
  size_t description_size = 0;
  description_size = 123;
  assert(llama_dart_model_get_description(model, nullptr, 1,
                                          &description_size) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(description_size == 0);
  assert(llama_dart_model_get_description(model, nullptr, 0,
                                          &description_size) ==
         LLAMA_DART_ERROR_BUFFER_TOO_SMALL);
  assert(description_size == 5);
  char description[16]{};
  assert(llama_dart_model_get_description(model, description,
                                          sizeof(description),
                                          &description_size) ==
         LLAMA_DART_SUCCESS);
  assert(std::strcmp(description, "gpt-2") == 0);
  size_t metadata_count = 0;
  assert(llama_dart_model_metadata_count(model, &metadata_count) ==
         LLAMA_DART_SUCCESS);
  assert(metadata_count > 0);
  size_t metadata_key_size = 0;
  size_t metadata_value_size = 0;
  assert(llama_dart_model_metadata_get(
             model, 0, nullptr, 0, &metadata_key_size, nullptr, 0,
             &metadata_value_size) == LLAMA_DART_ERROR_BUFFER_TOO_SMALL);
  assert(metadata_key_size > 0);
  char metadata_key[256]{};
  char metadata_value[512]{};
  assert(llama_dart_model_metadata_get(
             model, 0, metadata_key, sizeof(metadata_key), &metadata_key_size,
             metadata_value, sizeof(metadata_value), &metadata_value_size) ==
         LLAMA_DART_SUCCESS);
  assert(std::strlen(metadata_key) == metadata_key_size);
  metadata_key_size = 123;
  metadata_value_size = 456;
  assert(llama_dart_model_metadata_get(
             model, metadata_count, metadata_key, sizeof(metadata_key),
             &metadata_key_size, metadata_value, sizeof(metadata_value),
             &metadata_value_size) == LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(metadata_key_size == 0);
  assert(metadata_value_size == 0);
  assert(metadata_key[0] == '\0');
  assert(metadata_value[0] == '\0');

  const uint8_t text[] = {'h', 'e', 'l', 'l', 'o'};
  const uint8_t role[] = {'u', 's', 'e', 'r'};
  llama_dart_chat_message chat_message{};
  chat_message.struct_size = sizeof(chat_message);
  chat_message.role_data = role;
  chat_message.role_size = sizeof(role);
  chat_message.content_data = text;
  chat_message.content_size = sizeof(text);
  llama_dart_buffer prompt{};
  assert(llama_dart_model_apply_chat_template(model, &chat_message, 1, 1,
                                              &prompt) ==
         LLAMA_DART_SUCCESS);
  assert(prompt.data != nullptr);
  assert(prompt.size > sizeof(text));
  llama_dart_buffer_free(prompt.data);
  assert(llama_dart_model_apply_chat_template(model, nullptr, 0, 1,
                                              &prompt) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(prompt.data == nullptr);
  assert(prompt.size == 0);
  const uint8_t chat_text_with_nul[] = {'b', 'a', 'd', '\0', 'c', 'h', 'a',
                                        't'};
  chat_message.content_data = chat_text_with_nul;
  chat_message.content_size = sizeof(chat_text_with_nul);
  assert(llama_dart_model_apply_chat_template(model, &chat_message, 1, 1,
                                              &prompt) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(prompt.data == nullptr);
  assert(prompt.size == 0);
  const uint8_t role_with_line_break[] = {'u', 's', 'e', 'r', '\n'};
  chat_message.role_data = role_with_line_break;
  chat_message.role_size = sizeof(role_with_line_break);
  chat_message.content_data = text;
  chat_message.content_size = sizeof(text);
  assert(llama_dart_model_apply_chat_template(model, &chat_message, 1, 1,
                                              &prompt) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(prompt.data == nullptr);
  assert(prompt.size == 0);
  const uint8_t unsupported_role[] = {'d', 'e', 'v', 'e', 'l', 'o', 'p', 'e',
                                      'r'};
  chat_message.role_data = unsupported_role;
  chat_message.role_size = sizeof(unsupported_role);
  chat_message.content_data = text;
  chat_message.content_size = sizeof(text);
  assert(llama_dart_model_apply_chat_template(model, &chat_message, 1, 1,
                                              &prompt) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(prompt.data == nullptr);
  assert(prompt.size == 0);
  const uint8_t blank_role[] = {' ', '\t'};
  chat_message.role_data = blank_role;
  chat_message.role_size = sizeof(blank_role);
  chat_message.content_data = text;
  chat_message.content_size = sizeof(text);
  assert(llama_dart_model_apply_chat_template(model, &chat_message, 1, 1,
                                              &prompt) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(prompt.data == nullptr);
  assert(prompt.size == 0);
  const uint8_t blank_chat_text[] = {' ', '\t'};
  chat_message.role_data = role;
  chat_message.role_size = sizeof(role);
  chat_message.content_data = blank_chat_text;
  chat_message.content_size = sizeof(blank_chat_text);
  assert(llama_dart_model_apply_chat_template(model, &chat_message, 1, 1,
                                              &prompt) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(prompt.data == nullptr);
  assert(prompt.size == 0);
  chat_message.role_data = role;
  chat_message.role_size = sizeof(role);
  chat_message.content_data = text;
  chat_message.content_size = std::numeric_limits<size_t>::max();
  assert(llama_dart_model_apply_chat_template(model, &chat_message, 1, 1,
                                              &prompt) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(prompt.data == nullptr);
  assert(prompt.size == 0);
  chat_message.content_size = sizeof(text);

  llama_dart_chat_template_capabilities chat_capabilities{};
  chat_capabilities.struct_size = sizeof(chat_capabilities);
  assert(llama_dart_model_get_chat_template_capabilities(
             model, &chat_capabilities) == LLAMA_DART_SUCCESS);
  assert(chat_capabilities.struct_size == sizeof(chat_capabilities));
  assert(chat_capabilities.supports_tools == 0);
  assert(chat_capabilities.supports_tool_calls == 0);

  const char simple_chat_request[] =
      R"({"messages":[{"role":"user","content":"hello"}],"tools":[],"tool_choice":"auto","parallel_tool_calls":false,"add_generation_prompt":true})";
  llama_dart_buffer chat_plan{};
  assert(llama_dart_model_create_chat_plan(
             model, reinterpret_cast<const uint8_t *>(simple_chat_request),
             std::strlen(simple_chat_request), &chat_plan) ==
         LLAMA_DART_SUCCESS);
  assert(chat_plan.data != nullptr);
  assert(chat_plan.size > 0);
  const std::string chat_plan_text(
      reinterpret_cast<const char *>(chat_plan.data), chat_plan.size);
  assert(chat_plan_text.find("\"version\":1") != std::string::npos);
  assert(chat_plan_text.find("\"prompt\"") != std::string::npos);
  const char assistant_output[] = "hello";
  llama_dart_buffer assistant_message{};
  assert(llama_dart_chat_parse_output(
             chat_plan.data, chat_plan.size,
             reinterpret_cast<const uint8_t *>(assistant_output),
             std::strlen(assistant_output), &assistant_message) ==
         LLAMA_DART_SUCCESS);
  assert(assistant_message.data != nullptr);
  const std::string assistant_message_text(
      reinterpret_cast<const char *>(assistant_message.data),
      assistant_message.size);
  assert(assistant_message_text.find("\"role\":\"assistant\"") !=
         std::string::npos);
  llama_dart_buffer_free(assistant_message.data);
  llama_dart_buffer_free(chat_plan.data);

  const char markerless_thinking_off_request[] =
      R"({"messages":[{"role":"user","content":"hello"}],"tools":[],"tool_choice":"auto","parallel_tool_calls":false,"add_generation_prompt":true,"enable_thinking":false})";
  chat_plan = {};
  assert(llama_dart_model_create_chat_plan(
             model,
             reinterpret_cast<const uint8_t *>(
                 markerless_thinking_off_request),
             std::strlen(markerless_thinking_off_request), &chat_plan) ==
         LLAMA_DART_SUCCESS);
  assert(chat_plan.data != nullptr);
  const std::string markerless_thinking_off_plan(
      reinterpret_cast<const char *>(chat_plan.data), chat_plan.size);
  assert(markerless_thinking_off_plan.find(
             R"("reasoning_budget_tokens":-1)") != std::string::npos);
  llama_dart_buffer_free(chat_plan.data);

  const char reasoning_budget_without_thinking[] =
      R"({"messages":[{"role":"user","content":"hello"}],"tools":[],"tool_choice":"auto","parallel_tool_calls":false,"add_generation_prompt":true,"reasoning_budget_tokens":8})";
  chat_plan = {};
  assert(llama_dart_model_create_chat_plan(
             model,
             reinterpret_cast<const uint8_t *>(
                 reasoning_budget_without_thinking),
             std::strlen(reasoning_budget_without_thinking), &chat_plan) ==
         LLAMA_DART_ERROR_UNSUPPORTED);
  assert(chat_plan.data == nullptr);
  assert(chat_plan.size == 0);

  const char non_integer_reasoning_budget[] =
      R"({"messages":[{"role":"user","content":"hello"}],"tools":[],"tool_choice":"auto","parallel_tool_calls":false,"add_generation_prompt":true,"enable_thinking":true,"reasoning_budget_tokens":1.5})";
  assert(llama_dart_model_create_chat_plan(
             model,
             reinterpret_cast<const uint8_t *>(non_integer_reasoning_budget),
             std::strlen(non_integer_reasoning_budget), &chat_plan) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(chat_plan.data == nullptr);
  assert(chat_plan.size == 0);

  const char unsupported_reasoning_budget_template[] =
      R"({"messages":[{"role":"user","content":"hello"}],"tools":[],"tool_choice":"auto","parallel_tool_calls":false,"add_generation_prompt":true,"enable_thinking":true,"reasoning_budget_tokens":8})";
  assert(llama_dart_model_create_chat_plan(
             model,
             reinterpret_cast<const uint8_t *>(
                 unsupported_reasoning_budget_template),
             std::strlen(unsupported_reasoning_budget_template), &chat_plan) ==
         LLAMA_DART_ERROR_UNSUPPORTED);
  assert(chat_plan.data == nullptr);
  assert(chat_plan.size == 0);

  const char tool_chat_request[] =
      R"({"messages":[{"role":"user","content":"hello"}],"tools":[{"type":"function","function":{"name":"lookup","description":"Look up local data","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"],"additionalProperties":false}}}],"tool_choice":"auto","parallel_tool_calls":true,"add_generation_prompt":true})";
  assert(llama_dart_model_create_chat_plan(
             model, reinterpret_cast<const uint8_t *>(tool_chat_request),
             std::strlen(tool_chat_request), &chat_plan) ==
         LLAMA_DART_ERROR_UNSUPPORTED);
  assert(chat_plan.data == nullptr);
  assert(chat_plan.size == 0);

  const char *tool_fixture_model_path =
      std::getenv("LLAMA_DART_TEST_TOOL_VOCAB_MODEL");
  if (tool_fixture_model_path == nullptr ||
      std::strlen(tool_fixture_model_path) == 0) {
    tool_fixture_model_path = LLAMA_DART_TEST_TOOL_VOCAB_MODEL;
  }
  llama_dart_model_load_config tool_fixture_config = fixture_config;
  tool_fixture_config.model_path_data =
      reinterpret_cast<const uint8_t *>(tool_fixture_model_path);
  tool_fixture_config.model_path_size = std::strlen(tool_fixture_model_path);
  tool_fixture_config.chat_template_data = nullptr;
  tool_fixture_config.chat_template_size = 0;
  llama_dart_model *tool_model = nullptr;
  assert(llama_dart_model_load(&tool_fixture_config, &tool_model) ==
         LLAMA_DART_SUCCESS);
  assert(tool_model != nullptr);
  verify_token_piece_bound(tool_model);
  effective_template = {};
  assert(llama_dart_model_get_chat_template(tool_model, &effective_template) ==
         LLAMA_DART_SUCCESS);
  assert(effective_template.data != nullptr);
  assert(effective_template.size > 0);
  llama_dart_buffer_free(effective_template.data);
  chat_capabilities = {};
  chat_capabilities.struct_size = sizeof(chat_capabilities);
  assert(llama_dart_model_get_chat_template_capabilities(
             tool_model, &chat_capabilities) == LLAMA_DART_SUCCESS);
  assert(chat_capabilities.supports_tools == 1);
  assert(chat_capabilities.supports_tool_calls == 1);
  assert(chat_capabilities.supports_parallel_tool_calls == 1);

  prompt = {};
  assert(llama_dart_model_apply_chat_template(
             tool_model, &chat_message, 1, 1, &prompt) ==
         LLAMA_DART_ERROR_UNSUPPORTED);
  assert(prompt.data == nullptr);
  assert(prompt.size == 0);

  assert(llama_dart_model_create_chat_plan(
             tool_model,
             reinterpret_cast<const uint8_t *>(simple_chat_request),
             std::strlen(simple_chat_request), &chat_plan) ==
         LLAMA_DART_SUCCESS);
  assert(chat_plan.data != nullptr);
  assert(chat_plan.size > 0);
  const std::string plain_gemma4_chat_plan(
      reinterpret_cast<const char *>(chat_plan.data), chat_plan.size);
  assert(plain_gemma4_chat_plan.find("hello") != std::string::npos);
  llama_dart_buffer_free(chat_plan.data);

  const char gemma4_reasoning_budget_request[] =
      R"({"messages":[{"role":"user","content":"hello"}],"tools":[],"tool_choice":"auto","parallel_tool_calls":false,"add_generation_prompt":true,"reasoning_budget_tokens":8})";
  chat_plan = {};
  assert(llama_dart_model_create_chat_plan(
             tool_model,
             reinterpret_cast<const uint8_t *>(
                 gemma4_reasoning_budget_request),
             std::strlen(gemma4_reasoning_budget_request), &chat_plan) ==
         LLAMA_DART_SUCCESS);
  assert(chat_plan.data != nullptr);
  const std::string gemma4_reasoning_chat_plan(
      reinterpret_cast<const char *>(chat_plan.data), chat_plan.size);
  assert(gemma4_reasoning_chat_plan.find(
             R"("thinking_start_tag":"<|channel>thought")") !=
         std::string::npos);
  assert(gemma4_reasoning_chat_plan.find(
             R"("thinking_end_tags":["<channel|>"])") !=
         std::string::npos);
  assert(gemma4_reasoning_chat_plan.find(
             R"("reasoning_budget_tokens":8)") != std::string::npos);
  llama_dart_buffer_free(chat_plan.data);

  const char gemma4_thinking_off_request[] =
      R"({"messages":[{"role":"user","content":"hello"}],"tools":[],"tool_choice":"auto","parallel_tool_calls":false,"add_generation_prompt":true,"enable_thinking":false})";
  chat_plan = {};
  assert(llama_dart_model_create_chat_plan(
             tool_model,
             reinterpret_cast<const uint8_t *>(gemma4_thinking_off_request),
             std::strlen(gemma4_thinking_off_request), &chat_plan) ==
         LLAMA_DART_SUCCESS);
  assert(chat_plan.data != nullptr);
  const std::string gemma4_thinking_off_plan(
      reinterpret_cast<const char *>(chat_plan.data), chat_plan.size);
  assert(gemma4_thinking_off_plan.find(
             R"("thinking_start_tag":"<|channel>thought")") !=
         std::string::npos);
  assert(gemma4_thinking_off_plan.find(
             R"("thinking_end_tags":["<channel|>"])") !=
         std::string::npos);
  assert(gemma4_thinking_off_plan.find(
             R"("reasoning_budget_tokens":0)") != std::string::npos);
  llama_dart_buffer_free(chat_plan.data);

  const char gemma4_thinking_start[] = "<|channel>thought";
  size_t gemma4_thinking_start_count = 0;
  assert(llama_dart_model_tokenize(
             tool_model,
             reinterpret_cast<const uint8_t *>(gemma4_thinking_start),
             std::strlen(gemma4_thinking_start), nullptr, 0,
             &gemma4_thinking_start_count, 0, 1) ==
         LLAMA_DART_ERROR_BUFFER_TOO_SMALL);
  assert(gemma4_thinking_start_count > 1);
  std::vector<int32_t> gemma4_thinking_start_tokens(
      gemma4_thinking_start_count);
  assert(llama_dart_model_tokenize(
             tool_model,
             reinterpret_cast<const uint8_t *>(gemma4_thinking_start),
             std::strlen(gemma4_thinking_start),
             gemma4_thinking_start_tokens.data(),
             gemma4_thinking_start_tokens.size(),
             &gemma4_thinking_start_count, 0, 1) ==
         LLAMA_DART_SUCCESS);

  const char required_tool_chat_request[] =
      R"({"messages":[{"role":"user","content":"hello"}],"tools":[{"type":"function","function":{"name":"lookup","description":"Look up local data","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"],"additionalProperties":false}}}],"tool_choice":"required","parallel_tool_calls":true,"add_generation_prompt":true})";
  chat_plan = {};
  assert(llama_dart_model_create_chat_plan(
             tool_model,
             reinterpret_cast<const uint8_t *>(required_tool_chat_request),
             std::strlen(required_tool_chat_request), &chat_plan) ==
         LLAMA_DART_SUCCESS);
  assert(chat_plan.data != nullptr);
  const std::string gemma4_tool_chat_plan(
      reinterpret_cast<const char *>(chat_plan.data), chat_plan.size);
  assert(gemma4_tool_chat_plan.find("query") != std::string::npos);

  const char valid_gemma4_call[] =
      R"(<|tool_call>call:lookup{query: <|"|>alpha<|"|>}<tool_call|>)";
  assistant_message = {};
  assert(llama_dart_chat_parse_output(
             chat_plan.data, chat_plan.size,
             reinterpret_cast<const uint8_t *>(valid_gemma4_call),
             std::strlen(valid_gemma4_call), &assistant_message) ==
         LLAMA_DART_SUCCESS);
  const std::string valid_gemma4_message(
      reinterpret_cast<const char *>(assistant_message.data),
      assistant_message.size);
  assert(valid_gemma4_message.find("{\\\"query\\\":\\\"alpha\\\"}") !=
         std::string::npos);
  llama_dart_buffer_free(assistant_message.data);

  // The pristine upstream Gemma 4 grammar accepts generic dictionary members.
  // The checked Dart wrapper validates this parsed call against the matching
  // tool schema before exposing it through GenerationChunk.assistantMessage.
  const char schema_invalid_gemma4_call[] =
      R"(<|tool_call>call:lookup{markdown: <|"|>alpha<|"|>}<tool_call|>)";
  assistant_message = {};
  assert(llama_dart_chat_parse_output(
             chat_plan.data, chat_plan.size,
             reinterpret_cast<const uint8_t *>(schema_invalid_gemma4_call),
             std::strlen(schema_invalid_gemma4_call), &assistant_message) ==
         LLAMA_DART_SUCCESS);
  const std::string schema_invalid_gemma4_message(
      reinterpret_cast<const char *>(assistant_message.data),
      assistant_message.size);
  assert(schema_invalid_gemma4_message.find("markdown") != std::string::npos);
  llama_dart_buffer_free(assistant_message.data);
  llama_dart_buffer_free(chat_plan.data);

  const char marker_chat_request[] =
      R"({"messages":[{"role":"user","content":"before<__media__>middle<__media__>after"}],"tools":[],"tool_choice":"auto","parallel_tool_calls":false,"add_generation_prompt":true})";
  chat_plan = {};
  assert(llama_dart_model_create_chat_plan(
             tool_model,
             reinterpret_cast<const uint8_t *>(marker_chat_request),
             std::strlen(marker_chat_request), &chat_plan) ==
         LLAMA_DART_SUCCESS);
  assert(chat_plan.data != nullptr);
  assert(chat_plan.size > 0);
  const std::string marker_chat_plan(
      reinterpret_cast<const char *>(chat_plan.data), chat_plan.size);
  const size_t before_pos = marker_chat_plan.find("before");
  const size_t first_marker_pos = marker_chat_plan.find("<__media__>");
  const size_t middle_pos = marker_chat_plan.find("middle");
  const size_t second_marker_pos = marker_chat_plan.find(
      "<__media__>", first_marker_pos + 1);
  const size_t after_pos = marker_chat_plan.find("after");
  assert(before_pos < first_marker_pos);
  assert(first_marker_pos < middle_pos);
  assert(middle_pos < second_marker_pos);
  assert(second_marker_pos < after_pos);
  assert(marker_chat_plan.find("lookup") == std::string::npos);
  llama_dart_buffer_free(chat_plan.data);

  const char simple_chat_request_without_generation_prompt[] =
      R"({"messages":[{"role":"user","content":"hello"}],"tools":[],"tool_choice":"auto","parallel_tool_calls":false,"add_generation_prompt":false})";
  chat_plan = {};
  assert(llama_dart_model_create_chat_plan(
             tool_model,
             reinterpret_cast<const uint8_t *>(
                 simple_chat_request_without_generation_prompt),
             std::strlen(simple_chat_request_without_generation_prompt),
             &chat_plan) == LLAMA_DART_SUCCESS);
  assert(chat_plan.data != nullptr);
  assert(chat_plan.size > 0);
  llama_dart_buffer_free(chat_plan.data);

  assert(llama_dart_model_create_chat_plan(
             tool_model, reinterpret_cast<const uint8_t *>(tool_chat_request),
             std::strlen(tool_chat_request), &chat_plan) ==
         LLAMA_DART_SUCCESS);
  assert(chat_plan.data != nullptr);
  assert(chat_plan.size > 0);
  const std::string tool_chat_plan_text(
      reinterpret_cast<const char *>(chat_plan.data), chat_plan.size);
  assert(tool_chat_plan_text.find("lookup") != std::string::npos);
  const char tool_output[] =
      R"(<|tool_call>call:lookup{query:<|"|>alpha<|"|>}<tool_call|>)";
  assistant_message = {};
  assert(llama_dart_chat_parse_output(
             chat_plan.data, chat_plan.size,
             reinterpret_cast<const uint8_t *>(tool_output),
             std::strlen(tool_output), &assistant_message) ==
         LLAMA_DART_SUCCESS);
  assert(assistant_message.data != nullptr);
  const std::string tool_message_text(
      reinterpret_cast<const char *>(assistant_message.data),
      assistant_message.size);
  assert(tool_message_text.find("lookup") != std::string::npos);
  assert(tool_message_text.find("alpha") != std::string::npos);
  llama_dart_buffer_free(assistant_message.data);
  llama_dart_buffer_free(chat_plan.data);
  llama_dart_model_free(tool_model);

  const char malformed_chat_request[] = "{";
  assert(llama_dart_model_create_chat_plan(
             model, reinterpret_cast<const uint8_t *>(malformed_chat_request),
             std::strlen(malformed_chat_request), &chat_plan) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(chat_plan.data == nullptr);
  assert(chat_plan.size == 0);

  llama_dart_embedding_config embedding_config{};
  embedding_config.struct_size = sizeof(embedding_config);
  embedding_config.text_data = text;
  embedding_config.text_size = sizeof(text);
  llama_dart_float_buffer embedding{};
  const uint8_t embedding_text_with_nul[] = {'b', 'a', 'd', '\0', 'e', 'm',
                                             'b'};
  embedding_config.text_data = embedding_text_with_nul;
  embedding_config.text_size = sizeof(embedding_text_with_nul);
  assert(llama_dart_context_embed(nullptr, &embedding_config, &embedding) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(embedding.data == nullptr);
  assert(embedding.length == 0);
  const uint8_t blank_text[] = {' ', '\t'};
  embedding_config.text_data = blank_text;
  embedding_config.text_size = sizeof(blank_text);
  assert(llama_dart_context_embed(nullptr, &embedding_config, &embedding) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(embedding.data == nullptr);
  assert(embedding.length == 0);
  embedding_config.text_data = text;
  embedding_config.text_size = std::numeric_limits<size_t>::max();
  assert(llama_dart_context_embed(nullptr, &embedding_config, &embedding) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(embedding.data == nullptr);
  assert(embedding.length == 0);
  embedding_config.text_data = text;
  embedding_config.text_size = sizeof(text);
  assert(llama_dart_context_embed(nullptr, &embedding_config, &embedding) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(embedding.data == nullptr);
  assert(embedding.length == 0);

  llama_dart_rerank_config rerank_config{};
  rerank_config.struct_size = sizeof(rerank_config);
  rerank_config.query_data = text;
  rerank_config.query_size = sizeof(text);
  rerank_config.document_data = text;
  rerank_config.document_size = sizeof(text);
  rerank_config.add_special = 1;
  rerank_config.parse_special = 1;
  float rerank_score = 0.0f;
  assert(llama_dart_context_rerank(nullptr, &rerank_config, &rerank_score) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  rerank_config.query_data = blank_text;
  rerank_config.query_size = sizeof(blank_text);
  assert(llama_dart_context_rerank(nullptr, &rerank_config, &rerank_score) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  rerank_config.query_data = text;
  rerank_config.query_size = sizeof(text);
  rerank_config.document_data = blank_text;
  rerank_config.document_size = sizeof(blank_text);
  assert(llama_dart_context_rerank(nullptr, &rerank_config, &rerank_score) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  rerank_config.document_data = text;
  rerank_config.document_size = sizeof(text);
  rerank_config.query_size = std::numeric_limits<size_t>::max();
  assert(llama_dart_context_rerank(nullptr, &rerank_config, &rerank_score) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  rerank_config.query_size = sizeof(text);

  llama_dart_lora_load_config lora_config{};
  lora_config.struct_size = sizeof(lora_config);
  lora_config.path_data = text;
  lora_config.path_size = sizeof(text);
  llama_dart_lora_adapter *adapter = nullptr;
  assert(llama_dart_lora_load(nullptr, &lora_config, &adapter) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(adapter == nullptr);
  lora_config.path_data = blank_text;
  lora_config.path_size = sizeof(blank_text);
  assert(llama_dart_lora_load(model, &lora_config, &adapter) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(adapter == nullptr);
  const uint8_t lora_path_with_line_break[] = {'b', 'a', 'd', '\n', 'l',
                                               'o', 'r', 'a'};
  lora_config.path_data = lora_path_with_line_break;
  lora_config.path_size = sizeof(lora_path_with_line_break);
  assert(llama_dart_lora_load(model, &lora_config, &adapter) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(adapter == nullptr);
  lora_config.path_data = text;
  lora_config.path_size = std::numeric_limits<size_t>::max();
  assert(llama_dart_lora_load(model, &lora_config, &adapter) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(adapter == nullptr);
  lora_config.path_size = sizeof(text);
  assert(llama_dart_context_set_lora_adapters(nullptr, nullptr, nullptr, 0) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);

  size_t token_count = 0;
  assert(llama_dart_model_tokenize(model, text, sizeof(text), nullptr, 0,
                                   &token_count, 0, 0) ==
         LLAMA_DART_ERROR_BUFFER_TOO_SMALL);
  assert(token_count > 0);
  const uint8_t multi_token_text[] = {'h', 'e', 'l', 'l', 'o', ' ', 'h',
                                      'e', 'l', 'l', 'o'};
  assert(llama_dart_model_tokenize(model, multi_token_text,
                                   sizeof(multi_token_text), nullptr, 0,
                                   &token_count, 0, 0) ==
         LLAMA_DART_ERROR_BUFFER_TOO_SMALL);
  assert(token_count > 1);
  int32_t too_small_tokens[] = {123};
  assert(llama_dart_model_tokenize(model, multi_token_text,
                                   sizeof(multi_token_text), too_small_tokens,
                                   1, &token_count, 0, 0) ==
         LLAMA_DART_ERROR_BUFFER_TOO_SMALL);
  assert(token_count > 1);
  assert(too_small_tokens[0] == 0);

  int32_t tokens[16]{};
  assert(llama_dart_model_tokenize(model, text, sizeof(text), tokens, 16,
                                   &token_count, 0, 0) == LLAMA_DART_SUCCESS);
  assert(token_count > 0);
  const size_t valid_token_count = token_count;
  tokens[0] = 123;
  assert(llama_dart_model_tokenize(model, text, sizeof(text), tokens, 16,
                                   &token_count, 2, 0) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(token_count == 0);
  assert(tokens[0] == 0);
  const uint8_t text_with_nul[] = {'b', 'a', 'd', '\0', 't', 'e', 'x', 't'};
  token_count = 123;
  tokens[0] = 123;
  assert(llama_dart_model_tokenize(model, text_with_nul, sizeof(text_with_nul),
                                   tokens, 16, &token_count, 0, 0) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(token_count == 0);
  assert(tokens[0] == 0);

  assert(llama_dart_model_tokenize(model, text, sizeof(text), tokens, 16,
                                   &token_count, 0, 0) == LLAMA_DART_SUCCESS);
  assert(token_count > 0);

  uint8_t decoded[32]{};
  size_t decoded_size = 0;
  assert(llama_dart_model_detokenize(model, tokens, valid_token_count, decoded,
                                     sizeof(decoded), &decoded_size, 0, 0) ==
         LLAMA_DART_SUCCESS);
  assert(decoded_size == sizeof(text));
  assert(std::memcmp(decoded, text, sizeof(text)) == 0);
  uint8_t too_small_decoded[] = {'x'};
  assert(llama_dart_model_detokenize(model, tokens, valid_token_count,
                                     too_small_decoded,
                                     sizeof(too_small_decoded), &decoded_size,
                                     0, 0) ==
         LLAMA_DART_ERROR_BUFFER_TOO_SMALL);
  assert(decoded_size > sizeof(too_small_decoded));
  assert(too_small_decoded[0] == 0);
  const int32_t negative_token[] = {-1};
  decoded_size = 123;
  decoded[0] = 'x';
  assert(llama_dart_model_detokenize(model, negative_token, 1, decoded,
                                     sizeof(decoded), &decoded_size, 0, 0) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(decoded_size == 0);
  assert(decoded[0] == 0);
  const int32_t too_large_token[] = {static_cast<int32_t>(info.n_vocab)};
  decoded_size = 123;
  decoded[0] = 'x';
  assert(llama_dart_model_detokenize(model, too_large_token, 1, decoded,
                                     sizeof(decoded), &decoded_size, 0, 0) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(decoded_size == 0);
  assert(decoded[0] == 0);

  llama_dart_context *context = nullptr;
  llama_dart_context_config context_config{};
  context_config.struct_size = sizeof(context_config);
  context_config.context_size = 128;
  context_config.batch_size = 16;
  context_config.ubatch_size = 16;
  context_config.threads = 1;
  context_config.batch_threads = 1;
  context_config.embeddings = 0;
  context_config.kv_cache_key_type = LLAMA_DART_KV_CACHE_F16;
  context_config.kv_cache_value_type = LLAMA_DART_KV_CACHE_F16;
  context_config.flash_attention = LLAMA_DART_FLASH_ATTENTION_AUTO;
  context_config.kv_cache_offload = 1;
  context_config.swa_full = 1;
  context_config.kv_unified = 0;
  llama_dart_context_config invalid_context_config = context_config;
  invalid_context_config.context_size = 0;
  assert(llama_dart_context_create(model, &invalid_context_config, &context) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(context == nullptr);
  invalid_context_config = context_config;
  invalid_context_config.batch_size = 0;
  assert(llama_dart_context_create(model, &invalid_context_config, &context) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(context == nullptr);
  invalid_context_config = context_config;
  invalid_context_config.ubatch_size = 0;
  assert(llama_dart_context_create(model, &invalid_context_config, &context) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(context == nullptr);
  invalid_context_config = context_config;
  invalid_context_config.ubatch_size = context_config.batch_size + 1;
  assert(llama_dart_context_create(model, &invalid_context_config, &context) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(context == nullptr);
  invalid_context_config = context_config;
  invalid_context_config.threads = -1;
  assert(llama_dart_context_create(model, &invalid_context_config, &context) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(context == nullptr);
  invalid_context_config = context_config;
  invalid_context_config.pooling_type = 999;
  assert(llama_dart_context_create(model, &invalid_context_config, &context) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(context == nullptr);
  invalid_context_config = context_config;
  invalid_context_config.attention_type = 999;
  assert(llama_dart_context_create(model, &invalid_context_config, &context) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(context == nullptr);
  invalid_context_config = context_config;
  invalid_context_config.kv_cache_key_type = 999;
  assert(llama_dart_context_create(model, &invalid_context_config, &context) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(context == nullptr);
  invalid_context_config = context_config;
  invalid_context_config.kv_cache_value_type = 999;
  assert(llama_dart_context_create(model, &invalid_context_config, &context) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(context == nullptr);
  invalid_context_config = context_config;
  invalid_context_config.flash_attention = 999;
  assert(llama_dart_context_create(model, &invalid_context_config, &context) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(context == nullptr);
  invalid_context_config = context_config;
  invalid_context_config.kv_cache_offload = 2;
  assert(llama_dart_context_create(model, &invalid_context_config, &context) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(context == nullptr);
  invalid_context_config = context_config;
  invalid_context_config.swa_full = 2;
  assert(llama_dart_context_create(model, &invalid_context_config, &context) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(context == nullptr);
  invalid_context_config = context_config;
  invalid_context_config.kv_unified = 2;
  assert(llama_dart_context_create(model, &invalid_context_config, &context) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(context == nullptr);
  invalid_context_config = context_config;
  invalid_context_config.kv_cache_value_type = LLAMA_DART_KV_CACHE_Q8_0;
  invalid_context_config.flash_attention = LLAMA_DART_FLASH_ATTENTION_DISABLED;
  assert(llama_dart_context_create(model, &invalid_context_config, &context) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(context == nullptr);
  invalid_context_config = context_config;
  invalid_context_config.batch_threads = -1;
  assert(llama_dart_context_create(model, &invalid_context_config, &context) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(context == nullptr);
  invalid_context_config = context_config;
  invalid_context_config.embeddings = 2;
  assert(llama_dart_context_create(model, &invalid_context_config, &context) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(context == nullptr);
  invalid_context_config = context_config;
  invalid_context_config.mmproj_use_gpu = 2;
  assert(llama_dart_context_create(model, &invalid_context_config, &context) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(context == nullptr);
  invalid_context_config = context_config;
  invalid_context_config.mmproj_use_gpu = 1;
  assert(llama_dart_context_create(model, &invalid_context_config, &context) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(context == nullptr);
  invalid_context_config = context_config;
  invalid_context_config.mmproj_path_data = blank_text;
  invalid_context_config.mmproj_path_size = sizeof(blank_text);
  assert(llama_dart_context_create(model, &invalid_context_config, &context) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(context == nullptr);
  invalid_context_config = context_config;
  invalid_context_config.mmproj_path_data = path_with_line_break;
  invalid_context_config.mmproj_path_size = sizeof(path_with_line_break);
  assert(llama_dart_context_create(model, &invalid_context_config, &context) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(context == nullptr);
  invalid_context_config = context_config;
  invalid_context_config.embeddings = 1;
  invalid_context_config.mmproj_path_data = text;
  invalid_context_config.mmproj_path_size = sizeof(text);
  assert(llama_dart_context_create(model, &invalid_context_config, &context) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(context == nullptr);
  invalid_context_config = context_config;
  invalid_context_config.speculative_ngram_n = 16;
  invalid_context_config.speculative_ngram_m = 8;
  invalid_context_config.speculative_type =
      LLAMA_DART_SPECULATIVE_NGRAM_SIMPLE;
  assert(llama_dart_context_create(model, &invalid_context_config, &context) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(context == nullptr);
  invalid_context_config = context_config;
  invalid_context_config.embeddings = 1;
  invalid_context_config.speculative_ngram_n = 8;
  invalid_context_config.speculative_ngram_m = 8;
  invalid_context_config.speculative_type =
      LLAMA_DART_SPECULATIVE_NGRAM_SIMPLE;
  assert(llama_dart_context_create(model, &invalid_context_config, &context) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(context == nullptr);
  for (const uint32_t ngram_type : {
           static_cast<uint32_t>(LLAMA_DART_SPECULATIVE_NGRAM_MAP_K),
           static_cast<uint32_t>(LLAMA_DART_SPECULATIVE_NGRAM_MAP_K4V)}) {
    invalid_context_config = context_config;
    invalid_context_config.speculative_ngram_n = 8;
    invalid_context_config.speculative_ngram_m = 16;
    invalid_context_config.speculative_type = ngram_type;
    assert(llama_dart_context_create(model, &invalid_context_config,
                                     &context) ==
           LLAMA_DART_ERROR_CONTEXT_CREATE);
    assert(context == nullptr);
  }
  invalid_context_config = context_config;
  invalid_context_config.speculative_ngram_n = 24;
  invalid_context_config.speculative_ngram_m = 64;
  invalid_context_config.speculative_ngram_min_draft = 48;
  invalid_context_config.speculative_type = LLAMA_DART_SPECULATIVE_NGRAM_MOD;
  assert(llama_dart_context_create(model, &invalid_context_config, &context) ==
         LLAMA_DART_ERROR_CONTEXT_CREATE);
  assert(context == nullptr);
  invalid_context_config.speculative_ngram_min_draft = 65;
  assert(llama_dart_context_create(model, &invalid_context_config, &context) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(context == nullptr);
  invalid_context_config = context_config;
  invalid_context_config.speculative_type =
      LLAMA_DART_SPECULATIVE_NGRAM_CACHE;
  assert(llama_dart_context_create(model, &invalid_context_config, &context) ==
         LLAMA_DART_ERROR_CONTEXT_CREATE);
  assert(context == nullptr);
  invalid_context_config.speculative_ngram_n = 1;
  assert(llama_dart_context_create(model, &invalid_context_config, &context) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(context == nullptr);
  invalid_context_config = context_config;
  invalid_context_config.speculative_type = 99;
  assert(llama_dart_context_create(model, &invalid_context_config, &context) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(context == nullptr);
  invalid_context_config = context_config;
  invalid_context_config.speculative_type =
      LLAMA_DART_SPECULATIVE_DRAFT_MODEL;
  invalid_context_config.speculative_model_path_data = text;
  invalid_context_config.speculative_model_path_size = sizeof(text);
  assert(llama_dart_context_create(model, &invalid_context_config, &context) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(context == nullptr);
  invalid_context_config.speculative_draft_max = 3;
  assert(llama_dart_context_create(model, &invalid_context_config, &context) ==
         LLAMA_DART_ERROR_CONTEXT_CREATE);
  assert(context == nullptr);
  invalid_context_config.context_size = 0x80000000u;
  assert(llama_dart_context_create(model, &invalid_context_config, &context) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(context == nullptr);
  invalid_context_config = context_config;
  invalid_context_config.speculative_type = LLAMA_DART_SPECULATIVE_MTP;
  invalid_context_config.speculative_draft_max = 1025;
  assert(llama_dart_context_create(model, &invalid_context_config, &context) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(context == nullptr);
  invalid_context_config = context_config;
  invalid_context_config.speculative_type = LLAMA_DART_SPECULATIVE_DFLASH;
  invalid_context_config.speculative_draft_max = 15;
  assert(llama_dart_context_create(model, &invalid_context_config, &context) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(context == nullptr);
  const llama_dart_result context_result =
      llama_dart_context_create(model, &context_config, &context);
  if (context_result == LLAMA_DART_SUCCESS) {
    assert(context != nullptr);
    llama_dart_context_info context_info{};
    context_info.struct_size = sizeof(context_info);
    assert(llama_dart_context_get_info(context, &context_info) ==
           LLAMA_DART_SUCCESS);
    assert(context_info.context_size > 0);
    assert(context_info.supports_vision == 0);
    assert(context_info.supports_audio == 0);
    assert(context_info.gpu_backend == LLAMA_DART_GPU_BACKEND_CPU);
    assert(context_info.used_tokens == 0);
    assert(context_info.kv_cache_key_type == LLAMA_DART_KV_CACHE_F16);
    assert(context_info.kv_cache_value_type == LLAMA_DART_KV_CACHE_F16);
    assert(context_info.flash_attention ==
           LLAMA_DART_FLASH_ATTENTION_AUTO);
    assert(context_info.kv_cache_offload == 0);
    assert(context_info.swa_full == 1);
    assert(context_info.kv_unified == 0);
    discarded_tokens = 123;
    assert(llama_dart_context_shift(context, 0, 0, &discarded_tokens) ==
           LLAMA_DART_ERROR_INVALID_ARGUMENT);
    assert(discarded_tokens == 0);
    assert(llama_dart_context_cancel(context) == LLAMA_DART_SUCCESS);
    assert(llama_dart_context_complete(context, &completion_config,
                                       &completion, nullptr) ==
           LLAMA_DART_ERROR_CANCELLED);
    completion_config.max_tokens = 0;
    assert(llama_dart_context_complete(context, &completion_config,
                                       &completion, nullptr) ==
           LLAMA_DART_ERROR_GENERATION);
    completion_config.max_tokens = 1;
    completion_config.temperature = std::numeric_limits<float>::quiet_NaN();
    assert(llama_dart_context_complete(context, &completion_config,
                                       &completion, nullptr) ==
           LLAMA_DART_ERROR_GENERATION);
    assert(llama_dart_generation_start(context, &completion_config,
                                       &generation) ==
           LLAMA_DART_ERROR_GENERATION);
    assert(generation == nullptr);
    completion_config.temperature = 0.0f;
    completion_config.frequency_penalty =
        std::numeric_limits<float>::infinity();
    assert(llama_dart_context_complete(context, &completion_config,
                                       &completion, nullptr) ==
           LLAMA_DART_ERROR_INVALID_ARGUMENT);
    completion_config.frequency_penalty = 0.0f;
    completion_config.mirostat = 3;
    assert(llama_dart_context_complete(context, &completion_config,
                                       &completion, nullptr) ==
           LLAMA_DART_ERROR_INVALID_ARGUMENT);
    completion_config.mirostat = 1;
    completion_config.mirostat_tau = 0.0f;
    completion_config.mirostat_eta = 0.1f;
    assert(llama_dart_context_complete(context, &completion_config,
                                       &completion, nullptr) ==
           LLAMA_DART_ERROR_INVALID_ARGUMENT);
    completion_config.mirostat = 0;
    assert(llama_dart_context_reset(context) == LLAMA_DART_SUCCESS);
    llama_dart_completion_config stop_config{};
    stop_config.struct_size = sizeof(stop_config);
    stop_config.prompt_data = text;
    stop_config.prompt_size = sizeof(text);
    stop_config.max_tokens = 1;
    stop_config.temperature = 0.0f;
    stop_config.top_p = 1.0f;
    stop_config.typical_p = 1.0f;
    stop_config.repeat_penalty = 1.0f;
    const int32_t negative_stop_token = -1;
    stop_config.stop_tokens = &negative_stop_token;
    stop_config.stop_token_count = 1;
    assert(llama_dart_context_complete(context, &stop_config, &completion,
                                       nullptr) ==
           LLAMA_DART_ERROR_INVALID_ARGUMENT);
    const int32_t oversized_stop_token = info.n_vocab;
    stop_config.stop_tokens = &oversized_stop_token;
    assert(llama_dart_context_complete(context, &stop_config, &completion,
                                       nullptr) ==
           LLAMA_DART_ERROR_INVALID_ARGUMENT);
    const int32_t duplicate_stop_tokens[] = {tokens[0], tokens[0]};
    stop_config.stop_tokens = duplicate_stop_tokens;
    stop_config.stop_token_count = 2;
    assert(llama_dart_context_complete(context, &stop_config, &completion,
                                       nullptr) ==
           LLAMA_DART_ERROR_INVALID_ARGUMENT);
    const llama_dart_result warm_up_result =
        llama_dart_context_warm_up(context);
    assert(warm_up_result == LLAMA_DART_SUCCESS ||
           warm_up_result == LLAMA_DART_ERROR_GENERATION);
    assert(llama_dart_context_reset(context) == LLAMA_DART_SUCCESS);
    assert(llama_dart_context_cancel(context) == LLAMA_DART_SUCCESS);
    assert(llama_dart_context_embed(context, &embedding_config, &embedding) ==
           LLAMA_DART_ERROR_CANCELLED);
    assert(embedding.data == nullptr);
    assert(embedding.length == 0);
    assert(llama_dart_context_cancel(context) == LLAMA_DART_SUCCESS);
    rerank_score = 123.0f;
    assert(llama_dart_context_rerank(context, &rerank_config,
                                     &rerank_score) ==
           LLAMA_DART_ERROR_CANCELLED);
    assert(rerank_score == 0.0f);
    const uint8_t invalid_grammar[] = {'r', 'o', 'o', 't', ' ', ':',
                                       ':', '=', ' ', '['};
    const uint8_t grammar_root[] = {'r', 'o', 'o', 't'};
    completion_config.prompt_data = text;
    completion_config.prompt_size = sizeof(text);
    completion_config.grammar_root_data = grammar_root;
    completion_config.grammar_root_size = sizeof(grammar_root);
    assert(llama_dart_context_complete(context, &completion_config,
                                       &completion, nullptr) ==
           LLAMA_DART_ERROR_INVALID_ARGUMENT);
    completion_config.grammar_root_data = nullptr;
    completion_config.grammar_root_size = 0;
    const uint8_t grammar_root_with_line_break[] = {'r', 'o', 'o',
                                                    't', '\n'};
    completion_config.prompt_data = text;
    completion_config.prompt_size = sizeof(text);
    completion_config.grammar_data = invalid_grammar;
    completion_config.grammar_size = sizeof(invalid_grammar);
    completion_config.grammar_root_data = grammar_root_with_line_break;
    completion_config.grammar_root_size = sizeof(grammar_root_with_line_break);
    assert(llama_dart_context_complete(context, &completion_config,
                                       &completion, nullptr) ==
           LLAMA_DART_ERROR_INVALID_ARGUMENT);
    assert(llama_dart_generation_start(context, &completion_config,
                                       &generation) ==
           LLAMA_DART_ERROR_INVALID_ARGUMENT);
    assert(generation == nullptr);
    completion_config.grammar_root_data = grammar_root;
    completion_config.grammar_root_size = sizeof(grammar_root);
    assert(llama_dart_context_complete(context, &completion_config,
                                       &completion, nullptr) ==
           LLAMA_DART_ERROR_GENERATION);
    completion_config.prompt_data = nullptr;
    completion_config.prompt_size = 0;
    completion_config.grammar_data = nullptr;
    completion_config.grammar_size = 0;
    completion_config.grammar_root_data = nullptr;
    completion_config.grammar_root_size = 0;
    assert(llama_dart_context_complete(context, &completion_config,
                                       &completion, nullptr) ==
           LLAMA_DART_ERROR_INVALID_ARGUMENT);
    completion_config.prompt_data = text;
    completion_config.prompt_size = sizeof(text);
    completion_config.grammar_data = invalid_grammar;
    completion_config.grammar_size = sizeof(invalid_grammar);
    completion_config.grammar_root_data = grammar_root;
    completion_config.grammar_root_size = sizeof(grammar_root);
    assert(llama_dart_generation_start(context, &completion_config,
                                       &generation) ==
           LLAMA_DART_ERROR_GENERATION);
    assert(generation == nullptr);
    completion_config.prompt_data = nullptr;
    completion_config.prompt_size = 0;
    completion_config.grammar_data = nullptr;
    completion_config.grammar_size = 0;
    completion_config.grammar_root_data = nullptr;
    completion_config.grammar_root_size = 0;
    assert(llama_dart_context_complete(context, &completion_config,
                                       &completion, nullptr) ==
           LLAMA_DART_ERROR_INVALID_ARGUMENT);
    completion_config.prompt_data = text;
    completion_config.prompt_size = sizeof(text);
    const llama_dart_result active_generation_result =
        llama_dart_generation_start(context, &completion_config, &generation);
    if (active_generation_result == LLAMA_DART_SUCCESS) {
      assert(generation != nullptr);
      assert(llama_dart_context_warm_up(context) ==
             LLAMA_DART_ERROR_INVALID_ARGUMENT);
      discarded_tokens = 123;
      assert(llama_dart_context_shift(context, 0, 0, &discarded_tokens) ==
             LLAMA_DART_ERROR_INVALID_ARGUMENT);
      assert(discarded_tokens == 0);
      llama_dart_buffer active_state{};
      assert(llama_dart_context_state_get(context, &active_state) ==
             LLAMA_DART_ERROR_INVALID_ARGUMENT);
      assert(active_state.data == nullptr);
      assert(llama_dart_context_embed(context, &embedding_config, &embedding) ==
             LLAMA_DART_ERROR_INVALID_ARGUMENT);
      assert(llama_dart_context_rerank(context, &rerank_config,
                                       &rerank_score) ==
             LLAMA_DART_ERROR_INVALID_ARGUMENT);
      assert(llama_dart_context_set_lora_adapters(context, nullptr, nullptr,
                                                  0) ==
             LLAMA_DART_ERROR_INVALID_ARGUMENT);
      llama_dart_clear_last_error();
      llama_dart_context_free(context);
      assert(std::strlen(llama_dart_last_error_message()) > 0);
      assert(llama_dart_context_get_info(context, &context_info) ==
             LLAMA_DART_SUCCESS);
      assert(llama_dart_context_cancel(context) == LLAMA_DART_SUCCESS);
      assert(llama_dart_context_reset(nullptr) ==
             LLAMA_DART_ERROR_INVALID_ARGUMENT);
      llama_dart_generation_free(generation);
      assert(std::strlen(llama_dart_last_error_message()) == 0);
      context_info = {};
      context_info.struct_size = sizeof(context_info);
      assert(llama_dart_context_get_info(context, &context_info) ==
             LLAMA_DART_SUCCESS);
      assert(context_info.used_tokens == 0);
      generation = nullptr;
      llama_dart_generation *next_generation = nullptr;
      const llama_dart_result next_generation_result =
          llama_dart_generation_start(context, &completion_config,
                                      &next_generation);
      assert(next_generation_result != LLAMA_DART_ERROR_CANCELLED);
      if (next_generation_result == LLAMA_DART_SUCCESS) {
        llama_dart_generation_free(next_generation);
      }
    }
    completion_config.prompt_data = nullptr;
    completion_config.prompt_size = 0;
    llama_dart_buffer state{};
    assert(llama_dart_context_state_get(context, &state) ==
           LLAMA_DART_SUCCESS);
    assert(state.data != nullptr);
    assert(state.size > 0);
    assert(llama_dart_context_state_set(context, state.data, state.size) ==
           LLAMA_DART_SUCCESS);
    assert(llama_dart_context_state_set(context, nullptr, 0) ==
           LLAMA_DART_ERROR_INVALID_ARGUMENT);
    llama_dart_buffer_free(state.data);
    llama_dart_model_free(model);
    assert(std::strlen(llama_dart_last_error_message()) > 0);
    assert(llama_dart_context_get_info(context, &context_info) ==
           LLAMA_DART_SUCCESS);
    assert(llama_dart_context_reset(nullptr) ==
           LLAMA_DART_ERROR_INVALID_ARGUMENT);
    llama_dart_context_free(context);
    assert(std::strlen(llama_dart_last_error_message()) == 0);
  } else {
    assert(context_result == LLAMA_DART_ERROR_CONTEXT_CREATE);
    assert(context == nullptr);
    assert(std::strlen(llama_dart_last_error_message()) > 0);
  }

  assert(llama_dart_context_create(nullptr, &context_config, &context) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(llama_dart_context_reset(nullptr) == LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(llama_dart_context_warm_up(nullptr) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  assert(llama_dart_context_cancel(nullptr) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);
  llama_dart_buffer state{};
  assert(llama_dart_context_state_get(nullptr, &state) ==
         LLAMA_DART_ERROR_INVALID_ARGUMENT);

  llama_dart_model_free(model);
  assert(std::strlen(llama_dart_last_error_message()) == 0);

  size_t captured_log_count = 0;
  while (true) {
    log_level = LLAMA_DART_LOG_DISABLED;
    log_message = {};
    assert(llama_dart_log_next(&log_level, &log_message) ==
           LLAMA_DART_SUCCESS);
    if (log_message.data == nullptr) {
      assert(log_message.size == 0);
      assert(log_level == LLAMA_DART_LOG_DISABLED);
      break;
    }
    assert(log_message.size > 0);
    assert(log_level >= LLAMA_DART_LOG_DEBUG);
    assert(log_level <= LLAMA_DART_LOG_ERROR);
    captured_log_count += 1;
    llama_dart_buffer_free(log_message.data);
  }
  assert(captured_log_count > 0);
  assert(llama_dart_log_set_level(LLAMA_DART_LOG_WARNING) ==
         LLAMA_DART_SUCCESS);
  assert(llama_dart_log_next(&log_level, &log_message) ==
         LLAMA_DART_SUCCESS);
  assert(log_message.data == nullptr);
  assert(llama_dart_log_set_level(LLAMA_DART_LOG_DISABLED) ==
         LLAMA_DART_SUCCESS);

  llama_dart_clear_last_error();
  assert(std::strlen(llama_dart_last_error_message()) == 0);

  return 0;
}

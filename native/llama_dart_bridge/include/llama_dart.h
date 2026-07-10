#ifndef LLAMA_DART_H_
#define LLAMA_DART_H_

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define LLAMA_DART_EXPORT __declspec(dllexport)
#else
#define LLAMA_DART_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define LLAMA_DART_ABI_VERSION 36u

typedef enum llama_dart_result {
  LLAMA_DART_SUCCESS = 0,
  LLAMA_DART_ERROR_INVALID_ARGUMENT = 1,
  LLAMA_DART_ERROR_UNSUPPORTED = 2,
  LLAMA_DART_ERROR_INTERNAL = 3,
  LLAMA_DART_ERROR_MODEL_LOAD = 4,
  LLAMA_DART_ERROR_BUFFER_TOO_SMALL = 5,
  LLAMA_DART_ERROR_CONTEXT_CREATE = 6,
  LLAMA_DART_ERROR_GENERATION = 7,
  LLAMA_DART_ERROR_EMBEDDING = 8,
  LLAMA_DART_ERROR_CANCELLED = 9,
  LLAMA_DART_ERROR_RERANKING = 10,
  LLAMA_DART_ERROR_LORA = 11,
} llama_dart_result;

typedef enum llama_dart_capability_flags {
  LLAMA_DART_CAP_MODEL_LOADING = 1ull << 7,
  LLAMA_DART_CAP_TOKENIZATION = 1ull << 8,
  LLAMA_DART_CAP_STRUCTURED_OUTPUT = 1ull << 9,
  LLAMA_DART_CAP_TEXT_GENERATION = 1ull << 0,
  LLAMA_DART_CAP_EMBEDDINGS = 1ull << 1,
  LLAMA_DART_CAP_RERANKING = 1ull << 2,
  LLAMA_DART_CAP_MULTIMODAL = 1ull << 3,
  LLAMA_DART_CAP_LORA = 1ull << 4,
  LLAMA_DART_CAP_SPECULATIVE_DECODING = 1ull << 5,
  LLAMA_DART_CAP_MTP = 1ull << 6,
  LLAMA_DART_CAP_METAL = 1ull << 10,
  LLAMA_DART_CAP_VULKAN = 1ull << 11,
  LLAMA_DART_CAP_TOOL_CALLING = 1ull << 12,
  LLAMA_DART_CAP_LOGGING = 1ull << 13,
  LLAMA_DART_CAP_PREFILL = 1ull << 14,
} llama_dart_capability_flags;

typedef struct llama_dart_capabilities {
  uint32_t struct_size;
  uint32_t abi_version;
  uint64_t flags;
} llama_dart_capabilities;

typedef struct llama_dart_model llama_dart_model;
typedef struct llama_dart_context llama_dart_context;
typedef struct llama_dart_generation llama_dart_generation;
typedef struct llama_dart_lora_adapter llama_dart_lora_adapter;

typedef enum llama_dart_gpu_backend {
  LLAMA_DART_GPU_BACKEND_AUTO = 0,
  LLAMA_DART_GPU_BACKEND_CPU = 1,
  LLAMA_DART_GPU_BACKEND_METAL = 2,
  LLAMA_DART_GPU_BACKEND_VULKAN = 3,
} llama_dart_gpu_backend;

typedef enum llama_dart_log_level {
  LLAMA_DART_LOG_DISABLED = 0,
  LLAMA_DART_LOG_DEBUG = 1,
  LLAMA_DART_LOG_INFO = 2,
  LLAMA_DART_LOG_WARNING = 3,
  LLAMA_DART_LOG_ERROR = 4,
} llama_dart_log_level;

typedef enum llama_dart_media_type {
  LLAMA_DART_MEDIA_IMAGE = 1,
  LLAMA_DART_MEDIA_AUDIO = 2,
} llama_dart_media_type;

typedef enum llama_dart_speculative_type {
  LLAMA_DART_SPECULATIVE_NONE = 0,
  LLAMA_DART_SPECULATIVE_NGRAM_SIMPLE = 1,
  LLAMA_DART_SPECULATIVE_DRAFT_MODEL = 2,
  LLAMA_DART_SPECULATIVE_EAGLE3 = 3,
  LLAMA_DART_SPECULATIVE_MTP = 4,
  LLAMA_DART_SPECULATIVE_NGRAM_MAP_K = 5,
  LLAMA_DART_SPECULATIVE_NGRAM_MAP_K4V = 6,
  LLAMA_DART_SPECULATIVE_DFLASH = 7,
  LLAMA_DART_SPECULATIVE_NGRAM_MOD = 8,
  LLAMA_DART_SPECULATIVE_NGRAM_CACHE = 9,
} llama_dart_speculative_type;

typedef enum llama_dart_add_special_mode {
  LLAMA_DART_ADD_SPECIAL_NEVER = 0,
  LLAMA_DART_ADD_SPECIAL_ALWAYS = 1,
  LLAMA_DART_ADD_SPECIAL_IF_CONTEXT_EMPTY = 2,
} llama_dart_add_special_mode;

typedef enum llama_dart_kv_cache_type {
  LLAMA_DART_KV_CACHE_DEFAULT = 0,
  LLAMA_DART_KV_CACHE_F32 = 1,
  LLAMA_DART_KV_CACHE_F16 = 2,
  LLAMA_DART_KV_CACHE_BF16 = 3,
  LLAMA_DART_KV_CACHE_Q8_0 = 4,
  LLAMA_DART_KV_CACHE_Q4_0 = 5,
  LLAMA_DART_KV_CACHE_Q4_1 = 6,
  LLAMA_DART_KV_CACHE_IQ4_NL = 7,
  LLAMA_DART_KV_CACHE_Q5_0 = 8,
  LLAMA_DART_KV_CACHE_Q5_1 = 9,
} llama_dart_kv_cache_type;

typedef enum llama_dart_flash_attention_mode {
  LLAMA_DART_FLASH_ATTENTION_AUTO = 0,
  LLAMA_DART_FLASH_ATTENTION_DISABLED = 1,
  LLAMA_DART_FLASH_ATTENTION_ENABLED = 2,
} llama_dart_flash_attention_mode;

typedef struct llama_dart_media_input {
  uint32_t struct_size;
  uint32_t type;
  const uint8_t *path_data;
  size_t path_size;
  const uint8_t *content_data;
  size_t content_size;
} llama_dart_media_input;

typedef struct llama_dart_model_load_config {
  uint32_t struct_size;
  const uint8_t *model_path_data;
  size_t model_path_size;
  int32_t n_gpu_layers;
  uint8_t vocab_only;
  uint8_t use_mmap;
  uint8_t use_mlock;
  uint8_t check_tensors;
  uint32_t gpu_backend;
} llama_dart_model_load_config;

typedef struct llama_dart_model_info {
  uint32_t struct_size;
  int32_t vocab_type;
  int32_t n_vocab;
  int32_t n_ctx_train;
  int32_t n_embd;
  int32_t n_embd_inp;
  int32_t n_embd_out;
  int32_t n_layer;
  int32_t n_layer_nextn;
  int32_t n_head;
  int32_t n_head_kv;
  int32_t ftype;
  uint64_t size_bytes;
  uint64_t n_params;
  int32_t token_bos;
  int32_t token_eos;
  int32_t token_eot;
  int32_t token_sep;
  int32_t token_nl;
  int32_t token_pad;
  int32_t token_mask;
  uint8_t has_encoder;
  uint8_t has_decoder;
  uint8_t is_recurrent;
  uint8_t is_hybrid;
  uint8_t is_diffusion;
  uint8_t add_bos;
  uint8_t add_eos;
  uint8_t add_sep;
} llama_dart_model_info;

typedef struct llama_dart_context_config {
  uint32_t struct_size;
  uint32_t context_size;
  uint32_t batch_size;
  uint32_t ubatch_size;
  int32_t threads;
  int32_t batch_threads;
  uint8_t embeddings;
  int32_t pooling_type;
  int32_t attention_type;
  uint32_t speculative_ngram_n;
  uint32_t speculative_ngram_m;
  const uint8_t *mmproj_path_data;
  size_t mmproj_path_size;
  uint8_t mmproj_use_gpu;
  uint32_t speculative_type;
  const uint8_t *speculative_model_path_data;
  size_t speculative_model_path_size;
  uint32_t speculative_draft_max;
  uint32_t kv_cache_key_type;
  uint32_t kv_cache_value_type;
  uint32_t flash_attention;
  uint8_t kv_cache_offload;
  uint8_t swa_full;
  uint8_t kv_unified;
  uint32_t speculative_ngram_min_draft;
} llama_dart_context_config;

typedef struct llama_dart_context_info {
  uint32_t struct_size;
  uint32_t context_size;
  uint32_t sequence_context_size;
  uint32_t batch_size;
  uint32_t ubatch_size;
  uint32_t max_sequences;
  uint8_t supports_vision;
  uint8_t supports_audio;
  uint32_t gpu_backend;
  uint8_t supports_context_shift;
  uint32_t used_tokens;
  uint32_t kv_cache_key_type;
  uint32_t kv_cache_value_type;
  uint32_t flash_attention;
  uint8_t kv_cache_offload;
  uint8_t swa_full;
  uint8_t kv_unified;
} llama_dart_context_info;

typedef struct llama_dart_chat_template_capabilities {
  uint32_t struct_size;
  uint8_t supports_tools;
  uint8_t supports_tool_calls;
  uint8_t supports_parallel_tool_calls;
} llama_dart_chat_template_capabilities;

typedef struct llama_dart_string_view {
  const uint8_t *data;
  size_t size;
} llama_dart_string_view;

typedef struct llama_dart_completion_config {
  uint32_t struct_size;
  const uint8_t *prompt_data;
  size_t prompt_size;
  /* Zero evaluates the prompt without sampling a token. */
  uint32_t max_tokens;
  float temperature;
  int32_t top_k;
  float top_p;
  float min_p;
  float typical_p;
  int32_t penalty_last_n;
  float repeat_penalty;
  float frequency_penalty;
  float presence_penalty;
  uint8_t mirostat;
  float mirostat_tau;
  float mirostat_eta;
  const uint8_t *grammar_data;
  size_t grammar_size;
  const uint8_t *grammar_root_data;
  size_t grammar_root_size;
  const llama_dart_string_view *stop_sequences;
  size_t stop_sequence_count;
  uint32_t seed;
  /* A llama_dart_add_special_mode value. */
  uint8_t add_special;
  uint8_t parse_special;
  const llama_dart_media_input *media_inputs;
  size_t media_input_count;
  const uint8_t *json_schema_data;
  size_t json_schema_size;
  const uint8_t *chat_plan_data;
  size_t chat_plan_size;
  const int32_t *stop_tokens;
  size_t stop_token_count;
} llama_dart_completion_config;

typedef struct llama_dart_completion_stats {
  uint32_t struct_size;
  uint32_t prompt_tokens;
  uint32_t generated_tokens;
  double prompt_eval_ms;
  double decode_ms;
  double total_ms;
  double time_to_first_token_ms;
  uint32_t speculative_draft_tokens;
  uint32_t speculative_accepted_tokens;
  double speculative_draft_ms;
  double speculative_verify_ms;
} llama_dart_completion_stats;

typedef struct llama_dart_buffer {
  uint8_t *data;
  size_t size;
} llama_dart_buffer;

typedef struct llama_dart_float_buffer {
  float *data;
  size_t length;
} llama_dart_float_buffer;

typedef struct llama_dart_embedding_config {
  uint32_t struct_size;
  const uint8_t *text_data;
  size_t text_size;
  uint8_t add_special;
  uint8_t parse_special;
} llama_dart_embedding_config;

typedef struct llama_dart_rerank_config {
  uint32_t struct_size;
  const uint8_t *query_data;
  size_t query_size;
  const uint8_t *document_data;
  size_t document_size;
  uint8_t add_special;
  uint8_t parse_special;
} llama_dart_rerank_config;

typedef struct llama_dart_lora_load_config {
  uint32_t struct_size;
  const uint8_t *path_data;
  size_t path_size;
} llama_dart_lora_load_config;

typedef struct llama_dart_chat_message {
  uint32_t struct_size;
  const uint8_t *role_data;
  size_t role_size;
  const uint8_t *content_data;
  size_t content_size;
} llama_dart_chat_message;

LLAMA_DART_EXPORT uint32_t llama_dart_abi_version(void);

LLAMA_DART_EXPORT const char *llama_dart_upstream_commit(void);

LLAMA_DART_EXPORT const char *llama_dart_build_flags(void);

LLAMA_DART_EXPORT const char *llama_dart_model_file_type_name(int32_t ftype);

LLAMA_DART_EXPORT const char *llama_dart_multimodal_marker(void);

/* Configures bounded in-memory native log capture. Disabled by default. */
LLAMA_DART_EXPORT llama_dart_result
llama_dart_log_set_level(uint32_t minimum_level);

/* Pops one captured log message. Empty data means the queue is drained. */
LLAMA_DART_EXPORT llama_dart_result llama_dart_log_next(
    uint32_t *out_level, llama_dart_buffer *out_message);

LLAMA_DART_EXPORT llama_dart_result llama_dart_backend_init(void);

LLAMA_DART_EXPORT llama_dart_result llama_dart_backend_free(void);

LLAMA_DART_EXPORT llama_dart_result
llama_dart_get_capabilities(llama_dart_capabilities *out_capabilities);

LLAMA_DART_EXPORT llama_dart_result llama_dart_model_load(
    const llama_dart_model_load_config *config, llama_dart_model **out_model);

/* Releases a model returned by llama_dart_model_load. */
LLAMA_DART_EXPORT void llama_dart_model_free(llama_dart_model *model);

LLAMA_DART_EXPORT llama_dart_result
llama_dart_model_get_info(const llama_dart_model *model,
                          llama_dart_model_info *out_info);

LLAMA_DART_EXPORT llama_dart_result llama_dart_model_get_description(
    const llama_dart_model *model, char *buffer, size_t buffer_size,
    size_t *out_size);

LLAMA_DART_EXPORT llama_dart_result llama_dart_model_metadata_count(
    const llama_dart_model *model, size_t *out_count);

LLAMA_DART_EXPORT llama_dart_result llama_dart_model_metadata_get(
    const llama_dart_model *model, size_t index, char *key_buffer,
    size_t key_buffer_size, size_t *out_key_size, char *value_buffer,
    size_t value_buffer_size, size_t *out_value_size);

LLAMA_DART_EXPORT llama_dart_result llama_dart_model_tokenize(
    const llama_dart_model *model, const uint8_t *text_data, size_t text_size,
    int32_t *tokens, size_t tokens_capacity, size_t *out_token_count,
    uint8_t add_special, uint8_t parse_special);

LLAMA_DART_EXPORT llama_dart_result llama_dart_model_detokenize(
    const llama_dart_model *model, const int32_t *tokens, size_t token_count,
    uint8_t *text_data, size_t text_capacity, size_t *out_text_size,
    uint8_t remove_special, uint8_t unparse_special);

LLAMA_DART_EXPORT llama_dart_result llama_dart_model_apply_chat_template(
    const llama_dart_model *model, const llama_dart_chat_message *messages,
    size_t message_count, uint8_t add_assistant_prompt,
    llama_dart_buffer *out_prompt);

LLAMA_DART_EXPORT llama_dart_result
llama_dart_model_get_chat_template_capabilities(
    const llama_dart_model *model,
    llama_dart_chat_template_capabilities *out_capabilities);

/*
 * Renders an OpenAI-shaped JSON chat request through the pinned upstream
 * Jinja template machinery. The result is an opaque JSON plan consumed by
 * completion and llama_dart_chat_parse_output.
 */
LLAMA_DART_EXPORT llama_dart_result llama_dart_model_create_chat_plan(
    const llama_dart_model *model, const uint8_t *request_data,
    size_t request_size, llama_dart_buffer *out_plan);

LLAMA_DART_EXPORT llama_dart_result llama_dart_chat_parse_output(
    const uint8_t *plan_data, size_t plan_size, const uint8_t *output_data,
    size_t output_size, llama_dart_buffer *out_message_json);

LLAMA_DART_EXPORT llama_dart_result llama_dart_json_schema_to_grammar(
    const uint8_t *schema_data, size_t schema_size,
    llama_dart_buffer *out_grammar);

/* Creates a context owned by the caller until llama_dart_context_free. */
LLAMA_DART_EXPORT llama_dart_result llama_dart_context_create(
    llama_dart_model *model, const llama_dart_context_config *config,
    llama_dart_context **out_context);

/* Releases a context returned by llama_dart_context_create. */
LLAMA_DART_EXPORT void llama_dart_context_free(llama_dart_context *context);

LLAMA_DART_EXPORT llama_dart_result
llama_dart_context_reset(llama_dart_context *context);

/* Warms an empty context and restores it to the empty state. */
LLAMA_DART_EXPORT llama_dart_result
llama_dart_context_warm_up(llama_dart_context *context);

LLAMA_DART_EXPORT llama_dart_result
llama_dart_context_cancel(llama_dart_context *context);

LLAMA_DART_EXPORT llama_dart_result llama_dart_context_get_info(
    const llama_dart_context *context, llama_dart_context_info *out_info);

LLAMA_DART_EXPORT llama_dart_result llama_dart_context_state_get(
    llama_dart_context *context, llama_dart_buffer *out_state);

LLAMA_DART_EXPORT llama_dart_result llama_dart_context_state_set(
    llama_dart_context *context, const uint8_t *state_data, size_t state_size);

/*
 * Removes tokens [keep_tokens, keep_tokens + discard_tokens) and shifts the
 * preserved tail left. A zero discard_tokens value discards half of the
 * removable suffix. At least one tail token is always retained.
 */
LLAMA_DART_EXPORT llama_dart_result llama_dart_context_shift(
    llama_dart_context *context, uint32_t keep_tokens,
    uint32_t discard_tokens, uint32_t *out_discarded_tokens);

LLAMA_DART_EXPORT llama_dart_result llama_dart_context_complete(
    llama_dart_context *context, const llama_dart_completion_config *config,
    llama_dart_buffer *out_text, llama_dart_completion_stats *out_stats);

LLAMA_DART_EXPORT llama_dart_result llama_dart_generation_start(
    llama_dart_context *context, const llama_dart_completion_config *config,
    llama_dart_generation **out_generation);

LLAMA_DART_EXPORT llama_dart_result llama_dart_generation_next(
    llama_dart_generation *generation, llama_dart_buffer *out_text,
    llama_dart_completion_stats *out_stats, uint8_t *out_done);

/* Releases a generation returned by llama_dart_generation_start. */
LLAMA_DART_EXPORT void
llama_dart_generation_free(llama_dart_generation *generation);

LLAMA_DART_EXPORT llama_dart_result llama_dart_context_embed(
    llama_dart_context *context, const llama_dart_embedding_config *config,
    llama_dart_float_buffer *out_embedding);

LLAMA_DART_EXPORT llama_dart_result llama_dart_context_rerank(
    llama_dart_context *context, const llama_dart_rerank_config *config,
    float *out_score);

LLAMA_DART_EXPORT llama_dart_result llama_dart_lora_load(
    llama_dart_model *model, const llama_dart_lora_load_config *config,
    llama_dart_lora_adapter **out_adapter);

/* Releases a LoRA adapter returned by llama_dart_lora_load. */
LLAMA_DART_EXPORT void llama_dart_lora_free(
    llama_dart_lora_adapter *adapter);

LLAMA_DART_EXPORT llama_dart_result llama_dart_context_set_lora_adapters(
    llama_dart_context *context, llama_dart_lora_adapter **adapters,
    const float *scales, size_t adapter_count);

/* Releases data returned in llama_dart_buffer fields. */
LLAMA_DART_EXPORT void llama_dart_buffer_free(uint8_t *data);

/* Releases data returned in llama_dart_float_buffer fields. */
LLAMA_DART_EXPORT void llama_dart_float_buffer_free(float *data);

/* Returns a borrowed thread-local UTF-8 message, valid until the next bridge
 * call on this thread. Messages are truncated to 4095 bytes. */
LLAMA_DART_EXPORT const char *llama_dart_last_error_message(void);

LLAMA_DART_EXPORT void llama_dart_clear_last_error(void);

#ifdef __cplusplus
}
#endif

#endif

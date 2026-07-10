# Upstream sync

- Upstream: `https://github.com/ggml-org/llama.cpp`
- Local snapshot: `third_party/llama.cpp`
- Commit: `f5525f7e7a7e7cbecd386144299493ea40499bd3`
- Observed tag description: `gguf-v0.19.0-864-gf5525f7e7`
- Sync date: 2026-07-08
- Notices: keep `third_party/llama.cpp/LICENSE`,
  `third_party/llama.cpp/AUTHORS`, and files under
  `third_party/llama.cpp/licenses/` with redistributed source or binaries.

The repository vendors a curated snapshot instead of a submodule so source
installs and pub.dev archives are self-contained. The snapshot contains the
CMake, bridge-facing runtime, CPU/Metal/Vulkan, multimodal, license, conversion,
and native-test fixture files used by this package. Unrelated upstream apps,
benches, documentation, server tools, and disabled native backends are omitted.
When syncing, export the selected paths from the exact commit, remove upstream
ignore files, then run the native, Dart, Flutter, and publish checks before
updating the commit and date above.

Current bridge integration uses these public upstream C APIs:

- `llama_backend_init`
- `llama_backend_free`
- `llama_model_default_params`
- `llama_model_load_from_file`
- `llama_model_free`
- `llama_model_get_vocab`
- `llama_model_chat_template`
- `llama_model_meta_count`
- `llama_model_meta_key_by_index`
- `llama_model_meta_val_str_by_index`
- `llama_vocab_type`
- `llama_vocab_n_tokens`
- `llama_vocab_bos`, `llama_vocab_eos`, `llama_vocab_eot`,
  `llama_vocab_sep`, `llama_vocab_nl`, `llama_vocab_pad`, `llama_vocab_mask`
- `llama_vocab_get_add_bos`, `llama_vocab_get_add_eos`,
  `llama_vocab_get_add_sep`
- `llama_model_*` metadata accessors used by `llama_dart_model_info`
- `llama_ftype_name`
- `llama_model_n_layer_nextn`
- `llama_tokenize`
- `llama_detokenize`
- `llama_chat_apply_template`
- `llama_context_default_params`
- `llama_init_from_model`
- `llama_free`
- `llama_n_ctx`, `llama_n_ctx_seq`, `llama_n_batch`, `llama_n_ubatch`,
  `llama_n_seq_max`
- `llama_state_get_size`
- `llama_state_get_data`
- `llama_state_set_data`
- `llama_memory_seq_pos_max`
- `llama_decode`
- `llama_batch_get_one`
- `llama_batch_init`
- `llama_batch_free`
- `llama_get_memory`
- `llama_memory_clear`
- `llama_memory_seq_rm`
- `llama_pooling_type`
- `llama_model_has_encoder`
- `llama_model_has_decoder`
- `llama_model_n_embd_out`
- `llama_get_embeddings_seq`
- `llama_sampler_chain_default_params`
- `llama_sampler_chain_init`
- `llama_sampler_chain_add`
- `llama_sampler_init_top_k`
- `llama_sampler_init_top_p`
- `llama_sampler_init_min_p`
- `llama_sampler_init_typical`
- `llama_sampler_init_penalties`
- `llama_sampler_init_mirostat`
- `llama_sampler_init_mirostat_v2`
- `llama_sampler_init_grammar`
- `llama_sampler_init_grammar_lazy_patterns`
- `llama_sampler_init_greedy`
- `llama_sampler_init_temp`
- `llama_sampler_init_dist`
- `llama_sampler_sample`
- `llama_sampler_accept`
- `llama_sampler_free`
- `llama_vocab_is_eog`
- `llama_token_to_piece`

`LlamaEngine.warmUp()` follows pinned `common_init_from_params` semantics: it
uses BOS/EOS tokens with token zero as a fallback, runs encoder and decoder
graphs where present, synchronizes, clears context memory, and resets
performance counters. The deprecated `llama_set_warmup` toggle is not used.

Context shifting uses pinned `llama_memory_can_shift`, `llama_memory_seq_rm`,
`llama_memory_seq_add`, and sequence-position inspection. ABI 32 reports
dynamic shift support/token use and exposes a transactional single-sequence
shift; no upstream cache internals cross the bridge.

The bridge also statically links pinned `llama-common` and normalizes
model-backed settings with `common_base_params_to_speculative` before calling
`common_init_speculative_from_params`. Generation then uses
`common_speculative_init`, `common_speculative_process`,
`common_speculative_draft`, `common_speculative_accept`, and the
context-removal capability probe. These C++ helpers remain private bridge
implementation details and are not exposed to Dart or through the public C
ABI.

`GenerationConfig.jsonSchema` uses pinned `json_schema_to_grammar` from
`llama-common`. The bridge parses ordered JSON, converts it to GBNF on the
inference worker isolate, and exposes a narrow C conversion function for native
smoke coverage.

Tool-aware chat uses pinned `common_chat_templates_init`,
`common_chat_templates_get_caps`, `common_chat_templates_apply`, and
`common_chat_parse`. The bridge serializes the resulting prompt, lazy grammar
triggers, generation prefix, parser, and additional stops through ABI 29; these
upstream C++ types remain private implementation details.

ABI 30 adds request-owned stop token IDs. The bridge validates each ID against
the loaded upstream vocabulary and terminates before converting or decoding the
sampled token.

ABI 34 defines a zero generated-token limit as prompt-only prefill and adds a
tokenization mode that inserts model special tokens only when the context is
empty. Both prefill and ordinary completion continue to use pinned
`llama_tokenize` and `llama_decode`; chat-template prompts enable trusted
special-token parsing.

ABI 35 maps typed Dart KV-cache controls to pinned `llama_context_params`:
`type_k`, `type_v`, `offload_kqv`, `flash_attn_type`, `swa_full`, and
`kv_unified`. The stable bridge enum is translated explicitly instead of
exposing `ggml_type` values. Model-backed speculative contexts receive the same
policy through `common_params` and `common_params_speculative_draft`.

ABI 36 adds stable strategy identifiers for pinned upstream `ngram-map-k`,
`ngram-map-k4v`, `ngram-mod`, request-local `ngram-cache`, and DFlash. All
exposed speculative modes now use `common_speculative`; the bridge no longer
carries a separate n-gram drafting implementation. N-gram cache files are not
exposed because the pinned upstream loader aborts on malformed external cache
data, which cannot be allowed across the bridge C ABI.

Note: pinned upstream documents `llama_state_get_size()` as a save-only sizing
helper. Do not use it to preflight `llama_state_set_data()` restores; it can be
too small for the serialized state being restored.

Bridge CMake enables the static upstream common library for speculative
decoding while disabling upstream tools, UI/prebuilt UI, OpenSSL integration,
external LLGuidance, native CPU tuning, OpenMP, LLAMAFILE, HBM, KleidiAI, BLAS,
Accelerate, and non-Apple accelerator backends for the current smoke-tested
baseline. `LLAMA_DART_NO_NETWORK=ON` also enables disconnected CMake fetches.
Apple targets build the pinned Metal backend with embedded kernels. Vulkan
remains an opt-in CMake variant until Android device validation is available.
Native-assets builds use `RelWithDebInfo`; the effective CMake build type and
feature flags are exposed in runtime/benchmark metadata.

Dart bindings are generated with:

```sh
dart run ffigen --config ffigen.yaml
```

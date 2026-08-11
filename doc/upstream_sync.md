# Upstream sync

- Upstream: `https://github.com/ggml-org/llama.cpp`
- Submodule: `third_party/llama.cpp`
- Commit: `ddd4ec1428a6201e18975ea52b07c71e0f9aef26`
- Upstream build tag: `b10217`
- Sync date: 2026-08-01
- Notices: keep `third_party/llama.cpp/LICENSE`,
  `third_party/llama.cpp/AUTHORS`, and files under
  `third_party/llama.cpp/licenses/`, plus the package-maintained notices under
  `third_party/licenses/` and the independently pinned
  `third_party/vulkan_headers/LICENSE.md`, with redistributed source or
  binaries.

Android Vulkan builds use the complete `include/` tree from
KhronosGroup/Vulkan-Headers tag `vulkan-sdk-1.4.357.0`. The upstream source
archive SHA-256 is
`e87dce08116151f6b6d7de6b6faf41498e87e6cf848ff16fa3bd5402190ad4a3`;
`third_party/vulkan_headers/README.fllamer.md` records the imported subset and
update contract. This pin is independent of the llama.cpp gitlink.

The parent repository's gitlink is the source of truth for the upstream pin.
Initialize a repository checkout with:

```sh
git submodule update --init --checkout
```

The build hook never initializes or fetches the submodule. Published archives
instead include the checked-out source files selected by the root `.pubignore`,
so builds from pub packages remain self-contained and require no Git or network
access. The publish checkout must initialize the submodule before running
`dart pub publish`.

No package-local changes are carried inside `third_party/llama.cpp`. Opt-in
Android Vulkan builds instead create a temporary version-3 build-output overlay.
Its pure transform normalizes CRLF to LF, accepts only recorded full-file hashes
from this exact pin, and applies five q-payload-only shader replacements while
preserving Q4_0/Q4_1 scale and min fields. It also transforms exactly one
`ggml-vulkan.cpp` source to install a Qualcomm-vendor plus
Qualcomm-proprietary-driver K-quant correctness policy: restricted Q4_K routing
and CPU scheduling fallback for Q5_K/Q6_K matrix operations. An upstream sync
must revalidate, update, or remove every external transform and its expected
output hashes, then bump the hook's build-output overlay version; never apply it
inside the submodule. Metal capability reporting and Gemma 4 generation grammar
therefore match the pinned upstream commit. fllamer still validates parsed tool
arguments against the declared schema before exposing them to application code.

To sync upstream, fetch and inspect an explicit commit, check it out detached
inside the submodule, and stage the updated gitlink:

```sh
git -C third_party/llama.cpp fetch --tags origin
git -C third_party/llama.cpp checkout --detach <commit>
git add third_party/llama.cpp
```

Do not configure a tracking branch or use `git submodule update --remote`.
After confirming changed public APIs and behavior, update the commit, build
tag, sync date, bridge CMake metadata, feature status, and CHANGELOG together.
Keep the submodule worktree clean, then run the native, Dart, Flutter, and
publish checks. Review the dry-run archive contents because `.pubignore`
filters a full upstream checkout and new upstream paths may otherwise increase
the package payload.

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
- `llama_vocab_get_suppress_tokens`
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
- `llama_sampler_init_logit_bias`
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
`common_speculative_init_from_params`. Generation then uses
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

ABI 37 adds an optional validated chat-template override to model loading and
an effective-template getter. Plain and tool-aware chat paths now reject a
missing template before upstream's generic ChatML fallback can apply.

ABI 38 adds a stable native generation stop-reason enum and reports it in
completion statistics. Dart terminal chunks distinguish end-of-generation,
stop sequences, stop tokens, and maximum-token exhaustion.

ABI 39 adds request-owned nullable `enable_thinking` input to native chat
planning. Explicit values are passed as typed Jinja-template inputs without
changing the model-owned effective chat template.

ABI 40 changes the serialized chat plan from one reasoning end tag to a
bounded `thinking_end_tags` list and appends explicit integrated-MTP load intent
to model configuration. The bridge maps its existing mmap/mlock booleans to
pinned `llama_load_mode` values exactly: neither, mmap, mlock, or mmap+mlock.
Only target models with an integrated MTP head request `load_mtp`; ordinary and
sidecar-backed model loads retain upstream's reduced-memory default.
When a quantized V cache is paired with Flash Attention `auto`, the bridge
records the upstream-promoted `enabled` mode so `contextInfo()` continues to
report the applied cache policy rather than the original request.

ABI 41 adds exact prompt-prefix reuse. A completion may reuse committed KV
state only when its committed token history is an exact prefix of the newly
tokenized full prompt; a mismatch clears the context before evaluating the
full prompt.

Bounded reasoning uses pinned `common_reasoning_budget_init` with vectors of
start and end token sequences, plus `common_reasoning_budget_get_state` and
`common_reasoning_budget_get_end_match`. All template-provided end alternatives
are retained. When a natural end closes reasoning, the exact matched sequence
is replayed into deferred lazy grammar so an alternate that begins a tool call
can activate its grammar trigger. The manual bridge sampler also reads
`llama_vocab_get_suppress_tokens` and applies those entries through
`llama_sampler_init_logit_bias` with negative-infinity bias, matching pinned
`llama-common` sampling behavior.

Note: pinned upstream documents `llama_state_get_size()` as a save-only sizing
helper. Do not use it to preflight `llama_state_set_data()` restores; it can be
too small for the serialized state being restored.

Bridge CMake enables the static upstream common library for speculative
decoding while disabling upstream tools, UI/prebuilt UI, OpenSSL integration,
external LLGuidance, native CPU tuning, OpenMP, LLAMAFILE, HBM, KleidiAI, BLAS,
and Accelerate. It enables only the target-selected Metal or Vulkan accelerator
backend; CPU-only variants disable both. `LLAMA_DART_NO_NETWORK=ON` also enables
disconnected CMake fetches. Apple targets build the pinned Metal backend with
embedded kernels. Vulkan is strict by default for native Linux and Windows
builds while the CPU backend remains available; an explicit consuming-workspace
override builds CPU-only. Android remains CPU-only by default and can opt into
Vulkan with package-pinned Vulkan-Headers 1.4.357.0 plus the Flutter-selected
NDK's target loader, SPIR-V headers, and host `glslc`. That opt-in build also
requires the
hash-gated build-output shader overlay and reports both
`GGML_VULKAN_ANDROID_SAFE_QUANT=1` and
`GGML_VULKAN_ANDROID_SAFE_K_QUANT=1` in bridge metadata; other builds report
`0` for both.
Cross-architecture desktop builds are rejected because the pinned Vulkan
shader-generator toolchain has no separate host-tool contract. Android native
Vulkan cross-compilation does not establish physical-device inference.
Native-assets builds use `RelWithDebInfo`; the effective CMake build type and
feature flags are exposed in runtime/benchmark metadata.

Dart bindings are generated with:

```sh
dart run ffigen --config ffigen.yaml
```

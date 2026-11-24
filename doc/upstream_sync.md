# Upstream sync

- Upstream: `https://github.com/ggml-org/llama.cpp`
- Submodule: `third_party/llama.cpp`
- Commit: `b1b770d90d94cad9e9ea5b993bdbc383a522fb28`
- Upstream release tag: `v1.0.0` (build number 12359)
- Sync date: 2026-11-23
- Notices: keep `third_party/llama.cpp/LICENSE`,
  `third_party/llama.cpp/AUTHORS`, and files under
  `third_party/llama.cpp/licenses/`, the vendored notices under
  `third_party/llama.cpp/vendor/cpp-httplib/` and
  `third_party/llama.cpp/vendor/hash/`, plus the package-maintained notices under
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
Android Vulkan builds instead create a temporary version-7 build-output overlay.
Its pure transform normalizes CRLF to LF, accepts only recorded full-file hashes
from this exact pin, and applies five q-payload-only shader replacements while
preserving Q4_0/Q4_1 scale and min fields. The matrix shader `mul_mm.comp`
gains byte views of the Q4_0, Q4_1 and Q8_0 blocks for those replacements,
because the pinned shader declares only packed views. It also transforms exactly one
`ggml-vulkan.cpp` source to install a Qualcomm-vendor plus
Qualcomm-proprietary-driver K-quant correctness policy: restricted Q4_K routing
and CPU scheduling fallback for Q5_K/Q6_K matrix operations. The pin
includes upstream commit `47f5a64ae12833b9092c6c47b54aa9f310c4f6f1` (#29766,
2026-10-15), which publishes a Vulkan device into `vk_instance.devices` only
after it is fully built; overlay version 5 carried a port of it for the
previous pin, and a test now asserts it on the pinned source. An upstream sync
must revalidate, update, or remove every external transform and its expected
output hashes, then bump the hook's build-output overlay version; never apply it
inside the submodule. Metal capability reporting and Gemma 4 generation grammar
therefore match the pinned upstream commit. fllamer still validates parsed tool
arguments against the declared schema before exposing them to application code.

The `b11371` sync ported that overlay to upstream's reorganized Vulkan backend
without changing its intent. The ported overlay cross-compiles for Android, but
it has not run on a device: the device results in
[native builds](native_builds.md) were measured on `b10217`. Until an affected
Qualcomm device repeats the CPU-oracle check, Android Vulkan correctness on
that pin was unverified in both directions: the patches may no longer be
needed, or may no longer be enough. Overlay version 5 ran on the arm64
Android 15 emulator through Kokage's native test harness on 2026-11-09: the
first model load fails on Vulkan and recovers on CPU, and a second load in
the same process completes and generates. That run exercises the device
creation path only; the emulator has no Vulkan device that ggml accepts.

The `v0.9.0` sync made the overlay version 6. `dequant_funcs.glsl` and
`mul_mm_funcs.glsl` are unchanged upstream, and the `mul_mm.comp` byte views
apply to its new text. Upstream rewrote how `ggml_vk_mul_mat_q_f16` chooses
the kernel of the second operand (`ggml_vk_mul_mat_b_type`), so the Q4_K
replacement there is new: for Q4_K weights on the Qualcomm proprietary driver
no Q4_K matrix kernel is looked up, the second operand is not quantized, and
its kernel type is chosen among the f16 kernels the dequantized weights use.
The other Qualcomm K-quant replacements apply unchanged. Version 6 has the
same standing as the port before it: it cross-compiles for Android and has
not run on a device ([native builds](native_builds.md)).

The `v1.0.0` sync made the overlay version 7. Every transform is kept and
none is replaced by an upstream change: between the pins upstream changed
neither the three shader files nor the quantized matrix routes the policy
guards (the Q4_K matrix-vector registrations, the kernel choice in
`ggml_vk_mul_mat_q_f16`, the matrix-vector dispatch and the `MUL_MAT`+`ADD`
fusion check). The `MUL_MAT`/`MUL_MAT_ID` support check changed in one line
after the place the policy is inserted: it now also refuses a BF16 precision
request for the second operand (#31851), which sends more operations to the
CPU and none to a quantized kernel. The three shader transforms and their
hashes are unchanged. In `ggml-vulkan.cpp` only the
place where `ggml_vk_is_qualcomm_proprietary` is inserted moved: upstream
added `ggml_vk_amd_large_tile_supported` (#31587) directly before
`ggml_vk_load_shaders`, so the predicate now follows that function. Upstream's
own Qualcomm change in this range (#30829) caps the rows of scalar flash
attention for head sizes above 128 and does not touch K-quant weights. Its
new prefill flash-attention kernel (#29357) is enabled on Intel Xe, or on
another device with 32-wide subgroups and 16x16x16 cooperative matrices when
`GGML_VK_FORCE_XE_FA` is set; it reads F16 keys and values and no weights.
As before, the overlay does not guard `MUL_MAT_ID` for Q4_K weights; that
route is the same at both pins. Version 7 has the same standing as version
6.

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

The `b11371` sync adapted the bridge to four upstream API changes without an
ABI change:

- `llama_sampler_init_penalties` takes the vocabulary size first. The bridge
  still adds the sampler only for a positive `penalty_last_n`.
- `mtmd_helper_bitmap_init_from_file` and `_from_buf` take init options; the
  bridge passes `mtmd_helper_init_opt_default()`.
- llama-common takes its own `common_json` type. The bridge still parses
  requests and schemas with nlohmann JSON, which keeps its parse-error
  contract, and converts at the three calls into llama-common.
- `common_speculative_process` takes a `common_batch`. The bridge still decodes
  the target with `llama_decode` and hands llama-common a mirror of the decoded
  entries, built only when a speculator exists.

The `v0.9.0` sync adapted the bridge to upstream's reworked chat API
(#30210, #30389, #29876):

- `common_peg_arena` can no longer be saved or loaded, so a chat plan cannot
  carry its output parser as text. ABI 44 hands the parser out as a handle
  instead, from the call that makes the plan. The plan (version 2) no longer
  has `parser` or `format`.
- `common_chat_params::prompt` is a list of parts that marks user input. The
  sync joined the parts into one string, so the whole prompt was still
  tokenized with special-token parsing, as before. ABI 45 uses the marks
  (below).
- `common_chat_parse` takes `common_chat_input`, text with the token each
  piece came from. The bridge passes text without tokens; the pinned parser
  carries the tokens and does not read them.
- `common_chat_parser_params` lost `parse_tool_calls` (tool calls are always
  parsed), and `common_chat_templates_get_caps` returns `common_json`.

Two behaviors changed with that pin. The pinned parser leaves out whitespace
between a reasoning block's text and its end marker, which it kept before
(Qwen 3.5: `weigh\n</think>` is the reasoning `weigh`). Upstream's session
format is version 11, so `llama_state_set_data` rejects state saved by an
earlier pin; the `b11371` pin had version 10.

The `v1.0.0` sync needed no bridge source change and no ABI change. Upstream
removed its deprecated public API (#30565): `llama_set_warmup`, 35 deprecated
functions such as `llama_load_model_from_file`, `llama_n_vocab`, the
`llama_token_*` accessors and `llama_sampler_init_grammar_lazy`, and
`llama_context_params::defrag_thold`. The bridge called none of them and
starts every context from `llama_context_default_params`, so the new
`moe_cache_cpu_share` field keeps its default. The NextN functions moved from
`src/llama-ext.h` to `llama.h`; `llama-common` uses them and the bridge does
not. `tools/mtmd/clip.h` is gone (#31865); the bridge includes only `mtmd.h`
and `mtmd-helper.h`. `LLAMA_SESSION_VERSION` is still 11 and
`LLAMA_STATE_SEQ_VERSION` still 4.

These upstream behaviors changed with the pin:

- The Gemma 4 parser and grammar follow each tool's parameter schema
  (#31915, `common/parsers/gemma4.cpp`) when its root is an object that
  upstream can convert; otherwise they keep the generic dictionary. Declared
  property names come in the template's order, sorted by name without
  regard to case, with the required ones present. Other keys are accepted
  only after the declared ones and only when the schema sets
  `additionalProperties`: upstream reads a schema that lists `properties`
  and leaves `additionalProperties` out as allowing none
  (`common/json-schema.cpp`), where fllamer's validator reads it as allowing
  any. A property declared as an array, a tuple or an object, or through
  `$ref`, `anyOf` or a one-member `allOf` of those, must have that shape,
  and an object's keys follow the same rules. The type of a string, number,
  boolean, null, enum or const value is not followed. A call outside these
  rules can not be generated under the grammar, and `common_chat_parse`
  rejects it in ended output, which the bridge reports as
  `LLAMA_DART_ERROR_GENERATION` with upstream's message. Before, any
  dictionary parsed and Dart's schema validation decided, naming the tool
  and the key. A partial parse while output streams still does not fail.
- The same change removed upstream's compatibility rewrite of tool responses
  for outdated Gemma 4 chat templates. No such template was run on this
  pin, so how one renders a tool result now is not known.
- `json_schema_to_grammar` no longer turns an `allOf` of plain types into an
  empty-object grammar (#31834).
- Tool-call arguments in the Python-like syntax (LFM2 and LFM2.5) escape raw
  control characters inside quoted strings (#31937), and tool ids and
  argument values replace invalid UTF-8 with U+FFFD (#31861).
- Context shifting rotates the kept keys with per-model RoPE parameters
  (#31683). Models that override them, LFM2 among them, shift to a different
  result than before.
- Arithmetic changed in several kernels (#31854, #29810 and #32128 on CPU,
  #30982 on Metal, #31851 for accumulation precision), so output for a fixed
  seed can differ between the pins ([native builds](native_builds.md)).
- `mtmd` computes audio log-mel in the graph (#31808) and refuses a Qwen-VL
  projector whose spatial merge is not 2x2 or that lacks its projector or
  patch-embedding tensors (#32129, #32130).
- With an integrated MTP head, `common_speculative` keeps the target's hidden
  rows in a device buffer when the two contexts allow it. It reads row
  `k - 1` of the last target decode there, asserts that the batch has at
  most `llama_n_batch` entries, and reads the buffer again in
  `common_speculative_accept`, so no target decode may come between the
  verifying decode and that call. The bridge hands it the entries of one
  `llama_decode` call directly after that call, with every row output and at
  most `llama_n_batch` entries, and accepts before it decodes a replacement
  token. No MTP model has run on this pin.

`LlamaEngine.warmUp()` follows pinned `common_init_from_params` semantics: it
uses BOS/EOS tokens with token zero as a fallback, runs encoder and decoder
graphs where present, synchronizes, clears context memory, and resets
performance counters. Upstream removed its `llama_set_warmup` toggle in
`v1.0.0`; the bridge never used it.

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
triggers, generation prefix, and additional stops through ABI 29; these
upstream C++ types remain private implementation details. Until ABI 44 the
plan also carried the parser.

ABI 30 adds request-owned stop token IDs. The bridge validates each ID against
the loaded upstream vocabulary and terminates before converting or decoding the
sampled token.

ABI 34 defines a zero generated-token limit as prompt-only prefill and adds a
tokenization mode that inserts model special tokens only when the context is
empty. Both prefill and ordinary completion continue to use pinned
`llama_tokenize` and `llama_decode`; chat-template prompts enable trusted
special-token parsing, which ABI 45 limits to the template's own text in a
planned chat.

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
full prompt. The prompt boundary (see [Architecture](architecture.md)) builds
on pinned `llama_state_seq_get_data_ext`/`llama_state_seq_set_data_ext` with
`LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY` and `llama_memory_seq_rm`, the calls
upstream's server uses for its context checkpoints; it adds no ABI.

ABI 43 adds the chat stream parser: `llama_dart_chat_stream_parser_create`
loads a chat plan's parser once, `llama_dart_chat_stream_parser_parse` runs
pinned `common_chat_parse` over all output generated so far, with
`is_partial` for output that stops anywhere, and returns the response text,
the reasoning and the size of the tool calls parsed so far. The Dart worker
calls it once per streamed chunk and publishes what each call adds. A partial
result never ends inside a UTF-8 sequence, and output may contain NUL bytes.
A chat request's `"separate_reasoning": true` selects reasoning-aware parsing
without setting `enable_thinking`, which keeps upstream's default `true`.
`llama_dart_chat_template_capabilities.marks_reasoning` reports whether the
template, rendered with thinking on, exposes a usable reasoning start marker
and end marker, and uses the start marker in its source or in the generation
prompt it renders. Upstream's LFM2, Kimi K2 and Ministral parsers name markers
for templates that have none.

ABI 44 takes the output parser out of the plan, because the pinned parser
cannot be serialized. `llama_dart_model_create_chat_plan` gains
`out_parser`: when it is not null the call also returns the chat stream
parser of the plan's output, built from the same template application, so
the template is applied once per chat as before. The parser keeps nothing of
the model. `llama_dart_chat_stream_parser_parse_message` reads ended output
into the terminal assistant message with that parser; it replaces
`llama_dart_chat_parse_output`, and `llama_dart_chat_stream_parser_create`
is gone. In Dart the plan of a chat that is generated owns the parser and
the streaming generation closes it; plans made only to format a prompt or
count tokens ask for none.

The bridge reads chat output through one function for the stream and for the
terminal message. Three of its steps depend on upstream details that a sync
must recheck:

- Upstream's `until` parser cuts text before a piece that could begin one of
  its delimiters while output streams (`common/peg-parser.cpp`). Since
  `v0.9.0` it keeps the piece as text when the output has ended
  (`COMMON_PEG_PARSE_FLAG_COMPLETE`), but a literal parser still waits for
  the rest of its marker then, because `common_chat_peg_parse` always parses
  leniently: without the next step the native test for Gemma 4 with tools
  fails on an ended `<`, which must be response text. What follows in a
  template's grammar may then read that piece as something else. Ling 3 and
  Kimi K3 read a possible end of reasoning as response text, and DeepSeek
  V3.2 and V4 read a reasoning block that ends in such a piece as response
  text, which a later parse takes back. The bridge therefore removes such an
  end from the output before a partial parse. It collects the markers from
  `common_peg_until_parser::delimiters` of every parser in the request's
  arena, and adds the template's reasoning start marker, because a parser that begins
  with an optional reasoning block waits for the rest of that literal in the
  same way.
- For output that has ended, the bridge parses the output followed by one
  character that occurs in none of those markers. When that adds the removed
  piece, or the end of it, and the character to the response text or to the
  reasoning, relative to the parse without the piece, the piece is text
  there and the character is taken off; otherwise the output is parsed as it
  stands. This relies on `common_chat_parse` failing only for output the
  grammar rejects, with or without `is_partial`.
- Upstream prepends the plan's generation prompt to the output before
  parsing. The bridge does that itself and leaves
  `common_chat_parser_params::generation_prompt` empty, so that for a plan
  with thinking off whose generation prompt holds a closed reasoning block,
  output that begins with the template's start marker is parsed against the
  generation prompt cut before that block.

A partial parse can still report text that a later parse takes back when the
output does not follow the template's format at all, for example plain text
for the Cohere Command R7B template, which requires its response marker, or a
tool call the template's parser rejects. Upstream then returns whatever it
parsed before the mismatch. The Dart stream keeps what it delivered, delivers
nothing that is taken back, and fails with "The chat parser changed output
that was already streamed" when a later parse reads different text in its
place.

ABI 45 keeps message text from becoming control tokens in a planned chat.
The plan (version 3) adds `message_text_spans`: the byte ranges of its joined
prompt that pinned `common_chat_prompt` marks `is_input`, without the parts
that are exactly one of the four role names.
`llama_dart_model_tokenize_chat_plan` tokenizes a plan's prompt as a
completion of the plan evaluates it; Dart counts a planned chat with it.
[Architecture](architecture.md#message-text-in-a-chat-prompt) describes the
rule. It rests on upstream details that a sync must recheck:

- Upstream has no consumer of `is_input` to follow. At `v1.0.0`
  `tools/server` still joins the parts
  (`out_session.prompt().str()` in `server-common.cpp`) and parses special
  tokens over the whole prompt. When upstream tokenizes the parts itself,
  compare its rule with this one. `common/jinja/README.md` names per-part
  tokenization as the intent and lists its costs: a space the template adds
  before message text becomes a token of its own, and a token built from
  request text stops working. The bridge avoids the first for prompts whose
  messages spell no control token, by leaving those prompts joined, and the
  second for role names.
- `global_from_json(ctx, inp, inputs.mark_input)` in `common/chat.cpp` marks
  every string of the request; `mark_input` defaults to true in
  `autoparser::generation_params` and the bridge cannot set it per field.
  A filter keeps the mark only where upstream's rule for it says so (a result
  made of several strings is marked when all of them are). Text a template
  derives from a message in a way that drops the mark is template text to
  the bridge.
- The control tokens are those `tokenizer_st_partition` in
  `src/llama-vocab.cpp` skips when `parse_special` is false:
  `LLAMA_TOKEN_ATTR_CONTROL` and `LLAMA_TOKEN_ATTR_UNKNOWN`. It matches a
  token's vocabulary text exactly, which is what the bridge searches for.
- Runs that are tokenized apart get BOS and EOS as `mtmd_tokenizer::tokenize`
  adds them: by `llama_vocab_get_add_bos` and `llama_vocab_get_add_eos`.
- `mtmd_tokenize_from_parts` reads `parse_special` per text part and ignores
  per-part `add_special`. The bridge cuts the prompt at `mtmd_get_marker`
  the way `mtmd_tokenizer::split_text` does.
- The legacy `llama_chat_apply_template` path has no parts at all.

ABI 42 appends `maximum_token_piece_bytes` to `llama_dart_model_info`.
Model loading scans each vocabulary ID once with pinned
`llama_token_to_piece(vocab, token, buffer, 0, 0, true)`. A negative return is
its exact required byte count; conversion uses signed 64-bit arithmetic before
negation. This uses the same zero stripping and special-token rendering as
`append_token_piece`, without allocating a buffer per token. The result stays
on the model handle and is copied by later metadata reads. Custom ABI 41
bridges require a rebuild; Dart checks ABI compatibility before struct access.

Bounded reasoning uses pinned `common_reasoning_budget_init` with vectors of
start and end token sequences, plus `common_reasoning_budget_get_state` and
`common_reasoning_budget_get_end_match`. All template-provided end alternatives
are retained. When a natural end closes reasoning, the exact matched sequence
is replayed into deferred lazy grammar so an alternate that begins a tool call
can activate its grammar trigger. The manual bridge sampler also reads
`llama_vocab_get_suppress_tokens` and applies those entries through
`llama_sampler_init_logit_bias` with negative-infinity bias, matching pinned
`llama-common` sampling behavior.

An explicit `enable_thinking=false` plan with paired template reasoning
markers uses the same sampler with a zero budget. This enforces Off if a model
unexpectedly emits a reasoning start while preserving later public output.
Markerless templates remain on their existing template-only path, and finite
reasoning budgets are unchanged.

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

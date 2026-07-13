## 0.2.3

- Forced automatic GPU selection to an explicit CPU device with zero offloaded
  layers on Apple Simulator targets. Explicit Metal requests remain unchanged
  for diagnostics, while context, KV, speculative, and multimodal projector
  setup now inherit the resolved CPU backend.

## 0.2.2

- Derived Gemma 4 tool-argument grammar from each declared parameter schema
  instead of accepting arbitrary dictionary keys and value types. Terminal
  tool calls are also validated against their matching schema before exposure,
  so missing, extra, mistyped, or otherwise invalid arguments fail with a
  typed generation error.
- Added `GenerationStopReason` to terminal chunks and generation telemetry,
  distinguishing end-of-generation, stop sequences, stop tokens, and maximum
  token exhaustion. Truncated chat/tool output now reports the token limit
  explicitly instead of only surfacing the downstream PEG parse failure.
- Bumped the bridge ABI to 38 for native stop-reason telemetry and regenerated
  both committed FFI binding variants. Custom native bridges must be rebuilt.

## 0.2.1

- Added a legacy-first Jinja chat-template fallback for ordinary chat, so
  models such as Gemma 4 can use their embedded template without changing the
  established legacy path for other models. The fallback keeps custom GBNF,
  JSON mode, JSON Schema, template stops, and media ordering intact, and plain
  fallback output is not forced through the strict tool-output parser.
- Corrected the pinned Metal backend's `MUL_MAT`, `MUL_MAT_ID`, and `GET_ROWS`
  capability checks to reject source types without compiled kernels, including
  TQ1_0 and TQ2_0, before scheduling can reach a missing pipeline.
- Added a multimodal preflight for non-causal media chunks whose physical
  decode batch would exceed `ubatchSize`, returning a typed generation error
  instead of reaching an upstream assertion.

## 0.2.0

- Fixed default bundled-bridge lookup by resolving the native code-asset ID,
  including Flutter's Apple framework packaging, while preserving explicit
  library-path and environment overrides.
- Rejected concurrent engine streams and context work instead of placing them
  in an unbounded deferred queue. Stream cancellation is now stream-scoped,
  awaits worker cleanup, resets partial context state, and cannot abort a
  different request; concurrent `close()` calls share one cleanup future.
- Bumped the bridge ABI to 37. Chat formatting now requires a valid embedded or
  explicit `LlamaModelConfig.chatTemplate`, exposes the effective template and
  loaded-model metadata, and never silently substitutes ChatML. Freeing an
  incomplete native generation resets its context.
- Raised generic Flutter iOS native-asset hook inputs to the bridge's real iOS
  15 minimum, capped default CMake parallelism at four jobs with an explicit
  `FLLAMER_BUILD_JOBS` override, and streamed build output.
- Registered vendored llama.cpp, nlohmann/json, cpp-httplib, stb, and miniaudio
  licenses/authorship as Flutter additional notices.

Migration notes:

- Rebuild custom native bridges for ABI 37; ABI 36 libraries are rejected.
- Supply `LlamaModelConfig.chatTemplate` for models without a valid embedded
  chat template.
- Serialize work per engine. Concurrent generations and context operations now
  fail immediately instead of entering a deferred queue.
- Treat stream cancellation as clearing the engine context; prefill or resend
  the intended history before continuing generation.

## 0.1.1

- Updated `code_assets` to 1.2.1, `hooks` to 2.0.2, and `ffigen` to
  20.1.1, then regenerated the internal Dart FFI bindings.
- Raised the example app to Flutter 3.44 and Dart 3.12, migrated Android to
  AGP 9.0.1, Kotlin 2.3.20, and a checksum-verified Gradle 9.1 wrapper, and
  replaced its CocoaPods-only plugin wiring with Swift Package Manager.
- Updated the curated `llama.cpp` snapshot to upstream build `b9967`, including
  tokenizer input hardening, Q2_0 support, and CPU/Metal/multimodal fixes, and
  adapted the private bridge integration to upstream's renamed speculative
  initialization helper.

## 0.1.0

- Prepared the package for warning-free pub.dev validation: vendored the
  curated pinned `llama.cpp` build sources, consolidated Dart CLIs under the
  conventional `example/` directory, and corrected published documentation
  paths and licensing metadata.
- Added loaded-engine `tokenize`, `countTokens`, `detokenize`,
  `formatChat`, `countChatTokens`, and chat-template capability methods. They
  reuse the engine worker's existing model and serialize with inference;
  static chat token counting now formats and tokenizes with one temporary
  vocab-only model load instead of two.
- Added `LlamaModel.architecture()` for typed access to the GGUF
  `general.architecture` metadata on the existing worker-isolate path.
- Enabled strict-casts, strict-inference, and strict-raw-types analysis for the
  package and Flutter example, and documented installation/build prerequisites.
- Changed native-assets builds from `Release` to `RelWithDebInfo` so the
  optimized bundled bridge retains debug information for local symbolication;
  runtime build metadata now records the CMake build type.
- Raised and enforced the minimum iOS deployment target to 15.0 because the
  pinned embedded Metal scheduler uses an event API introduced in iOS 15.
- Added a default-on native `LLAMA_DART_NO_NETWORK` build mode that disables
  CMake fetches and upstream's optional external LLGuidance project.
- Corrected the embedding/reranking `mmprojPath` feature-gate message and
  documented speculative-decoding restrictions for media-bearing requests.
- Documented Android `content://` handling and the required copy/validation
  step from temporary picker content into durable app-private model storage.
- Replaced nested `List<Float32List>` batch embedding results with row-major
  `EmbeddingBatch` typed storage and zero-copy vector views; vector indexes now
  accept iterable vectors so a flat batch can be indexed directly. Batches
  retain normalization and pooling metadata from their request config.
- Added a runnable app-owned-GGUF batch embedding CLI using `EmbeddingBatch`.
- Made benchmark/chat/RAG CLI error paths directly testable and removed all
  nested Dart process launches from the test suite.
- Added pinned upstream `ngram-map-k` and `ngram-map-k4v` speculative decoding
  alongside `ngram-simple`, removed the bridge's duplicate n-gram algorithm,
  and added synthetic upstream-draft plus weighted output/state tests. The
  benchmark now accepts `--spec-ngram STRATEGY`.
- Added typed pinned-upstream `NGramModSpeculation` and request-local
  `NGramCacheSpeculation`, including ABI validation, reset/restore/shift state
  recreation, synthetic and weighted equivalence coverage, and benchmark
  selection/tuning metadata. Cache files remain intentionally unsupported so
  malformed external data cannot trigger upstream abort paths across the C ABI.
- Added typed, upstream-dependent `DFlashSpeculation` configuration through
  pinned `common_speculative`, with stable ABI and validation coverage. It is
  documented as unverified until a compatible checksum-pinned fixture exists.
- Added typed K/V cache data types, KV offload, Flash Attention, full-size SWA,
  and unified-cache tuning across Dart, the stable bridge, target and
  draft/MTP contexts, and context inspection. Quantized V caches are rejected
  when Flash Attention is explicitly disabled.
- Aligned external draft-model and EAGLE-3 context initialization with pinned
  upstream `common_base_params_to_speculative`, including its required output
  capacity normalization and the target model's selected backend devices
  (including CPU-only mode). Target output masks are now strategy-aware, while
  batches forwarded to draft contexts request only their final output; this
  avoids upstream output-capacity assertions without a vocabulary-sized draft
  buffer. Oversized model-backed common parameters are rejected before
  narrowing to `int32`, and model-load failures are distinguished from
  context-creation failures.
- Fixed `GpuConfig.cpu()` to restrict upstream model devices to CPU and disable
  context operation/KV offload, preventing accelerator initialization after
  model-layer offload was disabled.
- Kept native lifecycle assertions active in Release CTest builds and verified
  the bridge with an ASan/UBSan Debug build.
- Made synthetic C bridge compilation errors fail Dart tests with bounded
  compiler diagnostics; only a genuinely unavailable compiler now skips those
  probes.
- Made native last-error recording non-throwing so allocation failure while
  reporting another failure cannot unwind a C++ exception across the C ABI;
  Dart now bounds and lossily decodes malformed native error bytes instead of
  throwing a secondary UTF-8 `FormatException`.
- Mapped native token-batch and multimodal-chunk allocation failures to the
  existing typed `NativeOutOfMemoryException` path instead of a generic bridge
  exception.
- Removed dynamic string allocation from native validation/catch-path error
  composition and added complete allocation/exception guards around JSON and
  chat-plan parsing helpers.
- Removed pre-catch allocation from completion stop-token/chat-role validation
  and guarded media-path size inspection so malformed requests cannot unwind
  through the C ABI under allocation or filesystem errors.
- Made arbitrary detokenized/generated bytes use replacement decoding, while
  invalid UTF-8 model metadata/chat templates and malformed native tool-parser
  JSON now surface typed model/bridge/generation exceptions.
- Fixed model loading to stop when process-level backend initialization fails,
  and preserved backend-release failures instead of clearing them on free;
  Dart invalidates a freed model pointer before surfacing cleanup errors.
- Added exact tokenizer-backed RAG context budgeting that includes headers and
  source labels, plus full chat-prompt budgeting through an app-supplied
  model-template token counter.
- Moved full model-file SHA-256 work and vector-index JSON persistence/loading
  off the caller isolate.
- Added opt-in checksum-pinned Apache-2.0 MiniLM GGUF integration coverage for
  real model inspection, normalized and raw single embeddings, same-context
  batch embeddings, deterministic output, and semantic ranking.
- Added opt-in checksum-pinned MIT-derived TinyLlama Q8/Q4 integration coverage
  for real draft-model speculative output equivalence, proposal/acceptance
  telemetry, and deterministic engine cleanup.
- Added opt-in checksum-pinned Apache-2.0 Qwen3.5 0.8B MTP integration coverage
  for lightweight NextN inspection, deterministic ordinary/MTP output
  equivalence, proposal/acceptance telemetry, and engine cleanup.
- Added opt-in checksum-pinned Apache-2.0 Qwen3 1.7B EAGLE-3 integration
  coverage using the pinned upstream converter, with deterministic
  ordinary/EAGLE output equivalence, proposal/acceptance telemetry, and engine
  cleanup.
- Replaced the Flutter capability-label stub with a mobile-first local text and
  image chat workspace: platform file selection, explicit model lifecycle,
  capability gates, cancellable streaming, context reset, compact-phone layout
  coverage, and final generation telemetry.
- Restricted shared-library exports to the versioned `llama_dart_*` C ABI on
  Apple and ELF targets, hiding statically linked llama.cpp/ggml/common/mtmd
  symbols and adding a host symbol-table regression test.
- Extended the benchmark harness with native warm-up, context-reset sustained
  iterations, per-run RSS/output/telemetry records, and aggregate throughput,
  latency, and speculative-acceptance metrics.
- Extended benchmark schema 3 with typed KV-cache/Flash-Attention CLI controls
  and requested-versus-applied cache metadata.
- Added explicit benchmark device-model metadata through `--device-model`, with
  a visible `unspecified` fallback instead of guessing from the hostname.
- Added opt-in checksum-pinned Apache-2.0 Jina tiny reranker GGUF coverage for
  real pair scoring, same-context batches, deterministic semantic ordering,
  and `LlamaReranker` candidate reordering.
- Added opt-in checksum-pinned MIT base/LoRA GGUF coverage for real adapter
  loading, global scale changes, per-request disable and restoration, unload,
  and deterministic baseline recovery, plus focused lifecycle/performance docs.
- Added opt-in checksum-pinned TinyGemma3 model/mmproj/image coverage for real
  `mtmd` vision capability reporting, deterministic file/byte image chat,
  prompt telemetry, and native cleanup.
- Added opt-in checksum-pinned official Apache-2.0 Qwen3 ASR model/mmproj/audio
  coverage for real `mtmd` audio capability reporting, deterministic file/byte
  transcription, telemetry, and cleanup, plus a runnable audio CLI example.
- Added opt-in checksum-pinned official Apache-2.0 Qwen2.5 0.5B coverage for
  weighted named tool-call generation, native typed parsing, generated call
  ids, correlated result history, and final assistant response generation.
- Fixed sampler state advancing twice for generated tokens and existing context
  history, which could reject valid grammar-constrained output and over-apply
  repetition, frequency, and presence penalties.
- Fixed lightweight vocab-only model inspection to report standard GGUF
  architecture scalars, including context, embedding, layer/head, and NextN
  metadata, without allocating model weights.
- Bounded Dart allocations for untrusted model descriptions and metadata by
  entry, field, and total byte limits, with typed load failures before copying;
  empty/control-containing and duplicate metadata keys are rejected.
- Added worker-isolate prompt-only prefill with typed timing/token telemetry,
  automatic empty-context special-token insertion, exact continuation state,
  and real weighted-model coverage.
- Added cancellable decode-only `continueCompletion()` streams for prefetched
  or restored contexts, typed empty-context failures, and sampler penalty
  history across incremental requests.
- Added a non-retaining Dart finalizer safety net that requests cancellation
  and worker-owned native cleanup for leaked engines without replacing explicit
  `close()` or hiding deterministic cleanup failures.
- Added a shared engine-worker error/exit signal so pending commands and streams
  fail instead of hanging after unexpected isolate termination; close now
  completes cleanup even when its preliminary cancellation request fails and
  skips FFI cancellation through a context address after worker exit.
- Reworked generation streaming into bounded worker start/step/dispose commands,
  added validated `streamChunkTokens` coalescing, honored subscription
  pause/resume backpressure, released paused cancellation promptly, and kept
  parallel context work serialized with deterministic regression coverage.
- Added opt-in process-wide native log capture with typed severity levels,
  bounded oldest-first eviction, dropped-record reporting, immutable Dart
  drains, and silent defaults.
- Added transactional target/draft context shifting with automatic or explicit
  discard counts, dynamic capability and used-token reporting, rollback, and
  continued-generation integration coverage.
- Prevented successful terminal stream closure from issuing a late native
  cancellation that poisoned the next context operation.
- Added snapshotted per-request LoRA adapter selection and scales to
  `GenerationConfig`, including disable-all semantics and unconditional global
  scale restoration after generation.
- Added cancellable worker-isolate `LlamaEngine.warmUp()` for empty target and
  draft/MTP contexts, with session-state cleanup after the warm-up decode.
- Added validated custom stop token IDs to synchronous and streaming
  generation; terminating tokens are not decoded, emitted, or committed to the
  context.
- Added capability-gated upstream Jinja tool calling with typed assistant call
  and correlated result history, auto/required/none/named choice, parallel-call
  gating, lazy grammar triggers, template stop handling, native output parsing,
  terminal parsed assistant messages, and generated call IDs. Tool execution
  remains app-owned.
- Added model-backed draft, EAGLE-3, and integrated/separate MTP speculative
  decoding through pinned upstream `common_speculative`, with typed Dart
  configs, draft limits, compatibility errors, cancellation, and rejected-tail
  cleanup.
- Added speculative draft/verification timing telemetry alongside proposed and
  accepted token counts, and corrected sampler-state advancement across
  accepted n-gram drafts.
- Statically linked pinned `llama-common` and included its required sources in
  native-assets dependency tracking and the package payload without enabling
  runtime downloads.
- Removed redundant direct `llama` links from the bridge and native test target;
  CMake now receives that dependency once through `llama-common`/`mtmd` without
  duplicate-library linker warnings.
- Replaced target-only context snapshots with a checksummed versioned envelope
  that atomically captures target, draft/MTP, optional strategy-private,
  position, and token-history state; retained legacy restore for
  non-model-backed contexts.
- Routed `GenerationConfig.jsonSchema` through pinned upstream
  `json_schema_to_grammar` on the worker isolate, adding local `$defs`/`$ref`
  and broader schema support while retaining the pure-Dart subset helper.
- Added experimental mmproj-backed image/audio chat through pinned upstream
  `mtmd`, including file/byte inputs, worker-isolate preprocessing, streaming,
  model-specific capability reporting, and bounded media requests.
- Kept video input gated because pinned upstream video helpers require external
  ffmpeg executables, which are not appropriate for the mobile runtime.
- Bumped the generated native bridge ABI to 36 for multimodal requests, GPU
  backend selection, model-backed speculation, native JSON Schema conversion,
  expanded telemetry fields, serialized tool-aware chat plans, and custom stop
  token IDs, context warm-up, context shifting, token-use reporting, and native
  log capture, zero-token prompt prefill, and typed KV-cache tuning/reporting.
- Enabled the embedded upstream Metal backend on Apple builds, wired explicit
  Metal device selection, made auto mode offload all available layers, and
  exposed compiled/effective backend reporting.
- Included pinned mtmd sources and decoder headers in the published package and
  native-assets dependency tracking.
- Replaced the generated sample API with typed public configs, errors,
  capability reporting, and local RAG primitives.
- Reported ABI-mismatched native bridge libraries as unavailable during
  capability probing.
- Reported ABI-compatible but incomplete native bridge libraries as unavailable
  during capability probing.
- Rejected invalid Mirostat tau/eta values even when Mirostat sampling is
  disabled.
- Rejected whitespace in `NGramSpeculation.strategy` during config validation.
- Rejected blank RAG prompt headers before prompt assembly.
- Rejected malformed `LlamaReranker` candidate chunks before native scoring.
- Rejected blank tool descriptions in public tool definitions.
- Kept in-memory RAG vector index adds atomic when a later chunk or vector is
  invalid.
- Rejected duplicate chunks inside a single in-memory RAG vector index add
  batch.
- Made desktop benchmark `--help` return usage without parsing later flags.
- Rejected symbolic-link paths in `LlamaEngine.restoreStateFromFile()` so
  session state restores cannot read unexpected link targets.
- Rejected symbolic-link paths when loading persisted RAG vector indexes.
- Preferred native-assets CMake release library outputs in generator-specific
  build subdirectories such as iOS `Release-iphoneos`.
- Reported native context, model, and LoRA cleanup failures instead of silently
  clearing Dart handles after a failed native free.
- Added an experimental C ABI bridge for backend init/free, model load/free,
  model metadata, capabilities, and native smoke coverage.
- Derived the public Dart bridge ABI version from generated bridge bindings.
- Added a native generation handle ABI for worker-isolate token streaming.
- Added generated Dart FFI bindings and worker-isolate model metadata
  inspection.
- Added upstream file type/quantization names to model inspection.
- Added complete native-assets build-hook dependency tracking for CMake-used
  bridge and vendored upstream source files.
- Added a minimal Flutter Android example app for supported-ABI native-assets
  APK validation.
- Added iOS platform files to the minimal Flutter example for config-only build
  validation.
- Added Dart CLI examples for local chat and model-free local RAG usage.
- Removed the undocumented package-template `bin/fllamer.dart` executable.
- Made the native-assets build hook infer Android NDK location from the
  Flutter-provided compiler path before falling back to environment variables.
- Made the native-assets build hook choose the newest numeric NDK under
  `ANDROID_HOME/ndk` instead of using lexicographic path order.
- Added Dart build-mode/version and host processor-count metadata to desktop
  smoke benchmark JSON.
- Added current and peak process RSS snapshots to desktop smoke benchmark JSON.
- Added model file-name metadata to desktop smoke benchmark JSON.
- Added upstream quantization names to desktop smoke benchmark JSON.
- Added Dart tokenizer coverage against the vendored upstream GPT-2 golden
  fixture.
- Rejected empty byte buffers in byte-backed multimodal content-part
  constructors.
- Rejected line breaks in desktop smoke benchmark path flags.
- Reported desktop smoke benchmark CLI usage instead of a stack trace for
  invalid options.
- Rejected missing CLI option values before accidentally consuming the next
  long flag as a value.
- Reported incompatible native-library paths separately from missing bridge
  support.
- Aligned `ngram-simple` draft matching with pinned upstream behavior.
- Rejected speculative decoding and mmproj settings on native embedding and
  reranking contexts.
- Cleared generation-stream output buffers before `out_done` validation.
- Rejected oversize native ABI text/path buffers before scanning them.
- Added multimodal documentation for the current typed DTOs and native
  unsupported boundary.
- Added a validated `mmprojPath` model config field that remains gated until
  native multimodal loading is wired.
- Reported Dart chat CLI argument errors as usage text instead of stack traces.
- Reported Dart local RAG CLI query errors as usage text instead of stack traces.
- Preserved final streamed text in the Dart chat CLI example.
- Cleared native model/context info structs before null-handle validation
  failures.
- Cleared native completion stats before early validation failures.
- Kept native heap-buffer lengths empty until buffer allocation succeeds.
- Cleared stale native bridge errors after successful model, context,
  generation, and LoRA handle frees.
- Cleared stale native bridge errors after freeing result-owned buffers.
- Rejected non-finite native embedding vectors and rerank scores.
- Avoided native cancellation calls from stream subscriptions after engine
  close.
- Rejected unsupported root keys in OpenAI-shaped tool-call JSON.
- Reported malformed tool-call JSON as an argument error.
- Reported malformed OpenAI-shaped tool-call argument JSON as an argument
  error.
- Reported unsupported JSON Schema and parallel tool-call policies as typed
  unsupported feature errors.
- Reported out-of-range Dart chat CLI numeric options as usage errors.
- Rejected blank embedding and rerank text before Dart or native bridge work.
- Rejected blank raw completion prompts before Dart or native bridge work.
- Cleared native context cancellation state when freeing active generation
  handles.
- Returned native cancellation errors promptly from embedding and rerank jobs.
- Exposed direct `EmbeddingConfig.validate()` checks for unsupported rank
  pooling.
- Exposed reranking special-token parsing configuration through the Dart API.
- Prevented native model or LoRA adapter frees while bridge-owned LoRA handles
  are still live or active on a context.
- Rejected invalid document source URIs in built-in RAG text splitters.
- Kept RAG character splitting on Unicode scalar boundaries instead of
  splitting surrogate pairs.
- Rejected blank app-supplied RAG chunk text before indexing, retrieval, or
  prompt assembly.
- Rejected zero-length app-supplied RAG citation spans during prompt assembly.
- Rejected duplicate chunks returned from custom RAG vector indexes.
- Rejected below-threshold chunks returned from custom RAG vector indexes that
  ignore `minScore`.
- Rejected whitespace in custom structured-output grammar roots before Dart or
  native work.
- Rejected symbolic-link paths in `LlamaModel.deleteFile()` so safe deletion
  does not validate one file and remove only the link.
- Rejected symbolic-link paths in `LlamaEngine.saveStateToFile()` so session
  state persistence cannot silently overwrite another file through the link.
- Rejected symbolic-link paths in `InMemoryVectorIndex.persist()` so local RAG
  persistence cannot silently overwrite another file through the link.
- Rejected unsupported chat roles at the native C ABI before applying upstream
  chat templates.
- Snapshotted custom RAG vector-index dimensions once per retrieval before
  query embedding.
- Rejected invalid `LlamaReranker` queries before empty-candidate fast returns.
- Rejected malformed RAG retriever embedding vectors before vector-index search.
- Rejected blank structured-output grammar text and roots before native work.
- Rejected custom structured-output grammar roots when no grammar is provided.
- Rejected blank native bridge model and LoRA paths before upstream calls.
- Rejected empty native bridge chat-template message lists and blank roles/text
  before upstream calls.
- Rejected whitespace in tool names, tool-call ids, and named tool choices.
- Rejected non-`function` tool-call type values during tool-call parsing.
- Cleared native description and metadata string buffers before validation
  failures.
- Cleared native tokenization and detokenization output buffers before
  validation and buffer-too-small failures.
- Rejected native GPU layer counts below the upstream `-1` all-layers sentinel.
- Preserved full native model-description buffer capacities when calling the
  pinned upstream API.
- Returned immutable LoRA adapter lists from the worker-isolate session API.
- Labeled malformed persisted RAG vector-index JSON errors.
- Rejected duplicate chunk records when loading persisted in-memory vector
  indexes.
- Rejected unknown keys when loading persisted in-memory vector indexes.
- Rejected explicit `null` normalize fields when loading persisted in-memory
  vector indexes.
- Rejected explicit `null` record lists when loading persisted in-memory vector
  indexes.
- Rejected explicit `null` chunk metadata when loading persisted in-memory
  vector indexes.
- Added an actionable `--target-platform android-arm64,android-x64` hint when
  Flutter asks the native-assets hook for unsupported Android ABIs.
- Rejected custom vector indexes that return more retrieval results than
  requested by `topK`.
- Rejected custom RAG rerankers that drop vector-search candidates.
- Rejected partial character or token citation spans when building RAG prompts.
- Rejected unknown keys inside parsed tool-call payloads.
- Rejected explicit empty tool-call arrays during tool-call parsing and
  OpenAI-shaped serialization.
- Rejected explicit `null` or blank-string tool-call arguments instead of
  treating them as empty arguments.
- Rejected duplicate parsed or serialized tool-call ids.
- Rejected partial native context-state snapshots instead of returning
  truncated state buffers.
- Rejected JSON Schema object `required` lists with duplicate or undeclared
  properties.
- Rejected explicit `null` JSON Schema `type` values instead of inferring from
  other keywords.
- Rejected explicit `null` JSON Schema object keywords before grammar
  generation.
- Rejected explicit `null` JSON Schema `prefixItems` and `format` keywords
  before grammar generation.
- Rejected duplicate JSON Schema `enum` values, including numeric equivalents
  and object values with different key order, before grammar generation.
- Rejected rank pooling through the embedding APIs before native work; rank
  pooling remains exposed through reranking.
- Accepted generated tokens into the native sampler chain after successful
  decode so penalties and grammar state track streamed output.
- Rejected malformed custom vector-index dimensions before RAG retrieval.
- Snapshotted JSON-schema generation tool-calling definitions at config
  creation.
- Snapshotted tool-calling definitions before engine generation worker
  dispatch.
- Documented the current full iOS build blocker: local Xcode is missing the iOS
  26.5 platform, while iOS config-only generation still passes.
- Verified supported-ABI Android release APK packaging from the Flutter example.
- Snapshotted `LlamaReranker` candidate lists before native scoring.
- Snapshotted JSON-schema generation stop strings at config creation.
- Flushed persisted in-memory vector index writes to storage.
- Made native bridge tests skip fixture-backed coverage when the vocab fixture is
  absent from a published source package.
- Excluded the unused upstream model-download CMake helper from published
  payloads.
- Reduced pub dry-run warnings from vendored upstream ignore rules.
- Added native and Dart tokenization/detokenization using upstream model
  vocabularies.
- Added token-count helpers for raw text and chat-template prompts.
- Added native context create/free/info ABI with ownership checks.
- Wired `LlamaEngine.load`/`close` to a worker-isolate native model/context
  owner.
- Added model chat-template formatting through upstream `llama_chat_apply_template`.
- Added worker-isolate raw prompt and chat completion over upstream
  decode/sampler APIs.
- Added generation sampling controls for top-k, top-p, min-p, typical-p, and
  repeat/presence/frequency penalties.
- Added experimental GBNF grammar constraints for structured completion output.
- Added structured-output runtime capability reporting.
- Added an early Dart-side ABI version guard for native bridge loading.
- Added runtime upstream commit reporting from the native bridge.
- Added `LlamaEngine.reset()` to clear native context state through the worker
  isolate.
- Added validation and explicit unsupported-feature errors for speculative
  decoding configs.
- Added native-library-path NUL validation.
- Added `FLLAMER_NATIVE_LIBRARY` path validation before bridge lookup fallback.
- Added generation seed range validation.
- Added finite-number validation for generation sampling floats.
- Added non-negative LoRA scale validation.
- Added fixed-width native integer range validation for public configs.
- Added stop-string validation for generation configs.
- Added chat-message NUL validation before native chat-template calls.
- Added tool-call parser validation for empty parsed tool-call ids.
- Snapshot tool-call arguments before OpenAI-shaped JSON serialization.
- Added experimental text embeddings through the native bridge, returning
  `Float32List` vectors from pooled upstream sequence embeddings.
- Added same-context batch embedding helpers for RAG chunk batches.
- Added RAG `EmbeddingModel`/`Retriever` abstractions, `VectorIndexRetriever`,
  and `LlamaEmbeddingModel` to connect local embeddings to the in-memory index.
- Added app-owned RAG reranker hooks.
- Added async token-aware RAG splitting.
- Added JSON persistence for the in-memory vector index.
- Made built-in splitter chunk ids document-scoped and validated vector-index
  ids against NUL bytes.
- Made RAG prompt budgeting skip oversized chunks instead of dropping all later
  context.
- Added finite-vector validation for in-memory RAG search.
- Added persisted vector-index version validation.
- Reused vector normalization when loading persisted in-memory RAG indexes.
- Added positive RAG token-count validation.
- Added benchmark CLI validation for empty or NUL string option values.
- Reused benchmark option validation for programmatic benchmark calls.
- Rejected out-of-range numeric desktop benchmark options before native work.
- Mapped empty session-state files to `StateFileException`.
- Rejected persisted RAG vector dimension mismatches before vector allocation.
- Rejected line breaks in RAG ids before prompt source-marker assembly.
- Rejected line breaks in tool names and tool-call ids.
- Rejected malformed custom vector-index and reranker results before retrieval.
- Snapshotted custom vector-index and reranker chunks before retrieval returns.
- Cleared native C ABI size outputs on validation failures.
- Cleared native C ABI heap and flag outputs on validation failures.
- Rejected invalid RAG tokenizer token ids before detokenizing chunks.
- Rejected NUL bytes in RAG metadata keys and strings.
- Rejected blank or whitespace-containing multimodal MIME subtypes.
- Rejected non-finite scores returned by app-owned RAG rerankers.
- Rejected malformed speculative decoding strategy names before native setup.
- Rejected line breaks in grammar root names before native generation.
- Rejected line breaks in RAG source URIs before indexing or loading.
- Rejected line breaks in model, native-library, LoRA, session-state, media,
  and RAG index file paths before filesystem or native work.
- Snapshotted RAG tokenizer output before async detokenization.
- Snapshotted chat message lists before worker-isolate formatting/generation.
- Rejected native C ABI grammar root line breaks before grammar parsing.
- Rejected native C ABI chat role line breaks before template formatting.
- Verified the host package suite under `flutter test`.
- Added pre-embedding retriever search argument validation.
- Added RAG id validation across splitters, indexes, removals, and prompt
  sources.
- Added Dart-side detokenize token-id range validation before native calls.
- Added native completion sampler finite-float validation.
- Added native grammar-root validation for completion requests.
- Added native context-size and batch-size validation.
- Added native context thread-count validation.
- Added native context pooling and attention enum validation.
- Added native detokenize token-id vocabulary validation.
- Added a native-free empty-token detokenize fast path.
- Added Dart-side NUL validation for native text inputs.
- Added Dart and native NUL validation for raw completion prompts.
- Added native NUL validation for tokenization text.
- Added native NUL validation for embedding text.
- Mapped native allocation failures to `NativeOutOfMemoryException`.
- Added empty native-library-path validation.
- Mapped model-file type lookup failures to `ModelFileException`.
- Added cooperative native decode cancellation when a generation stream
  subscription is cancelled.
- Streamed generation chunks from the worker isolate instead of emitting only a
  final completion chunk.
- Fixed immediate generation-stream cancellation so the native worker cannot
  clear a pending cancel before decode starts.
- Added `GenerationConfig.jsonMode` using the pinned upstream JSON grammar.
- Allowed public APIs to use `FLLAMER_NATIVE_LIBRARY` or platform default
  native-library lookup when `nativeLibraryPath` is omitted.
- Preserved exact explicit native-library paths, including trailing filename
  whitespace, when opening the bridge.
- Made unavailable explicit Metal/Vulkan GPU backends fail with typed
  unsupported-feature errors before model loading.
- Excluded vendored upstream model fixtures from the published package payload.
- Added experimental native reranking scores and `LlamaReranker` for
  rank-pooling GGUF models.
- Added `RerankingException` for native reranking failures.
- Added experimental LoRA adapter load/list/scale/unload support on
  `LlamaEngine`.
- Moved generation stop-string handling into the native decode loop.
- Added `LlamaEngine.contextInfo()` for actual native context parameters.
- Added `LlamaModel.metadata()` for GGUF metadata key/value reads.
- Added experimental `LlamaEngine.saveState()` and `restoreState()` for
  in-memory context snapshots.
- Added Mirostat v1/v2 sampling configuration.
- Added final-chunk generation telemetry for prompt/generated token counts and
  elapsed prompt/decode/total time.
- Added tokenizer special-token metadata to `LlamaModelInfo`.
- Added `GenerationConfig.jsonSchema()` for a common JSON Schema-to-GBNF
  subset.
- Added local model file validation and SHA-256 checks through
  `LlamaModel.validateFile()` and `LlamaModel.sha256()`.
- Added computed tok/s rates to generation telemetry.
- Added architecture, native build, and mobile performance docs.
- Added file-backed context state helpers on `LlamaEngine`.
- Expanded the JSON Schema grammar helper to allow fixed-order optional object
  properties.
- Added typed map-object support through JSON Schema `additionalProperties`
  schemas.
- Allowed common JSON Schema metadata annotation keywords in structured-output
  grammar generation.
- Made persisted in-memory vector index record order deterministic.
- Rejected line breaks in native bridge model and LoRA paths before file loading.
- Made native CMake flags explicit for unused upstream network/tool/UI and
  accelerator backend options.
- Excluded disabled upstream backend source trees from the published package
  payload.
- Added `minProperties` and `maxProperties` support for map-style JSON Schema
  objects.
- Added keyword-based object, array, and string type inference for JSON Schema
  grammar generation.
- Added Dart-side tool definition DTOs and JSON tool-call parsing.
- Added tool-calling policy to `GenerationConfig`.
- Added typed multimodal chat content parts with non-text parts gated before
  native work.
- Added explicit validated local model file deletion through
  `LlamaModel.deleteFile()`.
- Added `RagPromptBuilder.buildChatMessages()` for source-tagged chat prompt
  assembly.
- Added JSON serialization helpers for tool definitions and parsed tool calls.
- Added native time-to-first-token telemetry for text generation.
- Added default GGUF magic-byte validation to local model file checks.
- Added typed RAG prompt citations with optional character/token source spans.
- Added scalar `const` support to the JSON Schema grammar helper.
- Expanded JSON Schema `const` and `enum` grammar helpers to support JSON
  array and object literals.
- Added simple `anyOf` and `oneOf` alternatives to the JSON Schema grammar
  helper.
- Added test and documentation coverage for primitive JSON Schema `type`
  unions.
- Added constrained nullable JSON Schema `type` union support, such as
  `["string", "null"]` with string bounds.
- Added `minLength` and `maxLength` string bounds to the JSON Schema grammar
  helper.
- Added selected JSON Schema string formats: `date`, `time`, `date-time`, and
  `uuid`.
- Added exact small finite integer ranges for JSON Schema `minimum` and
  `maximum`.
- Added `exclusiveMinimum` and `exclusiveMaximum` support for exact small
  integer ranges.
- Added positive `multipleOf` filters for exact small JSON Schema integer
  ranges.
- Added bounded homogeneous array support to the JSON Schema grammar helper.
- Added fixed tuple array support through JSON Schema `prefixItems`.
- Added OpenAI-shaped serialization for tool-calling config policy.
- Added typed tool-choice policy serialization for tool-calling configs.
- Exposed `ubatchSize` and `batchThreads` runtime tuning on `LlamaModelConfig`.
- Exposed mmap, mlock, and tensor-check model loading flags on
  `LlamaModelConfig`.
- Added exact RAG vector-index chunk removal with `documentId`.
- Made in-memory RAG vector-search tie ordering deterministic.
- Added NUL-byte validation at local RAG text boundaries.
- Added a CMake-backed native-assets build hook for bundling the native bridge.
- Added build-hook smoke coverage for bundled code asset output.
- Rejected empty text-only chat messages before native formatting.
- Added native build-flag reporting to runtime capabilities.
- Added a desktop smoke benchmark harness that records runtime, model, config,
  and generation telemetry metadata as JSON.
- Added benchmark CLI validation for paired batch and speculative-decoding
  tuning knobs.
- Added media MIME validation and clearer unsupported-feature errors for
  multimodal chat parts.
- Added experimental native `ngram-simple` speculative decoding with draft and
  accepted-token telemetry.
- Added Dart and benchmark tuning knobs for `ngram-simple` speculative
  decoding.
- Restricted native-assets Android builds to `arm64-v8a` and `x86_64` until
  other ABIs are intentionally validated.
- Aligned `GenerationConfig.maxTokens` validation with the native bridge's
  int32 completion limit.
- Exposed upstream NextN/MTP layer count through `LlamaModelInfo`.
- Fixed character-splitter citation spans to match trimmed chunk text.
- Kept native grammar/sampler setup failures from mutating context prompt state.
- Rejected native state, embedding, rerank, and LoRA context changes during
  active generation handles.
- Made generation streams created before engine close fail promptly when
  listened after close.
- Rejected persisted RAG vector-index files whose JSON root is not an object.
- Rejected malformed top-level persisted RAG vector-index fields with
  `FormatException`.
- Rejected malformed persisted RAG vector records with `FormatException`.
- Rejected NUL bytes in custom RAG prompt headers.
- Rejected empty or NUL-containing RAG vector-index persistence paths.
- Rejected non-JSON values in tool schemas and serialized tool-call arguments.
- Mapped session-state file read/write failures to `StateFileException`.
- Accepted string-keyed JSON maps in top-level tool-call parsing.
- Rejected malformed null top-level `tool_calls`.
- Validated explicit native-library paths in runtime capability probing.
- Rejected NUL bytes in tool schemas and serialized tool-call arguments.
- Rejected non-JSON values in parsed tool-call arguments.
- Skipped RAG reranking when vector search returns no candidates.
- Returned immutable RAG retriever results after app-owned reranking.
- Rejected non-JSON RAG vector metadata before indexing.
- Snapshotted RAG vector metadata during JSON export.
- Rejected cyclic RAG vector metadata before indexing.
- Deep-snapshotted tool schemas and direct tool-call arguments during JSON export.
- Returned immutable tool/OpenAI JSON adapter maps and lists.
- Ignored malformed Android NDK paths during native-assets build-hook discovery.
- Snapshotted generation stop markers before creating lazy generation streams.
- Snapshotted batch token, embedding, and reranking inputs before worker calls.
- Exposed immutable byte views from image, audio, and video chat content parts.
- Returned immutable RAG vector-index JSON export maps and lists.
- Pinned the Flutter example Android minimum SDK to API 28 and documented it.
- Removed Android internet permissions from Flutter example debug/profile builds.
- Excluded Flutter example Gradle/Xcode generated state from published payloads.
- Excluded Codex guidance and Xcode SwiftPM workspace state from published payloads.
- Documented local-by-default safety notes in the README.
- Added pub.dev topics for FFI, llama.cpp, local AI, and RAG discovery.
- Rejected malformed RAG vector records when loading persisted indexes.
- Mapped RAG vector-index file read/write failures to `RagIndexException`.
- Skipped RAG query embedding when the vector index is empty.
- Deep-snapshotted RAG vector chunks when adding them to the in-memory index.
- Deep-snapshotted RAG splitter metadata when producing chunks.
- Snapshotted RAG prompt-builder included chunk metadata.
- Deep-snapshotted nested RAG prompt-builder included chunk metadata.
- Rejected malformed persisted RAG vector source URIs.
- Rejected cyclic JSON Schema `const` and `enum` literals.
- Rejected cyclic nested JSON Schema objects.
- Rejected cyclic raw-map JSON Schema objects.
- Rejected unsupported sibling keys on JSON Schema `enum` literals.
- Rejected Dart and native configs whose `ubatchSize` exceeds `batchSize`.
- Rejected multimodal MIME types without a subtype.
- Rejected empty generation grammar strings.
- Rejected empty RAG vector source URIs.
- Rejected literal and percent-encoded NUL bytes in RAG vector source URIs.
- Rejected invalid RAG prompt-builder chunk source URIs before citation output.
- Bounded speculative acceptance-rate telemetry to the valid 0..1 range.
- Fixed JSON-mode number grammar for exponents containing zero.
- Fixed native completion decoding to use explicit context positions.
- Chunked native prompt decoding by the configured batch size.
- Seeded native sampler history with prompt tokens for unconstrained generation.
- Tracked native token history across completions for repeat penalties.

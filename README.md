# fllamer

`fllamer` is a Dart/Flutter package for local llama.cpp inference.

Current state: the public Dart API, typed errors, capability reporting,
Dart-only RAG primitives, and an experimental native C ABI bridge are in place.
Native generation streams decoded chunks through a worker isolate and uses
upstream chat-template formatting; cancelling the returned stream requests a
native decode abort. Pooled text embeddings are also available through an
experimental worker-isolate API when the model exposes upstream embedding
support. LoRA adapters can be loaded, listed, scaled, unloaded, and selected
per generation through the engine when the loaded model and upstream runtime
accept the adapter.

## Installation

Add the package to the application:

```yaml
dependencies:
  fllamer: ^0.2.4
```

Published packages include the pinned `llama.cpp` source needed by the native
build. Repository checkouts must initialize the exact submodule pin before
dependency resolution:

```sh
git submodule update --init --checkout
dart pub get
```

The native-assets hook builds the bundled bridge with CMake. Host Dart builds
therefore require CMake 3.16 or newer and a C++ toolchain; Android builds
additionally require the NDK, and Apple builds require Xcode. Linux/Windows
Vulkan builds require CMake 3.19 or newer, Vulkan development files, and
`glslc` unless the consuming workspace explicitly selects a CPU-only bridge.
Android Vulkan is an opt-in build variant. Its exact Vulkan-Headers pin is
bundled, and the build uses host `glslc` from Flutter's selected Android NDK.
See [native builds](doc/native_builds.md) for supported ABIs, minimum platform
versions, and platform-specific commands. Models, mmproj files, and LoRA
adapters are app-owned data and are never downloaded by the package.

## Quick start

```dart
import 'package:fllamer/fllamer.dart';

void main() async {
  final info = await LlamaModel.inspect(
    const LlamaModelConfig(
      modelPath: '/app/private/model.gguf',
    ),
  );

  print('${info.parameterCount} parameters');

  final embedding = await LlamaEmbeddings.embedText(
    const LlamaModelConfig(
      modelPath: '/app/private/embedding-model.gguf',
    ),
    'local retrieval context',
  );
  print('${embedding.length} embedding dimensions');

  final engine = await LlamaEngine.load(
    const LlamaModelConfig(
      modelPath: '/app/private/model.gguf',
    ),
  );
  try {
    await for (final chunk in engine.chat(
      messages: [ChatMessage.user('Write one sentence about local inference.')],
    )) {
      print(chunk.text);
    }
  } finally {
    await engine.close();
  }
}
```

## Supported today

- Typed model, GPU, generation, chat-message, and speculative-decoding configs.
- Typed exception hierarchy for native and feature-gating failures.
- Runtime capability reporting.
- Runtime upstream commit and native build-flag reporting.
- Native bridge lookup through the bundled code-asset ID by default, including
  Apple framework packaging. Explicit `nativeLibraryPath` and
  `FLLAMER_NATIVE_LIBRARY` remain available for custom/debug bridges.
- Opt-in process-wide native log capture with typed levels, bounded local
  buffering, and app-owned draining; logging is disabled by default.
- Experimental CMake-backed native-assets build hook that bundles the native
  bridge when Dart/Flutter builds request code assets.
- Mobile-first Flutter example app with app-owned GGUF/mmproj/image selection,
  model lifecycle controls, streaming text/image chat, cancellation, context
  reset, capability reporting, and generation telemetry.
- Local RAG building blocks: documents, deterministic character/token splitting,
  embedding/retriever/reranker interfaces, persistable in-memory vector search,
  and context/chat prompt assembly with typed citations. Tokenizer-backed
  context assembly counts headers and source labels exactly, while full chat
  assembly can use `LlamaEngine.countChatTokens()` to enforce the complete
  model-specific prompt budget without reopening the model. Vector-index JSON
  persistence and loading also encode/decode outside the caller isolate.
- Native ABI smoke bridge (current ABI 42): version, capabilities, backend
  init/free, model load/free, context create/free, model metadata, tokenization,
  detokenization, and last-error functions.
- Generated Dart FFI bindings from `native/llama_dart_bridge/include/llama_dart.h`.
- Worker-isolate model inspection that loads GGUF metadata with `vocab_only` and
  frees the native handle before returning, including tokenizer special-token
  metadata, maximum decoded token-piece bytes, file type/quantization name,
  and NextN/MTP layer count.
- Worker-isolate GGUF metadata reads through `LlamaModel.metadata()`, with a
  typed `LlamaModel.architecture()` accessor for `general.architecture`.
  Description and metadata allocations are bounded before copying untrusted
  GGUF text into Dart (1 MiB descriptions, 65,536 entries, 4 KiB keys, 16 MiB
  values, and 64 MiB total metadata).
- Local model file validation, GGUF magic checks, and SHA-256 checks through
  `LlamaModel.validateFile()` and `LlamaModel.sha256()`, plus explicit local
  file deletion with `LlamaModel.deleteFile()`. Full-file hashing runs outside
  the caller isolate.
- Worker-isolate tokenization, token counting, detokenization, chat-template
  capability inspection, and chat formatting. Methods on a loaded
  `LlamaEngine` reuse its worker-owned model; static `LlamaTokenizer` and
  `LlamaChatTemplate` helpers use temporary vocab-only model loads.
- Worker-owned `LlamaEmbeddingEngine` contexts reuse one loaded embedding
  model for tokenization, metadata inspection, and repeated embedding batches;
  `LlamaEmbeddings` remains the one-shot helper for occasional calls.
- Worker-isolate chat-template formatting with `llama_chat_apply_template`,
  plus capability-gated upstream Jinja rendering for tool-aware requests.
  Chat requests can set nullable `GenerationConfig.enableThinking`: an explicit
  value selects native Jinja chat planning and forwards the typed
  `enable_thinking` template input without reloading the model, while `null`
  retains the legacy formatter selection unless a reasoning budget requests a
  native plan. When explicit `false` is paired with template-exposed reasoning
  markers, native sampling also applies a zero reasoning budget: an unexpected
  reasoning start is immediately closed and the public suffix remains
  generatable. Markerless templates retain template-only Off behavior.
  Thinking chat cannot be combined with an app-supplied raw
  grammar, JSON mode, or JSON schema because that constraint would also apply
  to reasoning. Planner-owned tool grammar remains supported. Typed terminal
  parsing keeps reasoning separate from the normalized assistant value, but
  streamed `GenerationChunk.text` is raw model output and can contain reasoning;
  apps must filter it before display, history, or speech. Raw `complete()` and
  `continueCompletion()` calls reject thinking controls.
  Chat may also set `reasoningBudgetTokens` to cap each reasoning block
  independently of the total `maxTokens`. A budget leaves
  `enable_thinking` at llama.cpp's default `true` when `enableThinking` is
  `null`, and rejects an explicit `false`.
  Reasoning already present in the template's generation prefill counts toward
  the budget. A budget requires paired reasoning tags exposed by the selected
  template; otherwise chat fails with `UnsupportedFeatureException`. The bridge
  derives bounded, potentially multi-token start and end sequences from every
  end-tag alternative exposed by llama.cpp, uses its reasoning-budget state
  machine, and keeps lazy planner grammar out of active reasoning even when the
  budget is unlimited. A matched end sequence is replayed into the deferred
  grammar so alternatives that begin a tool call still activate its trigger.
  After budget exhaustion it completes any partial UTF-8 piece and then forces
  the first template-provided closing sequence. This is model-family-neutral;
  no marker spelling or token ID is hard-coded. `maxTokens` remains the separate
  total output cap and can stop generation first, so callers must reserve enough
  total space for model-specific markers and public output.
  Template-less models fail with `UnsupportedFeatureException` instead of
  silently inheriting llama.cpp's ChatML fallback. Apps can explicitly supply
  `LlamaModelConfig.chatTemplate`, and inspect the selected template through
  `LlamaModelInfo.chatTemplate`, `LlamaModel.chatTemplate()`, or the loaded
  engine without reopening its path.
- Worker-isolate `LlamaEngine.load`/`close` ownership for native model and
  context handles, plus `warmUp()` for predictable first-request latency,
  prompt-only `prefill()` with typed timing telemetry for reusable prefixes,
  transactional `shiftContext()` for long sessions, `reset()` to clear context
  state, and `contextInfo()` to inspect actual native parameters and token use.
  A non-retaining finalizer requests cancellation and worker cleanup for leaked
  engines, but apps must still call `close()` for deterministic release and
  cleanup errors.
- Experimental context state snapshots with `saveState()`, `restoreState()`,
  `saveStateToFile()`, and `restoreStateFromFile()`. New snapshots use a
  checksummed versioned envelope that preserves target, draft/MTP,
  strategy-private, position, and token-history state together; legacy
  target-only snapshots remain readable by non-model-backed contexts.
- Worker-isolate raw prompt and chat completion using upstream decode and
  sampler APIs, streaming decoded chunks with stop-string holdback and custom
  stop-token termination, plus typed terminal stop reasons and final-chunk
  telemetry for token counts, time to first token, elapsed time, and tok/s
  rates, speculative draft/accepted token counts, and draft/verification
  timings. Prompt evaluation adds model special
  tokens only when the context is empty; chat-template text parses trusted
  model control tokens. `continueCompletion()` samples directly from a
  prefetched or restored non-empty context, with prior-session sampler penalty
  history preserved across successful requests. Streaming uses a worker
  start/step/dispose protocol: `streamChunkTokens` coalesces up to four tokens by
  default, paused subscriptions stop requesting native steps, and a second
  active or pending stream is rejected with a typed generation error.
  Cancelling a subscription awaits native cleanup and resets its context before
  another request can start, so partial prompt/output state is not retained.
- Sampling controls for temperature, top-k, top-p, min-p, typical-p,
  repetition/presence/frequency penalties, Mirostat v1/v2, seed, stop strings,
  and model token IDs. The manual bridge sampler also applies every
  `tokenizer.ggml.suppress_tokens` entry from the loaded vocabulary.
- Experimental `ngram-simple`, `ngram-map-k`, and `ngram-map-k4v`
  self-speculation through `NGramSpeculation`, plus tuned
  `NGramModSpeculation` and request-local `NGramCacheSpeculation`. All five
  draftless strategies use pinned upstream `common_speculative`; cache mode is
  memory-only and never reads or writes cache files. The desktop benchmark can
  select each strategy with `--spec-ngram` and records its effective tuning.
  `DraftModelSpeculation`, `Eagle3Speculation`, `DFlashSpeculation`, and
  `MtpSpeculation` use the same upstream strategy layer and expose a bounded
  `draftLength`. DFlash is upstream-dependent and does not yet have a packaged
  fixture. An opt-in checksum-pinned Q8/Q4 TinyLlama pair covers
  the real ordinary draft-model path, and an opt-in checksum-pinned Qwen3.5
  model with an integrated NextN head covers real MTP output equivalence and
  acceptance telemetry. A pinned Qwen3 1.7B target and upstream-converted
  EAGLE-3 draft pair covers the EAGLE-3 path. Requests with stop strings or
  custom stop tokens fall back to ordinary decoding to preserve exact stop
  behavior. Integrated MTP heads are loaded only when `MtpSpeculation` has no
  separate sidecar, preserving upstream's ordinary-model memory saving. See
  [speculative decoding](doc/speculative_decoding.md).
- Runtime tuning knobs for mmap/mlock/tensor checks, context size, batch size,
  ubatch size, threads, batch threads, GPU layer count, K/V cache data types,
  KV offload, Flash Attention, full-size SWA cache, and unified KV cache.
  `contextInfo()` reports the applied cache settings. Explicit CPU mode forces
  context operation and KV offload off. Apple builds embed
  Metal and support `GpuConfig.auto()`/`metal()`; `GpuConfig.cpu()` forces the
  CPU path. On Apple Simulator targets, `GpuConfig.auto()` also resolves to an
  explicit CPU device with zero GPU layers because Simulator Metal is not a
  supported inference path. Explicit Metal requests remain available for
  diagnostics. Linux and Windows native-assets builds require and include
  Vulkan by default while retaining the CPU backend; consuming workspaces can
  explicitly build a CPU-only bridge. Android remains CPU-only by default and
  can opt into a Vulkan-enabled bridge without an app-specific SDK path; the
  package supplies its pinned headers and Flutter's selected NDK supplies the
  remaining target and host shader-tool inputs.
  Runtime capability metadata distinguishes the artifacts before an app
  requests Vulkan. One final-source Debug harness run on an Adreno 750 selected
  Vulkan, matched its bounded CPU oracle, and passed cancellation/reset/recovery
  and repeat-dispose checks. This is a narrow experimental receipt, not Android
  Vulkan support or a speed claim: the Qualcomm-proprietary K-quant policy uses
  a safe Q4_K matrix route and CPU fallback for Q5_K/Q6_K matrix operations.
  Android remains opt-in and unqualified; see [native builds](doc/native_builds.md)
  for the exact artifact and checker limitation.
- Experimental GBNF grammar constraints, JSON mode, and pinned-upstream JSON
  Schema conversion for `GenerationConfig.jsonSchema`, including local
  `$defs`/`$ref`, JSON
  `const`/`enum` literals, simple `anyOf`/`oneOf` alternatives, bounded
  homogeneous arrays, fixed tuple arrays, typed map objects with optional
  property-count bounds, and bounded strings with selected formats, primitive
  `type` unions including constrained nullable schemas, plus exact small
  inclusive/exclusive integer ranges with positive `multipleOf` filters. The
  standalone `llamaJsonSchemaGrammar()` compatibility helper delegates to the
  same pinned native converter. See
  [structured output](doc/structured_output.md).
- Experimental local image/audio chat from files or encoded bytes through a
  matching `mmprojPath`, with worker-isolate preprocessing, model-specific
  vision/audio capability reporting, ordered prompt markers, and bounded media
  inputs. Opt-in checksum-pinned TinyGemma3 and official Qwen3 ASR fixtures
  cover real image and audio chat. Video stays gated because upstream requires
  external ffmpeg tools.
- Experimental end-to-end function/tool calling through pinned upstream chat
  templates: model-specific capability inspection, typed definitions and
  history, auto/required/none/named choice, lazy grammar triggers, additional
  stops, upstream Gemma 4 argument grammar, post-parse schema validation,
  parsed terminal assistant messages, parallel-call gating, and stable
  generated call IDs. An opt-in checksum-pinned official Qwen2.5 fixture
  covers real weighted call generation and result consumption. Tool execution
  remains app-owned. See
  [tool calling](doc/tool_calling.md).
- Experimental pooled text embeddings returning `Float32List`; batch calls use
  a row-major `EmbeddingBatch` with one flat `Float32List` and zero-copy vector
  views instead of nested vector allocation, plus normalization and pooling
  metadata. Opt-in checksum-pinned real MiniLM GGUF integration coverage
  exercises same-context batches.
- Experimental native reranking scores for rank-pooling models, plus
  `LlamaReranker` for the RAG interface, with opt-in checksum-pinned real Jina
  tiny reranker integration coverage.
- Experimental LoRA adapter load/list/scale/unload support on `LlamaEngine`,
  with exact per-request adapter scale overrides in `GenerationConfig` and an
  opt-in checksum-pinned real base/adapter integration fixture. See
  [LoRA](doc/lora.md).
- Desktop benchmark harness with native warm-up, reset-between-run sustained
  iterations, KV-cache/Flash-Attention controls, requested-versus-applied
  context settings, explicit device-model metadata, RSS snapshots, and
  reproducible per-run plus aggregate JSON telemetry.
- Dart CLI examples for local text chat, real flat batch embeddings,
  image/audio chat, and model-free RAG smoke usage.

## Native logging

Native logging is silent by default. Enable bounded in-memory capture only
while needed, drain records into the app's logging system, then disable it:

```dart
LlamaRuntime.configureNativeLogging(LlamaLogLevel.warning);
final records = LlamaRuntime.drainNativeLogs();
LlamaRuntime.configureNativeLogging(LlamaLogLevel.disabled);
```

The setting and queue are process-wide because upstream callbacks are
process-wide. The queue retains at most 1 MiB or 4096 records, drops the oldest
records first, and reports drops as a warning record. Messages can contain
local model or adapter paths, so applications should apply their own redaction
before persistence or display.

## Not supported yet

- JSON Schema features outside the pinned upstream converter, including
  external references, and video input.
- Physical-device iOS runtime smoke tests. A dependent Flutter app passes an
  iOS 26.5 Simulator automatic-backend regression and an unsigned `iphoneos`
  Debug build, but the latter is compile/package evidence rather than a Metal
  runtime result from iPhone hardware.
- Android Vulkan product support beyond one Xiaomi 23127PN0CC/Adreno 750
  Android 16 Debug-harness receipt. That receipt covers a final-source bounded
  CPU-oracle run, cancellation, reset/recovery, and repeat disposal. It does not cover
  sustained or thermal behavior, more models, drivers, devices, Android
  lifecycle transitions, signed/release packaging, or performance. The strict
  backend-result checker still reports some Q4_0/Q4_1 intermediate relative
  errors, and Q5_K/Q6_K matrix work deliberately falls back to CPU on the
  affected Qualcomm proprietary driver.
- Android API levels below 28 and 32-bit Android `armeabi-v7a`; pass
  `--target-platform android-arm64,android-x64` for Android APK validation.
- iOS versions below 15.0. The pinned embedded Metal backend uses an event API
  introduced in iOS 15. The native-assets hook compiles the bridge for at least
  iOS 15 even when Flutter supplies its generic iOS 13 hook input; applications
  must also set their Xcode deployment target to iOS 15.0 or newer.
- Cross-architecture Linux or Windows builds. Build x64 artifacts on an x64
  host and arm64 artifacts on an arm64 host so the pinned Vulkan shader
  generator is executable during the native build.

## Safety notes

- Inference is local by default. The package does not download models, call
  hosted APIs, or request Android internet permission.
- Model files, LoRA adapters, mmproj files, and vector indexes are app-owned
  data. This package does not bundle model weights.
- `LlamaModel.deleteFile()` refuses symbolic-link paths after validating the
  target, so deletion cannot silently remove only the link.
- `LlamaEngine.saveStateToFile()` and `restoreStateFromFile()` refuse
  symbolic-link paths so session state cannot silently write to or read from
  another file through the link.
- `InMemoryVectorIndex.persist()` and `load()` refuse symbolic-link paths so
  vector indexes cannot silently write to or read from another file through the
  link.
- Treat GGUF and adapter files as untrusted input; validate local files with
  `LlamaModel.validateFile()` before loading when checksums are available.

## Native bridge checks

```sh
dart run ffigen --config ffigen.yaml
dart run ffigen --config ffigen.native_assets.yaml
cmake -S native/llama_dart_bridge -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native --config Release
ctest --test-dir build/native --output-on-failure
```

## Licensing

`fllamer` is GPL-3.0 licensed. The repository pins `llama.cpp` as a submodule,
and the published package includes its required source files. Keep upstream
notices from `third_party/llama.cpp/LICENSE`,
`third_party/llama.cpp/AUTHORS`, and `third_party/llama.cpp/licenses/` with any
redistribution. The package manifest registers these files plus cpp-httplib,
stb, and miniaudio notices as Flutter additional licenses so they are collected
into generated application notices. This does not decide whether a particular
application is compatible with GPL-3.0; distributors still need to make that
product/legal decision and meet the applicable source and notice obligations.
Model files, adapters, and mmproj files are app-supplied data with their own
licenses.

The upstream submodule is pinned at
`ddd4ec1428a6201e18975ea52b07c71e0f9aef26` (`b10217`).

See [doc/feature_matrix.md](doc/feature_matrix.md) and
[doc/upstream_sync.md](doc/upstream_sync.md). Architecture, native build, and
performance notes are in [doc/architecture.md](doc/architecture.md),
[doc/native_builds.md](doc/native_builds.md), and
[doc/mobile_performance.md](doc/mobile_performance.md). Multimodal status is
documented in [doc/multimodal.md](doc/multimodal.md).
Runnable CLI and Flutter examples live under [example](example).

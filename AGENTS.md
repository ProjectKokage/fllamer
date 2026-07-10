# AGENTS.md

This repository is a mobile-first Dart/Flutter wrapper around `llama.cpp`. Codex should use this file as standing project guidance when creating, reviewing, or modifying code in this repository.

## Core objective

Build a production-quality local inference library for Dart and Flutter that exposes the important runtime capabilities of `llama.cpp` through a safe, idiomatic Dart API, with first-class support for iOS and Android. The package must be practical for real mobile apps: predictable resource ownership, no UI-thread blocking, no hidden network calls, clear errors, measurable performance, and documented platform constraints.

Prefer correctness, safety, and maintainability over a broad but fragile feature surface. When upstream `llama.cpp` behavior or APIs are uncertain, inspect the pinned upstream source and headers before implementing. Do not invent native behavior that upstream does not support.

## Terminology

- `llama.cpp`: the upstream native inference engine under `third_party/llama.cpp` or an equivalent pinned submodule.
- Bridge: this repository's C ABI layer that Dart calls through `dart:ffi`.
- High-level API: the public Dart API used by Flutter and Dart apps.
- MTP: Multi-Token Prediction, exposed as a speculative decoding strategy when supported by the loaded model and the pinned upstream version.
- RAG: retrieval-augmented generation. In this repository, RAG means local embeddings, chunking, retrieval, optional reranking, and prompt assembly. It does not imply any cloud service.

## Non-negotiable design rules

1. Do not expose C++ ABI directly to Dart. Export a narrow, stable C ABI from the bridge and generate Dart bindings from that bridge header.
2. Do not call the `llama.cpp` CLI or launch `llama-server` from mobile apps. Mobile runtime integration must link the native library and call it in-process.
3. Do not block the Flutter UI isolate with model loading, prompt evaluation, token generation, embedding, reranking, LoRA operations, or multimodal preprocessing.
4. Do not bundle model weights, LoRA adapters, mmproj files, or vector databases in the package unless they are tiny test fixtures and license-compatible.
5. Do not make network requests by default. Model download helpers, if added, must be opt-in, checksum-verified, cancellable, and documented.
6. Do not rely on Python, shell scripts, executable downloads, or code generation at app runtime.
7. Do not hide native failures. Surface typed Dart exceptions with actionable messages and include the bridge/upstream error where available.
8. Do not commit generated files without also committing the command/configuration needed to regenerate them.
9. Do not change the public Dart API casually. Preserve semver and add migration notes for breaking changes.
10. Do not claim support for a feature or platform until there is a test, example, or clearly documented limitation.

## Preferred repository layout

If the repository already has a layout, preserve it unless there is a concrete reason to migrate. For a new repository, prefer this structure:

```text
.
├── AGENTS.md
├── README.md
├── CHANGELOG.md
├── LICENSE
├── analysis_options.yaml
├── native/
│   └── llama_dart_bridge/
│       ├── include/llama_dart.h
│       ├── src/
│       ├── CMakeLists.txt
│       └── tests/
├── packages/
│   ├── llama_cpp_dart/
│   │   ├── lib/
│   │   │   ├── llama_cpp_dart.dart
│   │   │   └── src/
│   │   │       ├── ffi/generated_bindings.dart
│   │   │       ├── engine/
│   │   │       ├── model/
│   │   │       ├── generation/
│   │   │       ├── embeddings/
│   │   │       ├── rag/
│   │   │       ├── multimodal/
│   │   │       └── errors.dart
│   │   ├── hook/
│   │   ├── test/
│   │   └── ffigen.yaml
│   └── llama_cpp_flutter/
│       ├── lib/
│       ├── android/
│       ├── ios/
│       ├── example/
│       └── test/
├── third_party/
│   └── llama.cpp/          # pinned submodule or vendored source
├── docs/
│   ├── architecture.md
│   ├── feature_matrix.md
│   ├── mobile_performance.md
│   ├── native_builds.md
│   ├── rag.md
│   └── upstream_sync.md
├── examples/
│   ├── flutter_chat/
│   ├── flutter_multimodal/
│   ├── flutter_rag/
│   └── dart_cli/
└── benchmarks/
    ├── mobile/
    └── desktop_smoke/
```

Keep the core Dart package usable from non-Flutter Dart where feasible. Keep Flutter-specific model pickers, storage helpers, permission handling, widgets, and platform integrations in a Flutter companion package or a clearly separated Flutter layer.

## Upstream `llama.cpp` policy

Pin upstream `llama.cpp` to an exact commit. Record the commit hash, sync date, build flags, and known feature status in `docs/upstream_sync.md` and `docs/feature_matrix.md`.

When syncing upstream:

1. Read upstream `include/llama.h`, relevant `common/` helpers, `tools/server/README.md`, multimodal docs, speculative decoding docs, and build docs for changed APIs or behavior.
2. Update the bridge only after confirming the current upstream C/C++ API. Avoid depending on unstable internal symbols unless there is no public alternative and the risk is documented.
3. Regenerate Dart FFI bindings from the bridge header, not directly from all of `llama.h` unless the project explicitly standardizes on that.
4. Run native tests, Dart tests, Flutter tests, and at least one smoke test for model load and token generation.
5. Update the feature matrix, examples, benchmark metadata, and CHANGELOG.

Do not track upstream `master` implicitly in a published package. Users must be able to reproduce the native build from a tagged package version.

## Native bridge architecture

The bridge is a small C ABI facade around `llama.cpp`. It should contain the complexity needed to make the Dart layer safe, but it must not become a fork of `llama.cpp`.

Bridge requirements:

- Use C-compatible exported functions, opaque handles, fixed-width integer types, explicit buffer lengths, and versioned structs.
- Include an ABI version function such as `llama_dart_abi_version()` and a runtime `llama_dart_get_capabilities()` call.
- Return explicit result codes. Never throw C++ exceptions across the C ABI.
- Convert all native exceptions, assertions that can be handled, and upstream errors into bridge errors.
- Provide a thread-safe last-error mechanism or result-owned error message buffers.
- Avoid global mutable state except required upstream backend initialization. Guard initialization and teardown.
- Make ownership clear: every handle returned by the bridge must have exactly one documented free/release function.
- Keep handles opaque to Dart. The Dart layer should never interpret native pointers beyond passing them back to the bridge.
- Hide internal symbols where possible. Export only the bridge ABI.
- Prefer UTF-8 paths and text. Always pass lengths with byte buffers; never rely on NUL-terminated Dart-owned strings unless explicitly allocated for that purpose.
- Avoid callbacks for high-frequency token streaming unless they are measured and safe. Prefer a native worker plus polling/event queue or batched callbacks. Do not call into Dart from arbitrary native threads without using the supported Dart native callback mechanism correctly.

Suggested bridge objects:

- `llama_dart_library` or process-level backend initializer.
- `llama_dart_model` for loaded GGUF model state.
- `llama_dart_context` for a context/KV cache/session.
- `llama_dart_generation` for an active generation job.
- `llama_dart_embedding_job` for embedding batches.
- `llama_dart_lora_adapter` for loaded LoRA adapter handles.
- `llama_dart_multimodal_projector` or equivalent if upstream requires a separate mmproj object.
- `llama_dart_speculative_config` for draft model, EAGLE-3, MTP, and n-gram speculative decoding options.

## Dart API principles

The public Dart API should be small, typed, and difficult to misuse. Generated FFI bindings belong under `src/ffi/` and must not be public API.

Use explicit lifecycle management:

```dart
final engine = await LlamaEngine.load(
  LlamaModelConfig(
    modelPath: modelPath,
    contextSize: 4096,
    gpu: GpuConfig.auto(),
  ),
);

try {
  await for (final chunk in engine.chat(
    messages: [ChatMessage.user('Write a haiku about local inference.')],
    config: const GenerationConfig(maxTokens: 128, temperature: 0.7),
  )) {
    // Render chunk.text in the UI layer.
  }
} finally {
  await engine.close();
}
```

API requirements:

- Every native-backed object must expose `close()` or `dispose()` and must guard against use after close.
- Dart `Finalizer`s are a safety net only. Tests should close resources explicitly.
- Long-running work must return `Future` or `Stream` and execute outside the UI isolate.
- Streaming APIs should support cancellation, backpressure, and coalescing chunks to avoid excessive UI rebuilds.
- Use typed configs rather than `Map<String, dynamic>` for public APIs. Maps are acceptable only for pass-through metadata or OpenAI-compatible request/response adapters.
- Provide typed exceptions: `LlamaException`, `ModelLoadException`, `ContextCreateException`, `GenerationException`, `EmbeddingException`, `UnsupportedFeatureException`, `NativeOutOfMemoryException`, `ResourceDisposedException`, and `CancelledException`.
- Expose model and runtime capabilities before feature use. For example, callers should be able to check whether multimodal, embeddings, reranking, LoRA, GPU offload, or a speculative strategy is available.
- Keep logs configurable. Do not print directly from library code.
- Provide OpenAI-shaped DTO adapters only as adapters. Do not let OpenAI compatibility dictate the core API if it would make local mobile use worse.

## Required feature coverage

Implement features incrementally, but design the API so these capabilities fit without breaking changes.

| Area | Expected support | Notes |
|---|---|---|
| Model loading | GGUF model load, metadata read, architecture info, tokenizer metadata, context size, batch size, GPU/offload config, memory estimates where feasible | File paths must work with app-private storage. On Android, `ContentResolver` URIs usually need copying to app-private files unless the bridge supports file descriptors. |
| Tokenization | Tokenize, detokenize, count tokens, BOS/EOS handling, chat template helpers | Preserve model-specific behavior. Avoid hand-written tokenization. |
| Completion | Raw prompt completion, chat completion, prefill, streaming, cancellation, stop strings/tokens, max tokens, seeds | Separate prompt formatting from generation. |
| Sampling | Temperature, top-k, top-p, min-p, typical-p, repeat penalties, presence/frequency penalties, Mirostat if supported, sampler chains | Keep config names close to upstream where possible but Dart-idiomatic. |
| Structured output | JSON mode, JSON schema or grammar/GBNF support where upstream supports it | Validate with tests. Document model limitations. |
| Function/tool calling | Tool definitions, tool-call parsing, parallel-tool-call flag where supported by templates | Tool execution remains app-owned; the library only formats and parses. |
| Embeddings | Embedding models, pooled embeddings, batch embedding, normalization metadata, `Float32List` output | Do not allocate nested Dart lists for large vector batches. |
| Reranking | Cross-encoder/reranker models where upstream supports rank pooling/rerank semantics | Expose as a separate API from embeddings. |
| RAG | Chunking interfaces, local vector index interface, simple in-memory baseline index, optional reranking, prompt assembly | Do not force a heavy vector DB dependency. Keep documents local by default. |
| Multimodal | Text + image/audio/video inputs where the pinned upstream supports them, mmproj loading, media capability detection | Treat multimodal as feature-gated. Validate prompt markers/templates per model family. |
| LoRA | Load adapters, unload adapters, list adapters, set global or per-request scales where feasible | Document batching/performance impact when adapter configs differ. |
| Speculative decoding | None, draft model, EAGLE-3, n-gram strategies, and other upstream-supported modes | Expose metrics such as accepted draft tokens and acceptance rate when available. |
| MTP | MTP speculative strategy when the model or separate MTP GGUF is available | Do not silently enable. Detect support and provide clear fallback errors. |
| KV/session features | Prompt cache/session reuse, context shift, KV cache clear/copy where supported | Critical for mobile latency and chat UX. |
| Performance telemetry | Load time, prompt eval tok/s, decode tok/s, time to first token, peak native memory if available, backend/device info | Include upstream commit and build flags in benchmark output. |
| Model management | File validation, checksum helpers, metadata preview, safe deletion helpers if requested | No model downloads by default. No license-bypassing workflows. |

## RAG requirements

RAG is an orchestration layer over local primitives. Keep it modular.

Required abstractions:

- `Document`: id, text, metadata, optional source URI/path.
- `TextChunk`: document id, chunk id, text, token count, metadata.
- `TextSplitter`: deterministic chunking by tokens or characters with overlap.
- `EmbeddingModel`: wraps local embedding inference.
- `VectorIndex`: add, remove, search, persist, load, and clear.
- `Retriever`: query-to-chunks with configurable top-k and score threshold.
- `Reranker`: optional reranking over candidate chunks.
- `RagPromptBuilder`: assembles retrieved context into a chat prompt without exceeding context budget.

Implementation guidance:

- Start with a simple in-memory flat index using cosine similarity or dot product, implemented over typed arrays.
- Keep persistent vector storage optional and interface-driven.
- Normalize vectors consistently and store whether vectors are normalized.
- Budget context in tokens, not characters.
- Provide citations/source spans in generated prompt context, but do not fabricate citations in model output.
- Add tests for chunking determinism, retrieval ranking, context-budget truncation, and metadata preservation.
- Keep all document processing local unless the user explicitly integrates a remote store.

## Multimodal requirements

Use upstream multimodal support rather than custom image/audio hacks.

Public API should support content parts such as:

- `TextPart`
- `ImagePart.fromFile`, `ImagePart.fromBytes`
- `AudioPart.fromFile`, `AudioPart.fromBytes` when supported
- `VideoPart.fromFile` or frame-based input when supported

Implementation guidance:

- Require a model capability check before multimodal requests.
- Require or auto-discover the mmproj file only when upstream supports that workflow. Expose explicit `mmprojPath` in config.
- Bound media sizes before native ingestion. Large images or audio should not be synchronously decoded on the UI isolate.
- Preserve model-specific prompt structure. OCR and VLM models often require specific templates; document known model families in `docs/multimodal.md`.
- Provide a small example app that selects an image and asks a local vision model a question.
- If a modality is unsupported by the pinned upstream version, return `UnsupportedFeatureException` with the needed upstream capability.

## Speculative decoding and MTP requirements

Expose speculative decoding as a typed config, not a collection of unrelated flags.

Suggested public shape:

```dart
sealed class SpeculativeDecodingConfig {
  const SpeculativeDecodingConfig();
}

final class NoSpeculativeDecoding extends SpeculativeDecodingConfig {
  const NoSpeculativeDecoding();
}

final class DraftModelSpeculation extends SpeculativeDecodingConfig {
  const DraftModelSpeculation({required this.draftModelPath});
}

final class Eagle3Speculation extends SpeculativeDecodingConfig {
  const Eagle3Speculation({required this.draftModelPath});
}

final class MtpSpeculation extends SpeculativeDecodingConfig {
  const MtpSpeculation({this.mtpModelPath});
}

final class NGramSpeculation extends SpeculativeDecodingConfig {
  const NGramSpeculation({required this.strategy});
}
```

Implementation guidance:

- Detect and report which speculative strategies are compiled and runtime-available.
- MTP requires model/head support. A model without MTP metadata must not be treated as MTP-capable.
- Draft models and EAGLE-3 models must be checked for tokenizer/architecture compatibility with the target model where upstream exposes the needed metadata.
- Provide metrics: draft tokens proposed, tokens accepted, acceptance rate, time spent drafting, time spent verifying, and net tok/s.
- Add tests for unsupported model combinations and clean fallback behavior.
- Document memory overhead clearly. Speculative decoding can improve throughput but may be inappropriate on memory-constrained phones.

## LoRA requirements

LoRA support must be explicit and observable.

Public API requirements:

- Load LoRA adapter from path.
- List loaded adapters with id/path/default scale.
- Set adapter scale globally where supported.
- Override adapter scales per generation request where supported.
- Unload/free adapters safely.

Implementation guidance:

- Validate adapter compatibility as much as upstream allows.
- Do not reload the base model to switch adapters unless upstream requires it; prefer loaded adapter handles and scale changes.
- Document performance consequences. Different LoRA configurations can reduce batching opportunities.
- Add tests for loading, scale changes, disabling via zero scale, and disposal order.

## Mobile platform requirements

### iOS

- Support physical iOS devices with `arm64` as the primary target.
- Support iOS simulator where practical, but do not block device support on simulator-only issues.
- Prefer Metal acceleration when available and validated. Provide CPU fallback.
- Package native artifacts as an `.xcframework` or through Flutter's current native asset/build-hook mechanism, depending on the package strategy.
- Do not download executable code or native libraries at runtime. Model files and adapters are data, but native code must be shipped through the app build.
- Avoid bitcode assumptions. Verify current Xcode defaults instead of copying old settings.
- Keep minimum iOS version explicit in docs and package metadata. If a feature requires a higher version, gate it.

### Android

- Support `arm64-v8a` as the mandatory production ABI.
- Consider `x86_64` only for emulator/developer convenience. Do not spend effort on `armeabi-v7a` unless explicitly requested.
- Use Android NDK/CMake builds. Keep `GGML_OPENMP=OFF` unless the project has verified a safe OpenMP packaging path.
- Use app-private model paths for best compatibility. If accepting shared-storage URIs, copy or map them through a safe, documented path.
- Provide CPU baseline builds. Treat Vulkan/OpenCL/GPU builds as feature-gated variants until tested across representative devices.
- Never globally compile with CPU flags that crash older supported devices. If optimized kernels are used, rely on upstream runtime dispatch or ship separate safe variants.
- Keep minimum Android API level explicit. If following upstream Android examples that use API 28, document that requirement and test it.

### Desktop and CI support

Desktop support is useful for tests and development, but mobile remains the main product requirement. Maintain macOS/Linux smoke tests if they help validate the FFI bridge quickly. Do not let desktop-only convenience leak into mobile runtime assumptions.

## Performance guidance

Mobile performance is a product requirement, not an afterthought.

Implementation rules:

- Keep all heavy native calls off the UI isolate.
- Prefer a dedicated inference isolate and a native worker model. Queue work per context; do not run concurrent decodes on the same context.
- Batch prompt ingestion and embedding inputs where possible.
- Stream generated text in UI-friendly chunks. Avoid rebuilding Flutter widgets for every byte or every token when batching improves UX.
- Use typed data (`Uint8List`, `Int32List`, `Float32List`) for large buffers. Avoid large `List<double>` and nested lists.
- Minimize UTF-8 conversion churn. Convert once per boundary and reuse buffers where safe.
- Cap default context size conservatively on mobile. A 4096-token default is a reasonable starting point unless the model requires otherwise.
- Expose tuning knobs: context size, batch size, thread count, batch thread count, GPU layers/offload mode, KV cache options, mmap/mlock where supported, and speculative decoding config.
- Provide safe `auto` defaults, but make them inspectable.
- Add a warm-up path for apps that want predictable first-token latency.
- Surface cancellation quickly. A cancelled generation should stop native work cooperatively and release the active job.
- Avoid excessive memory copies between Dart and native code.
- Watch thermal behavior. Benchmarks should include sustained generation, not only a short prompt.

Benchmark requirements:

- Benchmarks must report device model, OS version, app build mode, upstream commit, build flags, model name, quantization, context size, batch size, thread count, GPU/offload mode, prompt eval tok/s, decode tok/s, time to first token, and peak memory when available.
- Keep a reproducible benchmark harness under `benchmarks/`.
- Do not use benchmark-only flags in production defaults without documentation.
- Compare CPU baseline against accelerated builds before claiming a speedup.

## Build system guidance

Prefer Flutter/Dart's current FFI packaging path for new work: a `package_ffi`-style package with native assets/build hooks where supported. Use legacy plugin-specific FFI wiring only when compatibility requires it and document why.

Native build requirements:

- Use CMake for bridge and upstream `llama.cpp` builds.
- Keep platform flags centralized in scripts or build hook code.
- Make build output names stable: Android should load a predictable `.so`; Apple should expose a predictable framework/library name.
- Record all compile definitions used for release artifacts.
- Keep debug builds debuggable. Do not strip symbols in debug artifacts.
- Avoid runtime dependency surprises. If the native library depends on another native library, ensure it is packaged and loaded on the target platform.
- Add a `--no-network` or equivalent CI path that builds only from vendored/pinned sources.

Suggested checks:

```sh
dart format .
dart analyze --fatal-infos
dart test
flutter test
cmake -S native/llama_dart_bridge -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native --config Release
ctest --test-dir build/native --output-on-failure
```

For iOS and Android, add platform-specific CI jobs where the infrastructure permits:

```sh
flutter build ios --no-codesign --config-only
flutter build apk --debug
flutter build apk --release
```

If a command cannot run in the current environment because Xcode, Android SDK/NDK, CMake, or a model fixture is missing, report that exact limitation in the final response or PR notes. Do not mark it as passed.

## Generated bindings

Use `ffigen` or the standardized binding generation tool selected by the repository. Generate bindings from `native/llama_dart_bridge/include/llama_dart.h`.

Rules:

- Never hand-edit generated binding files.
- Keep `ffigen.yaml` committed.
- Keep generated bindings internal to the Dart package.
- Wrap every generated function in a checked high-level Dart method.
- Add tests for ABI version mismatch and missing symbols.
- Include a short comment in generated files identifying the generation command.

## Testing requirements

Testing must cover native lifecycle, Dart API behavior, and platform integration.

Minimum test classes:

- Pure Dart unit tests for config validation, prompt construction, RAG chunking, vector search, and error mapping.
- FFI smoke tests that load the native library and verify ABI version/capabilities.
- Native C/C++ tests for bridge object creation, error handling, invalid inputs, cancellation, and disposal.
- Integration tests with a tiny model fixture or a separately downloaded checksum-verified test model.
- Flutter example tests for UI integration and isolate behavior where feasible.
- Regression tests for every memory-safety bug fixed.

Model fixture rules:

- Do not commit large model files.
- Prefer tiny public GGUF fixtures with clear licenses and checksums for integration tests.
- If tests download fixtures, they must be opt-in or isolated from default offline CI.
- Always document exact model file, source, license, and checksum.

Memory and concurrency tests:

- Double-close must be safe.
- Use-after-close must throw a Dart `ResourceDisposedException`.
- Cancellation must complete promptly.
- Disposing a model while a context/generation is active must either be prevented or ordered safely.
- Parallel requests must be queued or rejected explicitly; they must not race the same native context.

## Documentation requirements

Update documentation in the same change that adds or changes behavior.

Required docs:

- `README.md`: concise installation, supported platforms, quick start, feature matrix link, model-file guidance, and safety notes.
- `docs/architecture.md`: native bridge, isolate model, lifecycle, and package boundaries.
- `docs/native_builds.md`: iOS/Android build flags, artifact layout, troubleshooting, and minimum versions.
- `docs/feature_matrix.md`: supported, experimental, unsupported, and upstream-dependent features.
- `docs/mobile_performance.md`: recommended model sizes, quantization guidance, context-size guidance, threading/offload tuning, and benchmark interpretation.
- `docs/rag.md`: local RAG concepts, embedding model selection, vector index options, and prompt budgeting.
- `examples/`: runnable examples for text chat, embeddings/RAG, and multimodal when supported.

Documentation must distinguish these states clearly:

- Supported and tested.
- Implemented but experimental.
- Supported by upstream but not exposed yet.
- Not supported by the pinned upstream version.
- Not appropriate for mobile runtime.

## Security, privacy, and licensing

- Keep inference local by default.
- Do not add telemetry, analytics, crash upload, or model download behavior without explicit opt-in API and documentation.
- Validate file paths and avoid surprising file deletion. Destructive model-management helpers must be explicit and well-tested.
- Treat GGUF/model files as untrusted inputs. Fail safely on invalid metadata or load errors.
- Do not bypass model licenses, gated model access, or provider terms.
- Keep third-party license notices for `llama.cpp`, ggml, build tools, and any added dependencies.
- Avoid dependencies that add network permissions, ad SDKs, telemetry, or large transitive trees.
- Keep Android and iOS permissions minimal. A local inference package should not require internet permission by default.

## Code style

Dart:

- Follow Effective Dart.
- Keep `analysis_options.yaml` strict. Do not silence lints globally to make a change pass.
- Prefer immutable config classes.
- Avoid `dynamic` in public API.
- Use `package:meta` annotations where helpful.
- Use `package:logging` or a small project logging abstraction; never `print` from library code.
- Keep examples simple and copy-pasteable.

Native C/C++:

- Match the upstream `llama.cpp` C++ standard and compiler requirements.
- Keep bridge code warning-clean where practical.
- Use RAII internally, but expose only C ABI functions.
- Do not throw through exported functions.
- Use sanitizers in native CI where available.
- Keep platform-specific code isolated behind small abstractions.
- Avoid broad `#ifdef` sprawl in core logic.

## Dependency policy

Add dependencies only when they materially improve correctness or maintainability.

Before adding a dependency, check:

- Is it necessary for mobile?
- Does it work on both iOS and Android?
- Does it add network permissions or large transitive dependencies?
- Is the license compatible?
- Can the behavior be implemented with Dart SDK, Flutter SDK, or a small internal utility instead?

For vector search, start simple. Do not add a native vector database dependency until there is a measured need and a clean mobile packaging story.

## Public API review checklist

Before changing public API, verify:

- Names are Dart-idiomatic and stable.
- The API does not leak native pointers or upstream-only jargon unless the jargon is unavoidable.
- Unsupported features fail with typed exceptions and clear messages.
- Asynchronous APIs cannot accidentally block the UI isolate.
- Cancellation and disposal behavior is documented.
- There is a migration path for existing users.
- README examples still compile.

## PR completion checklist for Codex

Before considering a task complete, run the relevant subset of:

```sh
dart format .
dart analyze --fatal-infos
dart test
flutter test
cmake --build build/native --config Release
ctest --test-dir build/native --output-on-failure
flutter build apk --debug
flutter build ios --no-codesign --config-only
```

Also verify:

- Native handles have clear ownership and tests for disposal.
- FFI bindings are regenerated if the bridge header changed.
- Feature matrix is updated if behavior changed.
- Examples are updated if public API changed.
- Benchmark metadata is updated if performance-sensitive code changed.
- CHANGELOG has an entry for user-visible changes.
- Any skipped command is reported with the exact reason.

## Common mistakes to avoid

- Creating a thin toy wrapper that only supports one prompt string and one model path.
- Putting all inference work on the UI isolate.
- Exposing raw `Pointer<Void>` handles in public API.
- Loading models from Android shared-storage URIs without a safe file-access plan.
- Assuming all GGUF chat templates behave the same.
- Claiming multimodal support without mmproj/model capability checks.
- Claiming MTP support without checking model/head availability.
- Enabling speculative decoding by default on memory-constrained devices.
- Adding a vector DB dependency before a simple typed-array index has been measured.
- Publishing artifacts built from an unpinned upstream commit.
- Swallowing native logs/errors and returning generic `Exception`.
- Committing large models, adapters, benchmark outputs, or generated build directories.

## When uncertain

Inspect the existing repository first. Then inspect the pinned upstream `llama.cpp` source. Make the smallest change that advances the requested task without weakening lifecycle safety, mobile performance, or API stability. If a feature is upstream-dependent or platform-dependent, implement capability detection and clear documentation rather than pretending it always works.


## Reference docs to consult when updating this repository

These links are reference starting points, not a substitute for inspecting the pinned source in this repository:

- OpenAI Codex AGENTS.md guidance: https://developers.openai.com/codex/guides/agents-md
- OpenAI Codex best practices: https://developers.openai.com/codex/learn/best-practices
- Dart C interop with `dart:ffi`: https://dart.dev/interop/c-interop
- Dart native assets/build hooks: https://dart.dev/tools/hooks
- Flutter FFI/native-code guidance: https://docs.flutter.dev/platform-integration/bind-native-code
- Flutter package/plugin development: https://docs.flutter.dev/packages-and-plugins/developing-packages
- llama.cpp build docs: https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md
- llama.cpp Android docs: https://github.com/ggml-org/llama.cpp/blob/master/docs/android.md
- llama.cpp server feature docs: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
- llama.cpp multimodal docs: https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md
- llama.cpp speculative decoding docs: https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md
- llama.cpp function-calling docs: https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md

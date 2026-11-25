# fllamer

Mobile-first local `llama.cpp` inference for Dart and Flutter.

`fllamer` runs models in-process through a versioned C ABI bridge and Dart
worker isolates—no external CLI or `llama-server` required. Inference is local
by default; model weights, `mmproj` files, and LoRA adapters are always
app-owned and never bundled or downloaded by the package.

> **Status:** The Dart API, typed error hierarchy, capability reporting, and
> Dart-only RAG primitives are stable. The native bridge and inference features
> are currently **experimental**. See the [feature matrix](doc/feature_matrix.md)
> for exact per-feature status.

## Features

- **Streaming chat & completion** — Off-UI-isolate generation via `LlamaEngine`
  with upstream chat-template formatting, thinking/reasoning separation and
  token budgets, stop strings/tokens, cancellation, context prefill/shift/reset,
  and versioned state snapshots ([architecture](doc/architecture.md)).
- **Structured output & tool calling** — GBNF grammars, JSON mode, pinned
  upstream JSON Schema constraints, and template-driven function/tool calling
  with post-parse schema validation
  ([structured output](doc/structured_output.md),
  [tool calling](doc/tool_calling.md)).
- **Embeddings, reranking & RAG** — Persistent `LlamaEmbeddingEngine` and
  one-shot `LlamaEmbeddings` returning flat `Float32List` batches, native
  rank-pooling reranking (`LlamaReranker`), and Dart-only RAG primitives
  (splitting, persistable vector index, token-budgeted prompt assembly, and
  typed citations) ([RAG](doc/rag.md)).
- **Multimodal** — Local image and audio chat from files or bytes using a
  matching `mmproj` file with worker-isolate preprocessing
  ([multimodal](doc/multimodal.md)).
- **LoRA & speculative decoding** — Runtime LoRA adapter management with
  per-request scale overrides ([LoRA](doc/lora.md)), plus n-gram, draft-model,
  EAGLE-3, and MTP speculative decoding
  ([speculative decoding](doc/speculative_decoding.md)).
- **Model inspection & validation** — Vocab-only GGUF inspection, bounded
  metadata reads, tokenization, GGUF magic/SHA-256 validation, and runtime
  capability and telemetry reporting.

## Platform support

The native-assets hook builds the bundled bridge with CMake automatically when
Dart or Flutter requests code assets.

| Platform | Backends | Build prerequisites | Status |
| --- | --- | --- | --- |
| **macOS** (arm64, x64) | CPU, Metal (embedded) | Xcode, CMake 3.16+ | Host smoke-tested |
| **iOS** (arm64, 15.0+) | CPU, Metal (embedded); Simulator defaults to CPU | Xcode (deployment target 15.0+), CMake 3.16+ | Simulator tested; physical device pending |
| **Android** (arm64-v8a, x86_64; API 28+) | CPU by default; opt-in Vulkan variant | Android NDK, CMake 3.16+ | CPU APK validated; Vulkan experimental |
| **Linux** (x64, arm64) | Vulkan by default (+ CPU); CPU-only opt-in | CMake 3.19+, C++ toolchain, Vulkan dev files, `glslc` | Build-policy tested; host runtime pending |
| **Windows** (x64, arm64) | Vulkan by default (+ CPU); CPU-only opt-in | Visual Studio, Ninja, CMake 3.19+, Vulkan SDK, `glslc` | Build-policy tested; host runtime pending |

See [native builds](doc/native_builds.md) for ABI details, Vulkan opt-in/opt-out
configuration, and platform build commands.

## Installation

Add `fllamer` to your `pubspec.yaml`:

```yaml
dependencies:
  fllamer: ^0.4.0
```

Published packages include the pinned `llama.cpp` sources needed by the native
build. If you are working from a Git checkout instead, initialize the submodule
before resolving dependencies:

```sh
git submodule update --init --checkout
dart pub get
```

## Quick start

### Inspect a model and stream chat

```dart
import 'package:fllamer/fllamer.dart';

Future<void> main() async {
  const config = LlamaModelConfig(
    modelPath: '/app/private/model.gguf',
  );

  final info = await LlamaModel.inspect(config);
  print('${info.parameterCount} parameters');

  final engine = await LlamaEngine.load(config);
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

### Generate text embeddings

Use `LlamaEmbeddings.embedText` for one-off calls, or `LlamaEmbeddingEngine` to
keep an embedding model loaded across repeated batches:

```dart
final embedding = await LlamaEmbeddings.embedText(
  const LlamaModelConfig(modelPath: '/app/private/embedding-model.gguf'),
  'local retrieval context',
);
print('${embedding.length} dimensions');
```

## Native logging

Native logging is silent by default. Enable bounded in-memory capture only when
needed, drain records into your app's logger, and disable it afterward:

```dart
LlamaRuntime.configureNativeLogging(LlamaLogLevel.warning);
final records = LlamaRuntime.drainNativeLogs();
LlamaRuntime.configureNativeLogging(LlamaLogLevel.disabled);
```

Capture is process-wide and capped at 1 MiB or 4,096 records (oldest dropped
first). Native messages may contain local file paths; redact them before
persisting or displaying logs.

## Known limitations

- **Video input** is unsupported (upstream requires external `ffmpeg` tools).
- **External JSON Schema `$ref`** and schema keywords outside the pinned
  upstream converter are unsupported.
- **Older mobile targets** are unsupported: Android requires API 28+ (`arm64-v8a`
  or `x86_64`, no 32-bit `armeabi-v7a`) and iOS requires deployment target
  15.0+.
- **Cross-architecture Linux/Windows builds** are unsupported; build x64 on x64
  and arm64 on arm64 so the Vulkan shader generator can execute during the
  build.
- **Mobile GPU qualification is incomplete:** physical-device iOS Metal runtime
  testing is pending, and Android Vulkan remains opt-in and unqualified.

## Privacy & safety

- **100% local:** `fllamer` never downloads models, calls remote APIs, or
  requests Android internet permission.
- **Untrusted file validation:** Validate user- or app-supplied GGUF files with
  `LlamaModel.validateFile()` and `LlamaModel.sha256()` before loading.
- **Symlink protection:** `LlamaModel.deleteFile()`,
  `LlamaEngine.saveStateToFile()` / `restoreStateFromFile()`, and
  `InMemoryVectorIndex.persist()` / `load()` refuse symbolic links.

## Documentation & examples

| Topic | Document |
| --- | --- |
| Full feature status | [doc/feature_matrix.md](doc/feature_matrix.md) |
| C ABI, isolates, lifecycle, cancellation | [doc/architecture.md](doc/architecture.md) |
| Native assets, platforms, smoke tests | [doc/native_builds.md](doc/native_builds.md) |
| Tool / function calling | [doc/tool_calling.md](doc/tool_calling.md) |
| GBNF grammars & JSON Schema | [doc/structured_output.md](doc/structured_output.md) |
| Image & audio chat | [doc/multimodal.md](doc/multimodal.md) |
| Local RAG & embeddings | [doc/rag.md](doc/rag.md) |
| LoRA adapters | [doc/lora.md](doc/lora.md) |
| Speculative decoding & MTP | [doc/speculative_decoding.md](doc/speculative_decoding.md) |
| Mobile performance & benchmarks | [doc/mobile_performance.md](doc/mobile_performance.md) |
| Upstream `llama.cpp` sync notes | [doc/upstream_sync.md](doc/upstream_sync.md) |
| Flutter & Dart CLI examples | [example/](example/) |

## Native bridge checks

When modifying `native/llama_dart_bridge`, regenerate bindings and run native
tests:

```sh
dart run ffigen --config ffigen.yaml
dart run ffigen --config ffigen.native_assets.yaml
cmake -S native/llama_dart_bridge -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native --config Release
ctest --test-dir build/native --output-on-failure
```

## Licensing

`fllamer` is licensed under **GPL-3.0**.

The repository pins `llama.cpp` at
`b1b770d90d94cad9e9ea5b993bdbc383a522fb28` (`v1.0.0`), and published packages
include its required source files. Redistributed builds must preserve the
upstream notices in `third_party/llama.cpp/LICENSE`,
`third_party/llama.cpp/AUTHORS`, and `third_party/llama.cpp/licenses/`.
`pubspec.yaml` registers copies of these and the bundled third-party notices,
written to `third_party/flutter_notices/` with the names each covers, under
`flutter.licenses` for Flutter's generated license bundle; distributors remain
responsible for verifying GPL-3.0 compatibility and notice compliance. Model
weights, `mmproj` files, and LoRA adapters are app-supplied data governed by
their own licenses.

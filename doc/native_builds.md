# Native Builds

Current local build:

```sh
cmake -S native/llama_dart_bridge -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native --config Release
ctest --test-dir build/native --output-on-failure
```

Package dry run:

```sh
dart pub publish --dry-run
```

The latest local dry run completed online validation with zero warnings and
assembled a 3 MiB compressed archive. The root `.pubignore` keeps local build
state, native test fixtures, conversion helpers, and unrelated upstream source
out of the published package.

Generated bindings:

```sh
dart run ffigen --config ffigen.yaml
dart run ffigen --config ffigen.native_assets.yaml
```

The first file preserves runtime `DynamicLibrary` lookup for explicit custom
bridge paths. The second generates `@Native` symbol addresses for the bundled
asset ID `package:fllamer/llama_dart_bridge`; both are internal implementation
details and must be regenerated after a bridge-header change.

The bridge builds against the curated, vendored `third_party/llama.cpp`
snapshot recorded in `doc/upstream_sync.md`. Do not point published builds at
upstream `master`.
`LLAMA_DART_NO_NETWORK=ON` is the default CMake path: it enables CMake's
disconnected fetch mode and disables upstream's optional external LLGuidance
project. The bridge does not require dependency or executable downloads.
`LlamaRuntime.currentCapabilities()` reports the bridge upstream commit and the
baseline native build flags, including `CMAKE_BUILD_TYPE`, exposed by the
loaded library.

For local development and explicit runtime testing, the Dart layer can still
open a native library by:

- pass `nativeLibraryPath`,
- set `FLLAMER_NATIVE_LIBRARY`.

When neither override is present, the Dart layer resolves the code asset by
its registered asset ID. It does not guess a platform filename; this is what
allows Flutter's Apple framework layout to work without an application-owned
path workaround.

## Native-assets hook

`hook/build.dart` builds the CMake bridge when a Dart or Flutter build requests
code assets. The hook emits a bundled dynamic library named
`llama_dart_bridge` and registers it as a `DynamicLoadingBundled` code asset.

The hook uses the same CMake project as the local smoke build, with
`BUILD_TESTING=OFF`. Native-assets inputs do not expose the enclosing
Flutter/Dart debug, profile, or release mode, so the hook uses
`RelWithDebInfo` for every bundled bridge: native code remains optimized while
retaining debug information for local symbolication. Final release packaging
may still strip the copied application artifact. CMake build parallelism is
bounded to four jobs by default to avoid scaling native compiler memory with a
high-core-count host; set `FLLAMER_BUILD_JOBS` to a positive integer when a CI
runner needs a different explicit limit. Command output is streamed instead of
retained in hook memory. It keeps the portable CPU flags from
`native/llama_dart_bridge/CMakeLists.txt`, including `GGML_OPENMP=OFF` and
`GGML_LLAMAFILE=OFF`. `LLAMA_BUILD_MTMD=ON` links image/audio preprocessing
into the bridge, while `MTMD_VIDEO=OFF` prevents runtime ffmpeg dependencies.
`LLAMA_BUILD_COMMON=ON` statically links the pinned upstream speculative
runtime used by draft-model, EAGLE-3, and MTP strategies. The bridge does not
call llama-common download helpers; model paths remain app supplied and local.
Apple targets also set `GGML_METAL=ON` and
`GGML_METAL_EMBED_LIBRARY=ON` so Metal kernels stay inside the bundled
library. Other upstream network/tool/UI and accelerator backends remain
disabled. A custom build can opt into Vulkan with
`-DLLAMA_DART_ENABLE_VULKAN=ON` when its SDK is available.

Target notes:

- Android requires an installed NDK. The verified minimum Android API is 28 in
  `example/android/app/build.gradle.kts`. The hook supports `arm64-v8a` for
  devices and `x86_64` for emulator/developer builds. It rejects `armeabi-v7a`,
  `x86`, and `riscv64` until those ABIs have an explicit support plan. The hook
  looks at the compiler path supplied by the native-assets build config, then
  `ANDROID_NDK`, `ANDROID_NDK_HOME`, `ANDROID_NDK_LATEST_HOME`,
  `ANDROID_NDK_ROOT`, then the newest numeric version under
  `ANDROID_HOME/ndk/*`. It passes the Android CMake toolchain, target ABI, NDK
  API level from the build config, and `c++_static`.
- iOS 15.0 is the explicit minimum in the example Xcode project. Flutter's
  native-assets driver currently supplies a generic iOS 13 hook target even
  when the Xcode project has a newer deployment target, so the hook raises the
  native bridge's effective CMake deployment target to 15.0. Applications must
  still set their own iOS deployment target to 15.0 or newer; the hook cannot
  rewrite an application's Xcode support policy. The example
  integrates Flutter plugins through Swift Package Manager and does not require
  CocoaPods. The pinned embedded Metal scheduler uses
  `MTLSharedEvent.waitUntilSignaledValue`, which was introduced in iOS 15. The
  hook passes `CMAKE_SYSTEM_NAME=iOS`, the target SDK and architecture, and the
  greater of the native-assets target or iOS 15.0. Xcode command line tools
  must be available.
- macOS and Linux host builds are supported by the same hook. Windows is not
  configured yet.

Model files remain app-owned data. The hook only packages native executable
code that is built from the vendored sources.

### Android model paths

The bridge accepts filesystem paths, not `content://` URIs. When a model,
mmproj, or LoRA adapter is selected through Android's Storage Access Framework,
copy the `ContentResolver` stream to an app-private persistent file, validate
its expected size/GGUF magic/checksum, and pass that local path to `fllamer`.
Do not retain the provider URI as `modelPath`, and do not depend on a picker
plugin's temporary cache copy surviving process restarts or storage pressure.

The example uses `file_selector`, whose Android implementation creates a
sanitized temporary cache copy for selected content. That is sufficient for an
interactive demonstration; production apps should move or copy it into their
own durable app-private model directory before loading it again later.

Current verification:

- The hook has host smoke-test coverage through `dart test` and `flutter test`.
- Native `ctest` always covers ABI/error handling; its assertions remain active
  in Release builds, and an ASan/UBSan Debug build passes locally.
  Fixture-backed vocab/context checks run when
  `third_party/llama.cpp/models/ggml-vocab-gpt-2.gguf` exists.
- Apple `ctest` asks the public Metal backend capability path about representative
  supported types and missing-kernel types, including TQ1_0/TQ2_0, so a future
  broad capability claim cannot silently reintroduce null-pipeline dispatch.
- Set `LLAMA_DART_TEST_MODEL` to an app-owned weighted GGUF before `dart test`
  to run opt-in warm-up, prompt-only prefill and continuation,
  explicit/automatic context-shift continuation, and deterministic stop-token
  generation coverage.
- Set `LLAMA_DART_TEST_EMBEDDING_MODEL` to the pinned Apache-2.0
  `all-MiniLM-L6-v2-Q4_K_M.gguf` fixture to run real single and same-context
  batch embedding coverage. The fixture is 20,999,104 bytes with SHA-256
  `2ec4cee28a27a9c973d5f5230930d6ef6e52694bd2bc71be26a9bef5b1d755e6`
  from
  `second-state/All-MiniLM-L6-v2-Embedding-GGUF` revision
  `544f204f2eaa2d71361ffc74d6df7170285b286a`:

  ```sh
  hf download second-state/All-MiniLM-L6-v2-Embedding-GGUF \
    all-MiniLM-L6-v2-Q4_K_M.gguf \
    --revision 544f204f2eaa2d71361ffc74d6df7170285b286a \
    --local-dir /tmp/fllamer-embedding-fixture
  LLAMA_DART_TEST_EMBEDDING_MODEL=/tmp/fllamer-embedding-fixture/all-MiniLM-L6-v2-Q4_K_M.gguf \
    dart test --plain-name 'real embedding fixture'
  ```

  The test validates the exact size and checksum before model loading. The
  default suite never downloads or packages either opt-in model.
- Set `LLAMA_DART_TEST_SPECULATIVE_TARGET` and
  `LLAMA_DART_TEST_SPECULATIVE_DRAFT` to the pinned 15M TinyLlama Q8 target and
  Q4 draft fixtures to run real model-backed speculative decoding. The GGUFs
  come from `ggml-org/tiny-llamas` revision
  `99dd1a73db5a37100bd4ae633f4cfce6560e1567`; the original
  `karpathy/tinyllamas` model is MIT-licensed.

  | File | Bytes | SHA-256 |
  | --- | ---: | --- |
  | `stories15M-q8_0.gguf` | 26,671,328 | `2eda49203f2f044f3dddf29a7dd7cc861ef5a0340f518a19613d73ba6d9c06b6` |
  | `stories15M-q4_0.gguf` | 19,077,344 | `6151b1929d7f5aa3385d9ddef3393e55587c0a55de661562322bc51dfda93a04` |

  ```sh
  hf download ggml-org/tiny-llamas \
    --include 'stories15M-q8_0.gguf' \
    --include 'stories15M-q4_0.gguf' \
    --revision 99dd1a73db5a37100bd4ae633f4cfce6560e1567 \
    --local-dir /tmp/fllamer-speculative-fixture
  LLAMA_DART_TEST_SPECULATIVE_TARGET=/tmp/fllamer-speculative-fixture/stories15M-q8_0.gguf \
  LLAMA_DART_TEST_SPECULATIVE_DRAFT=/tmp/fllamer-speculative-fixture/stories15M-q4_0.gguf \
    dart test --plain-name 'real target and draft fixtures'
  ```

  The test compares greedy output against non-speculative target generation,
  requires real draft proposals and accepted tokens, and closes both engines.
  These fixtures are test data only and are never redistributed by `fllamer`.
- Set `LLAMA_DART_TEST_EAGLE_TARGET` and `LLAMA_DART_TEST_EAGLE_DRAFT` to the
  checksum-pinned Apache-2.0 Qwen3 1.7B Q2_K target and EAGLE-3 BF16 draft to
  run real EAGLE-3 coverage. The target comes from
  `unsloth/Qwen3-1.7B-GGUF` revision
  `d7f544eead698dbd1f15126ef60b45a1e1933222`. The draft is converted with the
  vendored `llama.cpp` converter from `AngelSlim/Qwen3-1.7B_eagle3` revision
  `94441b48acc5804677ae12259617c83323b543a9` and target tokenizer/config files
  from `Qwen/Qwen3-1.7B` revision
  `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`:

  | File | Bytes | SHA-256 |
  | --- | ---: | --- |
  | `Qwen3-1.7B-Q2_K.gguf` | 777,796,160 | `62b3fb705434cb57fabc59d59aa7b4c6fb558fff7c7c4b2ce67456373bc30fd3` |
  | `Qwen3-1.7B-eagle3-bf16.gguf` | 279,901,088 | `3785856376f55cc634e637ff4bdbf8b319e62b78680de117eaf3ed728e26b526` |

  ```sh
  hf download AngelSlim/Qwen3-1.7B_eagle3 \
    --revision 94441b48acc5804677ae12259617c83323b543a9 \
    --local-dir /tmp/fllamer-eagle3-source
  hf download Qwen/Qwen3-1.7B \
    config.json generation_config.json merges.txt tokenizer.json \
    tokenizer_config.json vocab.json \
    --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
    --local-dir /tmp/fllamer-eagle3-target-hf
  hf download unsloth/Qwen3-1.7B-GGUF Qwen3-1.7B-Q2_K.gguf \
    --revision d7f544eead698dbd1f15126ef60b45a1e1933222 \
    --local-dir /tmp/fllamer-eagle3-fixture
  UV_CACHE_DIR=/tmp/fllamer-uv-cache uv run --isolated --python 3.12 \
    --with-requirements third_party/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt \
    python third_party/llama.cpp/convert_hf_to_gguf.py \
    /tmp/fllamer-eagle3-source \
    --target-model-dir /tmp/fllamer-eagle3-target-hf \
    --outtype bf16 \
    --outfile /tmp/fllamer-eagle3-fixture/Qwen3-1.7B-eagle3-bf16.gguf
  LLAMA_DART_TEST_EAGLE_TARGET=/tmp/fllamer-eagle3-fixture/Qwen3-1.7B-Q2_K.gguf \
  LLAMA_DART_TEST_EAGLE_DRAFT=/tmp/fllamer-eagle3-fixture/Qwen3-1.7B-eagle3-bf16.gguf \
    dart test --plain-name 'real EAGLE-3 fixtures'
  ```

  Conversion dependencies are fixture-preparation tools only; they are not
  package or application runtime dependencies. The test validates both files,
  compares deterministic greedy output with ordinary decoding, requires draft
  proposals and accepted tokens, checks timing telemetry, and closes both
  engines.
- Set `LLAMA_DART_TEST_MTP_MODEL` to the pinned Apache-2.0 Qwen3.5 0.8B
  Q4_K_M GGUF to run real integrated-MTP coverage. The 549,698,976-byte file
  has SHA-256
  `ac7c9d7a1b3e3695bb3bd50f8ceaa97f9c93e99ccc3d3d1a620301b6dd6d3d86`
  and is pinned at `unsloth/Qwen3.5-0.8B-MTP-GGUF` revision
  `cf8a611f6ed2c2060046219a19f12cd3d5ecd67c`:

  ```sh
  hf download unsloth/Qwen3.5-0.8B-MTP-GGUF \
    Qwen3.5-0.8B-Q4_K_M.gguf \
    --revision cf8a611f6ed2c2060046219a19f12cd3d5ecd67c \
    --local-dir /tmp/fllamer-mtp-fixture
  LLAMA_DART_TEST_MTP_MODEL=/tmp/fllamer-mtp-fixture/Qwen3.5-0.8B-Q4_K_M.gguf \
    dart test --plain-name 'real integrated MTP fixture'
  ```

  The test validates the exact file before loading, verifies that lightweight
  inspection exposes its single integrated NextN layer, compares deterministic
  greedy output with ordinary decoding, requires proposals and accepted
  tokens, checks draft/verification telemetry, and closes both engines. The
  default suite never downloads or packages the model.
- Set `LLAMA_DART_TEST_RERANKER_MODEL` to the pinned Apache-2.0
  `jina-reranker-v1-tiny-en` F16 GGUF to run real pair, same-context batch, and
  RAG reranker coverage. The 67,504,480-byte fixture has SHA-256
  `ad9f450c1053a431e2e3746d1f9f7768fb9183cf0796e046983a3449dca093c2`
  and comes from `ggml-org/models` revision
  `499bc8821c6b12b4e53c5bffcb21ec206f212d81`:

  ```sh
  hf download ggml-org/models \
    jina-reranker-v1-tiny-en/ggml-model-f16.gguf \
    --revision 499bc8821c6b12b4e53c5bffcb21ec206f212d81 \
    --local-dir /tmp/fllamer-reranker-fixture
  LLAMA_DART_TEST_RERANKER_MODEL=/tmp/fllamer-reranker-fixture/jina-reranker-v1-tiny-en/ggml-model-f16.gguf \
    dart test --plain-name 'real reranker fixture'
  ```

  The test validates size and checksum before loading and requires finite,
  deterministic scores with relevant-over-unrelated ranking. The default suite
  never downloads or packages the model.
- Set `LLAMA_DART_TEST_LORA_MODEL` and `LLAMA_DART_TEST_LORA_ADAPTER` to the
  pinned MIT-licensed 15M MoE Q8 base and Shakespeare LoRA adapter to run real
  load/list/global-scale/request-override/unload coverage. Both files come from
  `ggml-org/stories15M_MOE` revision
  `b6dd737497465570b5f5e962dbc9d9454ed1e0eb`:

  | File | Bytes | SHA-256 |
  | --- | ---: | --- |
  | `stories15M_MOE-Q8_0.gguf` | 39,390,272 | `c7aa6863f9a4b3cdf19716e2c95622dcbd3bd06989324bf1ac8e60486ef8e881` |
  | `moe_shakespeare15M.gguf` | 16,364,896 | `d1e0617d7e10de960639d18a4620ec8c6bb56343f45692830d3634a1a3e1fe1a` |

  ```sh
  hf download ggml-org/stories15M_MOE \
    --include 'stories15M_MOE-Q8_0.gguf' \
    --include 'moe_shakespeare15M.gguf' \
    --revision b6dd737497465570b5f5e962dbc9d9454ed1e0eb \
    --local-dir /tmp/fllamer-lora-fixture
  LLAMA_DART_TEST_LORA_MODEL=/tmp/fllamer-lora-fixture/stories15M_MOE-Q8_0.gguf \
  LLAMA_DART_TEST_LORA_ADAPTER=/tmp/fllamer-lora-fixture/moe_shakespeare15M.gguf \
    dart test --plain-name 'real LoRA fixture'
  ```

  The test validates both files before loading and proves that scale zero,
  per-request disable, and unload reproduce baseline generation while global
  scale one changes it. The default suite never downloads or packages either
  file.
- Set `LLAMA_DART_TEST_MULTIMODAL_MODEL`,
  `LLAMA_DART_TEST_MULTIMODAL_MMPROJ`, and
  `LLAMA_DART_TEST_MULTIMODAL_IMAGE` to the pinned TinyGemma3 CIFAR fixture to
  run real `mtmd` image-chat coverage. The fixture repository declares WTFPL
  and is pinned at `ggml-org/tinygemma3-GGUF` revision
  `c287502cd9e278dac8eed805c112cce5d0081e0b`:

  | File | Bytes | SHA-256 |
  | --- | ---: | --- |
  | `tinygemma3-Q8_0.gguf` | 47,227,552 | `7566ae7219c93ea2ecc692a931ee122d30c55261d0e2c3347acb8b939d2e9abd` |
  | `mmproj-tinygemma3.gguf` | 1,039,072 | `93c2ba8c34574dd8f2dfda64931fc20943de2f941bfe03e6e9eca68951b80604` |
  | `test/11_truck.png` | 3,134 | `2935c6e3f3d4d78284e8ab4fb89f271c057b77956f6fc3c43daf3d6374296108` |

  ```sh
  hf download ggml-org/tinygemma3-GGUF \
    --include 'tinygemma3-Q8_0.gguf' \
    --include 'mmproj-tinygemma3.gguf' \
    --include 'test/11_truck.png' \
    --revision c287502cd9e278dac8eed805c112cce5d0081e0b \
    --local-dir /tmp/fllamer-multimodal-fixture
  LLAMA_DART_TEST_MULTIMODAL_MODEL=/tmp/fllamer-multimodal-fixture/tinygemma3-Q8_0.gguf \
  LLAMA_DART_TEST_MULTIMODAL_MMPROJ=/tmp/fllamer-multimodal-fixture/mmproj-tinygemma3.gguf \
  LLAMA_DART_TEST_MULTIMODAL_IMAGE=/tmp/fllamer-multimodal-fixture/test/11_truck.png \
    dart test --plain-name 'real multimodal fixture'
  ```

  The test validates all three files before loading and verifies capability
  reporting, deterministic file/byte image preprocessing, streamed generation,
  prompt telemetry, and cleanup. The default suite never downloads or packages
  these files.
- Set `LLAMA_DART_TEST_GEMMA4_MODEL`,
  `LLAMA_DART_TEST_GEMMA4_MMPROJ`, and `LLAMA_DART_TEST_GEMMA4_IMAGE` to the
  Apache-2.0 Gemma 4 mobile fixture to verify that an undersized non-causal
  microbatch fails recoverably before upstream decode. The model repository is
  pinned at `unsloth/gemma-4-E2B-it-qat-mobile-GGUF` revision
  `ae6332216be5fea499f72bb6e484648ab3bdbb00`; the image is pinned test media
  from the vendored llama.cpp commit.

  | File | Bytes | SHA-256 |
  | --- | ---: | --- |
  | `gemma-4-E2B-it-qat-UD-Q2_K_XL.gguf` | 2,186,184,768 | `8279c8b153490e400831e89fc8162348911dfbe3c70d22055c70abaa9b05a0b4` |
  | `mmproj-BF16.gguf` | 986,833,728 | `38b33846f56426cd650e0e574d78de125abdfcedf35c0d7f6929f6ffe26efe02` |
  | `third_party/llama.cpp/tools/mtmd/test-1.jpeg` | 124,071 | `2dff664c0c8aaea18aff8cbe7e868845b775e90cdd7a0bac98df709b131deaa3` |

  ```sh
  hf download unsloth/gemma-4-E2B-it-qat-mobile-GGUF \
    gemma-4-E2B-it-qat-UD-Q2_K_XL.gguf mmproj-BF16.gguf \
    --revision ae6332216be5fea499f72bb6e484648ab3bdbb00 \
    --local-dir /tmp/fllamer-gemma4-fixture
  LLAMA_DART_TEST_GEMMA4_MODEL=/tmp/fllamer-gemma4-fixture/gemma-4-E2B-it-qat-UD-Q2_K_XL.gguf \
  LLAMA_DART_TEST_GEMMA4_MMPROJ=/tmp/fllamer-gemma4-fixture/mmproj-BF16.gguf \
  LLAMA_DART_TEST_GEMMA4_IMAGE=third_party/llama.cpp/tools/mtmd/test-1.jpeg \
    dart test --plain-name \
      'real Gemma 4 rejects an unsafe non-causal microbatch recoverably'
  ```

  The test validates all three files, loads a CPU context with
  `batchSize: 512` and `ubatchSize: 128`, and requires a typed
  `GenerationException` instead of a native assertion. The default suite never
  downloads or packages these files.
- Set `LLAMA_DART_TEST_AUDIO_MODEL`, `LLAMA_DART_TEST_AUDIO_MMPROJ`, and
  `LLAMA_DART_TEST_AUDIO_FILE` to run real `mtmd` audio-transcription coverage.
  The official conversion is derived from the Apache-2.0
  `Qwen/Qwen3-ASR-0.6B` model and pinned at
  `ggml-org/Qwen3-ASR-0.6B-GGUF` revision
  `928ab958557df9aa2ef1c93e0e83c7ad0933fae2`. The MP3 is pinned upstream test
  media from the vendored `llama.cpp` commit.

  | File | Bytes | SHA-256 |
  | --- | ---: | --- |
  | `Qwen3-ASR-0.6B-Q8_0.gguf` | 804,749,248 | `bca259818b50ca7c4c05e9bdb35a5dc04fa039653a6d6f3f0f331f96f6aa1971` |
  | `mmproj-Qwen3-ASR-0.6B-Q8_0.gguf` | 214,392,480 | `41a342b5e4c514e968cb756de6cd1b7be39eff43c44c57a2ef5fc6522e36603d` |
  | `third_party/llama.cpp/tools/mtmd/test-2.mp3` | 140,060 | `cdeac0ded280e18b99afbb7fac86130e4dda4b7d0b252cc22aa3d20679270a1e` |

  ```sh
  hf download ggml-org/Qwen3-ASR-0.6B-GGUF \
    Qwen3-ASR-0.6B-Q8_0.gguf \
    mmproj-Qwen3-ASR-0.6B-Q8_0.gguf \
    --revision 928ab958557df9aa2ef1c93e0e83c7ad0933fae2 \
    --local-dir /tmp/fllamer-audio-fixture
  LLAMA_DART_TEST_AUDIO_MODEL=/tmp/fllamer-audio-fixture/Qwen3-ASR-0.6B-Q8_0.gguf \
  LLAMA_DART_TEST_AUDIO_MMPROJ=/tmp/fllamer-audio-fixture/mmproj-Qwen3-ASR-0.6B-Q8_0.gguf \
  LLAMA_DART_TEST_AUDIO_FILE=third_party/llama.cpp/tools/mtmd/test-2.mp3 \
    dart test --plain-name 'real audio fixture'
  ```

  The test validates all three files, requires audio-only capability reporting,
  checks deterministic file/byte transcription content and telemetry, and
  closes the native model, projector, and context. The default suite never
  downloads or packages these files.
- Set `LLAMA_DART_TEST_TOOL_MODEL` to the official Apache-2.0 Qwen2.5 0.5B
  Instruct Q4_K_M GGUF to run weighted tool-call generation, native parsing,
  generated call-id, typed history, and correlated result-consumption coverage.
  The 491,400,032-byte file has SHA-256
  `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db`
  and is pinned at `Qwen/Qwen2.5-0.5B-Instruct-GGUF` revision
  `9217f5db79a29953eb74d5343926648285ec7e67`:

  ```sh
  hf download Qwen/Qwen2.5-0.5B-Instruct-GGUF \
    qwen2.5-0.5b-instruct-q4_k_m.gguf \
    --revision 9217f5db79a29953eb74d5343926648285ec7e67 \
    --local-dir /tmp/fllamer-tool-fixture
  LLAMA_DART_TEST_TOOL_MODEL=/tmp/fllamer-tool-fixture/qwen2.5-0.5b-instruct-q4_k_m.gguf \
    dart test --plain-name 'real tool fixture'
  ```

  The test validates the file before loading, requires the model's embedded
  template to report tool and call support, generates a named weather call,
  validates typed arguments and its stable id, then feeds a correlated tool
  result back for a normal assistant response. The default suite never
  downloads or packages the model.
- `example/` has Flutter widget coverage for the local model workflow and a
  compact phone viewport. It exercises app-owned GGUF/mmproj/image selection,
  model lifecycle controls, cancellable streaming chat, and telemetry in the
  UI while validating native-assets packaging.
- Android debug and release APK packaging pass from `example/` for the supported
  ABI set:

  ```sh
  flutter build apk --debug --target-platform android-arm64,android-x64
  flutter build apk --release --target-platform android-arm64,android-x64
  ```
- iOS Xcode configuration generation passes from `example/`:

  ```sh
  flutter build ios --no-codesign --config-only
  ```
- A direct offline CMake cross-compile of `llama_dart_bridge` for iOS 15 arm64
  passes with the embedded Metal backend. This validates the native source and
  deployment target independently of Flutter's Xcode destination selection.

Current limitations:

- `flutter build apk --debug` without `--target-platform` still asks Flutter's
  native-assets pipeline to build `android-arm`; the hook rejects that 32-bit
  ABI because this package only supports `arm64-v8a` and `x86_64`.
- Full iOS device and simulator builds were attempted from `example/` and are
  blocked by local Xcode platform availability: Xcode reports that the iOS
  26.5 platform component is not installed, even though its SDK is discoverable
  for the direct CMake cross-compile. The passing app-level check in this
  workspace is `flutter build ios --no-codesign --config-only`.
- Physical-device smoke tests have not been run in this workspace.
- Runtime loading still supports explicit `nativeLibraryPath` and
  `FLLAMER_NATIVE_LIBRARY`; app builds should verify the bundled library is
  discoverable on each target platform before release.

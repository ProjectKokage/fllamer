# Native Builds

Native CPU/Metal builds require CMake 3.16 or newer. Vulkan builds require
CMake 3.19 or newer, matching the pinned upstream shader build; the
native-assets hook checks this before configuration.

Current local build:

```sh
cmake -S native/llama_dart_bridge -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native --config Release
ctest --test-dir build/native --output-on-failure
```

Direct Linux and Windows CMake builds require Vulkan by default. Install the
target's Vulkan development files, SPIR-V headers, and `glslc`, or request a
deliberate CPU-only artifact with
`-DLLAMA_DART_ENABLE_VULKAN=OFF`. Configuration fails when Vulkan is enabled
but those build dependencies are incomplete; it never silently changes the
artifact to CPU-only.

Direct Android CMake builds remain CPU-only by default. An opt-in Vulkan build
sets `-DLLAMA_DART_ENABLE_VULKAN=ON` and supplies
`-DLLAMA_DART_ANDROID_VULKAN_HOST_ROOT=/path/to/fllamer/third_party/vulkan_headers`
plus the exact host `glslc` under the selected NDK's
`shader-tools/<host-tag>/` as `-DVulkan_GLSLC_EXECUTABLE`. It must also pass the
build-output directory prepared by the hook as
`-DLLAMA_DART_ANDROID_VULKAN_SHADER_OVERLAY_DIR`. The hook normalizes CRLF to
LF, verifies the exact pinned hashes of every transformed source, and refuses
unknown or already-patched input. The header root must contain
`Include/vulkan/vulkan.hpp` or `include/vulkan/vulkan.hpp` and the matching
Vulkan header set. The Android NDK remains authoritative for the target Vulkan
loader, bundled SPIR-V headers, and host shader compiler. CPU-only and desktop
builds neither prepare nor consume this overlay.

Package dry run:

```sh
dart pub publish --dry-run
```

Initialize the submodule before running the dry run and require zero warnings.
The root `.pubignore` keeps local build state, native test fixtures, conversion
helpers, and unrelated upstream source out of the published package. Inspect
the emitted file list and compressed size whenever the upstream pin changes.

Generated bindings:

```sh
dart run ffigen --config ffigen.yaml
dart run ffigen --config ffigen.native_assets.yaml
```

The first file preserves runtime `DynamicLibrary` lookup for explicit custom
bridge paths. The second generates `@Native` symbol addresses for the bundled
asset ID `package:fllamer/llama_dart_bridge`; both are internal implementation
details and must be regenerated after a bridge-header change.

The bridge builds against the exact `third_party/llama.cpp` submodule gitlink
recorded in `doc/upstream_sync.md`. Initialize repository checkouts with
`git submodule update --init --checkout`; published packages already contain
the required checked-out source files. Do not point builds at upstream
`master`.
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
library. Linux and Windows targets set `GGML_VULKAN=ON` by default and retain
the CPU backend. Android retains CPU by default and accepts an explicit Vulkan
variant. Other upstream network/tool/UI and accelerator backends remain
disabled.

The Vulkan policy is explicit native-assets input rather than an ambient
best-effort probe. In the consuming app's workspace-root `pubspec.yaml`, use:

```yaml
hooks:
  user_defines:
    fllamer:
      # Defaults to true on Linux/Windows and false on Android.
      vulkan: true
```

`vulkan` accepts only a boolean. `false` builds a reproducible CPU-only bridge.
On Linux and Windows, `true` is strict and fails CMake configuration if Vulkan,
`glslc`, or SPIR-V-Headers cannot be found. The optional `vulkan_sdk` directory
is forwarded as `Vulkan_ROOT` and `VULKAN_SDK` on those desktop targets. On
Android it overrides only the bundled header root; ordinary Android consumers
should omit it.

On Android, `vulkan: true` is self-contained: the hook uses the package's
complete Vulkan-Headers 1.4.357.0 `include/` tree, and derives host `glslc` from
the same Android NDK selected by Flutter. `vulkan_sdk` remains an optional
header-only override.
The hook never forwards either header root as `Vulkan_ROOT` or `VULKAN_SDK`, so
CMake cannot link a host loader into the Android artifact. The NDK supplies the
target `libvulkan.so` stub, SPIR-V headers, and host shader compiler. The hook
creates a hash-gated build-output overlay, leaving the submodule untouched:
five exact q-payload
load replacements cover Q4_0, Q4_1, and Q8_0 while preserving Q4_0/Q4_1
scale/min fields, and one exact `ggml-vulkan.cpp` transform installs the
Qualcomm-specific K-quant fallback policy below. CMake replaces only the four
generated shader units that include those shader files and the one Vulkan
source unit. Android Vulkan bridge metadata reports both
`GGML_VULKAN_ANDROID_SAFE_QUANT=1` and
`GGML_VULKAN_ANDROID_SAFE_K_QUANT=1`; all other builds report `0` for both.
When a later Android build disables Vulkan, the hook explicitly removes the
cached header-root, shader-overlay, and `glslc` entries before configuring the
CPU bridge. This keeps one native-assets cache from retaining a stale opt-in.
The hook records the shader sources, bundled or overridden Vulkan headers, and
exact NDK `glslc` as build inputs. It does not download executable code or
headers during the build.

The K-quant policy is deliberately narrow. It applies only when the Vulkan
vendor is Qualcomm and the driver ID is Qualcomm proprietary. For Q4_K it
suppresses the failing F32/F16 DMMV registrations and matvec route, forces the
dequant-to-F16 matrix route, and rejects `MUL_MAT`+`ADD` fusion. For Q5_K and
Q6_K it reports `MUL_MAT` and `MUL_MAT_ID` unsupported so the upstream
scheduler places those operations on the retained CPU backend. It is a
correctness fallback for that driver, not a generic K-quant acceleration or a
promise that an apparently fully offloaded model executes every operation on
Vulkan.

All Vulkan-enabled bridge variants retain the CPU backend, so an app may choose
`GpuConfig.cpu()` before loading a model or perform a separately owned CPU load
under its own bounded degradation policy. This is not an automatic retry of a
Vulkan-selected model load or compute failure. A Vulkan-enabled bridge still
has a target-system
dependency on the Vulkan loader (`vulkan-1.dll` on Windows or the platform
Vulkan loader on Linux, and `libvulkan.so` on Android). Systems without that
loader need the CPU-only artifact; they cannot load the Vulkan-linked bridge
merely to select CPU afterward.

Target notes:

- Android requires an installed NDK. The verified minimum Android API is 28 in
  `example/android/app/build.gradle.kts`. The hook supports `arm64-v8a` for
  devices and `x86_64` for emulator/developer builds. It rejects `armeabi-v7a`,
  `x86`, and `riscv64` until those ABIs have an explicit support plan. The hook
  looks at the compiler path supplied by the native-assets build config, then
  `ANDROID_NDK`, `ANDROID_NDK_HOME`, `ANDROID_NDK_LATEST_HOME`,
  `ANDROID_NDK_ROOT`, then the newest numeric version under
  `ANDROID_HOME/ndk/*`. It passes the Android CMake toolchain, target ABI, NDK
  API level from the build config, and `c++_static`. An enabled Vulkan build
  uses the package's pinned Vulkan-Hpp headers and that selected NDK's host
  `glslc`. Vulkan remains opt-in because it is an experimental, device-qualified
  path rather than a package default. The
  pinned backend requires a Vulkan 1.2-capable runtime before it can expose a
  usable device.
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
  must be available. On Apple Simulator targets, automatic GPU selection is
  normalized before model loading to an explicit CPU device with zero GPU
  layers. This also keeps context, KV, speculative, and multimodal projector
  work off Metal. Explicit Metal selection remains unchanged for diagnostics;
  physical-device Metal behavior must be validated on a physical device.
- The Linux hook is configured for native x64 and arm64 host builds.
  Debian/Ubuntu builders can provision the pinned upstream requirements with
  `libvulkan-dev`, `glslc`, and `spirv-headers`; other distributions need the
  equivalent loader development files, shader compiler, and SPIR-V headers.
  The optional LunarG SDK path above is also an accepted configuration input.
  The emitted code asset is `libllama_dart_bridge.so`.
- The Windows hook is configured for native x64 and arm64 host builds through
  the C compiler, linker, archiver, and Developer Command Prompt supplied by
  native-assets.
  Install a `flutter doctor -v`-accepted Visual Studio Desktop development with
  C++ toolchain, CMake 3.19 or newer, Ninja, and a LunarG Vulkan SDK containing
  headers, `vulkan-1.lib`, `glslc`, and SPIR-V headers. The hook configures
  Ninja with those exact native-assets tool paths and runs CMake inside the
  supplied Visual Studio environment. The emitted code asset is
  `llama_dart_bridge.dll`.
- Cross-architecture Linux and Windows builds are rejected. The pinned Vulkan
  build executes a target-built shader generator during compilation, so x64
  artifacts must be built on x64 and arm64 artifacts on arm64 until an
  independently provisioned host-toolchain contract is implemented.

Model files remain app-owned data. The hook only packages native executable
code that is built from the pinned source files.

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

- Model-free Dart tests cover the Linux/Windows strict-default Vulkan policy,
  Android's explicit opt-in, bundled Vulkan-Hpp pin, selected-NDK `glslc`,
  CPU-only overrides, target-specific header-root propagation, desktop
  architecture rejection,
  Windows Ninja/MSVC argument propagation, and Developer Command Prompt
  environment parsing. These policy tests do not constitute Vulkan runtime
  evidence.
- On 2026-08-11, direct offline CMake cross-compiles of the opt-in Vulkan
  bridge passed for `arm64-v8a` and `x86_64` at Android API 28 from macOS
  26.5.2 with CMake 4.4.2, Android NDK r28c (`28.2.13676358`),
  Vulkan-Headers 1.4.357.0, and shaderc/glslc 2026.3. Both ELF files are 64-bit
  and have 16 KiB `LOAD` alignment. Their only `DT_NEEDED` entries are
  Android system libraries: `libm.so`, `libdl.so`, `libvulkan.so`, and
  `libc.so`; no host or duplicate Vulkan loader is packaged. Stripped copies
  were 44,590,896 bytes for arm64-v8a and 45,787,088 bytes for x86_64. This is
  native compile/link evidence, not a Flutter APK, loaded-device, model-offload,
  correctness, or performance result.
- Later on 2026-08-11, an arm64 Debug APK loaded the opt-in bridge on a Xiaomi
  `23127PN0CC` (`houji`), Android 16/API 36, Snapdragon SM8650, and Adreno 750
  reporting Vulkan 1.3.128. The APK SHA-256 was
  `8dcca92fc0f6e532a9cf176f4a2b30bae67da65e5cd470fe5367a3d7628a537b`;
  its stripped bridge SHA-256 was
  `724a16f7a701466b7a8ce988f8d6c462928d8de6c07b46c48c48590ca4037778`.
  The exact SmolLM2 Q4_0 fixture was 91,893,088 bytes at SHA-256
  `bcc3af2849ad6095af57e9b5cd43775256efdc66e306acb529172f92d0c04b03`.
  Runtime diagnostics selected Vulkan, named Adreno 750, enabled KV offload,
  and reported 31/31 model layers offloaded. Bounded generation and
  cancellation/reset/recovery completed, but greedy Vulkan output diverged
  materially from CPU. Direct CLI reproduction remained incorrect after
  disabling FP16, async, fusion, graph optimization, integer dot, dot2, MMVQ,
  Flash Attention, or KV offload and with only one layer offloaded. Batch and
  ubatch 16 with one Vulkan node per submission also failed; the tested Q4_K_M
  fixture separately failed pipeline creation. This is negative correctness
  evidence, not Android Vulkan support or a performance result.
- The final 2026-08-11 overlay also patches `ggml-vulkan.cpp`, preserving the
  pinned submodule, and enables the Qualcomm-proprietary K-quant policy above.
  It pins the normalized source/output SHA-256 pairs: `ggml-vulkan.cpp`
  `34691a65d3d436342f26d9b464c49dd6ba3a9f15e5c7344f3176727184820c6b` /
  `877d2c2d0da802b84dc8962f0047f21c6bff033052fdcbfcfc730f3aec89fe80`,
  `dequant_funcs_cm2.comp`
  `d70cf26d67104b333fdd2dedefdb060ea1409091465cfc81829c1f6d1b14683a` /
  `79e3bed12bdb16181293a3123e09c58abe6b1d774d928f59407c5d2ac3eb8e25`,
  and `mul_mm_cm2.comp`
  `b48523e624ca55a8e4441c38e580b7109813a146265f2866f1238549caceebbe` /
  `bf170282a7fb3f17e7214814fd0e9ce1656e54d68fce28a8e917201537056d9e`.
  The transform rejects any other input or output.
- A dependent Flutter native-test harness then exercised the final-source
  arm64-v8a bridge on that same physical device. The Debug APK SHA-256 was
  `916db989df04dc68f4e426cfef238d3672c7b221c53311ad359c47731a816e18`; its
  29,976,312-byte bridge SHA-256 was
  `5b79932a6261a83a22e18fcc6eb4d0dbdd0952ee2f1e95a68ccf37e181a2614b`.
  It used NDK `30.0.14904198`, target API 35, `RelWithDebInfo`,
  Vulkan-Headers 1.4.357, and the explicit host `glslc`. Diagnostics selected
  Adreno 750 Vulkan, enabled KV offload, and contained both safe-overlay
  markers. The exact 91,893,088-byte SmolLM2 Q4_0 fixture at SHA-256
  `bcc3af2849ad6095af57e9b5cd43775256efdc66e306acb529172f92d0c04b03`
  produced the checked bounded greedy result, matched a CPU reference, and
  passed cancellation, reset/recovery, and repeat-dispose checks.
- Device probes of the same Qualcomm-proprietary policy also established the
  intended hybrid behavior: Q4_K uses the safe Vulkan matrix route, while
  Q5_K/Q6_K matrix operations fall back to CPU and surrounding eligible work
  can remain on Vulkan. This is correctness evidence for those bounded probes,
  not a general all-model or all-driver qualification. A report of all layers
  offloaded is layer placement metadata; it must not be read as proof that all
  operations ran on Vulkan.
- `GGML_VULKAN_CHECK_RESULTS` still exceeds its 1% relative threshold for some
  Q4_0/Q4_1 intermediate local/accumulated values. It is a strict diagnostic
  checker, not an acceptance bypass: the final bounded CPU-oracle result above
  is the only positive inference evidence recorded here. This remains
  experimental and unqualified; no sustained, thermal, multi-device,
  release-signed-package, or performance qualification has been completed.
- A final normal Kokage consumer build then exercised the self-contained
  header/tool contract with `VULKAN_SDK` unset. The target-API-36, two-ABI
  Debug APK was
  311,977,712 bytes at SHA-256
  `2c8b47cd13c44575a70f81d63c0eefeee3e8205dd3372fbe67055d5acb232424`.
  Its arm64-v8a bridge was 29,976,312 bytes at
  `bb78a1ba1b47af1edc3496b4791546166ab975969f199990cadc16dc67694692`;
  its x86_64 bridge was 31,072,464 bytes at
  `02971b6a0f88d50c4f2e16ce8bb1c7fdfbf88660a5a079c88af220f7ef0a309c`.
  Both return ABI 41, embed Vulkan plus both Android safe-policy markers, have
  16 KiB `LOAD` alignment, and depend only on Android system
  `libm.so`, `libdl.so`, `libvulkan.so`, and `libc.so`. The APK packages no
  Vulkan loader and passes `zipalign -P 16`. The build used macOS 26.5.2,
  Flutter 3.47.0-0.1.pre, CMake 4.4.2, NDK `30.0.14904198`, native API 35,
  bundled Vulkan-Headers 1.4.357.0, and the selected NDK's glslc v2022.3.
  This verifies normal consumer compilation and packaging, not device
  selection, model correctness, Release/AAB output, or performance.
- The hook has host smoke-test coverage through `dart test` and `flutter test`.
- Native `ctest` always covers ABI/error handling; its assertions remain active
  in Release builds, and an ASan/UBSan Debug build passes locally.
  Fixture-backed vocab/context checks run when
  `third_party/llama.cpp/models/ggml-vocab-gpt-2.gguf` exists.
- Set `LLAMA_DART_TEST_MODEL` to an app-owned weighted GGUF before `dart test`
  to run opt-in warm-up, prompt-only prefill and continuation,
  explicit/automatic context-shift continuation, and deterministic stop-token
  generation coverage.
- The 2026-09-06 prompt-source policy check on macOS 26.6.2 arm64 with
  Flutter 3.47.1/Dart 3.13.1 passed `dart analyze --fatal-infos`, the complete
  Dart and Flutter suites (207 tests and 12 explicit optional-fixture skips
  each), and host CTest (3/3). The separate weighted command below also passed
  with the exact 26,671,328-byte `stories15M-q8_0.gguf` fixture at SHA-256
  `2eda49203f2f044f3dddf29a7dd7cc861ef5a0340f518a19613d73ba6d9c06b6`.
  It used CPU, context 128, batch 32, one generation/batch thread and disabled
  KV offload. With the ChatML template and both `enableThinking` values,
  `contextSize * maximumTokenPieceBytes` bounded source staging, formatted
  counts matched tokenization, and generation telemetry matched those counts.
  This generic template check does not qualify a model's private reasoning.
  The compiled fake bridge separately tested request and returned-plan byte
  edges, including the numeric reasoning-budget metadata shared by counting
  and dispatch. No ABI or upstream revision changed. Initial Jinja rendering
  and the native returned-buffer allocation still precede the size check;
  these runs do not measure expansion, peak memory, mobile/GPU behavior,
  sustained performance, or signed-package qualification.
- The 2026-09-06 streaming-progress check on macOS 26.6.2 arm64 passed the
  complete Dart and Flutter suites (201 tests each; 12 explicit optional-fixture
  skips), native CTest (3/3), and the weighted TinyLlama Q8 CPU test below.
  The compiled fake bridge separately verified split UTF-8 pieces, unchanged
  token counts, absence of progress during a blocked native call, one batch
  across pause/resume, and cancellation followed by recovery. The weighted
  test used the exact `stories15M-q8_0.gguf` identity in the table below,
  context 128, batch 32, one generation/batch thread, and CPU with KV offload
  disabled; it checked completed-step counts against actual native telemetry.
  These checks cover the host bridge and stream ownership, not mobile device
  performance.

  ```sh
  LLAMA_DART_TEST_MODEL=/private/tmp/fllamer-speculative-fixture/stories15M-q8_0.gguf \
    dart test test/fllamer_test.dart \
    --plain-name 'weighted fixture warms prefills shifts and stops cleanly'
  ```
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
  pinned `llama.cpp` converter from `AngelSlim/Qwen3-1.7B_eagle3` revision
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
  from the pinned llama.cpp commit.

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
  media from the pinned `llama.cpp` commit.

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
- A clean dependent-app `flutter drive` run on an iPhone 17 Pro iOS 26.5
  Simulator passes with `GpuConfig.auto()`: context inspection reports the CPU
  backend with KV offload disabled, and the exact Qwen2.5 0.5B Instruct Q4_K_M
  fixture produces the same controlled greedy response as the CPU reference.
  A subsequent unsigned `iphoneos` Debug build passes from the same source.
  These are Simulator CPU runtime and device compile/package results, not a
  physical-iPhone Metal runtime result.

Current limitations:

- Linux and Windows bridge compilation, packaging, loader discovery, real GGUF
  offload, and CPU fallback still require results from those target hosts; a
  macOS policy test cannot establish them.
- `flutter build apk --debug` without `--target-platform` still asks Flutter's
  native-assets pipeline to build `android-arm`; the hook rejects that 32-bit
  ABI because this package only supports `arm64-v8a` and `x86_64`.
- The Android Vulkan receipt above is one physical-device Debug-harness result,
  not a release or fleet qualification. Physical Android CPU paths and all
  other target/device combinations still need their own evidence.
- Runtime loading still supports explicit `nativeLibraryPath` and
  `FLLAMER_NATIVE_LIBRARY`; app builds should verify the bundled library is
  discoverable on each target platform before release.

## 2026-09-06 decoded token-piece bound verification

The ABI 42 model-info change keeps the pinned llama.cpp revision unchanged.
Both committed ffigen configurations regenerated the dynamic and native-assets
bindings. The local macOS arm64 CMake build and CTest passed all three targets,
including ABI/error handling, reasoning sampler, and exported-symbol checks.
The bridge test decodes every vocabulary ID from the pinned GPT-2, Gemma 4,
and Qwen 3.5 vocabulary fixtures, verifies that the metadata bounds every
rendered piece, and witnesses equality for the largest piece. Repeated reads
return the cached value; invalid-handle output clears it.

The immutable TinyLlama `stories15M-q8_0.gguf` fixture documented above was
verified at 26,671,328 bytes and SHA-256
`2eda49203f2f044f3dddf29a7dd7cc861ef5a0340f518a19613d73ba6d9c06b6`.
With `LLAMA_DART_TEST_MODEL` pointing to its external temporary path, the
`weighted fixture warms prefills shifts and stops cleanly` test passed on
macOS arm64 CPU with context 128, batch 32, and one thread. It additionally
checks positive/repeated model metadata, generated-token allowance, and
output length against the decoded-piece bound. An explicit test ChatML
prompt additionally verifies both thinking-mode counts against the real
loaded generation's prompt-token telemetry. The static formatter/counter
parity test covers true and false modes for GPT-2, Gemma 4 and Qwen 3.5.
This is host correctness
coverage; it does not qualify mobile inference, GPU execution, model answer
quality, or physical devices.

Full `dart analyze --fatal-infos` passed. Final `dart test` and `flutter test`
each passed 199 tests with 12 separately provisioned fixture tests skipped.

The content-free native tokenizer probe also measured the following cached
model metadata and the difference between omitted and explicit thinking mode
for representative 128-message English, Japanese and mixed-language requests:

| Vocabulary fixture | Maximum decoded piece bytes | Explicit Off count delta | Explicit On count delta |
| --- | ---: | ---: | ---: |
| GPT-2, explicit test ChatML | 128 | 0 | 0 |
| Gemma 4, GGUF template | 48 | +3 | 0 |
| Qwen 3.5, GGUF template | 128 | 0 | 0 |

Gemma 4 tool replay produced the same mode deltas. The Qwen 3.5 vocabulary
fixture's template rejected tool replay, which remains outside this probe's
coverage. The measured deltas are fixture observations, not universal token
margins: callers now count the selected native mode directly. The probe used
vocabulary-only models and app planning arithmetic; it did not generate
answers or measure multimodal preprocessing, private reasoning throughput,
or model quality.

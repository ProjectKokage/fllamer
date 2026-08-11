# Architecture

`fllamer` has three layers:

- Public Dart API in `lib/fllamer.dart` and `lib/src/`.
- Generated FFI bindings in `lib/src/ffi/generated_bindings.dart`.
- A narrow C ABI bridge in `native/llama_dart_bridge/`.

The Dart API never exposes native pointers. Native-backed work is routed through
a worker isolate in `NativeLlamaEngineSession`, so model loading, tokenization,
generation, multimodal preprocessing, embeddings, reranking, LoRA changes, and
context state operations do not run on the caller isolate.
Tokenizer and chat-template methods on `LlamaEngine` reuse that worker's loaded
model and serialize with context work. The static `LlamaTokenizer` and
`LlamaChatTemplate` helpers remain available before an engine is opened; each
operation uses a short-lived worker and one temporary vocab-only model load.
`LlamaEmbeddingEngine` applies the same ownership model to repeated embedding
work: one worker owns one embedding-enabled context and serializes
tokenization, metadata inspection, and embedding batches until `close()`.
Model SHA-256 calculation and in-memory vector-index JSON persistence/loading
also use short-lived workers because those Dart-only operations can process
hundreds of megabytes in mobile apps.

Normal runtime lookup uses the registered code-asset ID
`package:fllamer/llama_dart_bridge`, including Flutter's Apple framework
layout. A separate generated lookup binding remains available for explicit
`nativeLibraryPath` or `FLLAMER_NATIVE_LIBRARY` overrides used by custom builds
and tests.

Upstream native logging is process-global, so the bridge installs one callback
that is silent unless `LlamaRuntime.configureNativeLogging()` enables a minimum
level. The callback writes only to a mutex-protected 1 MiB/4096-record FIFO;
Dart explicitly drains immutable `LlamaLogRecord` values. No native thread
calls Dart, and upstream common/CLI console logging remains disabled.

`LlamaEngine.load()` owns one native model handle and one native context handle.
When `mmprojPath` is configured, that context also owns one upstream `mtmd`
context and frees it before releasing the text model.
When model-backed speculation is configured, it additionally owns the pinned
upstream speculator, draft/MTP context, and optional draft model. The speculator
is released before the draft context, and the draft context before the target
context.
The public mmap/mlock booleans map explicitly to upstream's none, mmap, mlock,
and mmap+mlock load modes. Integrated MTP tensors are loaded only when the
target model owns the requested MTP head; ordinary and sidecar-backed loads
keep upstream's reduced-memory default.
Typed `KvCacheConfig` values are translated to pinned upstream cache types and
context flags inside the bridge. Model-backed speculative setup copies the same
cache type, offload, Flash Attention, SWA, and unified-cache policy to its draft
or MTP context so the two sides do not silently use different memory settings.
Explicit CPU selection, and automatic selection on Apple Simulator targets,
restricts the upstream model device list to CPU and overrides operation, KV,
and speculative offload. The Simulator automatic policy also forces the
multimodal projector onto CPU without changing the bridge's independent
projector choice for explicit configurations. This prevents a
Metal/Vulkan-enabled library from initializing accidental accelerator-backed
work after model-layer offload was disabled.
Call `close()` when finished. Double-close is safe; use after close throws
`ResourceDisposedException`. Context state can be saved in memory or to an
app-owned file for prompt/session reuse.
A Dart `Finalizer` is attached with a token that contains only the worker send
port, native library path, and context address, so it does not retain the
engine. If the engine is collected, the token requests native cancellation and
asks the worker to release its handles without calling Dart from a native
thread. This is an unobservable best-effort safety net: only explicit `close()`
provides deterministic timing and reports native cleanup failures.

One shared isolate lifecycle channel observes worker errors and exits. Every
command reply and active stream subscribes to that channel, then detaches after
completion, so an unexpected worker exit fails pending work instead of leaving
a `Future` or stream waiting forever. Explicit close keeps the worker alive
until its cleanup reply is delivered, then terminates it; cancellation failure
cannot skip cleanup after the Dart finalizer has been detached.

LoRA adapters are owned by the loaded model and may be selected by a context.
The bridge refuses to free a model while LoRA handles remain and refuses to free
an adapter while a context still has it active. Per-request scale maps are
snapshotted before dispatch, installed as the exact active adapter set around
one worker-serialized completion, and always replaced with the engine's global
scales after the native generation handle is released.

Native completion separates prompt decoding, sampler setup, and one-token
decode steps internally. `LlamaEngine.complete` and `chat` use a worker
start/step/dispose protocol for the native generation handle. Each caller
request permits at most one bounded native batch, `streamChunkTokens` coalesces
normal token steps, and a paused subscription stops requesting subsequent
batches. Cancellation can dispose an idle paused generation; context/state/LoRA
commands received meanwhile fail with a typed busy error, and a second stream is
rejected before it can affect the active context. Cancelling a stream awaits
worker disposal and resets the context before releasing the engine's generation
slot, so partially evaluated prompt or output tokens cannot leak into the next
request. Stop strings are held back at the byte boundary; custom stop token IDs
terminate before token-to-piece conversion, sampler acceptance, or context
decode, so the terminating token is neither emitted nor committed.
`LlamaEngine.prefill()` uses the same validated prompt decode path with a
zero-token sampling limit. Its default tokenization mode adds model special
tokens only at position zero, allowing later calls to append reusable prefixes
without duplicating BOS. Chat-template prompts enable parsing for trusted model
control tokens; raw application text keeps that parsing disabled by default.
`continueCompletion()` starts the same cancellable generation stream with no
new prompt and requires non-empty context state. Newly created unconstrained
sampler chains first accept the bridge's committed token history, preserving
repeat, presence, and frequency penalties across prefill and incremental
completion requests. They also apply every model-provided
`tokenizer.ggml.suppress_tokens` entry as a negative-infinity logit bias before
ordinary sampling.
`LlamaEngine.warmUp()` runs pinned upstream's manual BOS/EOS fallback decode on
the worker isolate for the target and any draft/MTP context, then clears native
memory and performance counters. It accepts only an empty context so warm-up
cannot discard a live session.

`LlamaEngine.shiftContext()` mirrors pinned upstream sequence removal and
position shifting. It preserves a caller-selected prefix and at least one tail
token, shifts target and draft/MTP caches together, updates bridge-owned token
history, and captures native state first so a partial failure can roll back.
Multimodal contexts are rejected because media chunks do not map one-to-one to
the text token history.

Tool-aware chat requests are serialized to a narrow native chat-plan ABI. The
bridge asks pinned `llama-common` to render the model's Jinja template and
returns the prompt, grammar mode and triggers, generation prefix, parser,
additional stops, and format metadata as an opaque plan. Completion validates
that the plan prompt matches the evaluated prompt, applies lazy grammar state,
and merges template stops. The worker retains raw streamed text and invokes the
pinned parser at completion, returning a normalized assistant message with
typed tool calls on the terminal chunk. Native pointers and tool execution
never cross into the public API.
When bounded reasoning is requested, the plan retains every template-provided
reasoning end alternative. The bridge keeps planner grammar deferred only while
reasoning is active and replays the exact naturally matched end sequence into
that grammar, allowing an alternate end that begins a tool call to activate its
trigger.

Chat formatting always selects either `LlamaModelConfig.chatTemplate` or the
GGUF's embedded default. The bridge validates and exposes that effective
template; absence is a typed unsupported error, never an implicit ChatML
choice. Loaded-engine model info, metadata, and template reads reuse the same
worker-owned model so callers do not need another pathname-based inspection
load.

Model-backed speculative completion mirrors every target prefill and verify
batch through `common_speculative_process`, lets the selected upstream strategy
produce drafts, verifies those drafts with the request sampler, reports
acceptance/timing telemetry, and removes rejected target and draft tails before
continuing. Contexts that cannot remove speculative tails fail capability setup
instead of risking inconsistent state.

The bridge exports C functions from `llama_dart.h`, returns result codes, and
keeps up to 4095 bytes of last-error text in an allocation-free thread-local
buffer for Dart exception mapping. Truncation preserves a UTF-8 codepoint
boundary, and recording an error cannot throw through the C ABI. ABI changes
require updating `LlamaRuntime.bridgeAbiVersion`, regenerating FFI bindings with
`ffigen`, and running native plus Dart tests. Linker export lists keep
statically linked `llama.cpp`, ggml, common, and mtmd symbols private; host
CTest fails when any defined public symbol does not use the `llama_dart_*`
bridge prefix.
The native test target explicitly undefines `NDEBUG`, so these lifecycle and
argument assertions still execute in Release CTest builds.

Unsupported combinations stay feature-gated. Today those include video input,
and multimodal plus model-backed speculation.

Context snapshots use a fixed, versioned native envelope with a payload
checksum. The envelope records strategy identity and section lengths before
target state, optional draft/MTP state, optional upstream strategy-private
state, and token history. Restore parses and validates every section first,
captures rollback copies, applies target and draft state, then recreates the
upstream speculator. Strategies without serializable private state perform one
ordinary synchronized token after restore before drafting resumes. Target-only
snapshots from ABI 26 and earlier still restore on non-model-backed contexts.
Native-assets packaging is experimental: the build hook exists, Android
supported-ABI debug/release APK packaging and iOS config-only generation pass
through the example app. A dependent Flutter app also passes an iOS 26.5
Simulator automatic-backend runtime regression and an unsigned `iphoneos`
Debug build. Those checks validate Simulator CPU runtime behavior and device
compile/package output, respectively; physical-device Metal runtime testing is
still pending. Linux and Windows native-assets builds use a strict,
configuration-visible Vulkan policy by default while preserving the CPU
backend. They accept an explicit CPU-only override and optional local SDK root,
and reject cross-architecture builds until the Vulkan host shader-generator
toolchain has a separate contract. Android stays CPU-only by default and has an
opt-in Vulkan artifact contract: the package supplies an exact bundled Vulkan
header set, while Flutter's selected NDK owns the target loader, SPIR-V headers,
and host `glslc`. An app may override only the header root when required; no
host loader enters the artifact. This defines
reproducible build inputs; it does not replace package, loader, GPU, model, or
physical-device validation.

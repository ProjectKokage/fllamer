# Architecture

`fllamer` has three layers:

- Public Dart API in `lib/fllamer.dart` and `lib/src/`.
- Generated FFI bindings in `lib/src/ffi/generated_bindings.dart`.
- A narrow C ABI bridge in `native/llama_dart_bridge/`.

The internal Dart side of the bridge is one library per owner. None of them is
exported from `lib/fllamer.dart`.

| File in `lib/src/` | Owns |
| --- | --- |
| `native_bridge.dart` | The opened bindings, the ABI check, error translation, process-wide capabilities and logging, and the static entry points `engine.dart` calls. |
| `native_config_mapping.dart` | Pure mappings between Dart configuration values and native numbers. |
| `native_model_ops.dart` | Work on a loaded model: scoped vocab-only loads, metadata and template reads, tokenization. |
| `native_chat.dart` | Chat rendering, tool-aware chat plans and parsing of chat output. |
| `native_one_shot.dart` | The bodies that run in short-lived workers for the static helpers. |
| `native_engine_handles.dart` | The worker-side owner of one model, context and its LoRA adapters, and one streaming generation. |
| `engine_worker_protocol.dart` | Messages between the caller and the worker, including the typed requests and the error codec. |
| `engine_worker.dart` | The worker isolate's entry point and its command loop. |
| `engine_session.dart` | The caller-side session: request sending, the stream protocol, close, cancel and the finalizer. |

The other libraries use four things from the opened bridge: its bindings,
`check`, `throwIfLastError` and `cancelContextAddress`.

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
Loaded-engine chat, formatting and counting accept an optional positive
`maximumPromptBytes` allocation policy, independently of the context's token
limit. They validate raw UTF-8 before snapshots, reserve JSON encoder chunks
before retention, and check returned native buffer lengths before decoding the
Dart prompt or chat plan. Media markers and plain-message native descriptors
count toward their respective staging buffers. A `PromptBufferException`
retains its type across the existing worker channel; callers can distinguish
source memory from token-space failures without reading exception text.
Loaded-engine formatting and counting also accept `reasoningBudgetTokens`;
pass the generation's value when checking the complete serialized plan. The
budget changes request/response metadata, so a count that omits it can fit a
buffer that the actual generation cannot. With explicit thinking enabled,
the pinned bridge does not include this numeric budget in template inputs;
a caller may reserve its largest configured budget to bound metadata digits
before the final output allowance is known, without changing the token count.

The pinned upstream template renderer still materializes its complete result,
and the bridge creates its initial native return buffer, before that output
check. This policy bounds downstream copying and tokenization input; it does
not claim to bound Jinja's internal expansion or prove any bytes-per-token
relationship. With an explicit policy, chat preflight uses the tokenizer's
existing size-query result and allocates no second owned token array. Native
internal tokenizer working memory remains owned by the pinned engine.

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
An expected close ends the channel without an event; a reply that is still
pending then ends with `ResourceDisposedException`, so a stream cancelled while
`close()` is in flight cannot wait on a worker that has already exited.

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
batches. After every successful native `generation_next` call whose cumulative
`generated_tokens` increased, the worker sends a content-free progress message
independently of text coalescing. This message becomes
`GenerationChunk.generatedTokens`; it never completes the outstanding request
or starts another native batch. One in-flight batch queues at most
`streamChunkTokens` progress events plus its one text/terminal reply, including
while paused. An incomplete UTF-8 piece can advance progress without text;
blocked native calls and prefill produce no synthetic heartbeat. Cancellation
retires the reply port, so late progress cannot revive a stream.
Cancellation can dispose an idle paused generation; context/state/LoRA
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
In a planned chat that parsing covers the template's own text only (see
[message text in a chat prompt](#message-text-in-a-chat-prompt)).
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
returns the prompt, grammar mode and triggers, generation prefix, and
additional stops as an opaque plan. The pinned output parser cannot be
serialized, so for a chat that is generated the same call also returns the
parser as a native handle, which the worker's plan owns and the streaming
generation closes. Completion validates
that the plan prompt matches the evaluated prompt, applies lazy grammar state,
and merges template stops. The worker retains raw streamed text and invokes the
pinned parser at completion, returning a normalized assistant message with
typed tool calls on the terminal chunk. For a plan that asked the template
for tools, a thinking mode, a reasoning budget or the separation itself, the
worker also runs the parser in partial mode over the retained text after each
streamed chunk and publishes what it adds to the response text and to the
reasoning. Every such plan is asked to separate reasoning. A partial parse
leaves an end of the output unread while it could still become one of the
template's markers. The last chunk is parsed as ended output, which delivers
that end as response text or as reasoning, when the parser accepts the output
and that reading continues what was streamed; output the parser rejects, or
whose ended reading divides it differently, ends as it was read while it
streamed. Native pointers and tool execution never cross into the public API.
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

## Generated-text resource bounds

`LlamaModelInfo.maximumTokenPieceBytes` reports the exact largest decoded
vocabulary piece under the bridge's generation settings (`lstrip=0`, special
rendering enabled). The model worker computes it once during loading and
retains it with that model. The nullable Dart field preserves existing custom
metadata constructors; native ABI 42 always provides the value. A model with
no usable vocabulary can report zero and cannot supply a positive generation
bound.

An actual total generation allowance of O tokens produces at most O times
this byte maximum. The total counts private reasoning, sampler-forced closing
delimiters, speculative accepted/replacement tokens, and visible text. Each
native step respects remaining total tokens; UTF-8 completion does not append
extra tokens after the total limit. Dart's malformed UTF-8 replacement uses
no more UTF-16 code units than input bytes. Consumers must check integer
multiplication before allocation and distinguish this decoded output bound
from arbitrary input text, JSON serialization, and re-encoded replacement
characters. This metadata adds no text-length or reply-token cap of its own.

## Message text in a chat prompt

A chat's messages come from outside the application: a user types them, a
character card is imported, a tool returns them. A model's control tokens,
such as `<|im_end|>` and `<|im_start|>` for ChatML templates, are written in
the prompt as text, and the tokenizer turns that text into the tokens when it
parses special tokens. A message that contains `<|im_end|>` followed by
`<|im_start|>system` would then end its own turn and open a system turn.

A chat plan therefore records where the request's text is in its prompt. The
pinned Jinja engine marks every string of the request (message content, tool
calls, tool results and tool definitions) and carries the mark through the
template, so the rendered prompt is a list of parts that are either template
text or request text. The plan (version 3) keeps the joined `prompt` and adds
`message_text_spans`, the byte ranges of the request's parts. A message's
role is marked too. A part that is exactly `system`, `user`, `assistant` or
`tool` is left out of the ranges: the caller chooses a role from those four
names, and some templates build their role marker from it
(`'<|' + message['role'] + '|>'`), which must stay one token.

The bridge tokenizes a plan's prompt in one function, for generation, for
exact prefix reuse and for `llama_dart_model_tokenize_chat_plan`, which
`countChatTokens` calls:

- When the model loads, the bridge collects the texts of the vocabulary's
  control and unknown tokens. These are the tokens `llama_tokenize` produces
  from text only when it parses special tokens. User-defined tokens are not
  among them: `llama_tokenize` reads those from any text, with or without
  parsing, and that is unchanged.
- A recorded range spells a control token when the prompt holds one of those
  texts at a place that shares at least one byte with the range. That
  includes a text the message completes together with the template text
  next to it.
- A range that spells one is tokenized on its own with special-token parsing
  off. Everything else, the template's text and the request text that spells
  none, is tokenized with parsing on, each stretch between two such ranges in
  one call.

So message text never becomes a control token, and the markers the template
writes still do. A prompt whose messages spell no control token is one call
over the joined prompt, with the tokens it had before the plan recorded
anything. When ranges are set apart, tokens cannot merge across their two
ends, a tokenizer that prefixes a space or `▁` to the start of its input
does so at the start of each piece, and the model's BOS and EOS go around all
pieces the way pinned `mtmd` puts them around its text chunks. This only
changes what a model sees for a chat with such a message, and it changes it
on every turn while that message is in the history.

With media, pinned `mtmd_tokenize` takes one parsing flag for the whole
text. A multimodal prompt with ranges to set apart is cut at its media
markers by the bridge, as `mtmd_tokenize` cuts it, and handed to
`mtmd_tokenize_from_parts` piece by piece; any other multimodal prompt goes
through `mtmd_tokenize` as before. A media marker typed into a message is
still a media marker, as before.

Three paths do not have this separation:

- The legacy formatter, `llama_chat_apply_template`, returns one string and
  no parts. A chat it renders, which is a chat that sets no tools, thinking
  mode, reasoning budget or `separateReasoning` on a template the formatter
  knows, is tokenized whole with special-token parsing on, so message text
  that spells a control token becomes that token. An application with
  untrusted messages sets one of those options to get a plan;
  `separateReasoning` alone keeps the template's default thinking mode.
- `formatChat` returns the prompt as a string. Passing that string to
  `tokenize`, `countTokens`, `prefill` or `complete` with `parseSpecial` on
  parses special tokens in all of it.
- A template that builds a control token out of request text other than a
  role name, for example a tool name between two marker halves, gets that
  text as text.

Chat formatting and token counting accept the same nullable `enableThinking`
mode as generation. An explicit value selects the native chat-plan renderer;
null preserves the original template-default path. Static helpers load a
vocabulary-only model in their worker, while `LlamaEngine` sends the mode to
its existing model worker. The count includes the rendered reasoning prefix,
using the exact native tokenizer with special tokens. A planned chat is
counted by tokenizing its plan, so the count is the number of prompt tokens
generation evaluates. A reasoning sampler
budget changes generation policy, not prompt text. Callers cache counts by
both messages and thinking mode and count the resolved mode again when an
automatic router makes its decision.

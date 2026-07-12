# Mobile Performance

Defaults are conservative except for available GPU offload: `useMmap` true,
`useMlock` false, `checkTensors` true, `contextSize` 4096, `batchSize` 512,
`ubatchSize` matching `batchSize`, and `GpuConfig.auto()` offloading all layers
when a compiled GPU backend is available. KV cache defaults are F16 keys and
values, automatic Flash Attention, KV offload enabled, full-size SWA enabled,
and unified-cache mode disabled. Use `GpuConfig.cpu()` for a deterministic CPU
baseline and tune explicit layer counts per model and device.

Measure at least:

- model name and quantization,
- device model and OS,
- mmap/mlock/tensor-check settings, context size, batch size, ubatch size,
  threads, batch threads, GPU/offload mode, and every `KvCacheConfig` field,
- prompt tokens and generated tokens,
- prompt eval tok/s,
- decode tok/s,
- time to first token,
- total elapsed time.

`GenerationTelemetry` reports prompt/generated token counts, elapsed times,
computed tok/s rates, speculative draft/accepted token counts, and cumulative
draft/verification milliseconds on the final generation chunk. Generated text
is streamed from the worker isolate in decoded chunks.

Call `await engine.warmUp()` immediately after loading when predictable first
request latency matters. Warm-up runs on the inference worker, primes both the
target and any draft/MTP execution graph, and restores an empty context. After
inference has populated the session, call `reset()` before warming again.

Use `contextInfo().usedTokens` to monitor long sessions. When
`supportsContextShift` is true, `shiftContext(keepTokens: ...)` preserves an
initial system/prefix region and automatically discards half of the remaining
old context; pass `discardTokens` for an exact range. Shifting snapshots native
state for rollback, so call it before the context is completely full rather
than on every request.

Use `prefill()` for a reusable system or document prefix when the app will save
the resulting state or append only the unevaluated suffix. The first prefill on
an empty context adds model special tokens automatically; later prefills do not.
Its telemetry reports prompt tokens, prompt-evaluation time, total time, and
prompt tok/s without sampling output.

After prefilling a complete prompt, use `continueCompletion()` to decode without
reevaluating text. The stream has the same subscription-cancellation behavior
as `complete()`, reports zero prompt tokens, and keeps sampler penalties aware
of committed tokens from earlier incremental requests.

`GenerationConfig.streamChunkTokens` defaults to four. Lower it for minimum
visible latency or raise it to reduce isolate messages and Flutter rebuilds.
Pausing a stream subscription stops later worker decode batches, with at most
the currently requested bounded batch in flight. A single speculative native
step may still contain several accepted draft tokens.

## Desktop Smoke Benchmark

Run the reproducible local smoke harness with an app-owned GGUF model:

```sh
dart run benchmark/desktop_smoke/benchmark.dart \
  --model /path/to/model.gguf \
  --native-library build/native/libllama_dart_bridge.dylib \
  --device-model 'MacBookPro18,3' \
  --max-tokens 128 \
  --iterations 5 \
  --kv-cache-key f16 \
  --kv-cache-value f16 \
  --json-out benchmark.json
```

Add `--spec-ngram ngram-simple --spec-ngram-size 12 --spec-draft-length 48`
when measuring self-speculation. Substitute `ngram-map-k` or
`ngram-map-k4v` to compare the pinned upstream map strategies. Use
`--spec-ngram ngram-mod --spec-ngram-size 24 --spec-min-draft-length 48
--spec-draft-length 64` for the adaptive mod strategy, or
`--spec-ngram ngram-cache` for the fixed-size request-local cache strategy.
The legacy `--spec-ngram-simple` flag remains an alias.

Draft-model and EAGLE-3 modes load a second GGUF model and context. Integrated
MTP creates an additional context against the target model; a separate
`mtpModelPath` loads another GGUF. Treat that memory as additive, start with the
default three-token draft limit, and compare net `decodeTokensPerSecond` plus
`speculativeAcceptanceRate` against a non-speculative run on the same device.

Always pass the exact phone or host identifier with `--device-model` for
records intended for comparison; omitted values are recorded as
`unspecified`, never guessed from a hostname. The harness warms the native
graphs once by default, resets the context between iterations, and runs three
generations unless `--iterations` overrides it. Use
enough iterations and output tokens to observe sustained throughput and thermal
behavior; pass `--no-warm-up` only when measuring cold-start behavior. The JSON
record includes device model, host OS, processor count, Dart build mode/version,
upstream commit, native build flags, model path/file name/quantization/metadata,
context/batch/thread settings, load and warm-up timing, process RSS snapshots,
requested and applied KV-cache settings, per-run output/telemetry, and aggregate
throughput, latency, and speculative acceptance totals. It does not download
models or write output unless `--json-out` is provided.

Keep large work off the UI isolate. `LlamaEngine`, tokenizer helpers,
multimodal preprocessing, embeddings, reranking, and model inspection use
worker isolates today. Full-file SHA-256 calculation and vector-index JSON
encode/decode also use workers. Image/audio requests are limited to 64 media
inputs of 64 MiB each, but apps should resize images and trim audio before
submission to reduce decode latency and peak memory.

Use smaller quantized models first on phones. Raise context size, batch size,
ubatch size, batch threads, or GPU layers only after measuring memory,
sustained decode speed, and thermal behavior on the target device. `ubatchSize`
must not exceed `batchSize`; raise `batchSize` first when larger physical
batches are needed. Some multimodal projectors use non-causal attention for a
media chunk and require that chunk's physical decode batch to fit in
`ubatchSize`; the bridge checks this before decode and reports a generation
error with an actionable tuning hint.

`LlamaRuntime.currentCapabilities().metal`/`vulkan` report compiled backends.
`LlamaEngine.contextInfo().gpuBackend` reports the backend selected for the
loaded model. The same context report exposes the configured K/V cache types,
effective KV offload, Flash Attention mode, full-size SWA mode, and
unified-cache mode. `GpuConfig.cpu()` forces both context operation offload and
KV offload off, even when `KvCacheConfig.offload` retains its portable default.

Quantized cache types can reduce context memory but are model- and backend-
dependent. Quantized V caches require Flash Attention; `KvCacheConfig`
rejects the explicitly disabled combination before native work. Start with
F16, compare output and throughput on the exact target model, and treat cache
quantization as an opt-in mobile memory tradeoff rather than a universal
default. Model-backed speculative decoding applies the same cache policy to
the target and draft/MTP contexts.

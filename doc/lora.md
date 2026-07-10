# LoRA adapters

LoRA support is experimental and local. The application owns the base GGUF and
adapter GGUF files; `fllamer` does not download or bundle either file. Adapter
compatibility is checked by pinned upstream `llama.cpp` when the adapter loads.

Load adapters once on a live engine, reuse them across requests, and unload
them before closing only when the memory needs to be reclaimed early:

```dart
final engine = await LlamaEngine.load(modelConfig);
try {
  final adapter = await engine.loadLora(
    const LoraAdapterConfig(
      path: '/app/private/adapter.gguf',
      scale: 1,
    ),
  );

  await for (final chunk in engine.complete(
    prompt: 'Continue this text:',
    config: GenerationConfig(
      loraScales: <int, double>{adapter.id: 0.75},
    ),
  )) {
    // Consume chunk.text in the application.
  }

  await engine.setLoraScale(adapter.id, 0.5);
  await engine.unloadLora(adapter.id);
} finally {
  await engine.close();
}
```

`GenerationConfig.loraScales` has three distinct meanings:

- `null` uses every loaded adapter at its current global scale.
- An empty map disables every adapter for that request.
- A non-empty map selects exactly those adapter ids and scales for the request.

Request overrides are snapshotted before worker execution. The worker restores
global scales after success, cancellation, or failure. Unknown ids, negative or
non-finite scales, incompatible adapters, and operations after engine disposal
produce typed errors.

Loaded adapters consume native memory. Loading or unloading per request adds
latency; keep a bounded working set loaded instead. Requests with different
adapter selections or scales cannot always share efficient native batching, so
measure throughput and memory on representative phones. A zero scale is useful
for deterministic disable/enable workflows but does not release adapter memory.

The opt-in integration test uses the MIT-licensed
`ggml-org/stories15M_MOE` Q8 base and Shakespeare adapter at immutable revision
`b6dd737497465570b5f5e962dbc9d9454ed1e0eb`. Exact sizes, SHA-256 values, and
the test command are recorded in [native_builds.md](native_builds.md).

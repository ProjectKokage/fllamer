# Working in fllamer

fllamer is a mobile-first Dart/Flutter library around a pinned `llama.cpp`.
The Dart package is at the repository root; the Flutter app is in `example/`.

## Start here

Read [README](README.md) and the relevant rows in the
[feature matrix](doc/feature_matrix.md). Then follow the owning document:

| Change | Read |
| --- | --- |
| C ABI, isolates, ownership, cancellation | [Architecture](doc/architecture.md) |
| Native assets, bindings, platforms, model smoke tests | [Native builds](doc/native_builds.md) |
| Upstream revision or native API | [Upstream sync](doc/upstream_sync.md), then the pinned source and headers |
| Retrieval | [RAG](doc/rag.md) |
| Media inputs | [Multimodal](doc/multimodal.md) |
| Templates and constrained generation | [Tool calling](doc/tool_calling.md), [structured output](doc/structured_output.md) |
| Adapters or speculation | [LoRA](doc/lora.md), [speculative decoding](doc/speculative_decoding.md) |
| Performance | [Mobile performance](doc/mobile_performance.md) |

Inspect the working tree and preserve unrelated edits and unpublished commits.
State the intended scope and verification. Make the smallest complete change;
reuse existing code and inspect pinned upstream behavior instead of guessing.

## Essential contracts

- Keep the core usable from Dart. Flutter UI and platform helpers belong in
  the example or an explicit adapter. Mobile inference links native code in
  process; do not invoke a CLI, Python or `llama-server` at application runtime.
- Expose a narrow, versioned C ABI with opaque handles and explicit buffer
  lengths. Never pass C++ exceptions through it or expose pointers in public
  Dart APIs. Preserve ABI checks, error translation and symbol visibility.
- Keep loading, inference and preprocessing off the Flutter UI isolate.
  Serialize work per native context; preserve cancellation, bounded queues,
  backpressure and stale-result suppression.
- Give each native resource one owner and explicit, idempotent cleanup,
  including partial initialization. Finalizers are a recovery mechanism.
  Reject use after disposal and retain dependencies while work is active.
- Validate untrusted model, path, metadata and media inputs before native use.
  Keep inference local and library logging quiet/configurable. Downloads or
  telemetry require explicit opt-in; download helpers verify checksums.
- Preserve typed APIs, model-specific templates and capability detection.
  Unsupported features fail explicitly. Public API breaks need versioning and
  migration notes; a missing test does not justify removing a working feature.
- Keep upstream pinned and native builds reproducible. Regenerate bindings
  through both committed ffigen configurations when the bridge header changes.
  Never hand-edit generated bindings or casually upgrade the dependency graph.
- Keep model weights, adapters, caches and build outputs out of Git/packages,
  except deliberately approved tiny fixtures. Preserve source/license notices;
  fixture downloads are opt-in and separate from ordinary tests.

## Verify the changed boundary

For Dart code changes, run targeted tests first, then from the repository root:

```sh
dart format --output=none --set-exit-if-changed .
dart analyze --fatal-infos
dart test
```

For Flutter example changes, run `flutter analyze` and `flutter test` from
`example/`. For bridge or native-build changes, run the CMake/CTest commands
and applicable artifact-backed smoke tests in [Native builds](doc/native_builds.md).
An upstream sync needs native, Dart and Flutter checks plus real model-load
and generation evidence. Performance claims need reproducible measurements.

For prose-only changes, check links, referenced paths/commands and
`git diff --check`; no inference build is required. Report skipped checks and
missing hardware/artifacts precisely. A host pass establishes only that target.

## Delivery

Branch names must not begin with `codex` (case-insensitive), including
`codex/` and `codex-`. Rename tool-generated defaults before committing or
pushing; use a descriptive name such as `docs-agent-guides`.

Keep work on a task branch and review the complete diff before handoff. Commit
only task files; push, publish or release only when requested. Do not merge
without owner approval. Routine fixes within the requested scope can proceed;
get approval before expanding product scope or changing recorded decisions.
Update the owning docs, feature matrix and changelog when their facts change.
Report the result, checks, compatibility impact and remaining evidence gaps.

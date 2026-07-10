# Speculative decoding

`fllamer` exposes draftless, draft-model, EAGLE-3, and MTP strategy families:

```dart
const NGramSpeculation(
  // Also accepts 'ngram-map-k' and 'ngram-map-k4v'.
  strategy: 'ngram-simple',
  ngramSize: 12,
  draftLength: 48,
)

const NGramModSpeculation(
  matchLength: 24,
  minimumDraftLength: 48,
  maximumDraftLength: 64,
)

const NGramCacheSpeculation()

const DraftModelSpeculation(
  draftModelPath: '/app/private/draft.gguf',
  draftLength: 3,
)

const Eagle3Speculation(
  draftModelPath: '/app/private/eagle3.gguf',
  draftLength: 3,
)

const DFlashSpeculation(
  draftModelPath: '/app/private/dflash.gguf',
  draftLength: 15,
)

const MtpSpeculation(
  // Omit mtpModelPath when the target GGUF contains usable NextN/MTP layers.
  mtpModelPath: '/app/private/mtp.gguf',
  draftLength: 3,
)
```

The bridge uses pinned upstream `common_speculative` for all five draftless
variants plus draft-model, EAGLE-3, DFlash, and MTP execution. Draftless modes
maintain request-local token-history/cache state and require target context
memory that can remove rejected tails. `NGramCacheSpeculation` is deliberately
memory-only: the bridge does not expose upstream cache files because malformed
external cache data can reach upstream abort paths. Model-backed modes load any
auxiliary GGUF during context creation, check the strategy-specific model shape
and vocabulary where upstream exposes those checks, mirror target batches into
the speculative context, and remove rejected draft tails from both contexts.

`LlamaRuntime.currentCapabilities().speculativeDecoding` and `.mtp` report
compiled bridge support. They do not prove that a particular model is
compatible. Inspect `LlamaModelInfo.nextnLayerCount` before selecting integrated
MTP; context creation returns a typed error when the required model/head or
memory behavior is unavailable.

The configured draft length must be between 1 and 1024. Each request further
caps it to the remaining generation budget, target context capacity, and native
batch capacity. Speculation is never enabled silently.

The final `GenerationTelemetry` reports proposed and accepted draft tokens,
acceptance rate, cumulative drafting time, cumulative verification time, and
net decode throughput. Compare net throughput and sustained memory/thermal
behavior against the same request with `NoSpeculativeDecoding`; a low acceptance
rate or a large draft model can make speculation slower on phones.

Current constraints:

- Model-backed speculation cannot share an mmproj context. Draftless modes may
  share one for text-only requests, but any request containing media inputs
  rejects speculative decoding.
- Requests with stop strings use ordinary one-token decoding to preserve exact
  stop-marker suppression.
- Context snapshots preserve target, draft/MTP, token-history, position, and
  available upstream strategy-private state in one checksummed envelope. When
  an upstream strategy does not expose private state, restore synchronizes one
  ordinary token before model-backed drafting resumes. Draftless strategy state
  is recreated from restored token history at the next generation boundary.
- The package does not bundle or download target, draft, EAGLE-3, or MTP GGUF
  files. The same applies to DFlash GGUFs. Apps own model selection, licenses,
  storage, and checksums.
- Default tests cover all five upstream draftless implementations with synthetic
  repeated token history, plus ABI, lifecycle, validation, and unsupported
  combinations. A weighted fixture test verifies greedy output equivalence and
  context-state restore for each draftless strategy.
  An opt-in checksum-pinned MIT-derived TinyLlama Q8/Q4 pair verifies ordinary
  draft-model output equivalence, proposal/acceptance telemetry, and cleanup
  through the real bridge. An opt-in checksum-pinned Qwen3.5 0.8B Q4_K_M GGUF
  with one integrated NextN layer verifies MTP model inspection, output
  equivalence, proposal/acceptance telemetry, and cleanup. A checksum-pinned
  Qwen3 1.7B Q2_K target and EAGLE-3 draft converted with the pinned upstream
  converter verify EAGLE-3 output equivalence, proposal/acceptance telemetry,
  and cleanup. DFlash has ABI/configuration tests but no checksum-pinned
  compatible fixture in this repository, so it remains upstream-dependent.
  Production throughput must still be measured with the app's models.

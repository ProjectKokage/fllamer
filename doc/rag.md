# RAG

`fllamer` keeps RAG local and dependency-light. The current layer provides:

- `Document` and `TextChunk` records with metadata and optional source URIs.
- `TextSplitter` plus deterministic character and token splitters.
- `EmbeddingModel`, implemented by `LlamaEmbeddingModel` for local GGUF
  embedding models.
- `VectorIndex`, with an in-memory flat `Float32List` implementation that can
  persist to JSON.
- `Reranker` for reranking vector candidates, with `LlamaReranker` for local
  rank-pooling models.
- `Retriever`, with `VectorIndexRetriever` for query embedding, vector search,
  and optional reranking.
- `RagPromptBuilder` for source-tagged context, typed citations, exact
  tokenizer-backed context budgets, and full chat-template token budgets.

The in-memory index normalizes vectors by default and ranks by dot product,
which is cosine similarity when both stored and query vectors are normalized.
`LlamaEmbeddingModel.embedTexts()` returns an `EmbeddingBatch`: one row-major
`Float32List` plus zero-copy, unmodifiable vector views and explicit
normalization/pooling metadata. Pass the batch directly to `VectorIndex.add()`
and use `batch.dimensions` when constructing an index; there is no need to
allocate a nested list of vectors.
Equal scores are ordered by document id and chunk id for deterministic results.
JSON persistence exports records in the same document-id/chunk-id order.
Persistence and loading perform JSON encode/decode in short-lived worker
isolates so large local indexes do not monopolize the Flutter UI isolate.
Native reranking is experimental and requires a GGUF model that supports
rank pooling. An opt-in checksum-pinned Apache-2.0 Jina tiny reranker fixture
verifies real pair scoring, same-context document batches, deterministic
scores, semantic ordering, and `LlamaReranker` candidate reordering without
bundling weights. Remove a specific duplicate chunk id with
`index.remove(chunkId, documentId: documentId)`; omitting `documentId` removes
all chunks with that chunk id for compatibility. `LlamaReranker` validates and
snapshots candidate chunks before native scoring. App-owned rerankers must
return every vector-search candidate exactly once with finite scores; missing,
unknown, duplicate, or malformed results are rejected before retrieval results
are exposed. Retrieved chunks are snapshotted before return so app-owned
indexes and rerankers cannot mutate result metadata after retrieval.

Built-in splitters generate deterministic chunk ids in the form
`documentId:index`. Character splitting uses Unicode scalar boundaries so it
does not split surrogate pairs. RAG ids and chunk text must not be empty or
contain NUL bytes, ids must not contain line breaks, source URIs must not be
empty or contain literal or percent-encoded NUL bytes or line breaks,
tokenizer output must contain non-negative int32 token ids, and chunk token
counts must be positive. Built-in splitters, the in-memory index, and retriever results
snapshot metadata through JSON-safe values; non-JSON values, non-string keys,
NUL-containing keys or strings, non-finite numbers, and cycles are rejected
before indexing, retrieval, or persistence. Custom vector indexes must honor
`topK` and `minScore`, and retriever and reranker results must not contain
duplicate chunks. App-supplied citation spans must provide both start and end
values, with end greater than start. JSON persistence and loading refuse
symbolic-link paths so they cannot silently write to or read from another file
through the link.

Example shape:

```dart
final embeddingModelConfig = LlamaModelConfig(
  modelPath: '/app/private/embedding-model.gguf',
  nativeLibraryPath: 'build/native/libllama_dart_bridge.dylib',
);
final generationEngine = await LlamaEngine.load(
  const LlamaModelConfig(
    modelPath: '/app/private/chat-model.gguf',
    nativeLibraryPath: 'build/native/libllama_dart_bridge.dylib',
  ),
);

try {
  final splitter = TokenTextSplitter(
    maxTokens: 256,
    overlap: 32,
    tokenize: generationEngine.tokenize,
    detokenize: generationEngine.detokenize,
  );
  final chunks = await splitter.split(document);

  final embeddingModel = LlamaEmbeddingModel(embeddingModelConfig);

  final vectors = await embeddingModel.embedTexts(
    chunks.map((chunk) => chunk.text).toList(),
  );

  final index = InMemoryVectorIndex(dimensions: vectors.dimensions)
    ..add(chunks, vectors);
  await index.persist('/app/private/rag-index.json');

  final retriever = VectorIndexRetriever(
    embeddingModel: embeddingModel,
    index: index,
    // reranker: appOwnedReranker,
  );
  final results = await retriever.retrieve('What does the document say?');

  final prompt = await const RagPromptBuilder().buildContextWithTokenizer(
    results.map((result) => result.chunk),
    maxContextTokens: 1200,
    tokenize: generationEngine.tokenize,
  );
  final citations = prompt.citations;

  final chatPrompt = await const RagPromptBuilder().buildChatPrompt(
    question: 'What does the document say?',
    chunks: results.map((result) => result.chunk),
    maxPromptTokens: 1600,
    countTokens: generationEngine.countChatTokens,
  );
  final messages = chatPrompt.messages;
} finally {
  await generationEngine.close();
}
```

Do not fabricate citations from model output. The prompt builder provides local
source markers plus `RagCitation` metadata for app-owned rendering and
downstream prompt assembly. Built-in splitters populate character or token
source spans when that information is available.

`buildContextWithTokenizer()` tokenizes each candidate assembled context, so
`usedTokens` and `maxContextTokens` include the header and every
`[source: document#chunk]` label. `buildChatPrompt()` calls an app-supplied
message token counter for the complete system/context/user message list. Prefer
`LlamaEngine.countChatTokens()` so the operation uses the engine's existing
worker-owned model; `LlamaChatTemplate.countTokens()` is available when no
engine is open and uses one temporary vocab-only model load. Both include the
model's template and assistant-generation prompt. The builder throws when the
base messages cannot fit and skips oversized retrieved chunks while still
considering later candidates.

The synchronous `buildContext()` and `buildChatMessages()` methods remain for
model-free pipelines. Their `maxContextTokens` budget uses the supplied
`TextChunk.tokenCount` values and therefore does not claim to include formatting
or chat-template overhead.

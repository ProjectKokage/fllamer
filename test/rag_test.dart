import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

import 'package:fllamer/fllamer.dart';
import 'package:test/test.dart';

void main() {
  group('character splitter', () {
    test('splits deterministically and snapshots metadata', () {
      final splitter = CharacterTextSplitter(maxLength: 8, overlap: 2);
      final tags = <Object?>['fixture'];
      final chunks = splitter.split(
        Document(
          id: 'doc',
          text: 'alpha beta gamma',
          metadata: <String, Object?>{'tags': tags},
        ),
      );
      tags[0] = 'changed';

      expect(chunks.map((chunk) => chunk.text), <String>[
        'alpha be',
        'beta gam',
        'amma',
      ]);
      expect(chunks.map((chunk) => chunk.id), <String>[
        'doc:0',
        'doc:1',
        'doc:2',
      ]);
      final storedTags = chunks.first.metadata['tags']! as List<Object?>;
      expect(storedTags.single, 'fixture');
      expect(() => storedTags[0] = 'changed', throwsUnsupportedError);
      expect(chunks.first.tokenCount, 2);
    });

    test('records trimmed character source spans', () {
      final splitter = CharacterTextSplitter(maxLength: 9);
      final chunks = splitter.split(
        const Document(id: 'doc', text: '  alpha  beta'),
      );

      expect(chunks.map((chunk) => chunk.text), <String>['alpha', 'beta']);
      expect(chunks.first.metadata['start'], 2);
      expect(chunks.first.metadata['end'], 7);
      expect(chunks.last.metadata['start'], 9);
      expect(chunks.last.metadata['end'], 13);
    });

    test('does not split surrogate pairs', () {
      final splitter = CharacterTextSplitter(maxLength: 2, overlap: 1);
      final chunks = splitter.split(const Document(id: 'doc', text: 'a😀b'));

      expect(chunks.map((chunk) => chunk.text), <String>['a😀', '😀b']);
      expect(chunks.first.metadata['end'], 3);
      expect(chunks.last.metadata['start'], 1);
    });

    test('rejects empty document ids', () {
      final splitter = CharacterTextSplitter(maxLength: 8);

      expect(
        () => splitter.split(const Document(id: '', text: 'alpha')),
        throwsArgumentError,
      );
      expect(
        () => splitter.split(const Document(id: 'doc', text: 'bad\u0000')),
        throwsArgumentError,
      );
    });

    test('rejects invalid document source URIs', () {
      final splitter = CharacterTextSplitter(maxLength: 8);

      expect(
        () => splitter.split(
          Document(id: 'doc', text: 'alpha', sourceUri: Uri.parse('')),
        ),
        throwsArgumentError,
      );
      expect(
        () => splitter.split(
          Document(
            id: 'doc',
            text: 'alpha',
            sourceUri: Uri.parse('file:///bad%00path'),
          ),
        ),
        throwsArgumentError,
      );
    });
  });

  group('token splitter', () {
    test('splits by token count with overlap and snapshots metadata', () async {
      final splitter = TokenTextSplitter(
        maxTokens: 2,
        overlap: 1,
        tokenize: (_) async => <int>[1, 2, 3, 4],
        detokenize: (tokens) async => tokens.join(' '),
      );
      final tags = <Object?>['fixture'];

      final chunks = await splitter.split(
        Document(
          id: 'doc',
          text: 'ignored by fake tokenizer',
          metadata: <String, Object?>{'tags': tags},
        ),
      );
      tags[0] = 'changed';

      expect(chunks.map((chunk) => chunk.text), <String>['1 2', '2 3', '3 4']);
      expect(chunks.map((chunk) => chunk.id), <String>[
        'doc:0',
        'doc:1',
        'doc:2',
      ]);
      expect(chunks.map((chunk) => chunk.tokenCount), <int>[2, 2, 2]);
      expect(chunks.first.metadata['tokenStart'], 0);
      expect(chunks.first.metadata['tokenEnd'], 2);
      final storedTags = chunks.first.metadata['tags']! as List<Object?>;
      expect(storedTags.single, 'fixture');
      expect(() => storedTags[0] = 'changed', throwsUnsupportedError);
    });

    test('rejects empty document ids', () async {
      final splitter = TokenTextSplitter(
        maxTokens: 2,
        tokenize: (_) async => <int>[1],
        detokenize: (_) async => 'x',
      );

      await expectLater(
        splitter.split(const Document(id: '', text: 'alpha')),
        throwsArgumentError,
      );
      await expectLater(
        splitter.split(const Document(id: 'doc', text: 'bad\u0000')),
        throwsArgumentError,
      );
    });

    test('rejects invalid document source URIs', () async {
      final splitter = TokenTextSplitter(
        maxTokens: 2,
        tokenize: (_) async => <int>[1],
        detokenize: (_) async => 'x',
      );

      await expectLater(
        splitter.split(
          Document(id: 'doc', text: 'alpha', sourceUri: Uri.parse('')),
        ),
        throwsArgumentError,
      );
      await expectLater(
        splitter.split(
          Document(
            id: 'doc',
            text: 'alpha',
            sourceUri: Uri.parse('file:///bad%0Apath'),
          ),
        ),
        throwsArgumentError,
      );
    });

    test('rejects detokenized NUL bytes', () async {
      final splitter = TokenTextSplitter(
        maxTokens: 2,
        tokenize: (_) async => <int>[1],
        detokenize: (_) async => 'bad\u0000',
      );

      await expectLater(
        splitter.split(const Document(id: 'doc', text: 'alpha')),
        throwsArgumentError,
      );
    });

    test('rejects invalid tokenizer token ids before detokenizing', () async {
      final splitter = TokenTextSplitter(
        maxTokens: 2,
        tokenize: (_) async => <int>[1, -1],
        detokenize: (_) async => throw StateError('detokenizer should not run'),
      );

      await expectLater(
        splitter.split(const Document(id: 'doc', text: 'alpha')),
        throwsArgumentError,
      );
    });

    test('snapshots tokenizer output before detokenizing', () async {
      final tokenBuffer = <int>[1, 2, 3];
      final splitter = TokenTextSplitter(
        maxTokens: 1,
        tokenize: (_) async {
          Timer.run(() {
            tokenBuffer[1] = 99;
          });
          return tokenBuffer;
        },
        detokenize: (tokens) async {
          await Future<void>.delayed(Duration.zero);
          return tokens.single.toString();
        },
      );

      final chunks = await splitter.split(const Document(id: 'doc', text: 'x'));

      expect(chunks.map((chunk) => chunk.text), <String>['1', '2', '3']);
    });
  });

  group('in-memory vector index', () {
    test('returns nearest chunks by cosine score', () {
      final chunks = <TextChunk>[
        const TextChunk(documentId: 'a', id: '0', text: 'alpha', tokenCount: 1),
        const TextChunk(documentId: 'b', id: '0', text: 'beta', tokenCount: 1),
      ];
      final index = InMemoryVectorIndex(dimensions: 2)
        ..add(chunks, <Float32List>[
          Float32List.fromList(<double>[1, 0]),
          Float32List.fromList(<double>[0, 1]),
        ]);

      final results = index.search(Float32List.fromList(<double>[0.8, 0.2]));

      expect(results.first.chunk.documentId, 'a');
      expect(results.first.score, greaterThan(results.last.score));
    });

    test('adds flat embedding batches without nested vector storage', () {
      final chunks = <TextChunk>[
        const TextChunk(documentId: 'a', id: '0', text: 'alpha', tokenCount: 1),
        const TextChunk(documentId: 'b', id: '0', text: 'beta', tokenCount: 1),
      ];
      final batch = EmbeddingBatch(
        count: 2,
        dimensions: 2,
        values: Float32List.fromList(<double>[1, 0, 0, 1]),
      );
      final index = InMemoryVectorIndex(dimensions: 2)..add(chunks, batch);

      expect(
        index.search(Float32List.fromList(<double>[1, 0])).first.chunk.text,
        'alpha',
      );
      expect(
        () => InMemoryVectorIndex(dimensions: 2).add(chunks, batch.take(1)),
        throwsArgumentError,
      );
    });

    test('orders equal vector scores deterministically', () {
      final index = InMemoryVectorIndex(dimensions: 2)
        ..add(
          const <TextChunk>[
            TextChunk(documentId: 'b', id: '1', text: 'beta', tokenCount: 1),
            TextChunk(documentId: 'a', id: '1', text: 'alpha', tokenCount: 1),
            TextChunk(documentId: 'a', id: '0', text: 'alpha', tokenCount: 1),
          ],
          <Float32List>[
            Float32List.fromList(<double>[1, 0]),
            Float32List.fromList(<double>[1, 0]),
            Float32List.fromList(<double>[1, 0]),
          ],
        );

      final results = index.search(Float32List.fromList(<double>[1, 0]));

      expect(
        results.map(
          (result) => '${result.chunk.documentId}:${result.chunk.id}',
        ),
        <String>['a:0', 'a:1', 'b:1'],
      );
    });

    test('does not partially add invalid batches', () {
      final index = InMemoryVectorIndex(dimensions: 2);

      expect(
        () => index.add(
          const <TextChunk>[
            TextChunk(documentId: 'a', id: '0', text: 'alpha', tokenCount: 1),
            TextChunk(documentId: 'b', id: '0', text: 'beta', tokenCount: 1),
          ],
          <Float32List>[
            Float32List.fromList(<double>[1, 0]),
            Float32List.fromList(<double>[double.nan, 0]),
          ],
        ),
        throwsArgumentError,
      );

      expect(index.isEmpty, isTrue);
    });

    test('rejects duplicate chunks in add batches', () {
      final index = InMemoryVectorIndex(dimensions: 2);

      expect(
        () => index.add(
          const <TextChunk>[
            TextChunk(documentId: 'a', id: '0', text: 'alpha', tokenCount: 1),
            TextChunk(documentId: 'a', id: '0', text: 'alpha', tokenCount: 1),
          ],
          <Float32List>[
            Float32List.fromList(<double>[1, 0]),
            Float32List.fromList(<double>[0, 1]),
          ],
        ),
        throwsArgumentError,
      );

      expect(index.isEmpty, isTrue);
    });

    test('retriever embeds query and searches the vector index', () async {
      final chunks = <TextChunk>[
        const TextChunk(documentId: 'a', id: '0', text: 'alpha', tokenCount: 1),
        const TextChunk(documentId: 'b', id: '0', text: 'beta', tokenCount: 1),
      ];
      final index = InMemoryVectorIndex(dimensions: 2)
        ..add(chunks, <Float32List>[
          Float32List.fromList(<double>[1, 0]),
          Float32List.fromList(<double>[0, 1]),
        ]);
      final retriever = VectorIndexRetriever(
        embeddingModel: _FakeEmbeddingModel(
          Float32List.fromList(<double>[0.9, 0.1]),
        ),
        index: index,
      );

      final results = await retriever.retrieve('query');

      expect(results.first.chunk.documentId, 'a');
    });

    test('retriever snapshots vector index dimensions once', () async {
      final index = _ChangingDimensionsVectorIndex();
      final retriever = VectorIndexRetriever(
        embeddingModel: _FakeEmbeddingModel(
          Float32List.fromList(<double>[1, 0]),
        ),
        index: index,
      );

      final results = await retriever.retrieve('query');

      expect(results.single.chunk.id, '0');
      expect(index.searchQueryLength, 2);
    });

    test('retriever rejects invalid search args before embedding', () async {
      final retriever = VectorIndexRetriever(
        embeddingModel: const _ThrowingEmbeddingModel(),
        index: InMemoryVectorIndex(dimensions: 2),
      );

      await expectLater(
        retriever.retrieve('query', topK: 0),
        throwsArgumentError,
      );
      await expectLater(
        retriever.retrieve('query', minScore: double.nan),
        throwsArgumentError,
      );
      await expectLater(
        retriever.retrieve('bad\u0000query'),
        throwsArgumentError,
      );
    });

    test('retriever rejects malformed vector index dimensions', () async {
      const retriever = VectorIndexRetriever(
        embeddingModel: _ThrowingEmbeddingModel(),
        index: _MalformedVectorIndex(dimensions: 0, isEmpty: true),
      );

      await expectLater(retriever.retrieve('query'), throwsArgumentError);
    });

    test('retriever rejects malformed vector index results', () async {
      final retriever = VectorIndexRetriever(
        embeddingModel: _FakeEmbeddingModel(
          Float32List.fromList(<double>[1, 0]),
        ),
        index: const _MalformedVectorIndex(),
      );

      await expectLater(retriever.retrieve('query'), throwsArgumentError);
      await expectLater(
        VectorIndexRetriever(
          embeddingModel: _FakeEmbeddingModel(
            Float32List.fromList(<double>[1, 0]),
          ),
          index: const _MalformedVectorIndex(
            resultCount: 2,
            score: 1,
            duplicateChunks: true,
          ),
        ).retrieve('query'),
        throwsArgumentError,
      );
    });

    test('retriever rejects vector indexes that ignore topK', () async {
      final retriever = VectorIndexRetriever(
        embeddingModel: _FakeEmbeddingModel(
          Float32List.fromList(<double>[1, 0]),
        ),
        index: const _MalformedVectorIndex(resultCount: 2, score: 1),
      );

      await expectLater(
        retriever.retrieve('query', topK: 1),
        throwsArgumentError,
      );
    });

    test('retriever rejects vector indexes that ignore minScore', () async {
      final retriever = VectorIndexRetriever(
        embeddingModel: _FakeEmbeddingModel(
          Float32List.fromList(<double>[1, 0]),
        ),
        index: const _MalformedVectorIndex(score: -1),
      );

      await expectLater(
        retriever.retrieve('query', minScore: 0),
        throwsArgumentError,
      );
    });

    test(
      'retriever rejects malformed embedding vectors before search',
      () async {
        await expectLater(
          VectorIndexRetriever(
            embeddingModel: _FakeEmbeddingModel(Float32List(1)),
            index: const _MalformedVectorIndex(),
          ).retrieve('query'),
          throwsArgumentError,
        );
        await expectLater(
          VectorIndexRetriever(
            embeddingModel: _FakeEmbeddingModel(
              Float32List.fromList(<double>[double.nan, 0]),
            ),
            index: const _MalformedVectorIndex(),
          ).retrieve('query'),
          throwsArgumentError,
        );
      },
    );

    test('retriever snapshots custom vector index results', () async {
      final tags = <Object?>['original'];
      final metadata = <String, Object?>{'title': 'fixture', 'tags': tags};
      final retriever = VectorIndexRetriever(
        embeddingModel: _FakeEmbeddingModel(
          Float32List.fromList(<double>[1, 0]),
        ),
        index: _MutableMetadataVectorIndex(metadata),
      );

      final results = await retriever.retrieve('query');
      metadata['title'] = 'changed';
      tags[0] = 'changed';

      final resultMetadata = results.single.chunk.metadata;
      expect(resultMetadata['title'], 'fixture');
      expect(resultMetadata['tags'], <Object?>['original']);
      expect(() => resultMetadata['title'] = 'changed', throwsUnsupportedError);
      final resultTags = resultMetadata['tags']! as List<Object?>;
      expect(() => resultTags[0] = 'changed', throwsUnsupportedError);
    });

    test('retriever skips embedding when vector index is empty', () async {
      final retriever = VectorIndexRetriever(
        embeddingModel: const _ThrowingEmbeddingModel(),
        index: InMemoryVectorIndex(dimensions: 2),
      );

      expect(await retriever.retrieve('query'), isEmpty);
    });

    test('retriever can rerank vector candidates', () async {
      final chunks = <TextChunk>[
        const TextChunk(documentId: 'a', id: '0', text: 'alpha', tokenCount: 1),
        const TextChunk(documentId: 'b', id: '0', text: 'beta', tokenCount: 1),
      ];
      final index = InMemoryVectorIndex(dimensions: 2)
        ..add(chunks, <Float32List>[
          Float32List.fromList(<double>[1, 0]),
          Float32List.fromList(<double>[0, 1]),
        ]);
      final retriever = VectorIndexRetriever(
        embeddingModel: _FakeEmbeddingModel(
          Float32List.fromList(<double>[0.9, 0.1]),
        ),
        index: index,
        reranker: const _ReverseReranker(),
      );

      final results = await retriever.retrieve('query');

      expect(results.first.chunk.documentId, 'b');
      expect(
        () => results.add(
          const VectorSearchResult(
            chunk: TextChunk(
              documentId: 'c',
              id: '0',
              text: 'gamma',
              tokenCount: 1,
            ),
            score: 0,
          ),
        ),
        throwsUnsupportedError,
      );
    });

    test('retriever rejects non-finite reranker scores', () async {
      final index = InMemoryVectorIndex(dimensions: 2)
        ..add(
          const <TextChunk>[
            TextChunk(documentId: 'a', id: '0', text: 'alpha', tokenCount: 1),
          ],
          <Float32List>[
            Float32List.fromList(<double>[1, 0]),
          ],
        );
      final retriever = VectorIndexRetriever(
        embeddingModel: _FakeEmbeddingModel(
          Float32List.fromList(<double>[1, 0]),
        ),
        index: index,
        reranker: const _NonFiniteReranker(),
      );

      await expectLater(retriever.retrieve('query'), throwsArgumentError);
    });

    test(
      'retriever rejects reranker chunks outside vector candidates',
      () async {
        final index = InMemoryVectorIndex(dimensions: 2)
          ..add(
            const <TextChunk>[
              TextChunk(documentId: 'a', id: '0', text: 'alpha', tokenCount: 1),
              TextChunk(documentId: 'b', id: '0', text: 'beta', tokenCount: 1),
            ],
            <Float32List>[
              Float32List.fromList(<double>[1, 0]),
              Float32List.fromList(<double>[0, 1]),
            ],
          );
        final embedding = _FakeEmbeddingModel(
          Float32List.fromList(<double>[1, 0]),
        );

        await expectLater(
          VectorIndexRetriever(
            embeddingModel: embedding,
            index: index,
            reranker: const _UnknownChunkReranker(),
          ).retrieve('query'),
          throwsArgumentError,
        );
        await expectLater(
          VectorIndexRetriever(
            embeddingModel: embedding,
            index: index,
            reranker: const _DuplicateReranker(),
          ).retrieve('query'),
          throwsArgumentError,
        );
        await expectLater(
          VectorIndexRetriever(
            embeddingModel: embedding,
            index: index,
            reranker: const _DroppingReranker(),
          ).retrieve('query'),
          throwsArgumentError,
        );
      },
    );

    test(
      'retriever skips reranking when vector search has no candidates',
      () async {
        final retriever = VectorIndexRetriever(
          embeddingModel: _FakeEmbeddingModel(
            Float32List.fromList(<double>[1, 0]),
          ),
          index: InMemoryVectorIndex(dimensions: 2),
          reranker: const _ThrowingReranker(),
        );

        final results = await retriever.retrieve('query');

        expect(results, isEmpty);
      },
    );

    test('removes generated chunk ids without touching other documents', () {
      final chunks = <TextChunk>[
        const TextChunk(
          documentId: 'a',
          id: 'a:0',
          text: 'alpha',
          tokenCount: 1,
        ),
        const TextChunk(
          documentId: 'b',
          id: 'b:0',
          text: 'beta',
          tokenCount: 1,
        ),
      ];
      final index = InMemoryVectorIndex(dimensions: 2)
        ..add(chunks, <Float32List>[
          Float32List.fromList(<double>[1, 0]),
          Float32List.fromList(<double>[0, 1]),
        ]);

      expect(index.remove('a:0'), isTrue);
      final results = index.search(Float32List.fromList(<double>[1, 0]));

      expect(results.single.chunk.documentId, 'b');
    });

    test('removes a chunk exactly when document id is provided', () {
      final index = InMemoryVectorIndex(dimensions: 2)
        ..add(
          const <TextChunk>[
            TextChunk(documentId: 'a', id: '0', text: 'alpha', tokenCount: 1),
            TextChunk(documentId: 'b', id: '0', text: 'beta', tokenCount: 1),
          ],
          <Float32List>[
            Float32List.fromList(<double>[1, 0]),
            Float32List.fromList(<double>[0, 1]),
          ],
        );

      expect(index.remove('0', documentId: 'a'), isTrue);
      final results = index.search(Float32List.fromList(<double>[0, 1]));

      expect(results.single.chunk.documentId, 'b');
      expect(index.remove('0', documentId: 'a'), isFalse);
    });

    test('rejects NUL bytes in vector index ids', () {
      final index = InMemoryVectorIndex(dimensions: 2);

      expect(
        () => index.add(
          const <TextChunk>[
            TextChunk(
              documentId: 'doc\u0000bad',
              id: '0',
              text: 'alpha',
              tokenCount: 1,
            ),
          ],
          <Float32List>[
            Float32List.fromList(<double>[1, 0]),
          ],
        ),
        throwsArgumentError,
      );
      expect(
        () => index.add(
          const <TextChunk>[
            TextChunk(
              documentId: 'doc',
              id: '0',
              text: 'bad\u0000',
              tokenCount: 1,
            ),
          ],
          <Float32List>[
            Float32List.fromList(<double>[1, 0]),
          ],
        ),
        throwsArgumentError,
      );
      expect(
        () => index.add(
          const <TextChunk>[
            TextChunk(documentId: 'doc', id: 'blank', text: ' ', tokenCount: 1),
          ],
          <Float32List>[
            Float32List.fromList(<double>[1, 0]),
          ],
        ),
        throwsArgumentError,
      );
      expect(
        () => index.add(
          const <TextChunk>[
            TextChunk(
              documentId: 'doc\nbad',
              id: '0',
              text: 'alpha',
              tokenCount: 1,
            ),
          ],
          <Float32List>[
            Float32List.fromList(<double>[1, 0]),
          ],
        ),
        throwsArgumentError,
      );
      expect(() => index.remove(''), throwsArgumentError);
      expect(() => index.remove('0', documentId: ''), throwsArgumentError);
      expect(() => index.removeDocument(''), throwsArgumentError);
    });

    test('rejects non-positive vector index token counts', () {
      final index = InMemoryVectorIndex(dimensions: 2);

      expect(
        () => index.add(
          const <TextChunk>[
            TextChunk(
              documentId: 'doc',
              id: '0',
              text: 'alpha',
              tokenCount: -1,
            ),
          ],
          <Float32List>[
            Float32List.fromList(<double>[1, 0]),
          ],
        ),
        throwsArgumentError,
      );
      expect(
        () => index.add(
          const <TextChunk>[
            TextChunk(
              documentId: 'doc',
              id: 'zero',
              text: 'alpha',
              tokenCount: 0,
            ),
          ],
          <Float32List>[
            Float32List.fromList(<double>[1, 0]),
          ],
        ),
        throwsArgumentError,
      );
      expect(
        () => InMemoryVectorIndex.fromJson(const <String, Object?>{
          'version': 1,
          'dimensions': 2,
          'records': <Object?>[
            <String, Object?>{
              'chunk': <String, Object?>{
                'documentId': 'doc',
                'id': 'json',
                'text': 'bad',
                'tokenCount': -1,
              },
              'vector': <Object?>[1, 0],
            },
          ],
        }),
        throwsFormatException,
      );
    });

    test('rejects non-finite vector search inputs', () {
      final index = InMemoryVectorIndex(dimensions: 2)
        ..add(
          const <TextChunk>[
            TextChunk(documentId: 'doc', id: '0', text: 'alpha', tokenCount: 1),
          ],
          <Float32List>[
            Float32List.fromList(<double>[1, 0]),
          ],
        );

      expect(
        () => index.add(
          const <TextChunk>[
            TextChunk(documentId: 'doc', id: '1', text: 'beta', tokenCount: 1),
          ],
          <Float32List>[
            Float32List.fromList(<double>[double.nan, 0]),
          ],
        ),
        throwsArgumentError,
      );
      expect(
        () => index.search(
          Float32List.fromList(<double>[1, 0]),
          minScore: double.nan,
        ),
        throwsArgumentError,
      );
      expect(
        () => index.search(Float32List.fromList(<double>[double.infinity, 0])),
        throwsArgumentError,
      );
      expect(
        () => InMemoryVectorIndex.fromJson(<String, Object?>{
          'version': 1,
          'dimensions': 2,
          'records': <Object?>[
            <String, Object?>{
              'chunk': <String, Object?>{
                'documentId': 'doc',
                'id': 'json',
                'text': 'bad',
                'tokenCount': 1,
              },
              'vector': <Object?>[double.nan, 0],
            },
          ],
        }),
        throwsFormatException,
      );
    });

    test('persists and loads vector records', () async {
      final temp = await Directory.systemTemp.createTemp('fllamer_index_');
      try {
        final path = '${temp.path}/index.json';
        final index = InMemoryVectorIndex(dimensions: 2)
          ..add(
            <TextChunk>[
              TextChunk(
                documentId: 'doc',
                id: '0',
                text: 'alpha',
                tokenCount: 1,
                metadata: const <String, Object?>{'title': 'fixture'},
                sourceUri: Uri.parse('file:///fixture.txt'),
              ),
            ],
            <Float32List>[
              Float32List.fromList(<double>[1, 0]),
            ],
          );

        await index.persist(path);
        final loaded = await InMemoryVectorIndex.load(path);
        final results = loaded.search(Float32List.fromList(<double>[1, 0]));

        expect(results.single.chunk.text, 'alpha');
        expect(results.single.chunk.metadata['title'], 'fixture');
        expect(
          results.single.chunk.sourceUri,
          Uri.parse('file:///fixture.txt'),
        );
      } finally {
        await temp.delete(recursive: true);
      }
    });

    test('rejects empty vector source URIs before indexing', () {
      expect(
        () => InMemoryVectorIndex(dimensions: 2).add(
          <TextChunk>[
            TextChunk(
              documentId: 'doc',
              id: '0',
              text: 'alpha',
              tokenCount: 1,
              sourceUri: Uri.parse(''),
            ),
          ],
          <Float32List>[
            Float32List.fromList(<double>[1, 0]),
          ],
        ),
        throwsArgumentError,
      );
      expect(
        () => InMemoryVectorIndex(dimensions: 2).add(
          <TextChunk>[
            TextChunk(
              documentId: 'doc',
              id: '0',
              text: 'alpha',
              tokenCount: 1,
              sourceUri: Uri.parse('file:///bad%00path'),
            ),
          ],
          <Float32List>[
            Float32List.fromList(<double>[1, 0]),
          ],
        ),
        throwsArgumentError,
      );
      expect(
        () => InMemoryVectorIndex(dimensions: 2).add(
          <TextChunk>[
            TextChunk(
              documentId: 'doc',
              id: '0',
              text: 'alpha',
              tokenCount: 1,
              sourceUri: Uri.parse('file:///bad%0Apath'),
            ),
          ],
          <Float32List>[
            Float32List.fromList(<double>[1, 0]),
          ],
        ),
        throwsArgumentError,
      );
    });

    test('rejects non-JSON vector metadata before indexing', () {
      final index = InMemoryVectorIndex(dimensions: 2);

      expect(
        () => index.add(
          <TextChunk>[
            TextChunk(
              documentId: 'doc',
              id: '0',
              text: 'alpha',
              tokenCount: 1,
              metadata: <String, Object?>{'bad': Object()},
            ),
          ],
          <Float32List>[
            Float32List.fromList(<double>[1, 0]),
          ],
        ),
        throwsArgumentError,
      );
    });

    test('rejects NUL bytes in vector metadata before indexing', () {
      final index = InMemoryVectorIndex(dimensions: 2);

      void addWithMetadata(Map<String, Object?> metadata) {
        index.add(
          <TextChunk>[
            TextChunk(
              documentId: 'doc',
              id: '0',
              text: 'alpha',
              tokenCount: 1,
              metadata: metadata,
            ),
          ],
          <Float32List>[
            Float32List.fromList(<double>[1, 0]),
          ],
        );
      }

      expect(
        () => addWithMetadata(<String, Object?>{'title': 'bad\u0000value'}),
        throwsArgumentError,
      );
      expect(
        () => addWithMetadata(<String, Object?>{'bad\u0000key': 'value'}),
        throwsArgumentError,
      );
    });

    test('rejects cyclic vector metadata before indexing', () {
      final metadata = <String, Object?>{};
      metadata['self'] = metadata;
      final index = InMemoryVectorIndex(dimensions: 2);

      expect(
        () => index.add(
          <TextChunk>[
            TextChunk(
              documentId: 'doc',
              id: '0',
              text: 'alpha',
              tokenCount: 1,
              metadata: metadata,
            ),
          ],
          <Float32List>[
            Float32List.fromList(<double>[1, 0]),
          ],
        ),
        throwsArgumentError,
      );
    });

    test('snapshots vector metadata at add and JSON export', () {
      final nested = <Object?>['fixture'];
      final metadata = <String, Object?>{'tags': nested};
      final index = InMemoryVectorIndex(dimensions: 2)
        ..add(
          <TextChunk>[
            TextChunk(
              documentId: 'doc',
              id: '0',
              text: 'alpha',
              tokenCount: 1,
              metadata: metadata,
            ),
          ],
          <Float32List>[
            Float32List.fromList(<double>[1, 0]),
          ],
        );

      nested[0] = 'changed';
      final results = index.search(Float32List.fromList(<double>[1, 0]));
      final storedTags =
          results.single.chunk.metadata['tags']! as List<Object?>;
      expect(storedTags.single, 'fixture');
      expect(() => storedTags[0] = 'changed', throwsUnsupportedError);

      final json = index.toJson();
      expect(() => json['records'] = <Object?>[], throwsUnsupportedError);
      final records = json['records']! as List<Object?>;
      expect(() => records.add(<String, Object?>{}), throwsUnsupportedError);
      final record = records.single! as Map<String, Object?>;
      expect(() => record['vector'] = <double>[0, 1], throwsUnsupportedError);
      final chunk = record['chunk']! as Map<String, Object?>;
      expect(() => chunk['id'] = 'changed', throwsUnsupportedError);
      final exportedVector = record['vector']! as List<Object?>;
      expect(() => exportedVector[0] = 0.5, throwsUnsupportedError);
      final exportedMetadata = chunk['metadata']! as Map<String, Object?>;
      final exportedTags = exportedMetadata['tags']! as List<Object?>;

      expect(exportedTags.single, 'fixture');
      expect(() => exportedTags[0] = 'changed', throwsUnsupportedError);
    });

    test('exports vector records deterministically', () {
      final index = InMemoryVectorIndex(dimensions: 2)
        ..add(
          const <TextChunk>[
            TextChunk(documentId: 'b', id: '0', text: 'beta', tokenCount: 1),
            TextChunk(documentId: 'a', id: '1', text: 'alpha', tokenCount: 1),
            TextChunk(documentId: 'a', id: '0', text: 'alpha', tokenCount: 1),
          ],
          <Float32List>[
            Float32List.fromList(<double>[0, 1]),
            Float32List.fromList(<double>[1, 0]),
            Float32List.fromList(<double>[1, 0]),
          ],
        );

      final records = index.toJson()['records']! as List<Object?>;

      expect(
        records.map((record) {
          final chunk =
              (record! as Map<String, Object?>)['chunk']!
                  as Map<String, Object?>;
          return '${chunk['documentId']}:${chunk['id']}';
        }),
        <String>['a:0', 'a:1', 'b:0'],
      );
    });

    test('rejects invalid persisted vector index paths', () async {
      final index = InMemoryVectorIndex(dimensions: 2);

      expect(() => index.persist(''), throwsArgumentError);
      expect(() => index.persist('bad\u0000path'), throwsArgumentError);
      expect(() => index.persist('bad\npath'), throwsArgumentError);
      await expectLater(InMemoryVectorIndex.load(''), throwsArgumentError);
      await expectLater(
        InMemoryVectorIndex.load('bad\u0000path'),
        throwsArgumentError,
      );
      await expectLater(
        InMemoryVectorIndex.load('bad\npath'),
        throwsArgumentError,
      );
    });

    test('maps persisted vector index file failures', () async {
      final temp = await Directory.systemTemp.createTemp('fllamer_index_');
      try {
        final index = InMemoryVectorIndex(dimensions: 2);

        await expectLater(
          InMemoryVectorIndex.load('${temp.path}/missing.json'),
          throwsA(isA<RagIndexException>()),
        );
        await expectLater(
          index.persist(temp.path),
          throwsA(isA<RagIndexException>()),
        );
        if (!Platform.isWindows) {
          final target = File('${temp.path}/target.json');
          await target.writeAsString('original');
          final link = Link('${temp.path}/linked.json');
          await link.create(target.path);
          await expectLater(
            index.persist(link.path),
            throwsA(isA<RagIndexException>()),
          );
          await expectLater(
            InMemoryVectorIndex.load(link.path),
            throwsA(isA<RagIndexException>()),
          );
          expect(await link.exists(), isTrue);
          expect(await target.readAsString(), 'original');
        }
      } finally {
        await temp.delete(recursive: true);
      }
    });

    test('labels malformed persisted vector index JSON', () async {
      final temp = await Directory.systemTemp.createTemp('fllamer_index_');
      try {
        final path = '${temp.path}/index.json';
        await File(path).writeAsString('{');

        await expectLater(
          InMemoryVectorIndex.load(path),
          throwsA(
            isA<FormatException>().having(
              (error) => error.message,
              'message',
              contains('vector index JSON is malformed'),
            ),
          ),
        );
      } finally {
        await temp.delete(recursive: true);
      }
    });

    test('rejects unsupported persisted vector index versions', () {
      expect(
        () => InMemoryVectorIndex.fromJson(const <String, Object?>{
          'version': 2,
          'dimensions': 2,
          'records': <Object?>[],
        }),
        throwsFormatException,
      );
    });

    test('rejects malformed persisted vector index headers', () {
      expect(
        () => InMemoryVectorIndex.fromJson(const <String, Object?>{
          'version': 1,
          'dimensions': '2',
        }),
        throwsFormatException,
      );
      expect(
        () => InMemoryVectorIndex.fromJson(const <String, Object?>{
          'version': 1,
          'dimensions': 0,
        }),
        throwsFormatException,
      );
      expect(
        () => InMemoryVectorIndex.fromJson(const <String, Object?>{
          'version': 1,
          'dimensions': 2,
          'normalize': null,
        }),
        throwsFormatException,
      );
      expect(
        () => InMemoryVectorIndex.fromJson(const <String, Object?>{
          'version': 1,
          'dimensions': 2,
          'normalize': 'true',
        }),
        throwsFormatException,
      );
      expect(
        () => InMemoryVectorIndex.fromJson(const <String, Object?>{
          'version': 1,
          'dimensions': 2,
          'extra': true,
        }),
        throwsFormatException,
      );
      expect(
        () => InMemoryVectorIndex.fromJson(const <String, Object?>{
          'version': 1,
          'dimensions': 2,
          'records': null,
        }),
        throwsFormatException,
      );
      expect(
        () => InMemoryVectorIndex.fromJson(const <String, Object?>{
          'version': 1,
          'dimensions': 2,
          'records': <String, Object?>{},
        }),
        throwsFormatException,
      );
    });

    test('rejects malformed persisted vector records', () {
      Map<String, Object?> indexWith(Object? record) {
        return <String, Object?>{
          'version': 1,
          'dimensions': 2,
          'records': <Object?>[record],
        };
      }

      expect(
        () => InMemoryVectorIndex.fromJson(indexWith('bad')),
        throwsFormatException,
      );
      expect(
        () => InMemoryVectorIndex.fromJson(
          indexWith(const <Object?, Object?>{1: 'bad'}),
        ),
        throwsFormatException,
      );
      expect(
        () => InMemoryVectorIndex.fromJson(
          indexWith(const <String, Object?>{'chunk': 'bad', 'vector': <int>[]}),
        ),
        throwsFormatException,
      );
      expect(
        () => InMemoryVectorIndex.fromJson(
          indexWith(const <String, Object?>{
            'chunk': <Object?, Object?>{1: 'bad'},
            'vector': <int>[],
          }),
        ),
        throwsFormatException,
      );
      expect(
        () => InMemoryVectorIndex.fromJson(
          indexWith(const <String, Object?>{
            'extra': true,
            'chunk': <String, Object?>{
              'documentId': 'doc',
              'id': '0',
              'text': 'alpha',
              'tokenCount': 1,
            },
            'vector': <int>[1, 0],
          }),
        ),
        throwsFormatException,
      );
      expect(
        () => InMemoryVectorIndex.fromJson(
          indexWith(const <String, Object?>{
            'chunk': <String, Object?>{
              'documentId': 'doc',
              'id': '0',
              'extra': true,
              'text': 'alpha',
              'tokenCount': 1,
            },
            'vector': <int>[1, 0],
          }),
        ),
        throwsFormatException,
      );
      expect(
        () => InMemoryVectorIndex.fromJson(const <String, Object?>{
          'version': 1,
          'dimensions': 2,
          'records': <Object?>[
            <String, Object?>{
              'chunk': <String, Object?>{
                'documentId': 'doc',
                'id': '0',
                'text': 'alpha',
                'tokenCount': 1,
              },
              'vector': <int>[1, 0],
            },
            <String, Object?>{
              'chunk': <String, Object?>{
                'documentId': 'doc',
                'id': '0',
                'text': 'alpha again',
                'tokenCount': 2,
              },
              'vector': <int>[0, 1],
            },
          ],
        }),
        throwsFormatException,
      );
      expect(
        () => InMemoryVectorIndex.fromJson(
          indexWith(const <String, Object?>{
            'chunk': <String, Object?>{
              'documentId': 'doc',
              'id': '0',
              'text': 'alpha',
              'tokenCount': 1,
              'sourceUri': 'file:///bad%0Dpath',
            },
            'vector': <int>[1, 0],
          }),
        ),
        throwsFormatException,
      );
      expect(
        () => InMemoryVectorIndex.fromJson(
          indexWith(const <String, Object?>{
            'chunk': <String, Object?>{
              'documentId': 'doc',
              'id': '0',
              'text': 'alpha',
              'tokenCount': 1,
              'metadata': null,
            },
            'vector': <int>[1, 0],
          }),
        ),
        throwsFormatException,
      );
      expect(
        () => InMemoryVectorIndex.fromJson(
          indexWith(const <String, Object?>{
            'chunk': <String, Object?>{
              'documentId': 'doc',
              'id': '0',
              'text': 'alpha',
              'tokenCount': 1,
              'metadata': <String, Object?>{'title': 'bad\u0000value'},
            },
            'vector': <int>[1, 0],
          }),
        ),
        throwsFormatException,
      );
      expect(
        () => InMemoryVectorIndex.fromJson(
          indexWith(const <String, Object?>{
            'chunk': <String, Object?>{
              'documentId': 'doc',
              'id': '0',
              'text': 'alpha',
              'tokenCount': 1,
              'metadata': <String, Object?>{'bad\u0000key': 'value'},
            },
            'vector': <int>[1, 0],
          }),
        ),
        throwsFormatException,
      );
      expect(
        () => InMemoryVectorIndex.fromJson(
          indexWith(const <String, Object?>{
            'chunk': <String, Object?>{
              'documentId': 'doc',
              'id': '0',
              'text': 'alpha',
              'tokenCount': 1,
              'sourceUri': '',
            },
            'vector': <int>[1, 0],
          }),
        ),
        throwsFormatException,
      );
      expect(
        () => InMemoryVectorIndex.fromJson(
          indexWith(const <String, Object?>{
            'chunk': <String, Object?>{
              'documentId': 'doc',
              'id': '0',
              'text': 'alpha',
              'tokenCount': '1',
            },
            'vector': <int>[1, 0],
          }),
        ),
        throwsFormatException,
      );
      expect(
        () => InMemoryVectorIndex.fromJson(
          indexWith(const <String, Object?>{
            'chunk': <String, Object?>{
              'documentId': 'doc',
              'id': '0',
              'text': ' ',
              'tokenCount': 1,
            },
            'vector': <int>[1, 0],
          }),
        ),
        throwsFormatException,
      );
      expect(
        () => InMemoryVectorIndex.fromJson(
          indexWith(const <String, Object?>{
            'chunk': <String, Object?>{
              'documentId': 'doc',
              'id': '0',
              'text': 'alpha',
              'tokenCount': 1,
              'sourceUri': 'file:///bad\u0000path',
            },
            'vector': <int>[1, 0],
          }),
        ),
        throwsFormatException,
      );
      expect(
        () => InMemoryVectorIndex.fromJson(
          indexWith(const <String, Object?>{
            'chunk': <String, Object?>{
              'documentId': 'doc',
              'id': '0',
              'text': 'alpha',
              'tokenCount': 1,
              'sourceUri': 'http://[::1',
            },
            'vector': <int>[1, 0],
          }),
        ),
        throwsFormatException,
      );
      expect(
        () => InMemoryVectorIndex.fromJson(
          indexWith(const <String, Object?>{
            'chunk': <String, Object?>{
              'documentId': 'doc',
              'id': '0',
              'text': 'alpha',
              'tokenCount': 1,
            },
            'vector': <Object?>[1],
          }),
        ),
        throwsFormatException,
      );
      expect(
        () => InMemoryVectorIndex.fromJson(
          indexWith(const <String, Object?>{
            'chunk': <String, Object?>{
              'documentId': 'doc',
              'id': '0',
              'text': 'alpha',
              'tokenCount': 1,
              'metadata': <String, Object?>{
                'nested': <Object?, Object?>{1: 'bad'},
              },
            },
            'vector': <int>[1, 0],
          }),
        ),
        throwsFormatException,
      );
      expect(
        () => InMemoryVectorIndex.fromJson(
          indexWith(<String, Object?>{
            'chunk': <String, Object?>{
              'documentId': 'doc',
              'id': '0',
              'text': 'alpha',
              'tokenCount': 1,
              'metadata': <String, Object?>{'bad': Object()},
            },
            'vector': <int>[1, 0],
          }),
        ),
        throwsFormatException,
      );
      expect(
        () => InMemoryVectorIndex.fromJson(
          indexWith(const <String, Object?>{
            'chunk': <String, Object?>{
              'documentId': 'doc',
              'id': '0',
              'text': 'alpha',
              'tokenCount': 1,
              'metadata': <Object?, Object?>{1: 'bad'},
            },
            'vector': <int>[1, 0],
          }),
        ),
        throwsFormatException,
      );
      expect(
        () => InMemoryVectorIndex.fromJson(
          indexWith(const <String, Object?>{
            'chunk': <String, Object?>{
              'documentId': 'doc',
              'id': '0',
              'text': 'alpha',
              'tokenCount': 1,
            },
            'vector': 'bad',
          }),
        ),
        throwsFormatException,
      );
      expect(
        () => InMemoryVectorIndex.fromJson(
          indexWith(const <String, Object?>{
            'chunk': <String, Object?>{
              'documentId': 'doc',
              'id': '0',
              'text': 'alpha',
              'tokenCount': 1,
            },
            'vector': <Object?>[1, 'bad'],
          }),
        ),
        throwsFormatException,
      );
    });

    test('rejects non-object persisted vector index roots', () async {
      final temp = await Directory.systemTemp.createTemp('fllamer_index_');
      try {
        final path = '${temp.path}/index.json';
        await File(path).writeAsString('[]');

        await expectLater(
          InMemoryVectorIndex.load(path),
          throwsFormatException,
        );
      } finally {
        await temp.delete(recursive: true);
      }
    });

    test('normalizes loaded vector records', () {
      final index = InMemoryVectorIndex.fromJson(const <String, Object?>{
        'version': 1,
        'dimensions': 2,
        'records': <Object?>[
          <String, Object?>{
            'chunk': <String, Object?>{
              'documentId': 'a',
              'id': '0',
              'text': 'scaled',
              'tokenCount': 1,
            },
            'vector': <Object?>[10, 0],
          },
          <String, Object?>{
            'chunk': <String, Object?>{
              'documentId': 'b',
              'id': '0',
              'text': 'unit',
              'tokenCount': 1,
            },
            'vector': <Object?>[0, 1],
          },
        ],
      });

      final results = index.search(Float32List.fromList(<double>[1, 0]));

      expect(results.first.score, 1);
      expect(results.first.score, greaterThan(results.last.score));
    });
  });

  group('RAG prompt builder', () {
    test('stops at the token budget', () {
      final prompt = const RagPromptBuilder().buildContext(const <TextChunk>[
        TextChunk(documentId: 'a', id: '0', text: 'one two', tokenCount: 2),
        TextChunk(documentId: 'b', id: '0', text: 'three', tokenCount: 1),
      ], maxContextTokens: 2);

      expect(prompt.usedTokens, 2);
      expect(prompt.includedChunks, hasLength(1));
      expect(prompt.citations.single.documentId, 'a');
      expect(prompt.citations.single.chunkId, '0');
      expect(
        () => prompt.includedChunks.add(
          const TextChunk(
            documentId: 'c',
            id: '0',
            text: 'extra',
            tokenCount: 1,
          ),
        ),
        throwsUnsupportedError,
      );
      expect(() => prompt.citations.clear(), throwsUnsupportedError);
      expect(prompt.context, contains('[source: a#0]'));
      expect(prompt.context, isNot(contains('three')));
    });

    test(
      'tokenizer-backed context budgets headers and source labels',
      () async {
        Future<List<int>> tokenize(String text) async {
          final count = text.trim().split(RegExp(r'\s+')).length;
          return List<int>.generate(count, (index) => index);
        }

        final prompt = await const RagPromptBuilder().buildContextWithTokenizer(
          const <TextChunk>[
            TextChunk(documentId: 'a', id: '0', text: 'one two', tokenCount: 2),
            TextChunk(documentId: 'b', id: '0', text: 'three', tokenCount: 1),
          ],
          maxContextTokens: 9,
          tokenize: tokenize,
        );

        expect(prompt.includedChunks.single.documentId, 'a');
        expect(prompt.context, isNot(contains('three')));
        expect(prompt.usedTokens, (await tokenize(prompt.context)).length);
        expect(prompt.usedTokens, 7);
        await expectLater(
          const RagPromptBuilder().buildContextWithTokenizer(
            const <TextChunk>[
              TextChunk(documentId: 'a', id: '0', text: 'one', tokenCount: 1),
            ],
            maxContextTokens: 8,
            tokenize: (_) async => const <int>[],
          ),
          throwsArgumentError,
        );
        await expectLater(
          const RagPromptBuilder().buildContextWithTokenizer(
            const <TextChunk>[
              TextChunk(documentId: 'a', id: '0', text: 'one', tokenCount: 1),
            ],
            maxContextTokens: 8,
            tokenize: (_) async => const <int>[-1],
          ),
          throwsArgumentError,
        );
      },
    );

    test('chat prompt budget counts the full formatted message list', () async {
      Future<int> countTokens(List<ChatMessage> messages) async {
        var count = 0;
        for (final message in messages) {
          count += 1 + message.text.trim().split(RegExp(r'\s+')).length;
        }
        return count;
      }

      final prompt = await const RagPromptBuilder().buildChatPrompt(
        question: 'What?',
        systemPrompt: 'Policy',
        chunks: const <TextChunk>[
          TextChunk(documentId: 'a', id: '0', text: 'one two', tokenCount: 2),
          TextChunk(documentId: 'b', id: '0', text: 'three', tokenCount: 1),
        ],
        maxPromptTokens: 14,
        countTokens: countTokens,
      );

      expect(prompt.usedTokens, await countTokens(prompt.messages));
      expect(prompt.usedTokens, 12);
      expect(prompt.includedChunks.single.documentId, 'a');
      expect(prompt.citations.single.chunkId, '0');
      expect(prompt.messages.map((message) => message.role), <ChatRole>[
        ChatRole.system,
        ChatRole.system,
        ChatRole.user,
      ]);
      expect(prompt.messages[1].text, isNot(contains('three')));
      expect(() => prompt.messages.clear(), throwsUnsupportedError);
      expect(() => prompt.includedChunks.clear(), throwsUnsupportedError);
      expect(() => prompt.citations.clear(), throwsUnsupportedError);

      await expectLater(
        const RagPromptBuilder().buildChatPrompt(
          question: 'What?',
          systemPrompt: 'Policy',
          chunks: const <TextChunk>[],
          maxPromptTokens: 3,
          countTokens: countTokens,
        ),
        throwsArgumentError,
      );
      await expectLater(
        const RagPromptBuilder().buildChatPrompt(
          question: 'What?',
          chunks: const <TextChunk>[],
          maxPromptTokens: 3,
          countTokens: (_) async => 0,
        ),
        throwsArgumentError,
      );
    });

    test('skips oversized chunks and keeps later fitting chunks', () {
      final prompt = const RagPromptBuilder().buildContext(const <TextChunk>[
        TextChunk(documentId: 'a', id: '0', text: 'too large', tokenCount: 4),
        TextChunk(documentId: 'b', id: '0', text: 'fits', tokenCount: 1),
      ], maxContextTokens: 2);

      expect(prompt.usedTokens, 1);
      expect(prompt.includedChunks.single.documentId, 'b');
      expect(prompt.context, contains('fits'));
      expect(prompt.context, isNot(contains('too large')));
    });

    test('builds chat messages from retrieved context', () {
      final messages = const RagPromptBuilder().buildChatMessages(
        question: 'What fits?',
        systemPrompt: 'Answer from local context.',
        chunks: <TextChunk>[
          TextChunk(documentId: 'a', id: '0', text: 'too large', tokenCount: 4),
          TextChunk(documentId: 'b', id: '0', text: 'fits', tokenCount: 1),
        ],
        maxContextTokens: 2,
      );

      expect(messages.map((message) => message.role), <ChatRole>[
        ChatRole.system,
        ChatRole.system,
        ChatRole.user,
      ]);
      expect(messages[1].text, contains('[source: b#0]'));
      expect(messages.last.text, 'What fits?');
      expect(
        () => const RagPromptBuilder().buildChatMessages(
          question: '',
          chunks: <TextChunk>[],
          maxContextTokens: 1,
        ),
        throwsArgumentError,
      );
    });

    test('preserves citation source spans', () {
      final sourceUri = Uri.parse('file:///fixture.txt');
      final prompt = const RagPromptBuilder().buildContext(<TextChunk>[
        TextChunk(
          documentId: 'doc',
          id: '0',
          text: 'alpha',
          tokenCount: 1,
          metadata: <String, Object?>{
            'start': 2,
            'end': 7,
            'tokenStart': 1,
            'tokenEnd': 2,
          },
          sourceUri: sourceUri,
        ),
      ], maxContextTokens: 1);

      final citation = prompt.citations.single;
      expect(citation.documentId, 'doc');
      expect(citation.chunkId, '0');
      expect(citation.sourceUri, sourceUri);
      expect(citation.start, 2);
      expect(citation.end, 7);
      expect(citation.tokenStart, 1);
      expect(citation.tokenEnd, 2);
    });

    test('snapshots included chunk metadata', () {
      final tags = <Object?>['fixture'];
      final metadata = <String, Object?>{'title': 'fixture', 'tags': tags};
      final prompt = const RagPromptBuilder().buildContext(<TextChunk>[
        TextChunk(
          documentId: 'doc',
          id: '0',
          text: 'alpha',
          tokenCount: 1,
          metadata: metadata,
        ),
      ], maxContextTokens: 1);
      metadata['title'] = 'changed';
      tags[0] = 'changed';

      expect(prompt.includedChunks.single.metadata['title'], 'fixture');
      final storedTags =
          prompt.includedChunks.single.metadata['tags']! as List<Object?>;
      expect(storedTags.single, 'fixture');
      expect(
        () => prompt.includedChunks.single.metadata['title'] = 'changed',
        throwsUnsupportedError,
      );
      expect(() => storedTags[0] = 'changed', throwsUnsupportedError);

      expect(
        () => const RagPromptBuilder().buildContext(<TextChunk>[
          TextChunk(
            documentId: 'doc',
            id: '0',
            text: 'alpha',
            tokenCount: 1,
            metadata: <String, Object?>{'bad': Object()},
          ),
        ], maxContextTokens: 1),
        throwsArgumentError,
      );

      final cyclicMetadata = <String, Object?>{};
      cyclicMetadata['self'] = cyclicMetadata;
      expect(
        () => const RagPromptBuilder().buildContext(<TextChunk>[
          TextChunk(
            documentId: 'doc',
            id: '0',
            text: 'alpha',
            tokenCount: 1,
            metadata: cyclicMetadata,
          ),
        ], maxContextTokens: 1),
        throwsArgumentError,
      );
    });

    test('rejects non-positive chunk token counts', () {
      expect(
        () => const RagPromptBuilder().buildContext(const <TextChunk>[
          TextChunk(documentId: 'a', id: '0', text: 'bad', tokenCount: -1),
        ], maxContextTokens: 2),
        throwsArgumentError,
      );
      expect(
        () => const RagPromptBuilder().buildContext(const <TextChunk>[
          TextChunk(documentId: 'a', id: '0', text: 'bad', tokenCount: 0),
        ], maxContextTokens: 2),
        throwsArgumentError,
      );
    });

    test('rejects empty chunk source ids', () {
      expect(
        () => const RagPromptBuilder().buildContext(const <TextChunk>[
          TextChunk(documentId: '', id: '0', text: 'bad', tokenCount: 1),
        ], maxContextTokens: 2),
        throwsArgumentError,
      );
      expect(
        () => const RagPromptBuilder().buildContext(const <TextChunk>[
          TextChunk(documentId: 'a\nb', id: '0', text: 'bad', tokenCount: 1),
        ], maxContextTokens: 2),
        throwsArgumentError,
      );
      expect(
        () => const RagPromptBuilder().buildContext(const <TextChunk>[
          TextChunk(documentId: 'a', id: '0', text: 'bad\u0000', tokenCount: 1),
        ], maxContextTokens: 2),
        throwsArgumentError,
      );
      expect(
        () => const RagPromptBuilder().buildContext(const <TextChunk>[
          TextChunk(documentId: 'a', id: '0', text: ' ', tokenCount: 1),
        ], maxContextTokens: 2),
        throwsArgumentError,
      );
      expect(
        () => const RagPromptBuilder().buildContext(<TextChunk>[
          TextChunk(
            documentId: 'a',
            id: '0',
            text: 'ok',
            tokenCount: 1,
            sourceUri: Uri(),
          ),
        ], maxContextTokens: 2),
        throwsArgumentError,
      );
      expect(
        () => RagPromptBuilder().buildContext(<TextChunk>[
          TextChunk(
            documentId: 'a',
            id: '0',
            text: 'ok',
            tokenCount: 1,
            sourceUri: Uri.parse('file:///bad%00path'),
          ),
        ], maxContextTokens: 2),
        throwsArgumentError,
      );
    });

    test('rejects invalid prompt headers', () {
      expect(
        () => const RagPromptBuilder(header: '  \n\t').buildContext(
          const <TextChunk>[
            TextChunk(documentId: 'a', id: '0', text: 'ok', tokenCount: 1),
          ],
          maxContextTokens: 2,
        ),
        throwsArgumentError,
      );
      expect(
        () => const RagPromptBuilder(header: 'bad\u0000header').buildContext(
          const <TextChunk>[
            TextChunk(documentId: 'a', id: '0', text: 'ok', tokenCount: 1),
          ],
          maxContextTokens: 2,
        ),
        throwsArgumentError,
      );
    });

    test('rejects invalid citation spans', () {
      expect(
        () => const RagPromptBuilder().buildContext(const <TextChunk>[
          TextChunk(
            documentId: 'a',
            id: '0',
            text: 'bad',
            tokenCount: 1,
            metadata: <String, Object?>{'start': '0'},
          ),
        ], maxContextTokens: 2),
        throwsArgumentError,
      );
      expect(
        () => const RagPromptBuilder().buildContext(const <TextChunk>[
          TextChunk(
            documentId: 'a',
            id: '0',
            text: 'bad',
            tokenCount: 1,
            metadata: <String, Object?>{'start': 0},
          ),
        ], maxContextTokens: 2),
        throwsArgumentError,
      );
      expect(
        () => const RagPromptBuilder().buildContext(const <TextChunk>[
          TextChunk(
            documentId: 'a',
            id: '0',
            text: 'bad',
            tokenCount: 1,
            metadata: <String, Object?>{'tokenEnd': 1},
          ),
        ], maxContextTokens: 2),
        throwsArgumentError,
      );
      expect(
        () => const RagPromptBuilder().buildContext(const <TextChunk>[
          TextChunk(
            documentId: 'a',
            id: '0',
            text: 'bad',
            tokenCount: 1,
            metadata: <String, Object?>{'start': 2, 'end': 1},
          ),
        ], maxContextTokens: 2),
        throwsArgumentError,
      );
      expect(
        () => const RagPromptBuilder().buildContext(const <TextChunk>[
          TextChunk(
            documentId: 'a',
            id: '0',
            text: 'bad',
            tokenCount: 1,
            metadata: <String, Object?>{'start': 1, 'end': 1},
          ),
        ], maxContextTokens: 2),
        throwsArgumentError,
      );
      expect(
        () => const RagPromptBuilder().buildContext(const <TextChunk>[
          TextChunk(
            documentId: 'a',
            id: '0',
            text: 'bad',
            tokenCount: 1,
            metadata: <String, Object?>{'tokenStart': 1, 'tokenEnd': 1},
          ),
        ], maxContextTokens: 2),
        throwsArgumentError,
      );
    });
  });
}

final class _FakeEmbeddingModel implements EmbeddingModel {
  const _FakeEmbeddingModel(this.embedding);

  final Float32List embedding;

  @override
  Future<Float32List> embedText(String text) async => embedding;
}

final class _ThrowingEmbeddingModel implements EmbeddingModel {
  const _ThrowingEmbeddingModel();

  @override
  Future<Float32List> embedText(String text) async {
    throw StateError('embedding should not run');
  }
}

final class _MalformedVectorIndex implements VectorIndex {
  const _MalformedVectorIndex({
    this.dimensions = 2,
    this.isEmpty = false,
    this.resultCount = 1,
    this.score = double.infinity,
    this.duplicateChunks = false,
  });

  @override
  final int dimensions;

  @override
  final bool isEmpty;

  final int resultCount;
  final double score;
  final bool duplicateChunks;

  @override
  void add(List<TextChunk> chunks, Iterable<Float32List> vectors) {
    throw UnsupportedError('not needed');
  }

  @override
  void clear() {}

  @override
  Future<void> persist(String path) async {}

  @override
  bool remove(String chunkId, {String? documentId}) => false;

  @override
  void removeDocument(String documentId) {}

  @override
  List<VectorSearchResult> search(
    Float32List query, {
    int topK = 5,
    double minScore = double.negativeInfinity,
  }) {
    return List<VectorSearchResult>.unmodifiable(<VectorSearchResult>[
      for (var i = 0; i < resultCount; i += 1)
        VectorSearchResult(
          chunk: TextChunk(
            documentId: 'a',
            id: duplicateChunks ? '0' : '$i',
            text: 'alpha',
            tokenCount: 1,
          ),
          score: score,
        ),
    ]);
  }
}

final class _ChangingDimensionsVectorIndex implements VectorIndex {
  int _reads = 0;
  int searchQueryLength = 0;

  @override
  int get dimensions {
    _reads += 1;
    return _reads == 1 ? 2 : 3;
  }

  @override
  bool get isEmpty => false;

  @override
  void add(List<TextChunk> chunks, Iterable<Float32List> vectors) {
    throw UnsupportedError('not needed');
  }

  @override
  void clear() {}

  @override
  Future<void> persist(String path) async {}

  @override
  bool remove(String chunkId, {String? documentId}) => false;

  @override
  void removeDocument(String documentId) {}

  @override
  List<VectorSearchResult> search(
    Float32List query, {
    int topK = 5,
    double minScore = double.negativeInfinity,
  }) {
    searchQueryLength = query.length;
    return const <VectorSearchResult>[
      VectorSearchResult(
        chunk: TextChunk(
          documentId: 'a',
          id: '0',
          text: 'alpha',
          tokenCount: 1,
        ),
        score: 1,
      ),
    ];
  }
}

final class _MutableMetadataVectorIndex implements VectorIndex {
  const _MutableMetadataVectorIndex(this.metadata);

  final Map<String, Object?> metadata;

  @override
  int get dimensions => 2;

  @override
  bool get isEmpty => false;

  @override
  void add(List<TextChunk> chunks, Iterable<Float32List> vectors) {
    throw UnsupportedError('not needed');
  }

  @override
  void clear() {}

  @override
  Future<void> persist(String path) async {}

  @override
  bool remove(String chunkId, {String? documentId}) => false;

  @override
  void removeDocument(String documentId) {}

  @override
  List<VectorSearchResult> search(
    Float32List query, {
    int topK = 5,
    double minScore = double.negativeInfinity,
  }) {
    return <VectorSearchResult>[
      VectorSearchResult(
        chunk: TextChunk(
          documentId: 'a',
          id: '0',
          text: 'alpha',
          tokenCount: 1,
          metadata: metadata,
        ),
        score: 1,
      ),
    ];
  }
}

final class _ReverseReranker implements Reranker {
  const _ReverseReranker();

  @override
  Future<List<VectorSearchResult>> rerank(
    String query,
    List<VectorSearchResult> results,
  ) async {
    return results.reversed.toList();
  }
}

final class _UnknownChunkReranker implements Reranker {
  const _UnknownChunkReranker();

  @override
  Future<List<VectorSearchResult>> rerank(
    String query,
    List<VectorSearchResult> results,
  ) async {
    return const <VectorSearchResult>[
      VectorSearchResult(
        chunk: TextChunk(
          documentId: 'other',
          id: '0',
          text: 'gamma',
          tokenCount: 1,
        ),
        score: 1,
      ),
    ];
  }
}

final class _DuplicateReranker implements Reranker {
  const _DuplicateReranker();

  @override
  Future<List<VectorSearchResult>> rerank(
    String query,
    List<VectorSearchResult> results,
  ) async {
    return <VectorSearchResult>[results.first, results.first];
  }
}

final class _DroppingReranker implements Reranker {
  const _DroppingReranker();

  @override
  Future<List<VectorSearchResult>> rerank(
    String query,
    List<VectorSearchResult> results,
  ) async {
    return <VectorSearchResult>[results.first];
  }
}

final class _ThrowingReranker implements Reranker {
  const _ThrowingReranker();

  @override
  Future<List<VectorSearchResult>> rerank(
    String query,
    List<VectorSearchResult> results,
  ) async {
    throw StateError('reranker should not run');
  }
}

final class _NonFiniteReranker implements Reranker {
  const _NonFiniteReranker();

  @override
  Future<List<VectorSearchResult>> rerank(
    String query,
    List<VectorSearchResult> results,
  ) async {
    return <VectorSearchResult>[
      VectorSearchResult(chunk: results.single.chunk, score: double.nan),
    ];
  }
}

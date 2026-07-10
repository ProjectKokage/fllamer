import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:fllamer/fllamer.dart';

Future<void> main(List<String> args) async {
  final status = await runLocalRagCli(args);
  if (status != 0) {
    exitCode = status;
  }
}

Future<int> runLocalRagCli(
  List<String> args, {
  StringSink? output,
  StringSink? errorOutput,
}) async {
  final out = output ?? stdout;
  final errors = errorOutput ?? stderr;
  final String query;
  try {
    query = _query(args);
  } on FormatException catch (error) {
    errors.writeln(error.message);
    errors.writeln();
    errors.writeln(_usage);
    return 64;
  }
  final splitter = CharacterTextSplitter(maxLength: 180, overlap: 20);
  final chunks = <TextChunk>[
    for (final document in _documents) ...splitter.split(document),
  ];
  final embedder = _TinyEmbedder();
  final index = InMemoryVectorIndex(dimensions: embedder.dimensions);
  index.add(chunks, [
    for (final chunk in chunks) await embedder.embedText(chunk.text),
  ]);

  final retriever = VectorIndexRetriever(
    embeddingModel: embedder,
    index: index,
  );
  final results = await retriever.retrieve(query, topK: 3);
  final prompt = await const RagPromptBuilder().buildContextWithTokenizer(
    results.map((result) => result.chunk),
    maxContextTokens: 80,
    tokenize: _wordTokens,
  );

  out.writeln('Query: $query');
  out.writeln(prompt.context);
  out.writeln('\nCitations:');
  for (final citation in prompt.citations) {
    out.writeln('- ${citation.documentId}#${citation.chunkId}');
  }
  return 0;
}

const _usage = '''
Usage: dart run example/dart_cli/local_rag.dart [query text]
''';

String _query(List<String> args) {
  final query = args.isEmpty
      ? 'How does fllamer handle mobile privacy?'
      : args.join(' ');
  if (query.trim().isEmpty) {
    throw const FormatException('Query must not be empty.');
  }
  if (query.contains('\u0000')) {
    throw const FormatException('Query must not contain NUL.');
  }
  return query;
}

Future<List<int>> _wordTokens(String text) async {
  final words = text.trim().split(RegExp(r'\s+'));
  return List<int>.generate(words.length, (index) => index);
}

const _documents = <Document>[
  Document(
    id: 'privacy',
    text:
        'fllamer keeps inference local by default. Models are app-owned files, '
        'and the library does not make hidden network requests.',
  ),
  Document(
    id: 'mobile',
    text:
        'Heavy model loading, tokenization, generation, embeddings, reranking, '
        'and LoRA work run away from the Flutter UI isolate.',
  ),
  Document(
    id: 'rag',
    text:
        'The RAG layer offers deterministic splitting, an in-memory vector '
        'index, retrieval, prompt assembly, and citations over local content.',
  ),
];

final class _TinyEmbedder implements EmbeddingModel {
  final int dimensions = 16;

  @override
  Future<Float32List> embedText(String text) async {
    final vector = Float32List(dimensions);
    for (final word in text.toLowerCase().split(RegExp(r'[^a-z0-9]+'))) {
      if (word.isEmpty) {
        continue;
      }
      vector[word.codeUnitAt(0) % dimensions] += 1;
    }
    var norm = 0.0;
    for (final value in vector) {
      norm += value * value;
    }
    if (norm == 0) {
      vector[0] = 1;
      return vector;
    }
    final scale = math.sqrt(norm);
    for (var i = 0; i < vector.length; i += 1) {
      vector[i] /= scale;
    }
    return vector;
  }
}

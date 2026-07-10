import 'dart:convert';
import 'dart:io';
import 'dart:isolate';
import 'dart:math' as math;
import 'dart:typed_data';

import 'config.dart';
import 'errors.dart';

final class Document {
  const Document({
    required this.id,
    required this.text,
    this.metadata = const <String, Object?>{},
    this.sourceUri,
  });

  final String id;
  final String text;
  final Map<String, Object?> metadata;
  final Uri? sourceUri;
}

final class TextChunk {
  const TextChunk({
    required this.documentId,
    required this.id,
    required this.text,
    required this.tokenCount,
    this.metadata = const <String, Object?>{},
    this.sourceUri,
  });

  final String documentId;
  final String id;
  final String text;
  final int tokenCount;
  final Map<String, Object?> metadata;
  final Uri? sourceUri;
}

abstract interface class TextSplitter {
  List<TextChunk> split(Document document);
}

abstract interface class AsyncTextSplitter {
  Future<List<TextChunk>> split(Document document);
}

typedef TextTokenizer = Future<List<int>> Function(String text);
typedef TokenDetokenizer = Future<String> Function(List<int> tokens);
typedef ChatTokenCounter = Future<int> Function(List<ChatMessage> messages);

abstract interface class EmbeddingModel {
  Future<Float32List> embedText(String text);
}

abstract interface class Retriever {
  Future<List<VectorSearchResult>> retrieve(
    String query, {
    int topK = 5,
    double minScore = double.negativeInfinity,
  });
}

abstract interface class Reranker {
  Future<List<VectorSearchResult>> rerank(
    String query,
    List<VectorSearchResult> results,
  );
}

final class CharacterTextSplitter implements TextSplitter {
  factory CharacterTextSplitter({required int maxLength, int overlap = 0}) {
    if (maxLength <= 0) {
      throw ArgumentError.value(maxLength, 'maxLength', 'must be positive');
    }
    if (overlap < 0 || overlap >= maxLength) {
      throw ArgumentError.value(
        overlap,
        'overlap',
        'must be non-negative and smaller than maxLength',
      );
    }
    return CharacterTextSplitter._(maxLength, overlap);
  }

  const CharacterTextSplitter._(this.maxLength, this.overlap);

  final int maxLength;
  final int overlap;

  @override
  List<TextChunk> split(Document document) {
    _validateIndexId(document.id, 'document.id');
    _validateText(document.text, 'document.text');
    _validateSourceUri(document.sourceUri, 'document.sourceUri');
    final text = document.text;
    if (text.trim().isEmpty) {
      return const <TextChunk>[];
    }
    final offsets = _runeOffsets(text);
    final length = offsets.length - 1;

    final chunks = <TextChunk>[];
    var start = 0;
    var index = 0;
    while (start < length) {
      final end = math.min(start + maxLength, length);
      final rawStart = offsets[start];
      final rawEnd = offsets[end];
      final rawText = text.substring(rawStart, rawEnd);
      final chunkText = rawText.trim();
      if (chunkText.isNotEmpty) {
        final spanStart = rawStart + rawText.length - rawText.trimLeft().length;
        final spanEnd = rawEnd - (rawText.length - rawText.trimRight().length);
        chunks.add(
          TextChunk(
            documentId: document.id,
            id: _chunkId(document.id, index),
            text: chunkText,
            tokenCount: _countWords(chunkText),
            metadata: _chunkMetadata(document.metadata, <String, Object?>{
              'start': spanStart,
              'end': spanEnd,
            }),
            sourceUri: document.sourceUri,
          ),
        );
        index += 1;
      }
      if (end == length) {
        break;
      }
      start = end - overlap;
    }
    return List<TextChunk>.unmodifiable(chunks);
  }
}

final class TokenTextSplitter implements AsyncTextSplitter {
  factory TokenTextSplitter({
    required int maxTokens,
    required TextTokenizer tokenize,
    required TokenDetokenizer detokenize,
    int overlap = 0,
  }) {
    if (maxTokens <= 0) {
      throw ArgumentError.value(maxTokens, 'maxTokens', 'must be positive');
    }
    if (overlap < 0 || overlap >= maxTokens) {
      throw ArgumentError.value(
        overlap,
        'overlap',
        'must be non-negative and smaller than maxTokens',
      );
    }
    return TokenTextSplitter._(maxTokens, overlap, tokenize, detokenize);
  }

  const TokenTextSplitter._(
    this.maxTokens,
    this.overlap,
    this.tokenize,
    this.detokenize,
  );

  final int maxTokens;
  final int overlap;
  final TextTokenizer tokenize;
  final TokenDetokenizer detokenize;

  @override
  Future<List<TextChunk>> split(Document document) async {
    _validateIndexId(document.id, 'document.id');
    _validateText(document.text, 'document.text');
    _validateSourceUri(document.sourceUri, 'document.sourceUri');
    if (document.text.trim().isEmpty) {
      return const <TextChunk>[];
    }
    final tokens = List<int>.unmodifiable(await tokenize(document.text));
    _validateTokenIds(tokens);
    if (tokens.isEmpty) {
      return const <TextChunk>[];
    }

    final chunks = <TextChunk>[];
    var start = 0;
    var index = 0;
    while (start < tokens.length) {
      final end = math.min(start + maxTokens, tokens.length);
      final text = (await detokenize(tokens.sublist(start, end))).trim();
      if (text.isNotEmpty) {
        _validateText(text, 'chunk.text');
        chunks.add(
          TextChunk(
            documentId: document.id,
            id: _chunkId(document.id, index),
            text: text,
            tokenCount: end - start,
            metadata: _chunkMetadata(document.metadata, <String, Object?>{
              'tokenStart': start,
              'tokenEnd': end,
            }),
            sourceUri: document.sourceUri,
          ),
        );
        index += 1;
      }
      if (end == tokens.length) {
        break;
      }
      start = end - overlap;
    }
    return List<TextChunk>.unmodifiable(chunks);
  }
}

abstract interface class VectorIndex {
  int get dimensions;

  bool get isEmpty;

  void add(List<TextChunk> chunks, Iterable<Float32List> vectors);

  List<VectorSearchResult> search(
    Float32List query, {
    int topK = 5,
    double minScore = double.negativeInfinity,
  });

  bool remove(String chunkId, {String? documentId});

  void removeDocument(String documentId);

  Future<void> persist(String path);

  void clear();
}

final class VectorIndexRetriever implements Retriever {
  const VectorIndexRetriever({
    required this.embeddingModel,
    required this.index,
    this.reranker,
  });

  final EmbeddingModel embeddingModel;
  final VectorIndex index;
  final Reranker? reranker;

  @override
  Future<List<VectorSearchResult>> retrieve(
    String query, {
    int topK = 5,
    double minScore = double.negativeInfinity,
  }) async {
    if (query.trim().isEmpty) {
      throw ArgumentError.value(query, 'query', 'must not be empty');
    }
    _validateText(query, 'query');
    if (topK <= 0) {
      throw ArgumentError.value(topK, 'topK', 'must be positive');
    }
    if (minScore.isNaN) {
      throw ArgumentError.value(minScore, 'minScore', 'must not be NaN');
    }
    final dimensions = index.dimensions;
    if (dimensions <= 0) {
      throw ArgumentError.value(
        dimensions,
        'index.dimensions',
        'must be positive',
      );
    }
    if (index.isEmpty) {
      return const <VectorSearchResult>[];
    }
    final vector = _embeddingVectorSnapshot(
      await embeddingModel.embedText(query),
      dimensions,
    );
    final results = _searchResultsSnapshot(
      index.search(vector, topK: topK, minScore: minScore),
      'index.result',
    );
    for (final result in results) {
      if (result.score < minScore) {
        throw ArgumentError.value(
          result.score,
          'index.result.score',
          'must not be below minScore',
        );
      }
    }
    if (results.length > topK) {
      throw ArgumentError.value(
        results.length,
        'index.results',
        'must not exceed topK',
      );
    }
    final reranker = this.reranker;
    if (reranker == null || results.isEmpty) {
      return results;
    }
    final reranked = _searchResultsSnapshot(
      await reranker.rerank(query, results),
      'reranker.result',
    );
    _validateRerankedCandidates(reranked, results);
    return reranked;
  }
}

final class VectorSearchResult {
  const VectorSearchResult({required this.chunk, required this.score});

  final TextChunk chunk;
  final double score;
}

final class InMemoryVectorIndex implements VectorIndex {
  factory InMemoryVectorIndex({
    required int dimensions,
    bool normalize = true,
  }) {
    if (dimensions <= 0) {
      throw ArgumentError.value(dimensions, 'dimensions', 'must be positive');
    }
    return InMemoryVectorIndex._(dimensions, normalize);
  }

  InMemoryVectorIndex._(this.dimensions, this.normalize);

  factory InMemoryVectorIndex.fromJson(Map<String, Object?> json) {
    _rejectUnexpectedKeys(json, const <String>{
      'version',
      'dimensions',
      'normalize',
      'records',
    }, 'vector index');
    if (json['version'] != 1) {
      throw FormatException(
        'unsupported vector index version ${json['version']}',
      );
    }
    final dimensions = json['dimensions'];
    if (dimensions is! int) {
      throw const FormatException('vector index dimensions must be an integer');
    }
    final normalize = json['normalize'];
    if (normalize is! bool && json.containsKey('normalize')) {
      throw const FormatException('vector index normalize must be a boolean');
    }
    final recordsValue = json['records'];
    if (recordsValue is! List<Object?> && json.containsKey('records')) {
      throw const FormatException('vector index records must be a list');
    }
    final index = _fromJsonValidation(
      () => InMemoryVectorIndex(
        dimensions: dimensions,
        normalize: normalize == null ? true : normalize as bool,
      ),
    );
    final records = recordsValue as List<Object?>? ?? const <Object?>[];
    for (final item in records) {
      if (item is! Map<Object?, Object?> ||
          item.keys.any((key) => key is! String)) {
        throw const FormatException('vector index record must be an object');
      }
      final record = Map<String, Object?>.from(item);
      _rejectUnexpectedKeys(record, const <String>{
        'chunk',
        'vector',
      }, 'vector index record');
      final chunkValue = record['chunk'];
      if (chunkValue is! Map<Object?, Object?> ||
          chunkValue.keys.any((key) => key is! String)) {
        throw const FormatException(
          'vector index record chunk must be an object',
        );
      }
      final chunkJson = Map<String, Object?>.from(chunkValue);
      _rejectUnexpectedKeys(chunkJson, const <String>{
        'documentId',
        'id',
        'text',
        'tokenCount',
        'metadata',
        'sourceUri',
      }, 'vector index chunk');
      final documentId = chunkJson['documentId'];
      final chunkId = chunkJson['id'];
      final chunkText = chunkJson['text'];
      final tokenCount = chunkJson['tokenCount'];
      final sourceUriValue = chunkJson['sourceUri'];
      final metadataValue = chunkJson['metadata'];
      if (metadataValue == null && chunkJson.containsKey('metadata')) {
        throw const FormatException(
          'vector index chunk metadata must be an object',
        );
      }
      if (documentId is! String ||
          chunkId is! String ||
          chunkText is! String ||
          tokenCount is! int ||
          sourceUriValue != null && sourceUriValue is! String ||
          metadataValue != null && metadataValue is! Map<Object?, Object?>) {
        throw const FormatException('vector index chunk fields are malformed');
      }
      final metadata = _metadataFromJson(metadataValue);
      final sourceUri = _sourceUriFromJson(sourceUriValue);
      final chunk = TextChunk(
        documentId: documentId,
        id: chunkId,
        text: chunkText,
        tokenCount: tokenCount,
        metadata: metadata,
        sourceUri: sourceUri,
      );
      _fromJsonValidation(() {
        _validateIndexId(chunk.documentId, 'chunk.documentId');
        _validateIndexId(chunk.id, 'chunk.id');
        _validateChunkText(chunk.text, 'chunk.text');
        _validateChunkTokenCount(chunk.tokenCount);
      });
      final vectorValue = record['vector'];
      if (vectorValue is! List<Object?>) {
        throw const FormatException(
          'vector index record vector must be a list',
        );
      }
      if (vectorValue.length != dimensions) {
        throw const FormatException('vector index vector dimension mismatch');
      }
      final vector = Float32List(vectorValue.length);
      for (var i = 0; i < vectorValue.length; i += 1) {
        final value = vectorValue[i];
        if (value is! num) {
          throw const FormatException(
            'vector index vector values must be numbers',
          );
        }
        vector[i] = value.toDouble();
      }
      final key = index._key(chunk.documentId, chunk.id);
      if (index._records.containsKey(key)) {
        throw const FormatException(
          'vector index records must not contain duplicate chunks',
        );
      }
      index._records[key] = _VectorRecord(
        chunk: _copyChunk(chunk),
        vector: _fromJsonValidation(() => index._copyVector(vector)),
      );
    }
    return index;
  }

  static Future<InMemoryVectorIndex> load(String path) async {
    _validateFilePath(path, 'path');
    return Isolate.run(() => _loadIndexInWorker(path));
  }

  @override
  final int dimensions;
  final bool normalize;
  final Map<String, _VectorRecord> _records = <String, _VectorRecord>{};

  @override
  bool get isEmpty => _records.isEmpty;

  @override
  void add(List<TextChunk> chunks, Iterable<Float32List> vectors) {
    final vectorIterator = vectors.iterator;
    final pending = <String, _VectorRecord>{};
    for (var i = 0; i < chunks.length; i += 1) {
      if (!vectorIterator.moveNext()) {
        throw ArgumentError(
          'vectors must contain exactly ${chunks.length} items',
        );
      }
      final vector = _copyVector(vectorIterator.current);
      final chunk = chunks[i];
      _validateIndexId(chunk.documentId, 'chunk.documentId');
      _validateIndexId(chunk.id, 'chunk.id');
      _validateChunkText(chunk.text, 'chunk.text');
      _validateChunkTokenCount(chunk.tokenCount);
      _validateSourceUri(chunk.sourceUri, 'chunk.sourceUri');
      final key = _key(chunk.documentId, chunk.id);
      if (pending.containsKey(key)) {
        throw ArgumentError.value(
          chunk.id,
          'chunk.id',
          'must not contain duplicate chunks',
        );
      }
      pending[key] = _VectorRecord(chunk: _copyChunk(chunk), vector: vector);
    }
    if (vectorIterator.moveNext()) {
      throw ArgumentError(
        'vectors must contain exactly ${chunks.length} items',
      );
    }
    _records.addAll(pending);
  }

  @override
  List<VectorSearchResult> search(
    Float32List query, {
    int topK = 5,
    double minScore = double.negativeInfinity,
  }) {
    if (topK <= 0) {
      throw ArgumentError.value(topK, 'topK', 'must be positive');
    }
    if (minScore.isNaN) {
      throw ArgumentError.value(minScore, 'minScore', 'must not be NaN');
    }
    final queryVector = _copyVector(query);
    final results = <VectorSearchResult>[];
    for (final record in _records.values) {
      final score = _dot(queryVector, record.vector);
      if (score >= minScore) {
        results.add(VectorSearchResult(chunk: record.chunk, score: score));
      }
    }
    results.sort((a, b) {
      final score = b.score.compareTo(a.score);
      if (score != 0) {
        return score;
      }
      final document = a.chunk.documentId.compareTo(b.chunk.documentId);
      if (document != 0) {
        return document;
      }
      return a.chunk.id.compareTo(b.chunk.id);
    });
    return List<VectorSearchResult>.unmodifiable(results.take(topK));
  }

  @override
  bool remove(String chunkId, {String? documentId}) {
    _validateIndexId(chunkId, 'chunkId');
    if (documentId != null) {
      _validateIndexId(documentId, 'documentId');
      return _records.remove(_key(documentId, chunkId)) != null;
    }
    var removed = false;
    _records.removeWhere((_, record) {
      final shouldRemove = record.chunk.id == chunkId;
      removed = removed || shouldRemove;
      return shouldRemove;
    });
    return removed;
  }

  @override
  void removeDocument(String documentId) {
    _validateIndexId(documentId, 'documentId');
    _records.removeWhere((_, record) => record.chunk.documentId == documentId);
  }

  Map<String, Object?> toJson() {
    final records = _records.values.toList()
      ..sort((a, b) {
        final document = a.chunk.documentId.compareTo(b.chunk.documentId);
        if (document != 0) {
          return document;
        }
        return a.chunk.id.compareTo(b.chunk.id);
      });
    return Map<String, Object?>.unmodifiable(<String, Object?>{
      'version': 1,
      'dimensions': dimensions,
      'normalize': normalize,
      'records': List<Map<String, Object?>>.unmodifiable(<Map<String, Object?>>[
        for (final record in records)
          Map<String, Object?>.unmodifiable(<String, Object?>{
            'chunk': Map<String, Object?>.unmodifiable(<String, Object?>{
              'documentId': record.chunk.documentId,
              'id': record.chunk.id,
              'text': record.chunk.text,
              'tokenCount': record.chunk.tokenCount,
              'metadata': _jsonMetadata(record.chunk.metadata),
              'sourceUri': record.chunk.sourceUri?.toString(),
            }),
            'vector': List<double>.unmodifiable(record.vector),
          }),
      ]),
    });
  }

  @override
  Future<void> persist(String path) async {
    _validateFilePath(path, 'path');
    await Isolate.run(() async {
      await _rejectIndexFileLink(path);
      await _writeIndexFile(path, jsonEncode(toJson()));
    });
  }

  @override
  void clear() {
    _records.clear();
  }

  Float32List _copyVector(Float32List vector) {
    if (vector.length != dimensions) {
      throw ArgumentError.value(
        vector.length,
        'vector',
        'must have $dimensions dimensions',
      );
    }
    final copy = Float32List.fromList(vector);
    _validateVectorValues(copy);
    if (!normalize) {
      return copy;
    }
    var sum = 0.0;
    for (final value in copy) {
      sum += value * value;
    }
    if (sum == 0) {
      throw ArgumentError.value(vector, 'vector', 'must not be all zeroes');
    }
    final length = math.sqrt(sum);
    for (var i = 0; i < copy.length; i += 1) {
      copy[i] /= length;
    }
    return copy;
  }

  double _dot(Float32List left, Float32List right) {
    var score = 0.0;
    for (var i = 0; i < dimensions; i += 1) {
      score += left[i] * right[i];
    }
    return score;
  }

  String _key(String documentId, String chunkId) =>
      _chunkKey(documentId, chunkId);
}

String _chunkId(String documentId, int index) => '$documentId:$index';

String _chunkKey(String documentId, String chunkId) =>
    '$documentId\u0000$chunkId';

List<int> _runeOffsets(String text) {
  final offsets = <int>[];
  var offset = 0;
  for (final rune in text.runes) {
    offsets.add(offset);
    offset += rune > 0xFFFF ? 2 : 1;
  }
  offsets.add(text.length);
  return offsets;
}

Float32List _embeddingVectorSnapshot(Float32List vector, int dimensions) {
  if (vector.length != dimensions) {
    throw ArgumentError.value(
      vector.length,
      'embedding',
      'must have $dimensions dimensions',
    );
  }
  final copy = Float32List.fromList(vector);
  _validateVectorValues(copy);
  return copy;
}

List<VectorSearchResult> _searchResultsSnapshot(
  Iterable<VectorSearchResult> results,
  String name,
) {
  final snapshot = <VectorSearchResult>[];
  final seen = <String>{};
  for (final result in results) {
    _validateIndexId(result.chunk.documentId, '$name.documentId');
    _validateIndexId(result.chunk.id, '$name.id');
    _validateChunkText(result.chunk.text, '$name.text');
    _validateChunkTokenCount(result.chunk.tokenCount);
    _validateSourceUri(result.chunk.sourceUri, '$name.sourceUri');
    if (!result.score.isFinite) {
      throw ArgumentError.value(result.score, '$name.score', 'must be finite');
    }
    final key = _chunkKey(result.chunk.documentId, result.chunk.id);
    if (!seen.add(key)) {
      throw ArgumentError.value(
        result.chunk.id,
        '$name.chunk',
        'must not contain duplicates',
      );
    }
    snapshot.add(
      VectorSearchResult(chunk: _copyChunk(result.chunk), score: result.score),
    );
  }
  return List<VectorSearchResult>.unmodifiable(snapshot);
}

void _validateRerankedCandidates(
  List<VectorSearchResult> reranked,
  List<VectorSearchResult> candidates,
) {
  if (reranked.length != candidates.length) {
    throw ArgumentError.value(
      reranked.length,
      'reranker.results',
      'must contain every vector search candidate exactly once',
    );
  }
  final allowed = <String>{
    for (final result in candidates)
      _chunkKey(result.chunk.documentId, result.chunk.id),
  };
  final seen = <String>{};
  for (final result in reranked) {
    final key = _chunkKey(result.chunk.documentId, result.chunk.id);
    if (!allowed.contains(key)) {
      throw ArgumentError.value(
        result.chunk.id,
        'reranker.chunk',
        'must come from vector search results',
      );
    }
    if (!seen.add(key)) {
      throw ArgumentError.value(
        result.chunk.id,
        'reranker.chunk',
        'must not contain duplicates',
      );
    }
  }
}

Map<String, Object?> _chunkMetadata(
  Map<String, Object?> metadata,
  Map<String, Object?> spans,
) {
  return _jsonMetadata(<String, Object?>{...metadata, ...spans});
}

TextChunk _copyChunk(TextChunk chunk) {
  return TextChunk(
    documentId: chunk.documentId,
    id: chunk.id,
    text: chunk.text,
    tokenCount: chunk.tokenCount,
    metadata: _jsonMetadata(chunk.metadata),
    sourceUri: chunk.sourceUri,
  );
}

TextChunk _copyPromptChunk(TextChunk chunk) {
  return TextChunk(
    documentId: chunk.documentId,
    id: chunk.id,
    text: chunk.text,
    tokenCount: chunk.tokenCount,
    metadata: _jsonMetadata(chunk.metadata),
    sourceUri: chunk.sourceUri,
  );
}

void _validateIndexId(String value, String name) {
  if (value.trim().isEmpty) {
    throw ArgumentError.value(value, name, 'must not be empty');
  }
  if (value.contains('\n') || value.contains('\r')) {
    throw ArgumentError.value(value, name, 'must not contain line breaks');
  }
  _validateText(value, name);
}

void _validateText(String value, String name) {
  if (value.contains('\u0000')) {
    throw ArgumentError.value(value, name, 'must not contain NUL');
  }
}

void _validateChunkText(String value, String name) {
  if (value.trim().isEmpty) {
    throw ArgumentError.value(value, name, 'must not be empty');
  }
  _validateText(value, name);
}

void _validateTokenIds(List<int> tokens) {
  for (final token in tokens) {
    if (token < 0 || token > 0x7FFFFFFF) {
      throw ArgumentError.value(
        tokens,
        'tokens',
        'must contain int32 token ids',
      );
    }
  }
}

void _validateFilePath(String value, String name) {
  if (value.trim().isEmpty) {
    throw ArgumentError.value(value, name, 'must not be empty');
  }
  _validateText(value, name);
  if (value.contains('\n') || value.contains('\r')) {
    throw ArgumentError.value(value, name, 'must not contain line breaks');
  }
}

Future<String> _readIndexFile(String path) async {
  try {
    return await File(path).readAsString();
  } on FileSystemException catch (error) {
    throw RagIndexException(
      'RAG vector index is not readable: $path',
      cause: error,
    );
  }
}

Future<InMemoryVectorIndex> _loadIndexInWorker(String path) async {
  await _rejectIndexFileLink(path);
  final text = await _readIndexFile(path);
  final Object? decoded;
  try {
    decoded = jsonDecode(text);
  } on FormatException catch (error) {
    throw FormatException(
      'vector index JSON is malformed: ${error.message}',
      text,
      error.offset,
    );
  }
  if (decoded is! Map<Object?, Object?>) {
    throw const FormatException('vector index JSON root must be an object');
  }
  return InMemoryVectorIndex.fromJson(Map<String, Object?>.from(decoded));
}

Future<void> _rejectIndexFileLink(String path) async {
  final FileSystemEntityType type;
  try {
    type = await FileSystemEntity.type(path, followLinks: false);
  } on FileSystemException catch (error) {
    throw RagIndexException(
      'RAG vector index path is not readable: $path',
      cause: error,
    );
  }
  if (type == FileSystemEntityType.link) {
    throw RagIndexException('RAG vector index path is a symbolic link: $path');
  }
}

Future<void> _writeIndexFile(String path, String json) async {
  try {
    await File(path).writeAsString(json, flush: true);
  } on FileSystemException catch (error) {
    throw RagIndexException(
      'RAG vector index could not be written: $path',
      cause: error,
    );
  }
}

void _validateVectorValues(Float32List vector) {
  for (final value in vector) {
    if (!value.isFinite) {
      throw ArgumentError.value(
        vector,
        'vector',
        'must contain only finite values',
      );
    }
  }
}

Map<String, Object?> _jsonMetadata(Map<Object?, Object?> metadata) {
  return _jsonObject(metadata, 'chunk.metadata', Set<Object>.identity());
}

Map<String, Object?> _metadataFromJson(Object? value) {
  if (value == null) {
    return const <String, Object?>{};
  }
  return _fromJsonValidation(
    () => _jsonMetadata(value as Map<Object?, Object?>),
  );
}

Uri? _sourceUriFromJson(Object? value) {
  if (value == null) {
    return null;
  }
  final text = value as String;
  if (text.trim().isEmpty) {
    throw const FormatException('vector index sourceUri must not be empty');
  }
  if (_containsNulOctet(text)) {
    throw const FormatException('vector index sourceUri must not contain NUL');
  }
  if (_containsLineBreakOctet(text)) {
    throw const FormatException(
      'vector index sourceUri must not contain line breaks',
    );
  }
  try {
    return Uri.parse(text);
  } on FormatException catch (error) {
    throw FormatException(
      'vector index sourceUri is malformed',
      text,
      error.offset,
    );
  }
}

void _validateSourceUri(Uri? value, String name) {
  final text = value?.toString();
  if (text == null) {
    return;
  }
  if (text.trim().isEmpty) {
    throw ArgumentError.value(value, name, 'must not be empty');
  }
  if (_containsNulOctet(text)) {
    throw ArgumentError.value(value, name, 'must not contain NUL');
  }
  if (_containsLineBreakOctet(text)) {
    throw ArgumentError.value(value, name, 'must not contain line breaks');
  }
}

bool _containsNulOctet(String value) {
  return value.contains('\u0000') || value.toLowerCase().contains('%00');
}

bool _containsLineBreakOctet(String value) {
  final lower = value.toLowerCase();
  return value.contains('\n') ||
      value.contains('\r') ||
      lower.contains('%0a') ||
      lower.contains('%0d');
}

T _fromJsonValidation<T>(T Function() parse) {
  try {
    return parse();
  } on ArgumentError catch (error) {
    throw FormatException(
      error.message?.toString() ?? 'malformed vector index',
    );
  }
}

void _rejectUnexpectedKeys(
  Map<String, Object?> value,
  Set<String> allowed,
  String name,
) {
  final unexpected = value.keys.where((key) => !allowed.contains(key));
  if (unexpected.isNotEmpty) {
    throw FormatException('$name contains unsupported key ${unexpected.first}');
  }
}

Map<String, Object?> _jsonObject(
  Map<Object?, Object?> value,
  String name,
  Set<Object> activeContainers,
) {
  if (!activeContainers.add(value)) {
    throw ArgumentError.value(value, name, 'must not contain cycles');
  }
  final result = <String, Object?>{};
  try {
    for (final entry in value.entries) {
      final key = entry.key;
      if (key is! String) {
        throw ArgumentError.value(key, name, 'keys must be strings');
      }
      if (key.contains('\u0000')) {
        throw ArgumentError.value(key, name, 'keys must not contain NUL');
      }
      result[key] = _jsonValue(entry.value, '$name.$key', activeContainers);
    }
    return Map<String, Object?>.unmodifiable(result);
  } finally {
    activeContainers.remove(value);
  }
}

Object? _jsonValue(Object? value, String name, Set<Object> activeContainers) {
  if (value == null || value is bool) {
    return value;
  }
  if (value is String) {
    _validateText(value, name);
    return value;
  }
  if (value is num) {
    if (!value.isFinite) {
      throw ArgumentError.value(value, name, 'must be finite');
    }
    return value;
  }
  if (value is List<Object?>) {
    if (!activeContainers.add(value)) {
      throw ArgumentError.value(value, name, 'must not contain cycles');
    }
    try {
      return List<Object?>.unmodifiable(<Object?>[
        for (var i = 0; i < value.length; i += 1)
          _jsonValue(value[i], '$name[$i]', activeContainers),
      ]);
    } finally {
      activeContainers.remove(value);
    }
  }
  if (value is Map<Object?, Object?>) {
    return _jsonObject(value, name, activeContainers);
  }
  throw ArgumentError.value(value, name, 'must be a JSON value');
}

void _validateChunkTokenCount(int tokenCount) {
  if (tokenCount <= 0) {
    throw ArgumentError.value(
      tokenCount,
      'chunk.tokenCount',
      'must be positive',
    );
  }
}

int? _metadataInt(Map<String, Object?> metadata, String key) {
  final value = metadata[key];
  if (value == null) {
    return null;
  }
  if (value is int && value >= 0) {
    return value;
  }
  throw ArgumentError.value(
    value,
    'chunk.metadata[$key]',
    'must be a non-negative int',
  );
}

void _validateRange(int? start, int? end, String startName, String endName) {
  if (start == null && end == null) {
    return;
  }
  if (start == null) {
    throw ArgumentError.value(end, startName, 'must be provided with end');
  }
  if (end == null) {
    throw ArgumentError.value(start, endName, 'must be provided with start');
  }
  if (end <= start) {
    throw ArgumentError.value(end, endName, 'must be greater than start');
  }
}

RagCitation _citationFor(TextChunk chunk) {
  final start = _metadataInt(chunk.metadata, 'start');
  final end = _metadataInt(chunk.metadata, 'end');
  final tokenStart = _metadataInt(chunk.metadata, 'tokenStart');
  final tokenEnd = _metadataInt(chunk.metadata, 'tokenEnd');
  _validateRange(start, end, 'chunk.metadata[start]', 'chunk.metadata[end]');
  _validateRange(
    tokenStart,
    tokenEnd,
    'chunk.metadata[tokenStart]',
    'chunk.metadata[tokenEnd]',
  );
  return RagCitation(
    documentId: chunk.documentId,
    chunkId: chunk.id,
    sourceUri: chunk.sourceUri,
    start: start,
    end: end,
    tokenStart: tokenStart,
    tokenEnd: tokenEnd,
  );
}

final class RagCitation {
  const RagCitation({
    required this.documentId,
    required this.chunkId,
    this.sourceUri,
    this.start,
    this.end,
    this.tokenStart,
    this.tokenEnd,
  });

  final String documentId;
  final String chunkId;
  final Uri? sourceUri;
  final int? start;
  final int? end;
  final int? tokenStart;
  final int? tokenEnd;
}

final class RagPrompt {
  const RagPrompt({
    required this.context,
    required this.includedChunks,
    required this.usedTokens,
    this.citations = const <RagCitation>[],
  });

  final String context;
  final List<TextChunk> includedChunks;
  final int usedTokens;
  final List<RagCitation> citations;
}

final class RagChatPrompt {
  const RagChatPrompt({
    required this.messages,
    required this.includedChunks,
    required this.usedTokens,
    this.citations = const <RagCitation>[],
  });

  final List<ChatMessage> messages;
  final List<TextChunk> includedChunks;
  final int usedTokens;
  final List<RagCitation> citations;
}

final class RagPromptBuilder {
  const RagPromptBuilder({this.header = 'Relevant local context:'});

  final String header;

  List<ChatMessage> buildChatMessages({
    required String question,
    required Iterable<TextChunk> chunks,
    required int maxContextTokens,
    String? systemPrompt,
  }) {
    _validateRagQuestion(question, systemPrompt);

    final prompt = buildContext(chunks, maxContextTokens: maxContextTokens);
    return _ragChatMessages(question, systemPrompt, prompt.context);
  }

  RagPrompt buildContext(
    Iterable<TextChunk> chunks, {
    required int maxContextTokens,
  }) {
    if (maxContextTokens <= 0) {
      throw ArgumentError.value(
        maxContextTokens,
        'maxContextTokens',
        'must be positive',
      );
    }
    _validateChunkText(header, 'header');

    final checkedChunks = _validatedPromptChunks(chunks);
    final included = <TextChunk>[];
    final citations = <RagCitation>[];
    var usedTokens = 0;
    for (final chunk in checkedChunks) {
      if (usedTokens + chunk.tokenCount > maxContextTokens) {
        continue;
      }
      included.add(chunk);
      citations.add(_citationFor(chunk));
      usedTokens += chunk.tokenCount;
    }

    return RagPrompt(
      context: _ragContext(header, included),
      includedChunks: List<TextChunk>.unmodifiable(included),
      usedTokens: usedTokens,
      citations: List<RagCitation>.unmodifiable(citations),
    );
  }

  Future<RagPrompt> buildContextWithTokenizer(
    Iterable<TextChunk> chunks, {
    required int maxContextTokens,
    required TextTokenizer tokenize,
  }) async {
    if (maxContextTokens <= 0) {
      throw ArgumentError.value(
        maxContextTokens,
        'maxContextTokens',
        'must be positive',
      );
    }
    _validateChunkText(header, 'header');
    final checkedChunks = _validatedPromptChunks(chunks);
    final included = <TextChunk>[];
    final citations = <RagCitation>[];
    var usedTokens = 0;
    for (final chunk in checkedChunks) {
      final candidate = <TextChunk>[...included, chunk];
      final candidateContext = _ragContext(header, candidate);
      final tokens = List<int>.unmodifiable(await tokenize(candidateContext));
      _validateTokenIds(tokens);
      if (tokens.isEmpty) {
        throw ArgumentError.value(
          tokens,
          'tokenize',
          'must return tokens for non-empty context',
        );
      }
      if (tokens.length > maxContextTokens) {
        continue;
      }
      included.add(chunk);
      citations.add(_citationFor(chunk));
      usedTokens = tokens.length;
    }
    return RagPrompt(
      context: _ragContext(header, included),
      includedChunks: List<TextChunk>.unmodifiable(included),
      usedTokens: usedTokens,
      citations: List<RagCitation>.unmodifiable(citations),
    );
  }

  Future<RagChatPrompt> buildChatPrompt({
    required String question,
    required Iterable<TextChunk> chunks,
    required int maxPromptTokens,
    required ChatTokenCounter countTokens,
    String? systemPrompt,
  }) async {
    _validateRagQuestion(question, systemPrompt);
    if (maxPromptTokens <= 0) {
      throw ArgumentError.value(
        maxPromptTokens,
        'maxPromptTokens',
        'must be positive',
      );
    }
    _validateChunkText(header, 'header');
    final checkedChunks = _validatedPromptChunks(chunks);
    final included = <TextChunk>[];
    final citations = <RagCitation>[];
    var messages = _ragChatMessages(question, systemPrompt, '');
    var usedTokens = await _countRagChatTokens(messages, countTokens);
    if (usedTokens > maxPromptTokens) {
      throw ArgumentError.value(
        maxPromptTokens,
        'maxPromptTokens',
        'must fit the $usedTokens-token base chat prompt',
      );
    }
    for (final chunk in checkedChunks) {
      final candidateChunks = <TextChunk>[...included, chunk];
      final candidateMessages = _ragChatMessages(
        question,
        systemPrompt,
        _ragContext(header, candidateChunks),
      );
      final candidateTokens = await _countRagChatTokens(
        candidateMessages,
        countTokens,
      );
      if (candidateTokens > maxPromptTokens) {
        continue;
      }
      included.add(chunk);
      citations.add(_citationFor(chunk));
      messages = candidateMessages;
      usedTokens = candidateTokens;
    }
    return RagChatPrompt(
      messages: messages,
      includedChunks: List<TextChunk>.unmodifiable(included),
      usedTokens: usedTokens,
      citations: List<RagCitation>.unmodifiable(citations),
    );
  }
}

List<TextChunk> _validatedPromptChunks(Iterable<TextChunk> chunks) {
  final checked = <TextChunk>[];
  for (final chunk in chunks) {
    _validateIndexId(chunk.documentId, 'chunk.documentId');
    _validateIndexId(chunk.id, 'chunk.id');
    _validateChunkText(chunk.text, 'chunk.text');
    _validateChunkTokenCount(chunk.tokenCount);
    _validateSourceUri(chunk.sourceUri, 'chunk.sourceUri');
    final snapshot = _copyPromptChunk(chunk);
    _citationFor(snapshot);
    checked.add(snapshot);
  }
  return List<TextChunk>.unmodifiable(checked);
}

String _ragContext(String header, List<TextChunk> chunks) {
  if (chunks.isEmpty) {
    return '';
  }
  final buffer = StringBuffer(header);
  for (final chunk in chunks) {
    buffer
      ..writeln()
      ..writeln('[source: ${chunk.documentId}#${chunk.id}]')
      ..writeln(chunk.text);
  }
  return buffer.toString().trimRight();
}

void _validateRagQuestion(String question, String? systemPrompt) {
  if (question.trim().isEmpty) {
    throw ArgumentError.value(question, 'question', 'must not be empty');
  }
  _validateText(question, 'question');
  final system = systemPrompt;
  if (system != null) {
    if (system.trim().isEmpty) {
      throw ArgumentError.value(
        systemPrompt,
        'systemPrompt',
        'must not be empty',
      );
    }
    _validateText(system, 'systemPrompt');
  }
}

List<ChatMessage> _ragChatMessages(
  String question,
  String? systemPrompt,
  String context,
) {
  return List<ChatMessage>.unmodifiable(<ChatMessage>[
    if (systemPrompt != null) ChatMessage.system(systemPrompt),
    if (context.isNotEmpty) ChatMessage.system(context),
    ChatMessage.user(question),
  ]);
}

Future<int> _countRagChatTokens(
  List<ChatMessage> messages,
  ChatTokenCounter countTokens,
) async {
  final count = await countTokens(messages);
  if (count <= 0) {
    throw ArgumentError.value(
      count,
      'countTokens',
      'must return a positive token count',
    );
  }
  return count;
}

final class _VectorRecord {
  const _VectorRecord({required this.chunk, required this.vector});

  final TextChunk chunk;
  final Float32List vector;
}

int _countWords(String text) {
  final trimmed = text.trim();
  if (trimmed.isEmpty) {
    return 0;
  }
  return trimmed.split(RegExp(r'\s+')).length;
}

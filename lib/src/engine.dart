import 'dart:io';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:crypto/crypto.dart' as crypto;

import 'config.dart';
import 'errors.dart';
import 'model_info.dart';
import 'native_bridge.dart';
import 'rag.dart';

abstract final class LlamaRuntime {
  static const bridgeAbiVersion = NativeLlamaBridge.expectedAbiVersion;

  static void configureNativeLogging(
    LlamaLogLevel minimumLevel, {
    String? nativeLibraryPath,
  }) {
    final path = nativeLibraryPath;
    if (path != null) {
      _validateFilePath(path, 'nativeLibraryPath');
    }
    final bridge = NativeLlamaBridge.tryOpen(path);
    if (bridge == null) {
      throw UnsupportedFeatureException(
        'Native bridge library was not found or is incompatible: '
        '${path ?? 'the default platform library path'}',
      );
    }
    bridge.configureLogging(minimumLevel);
  }

  static List<LlamaLogRecord> drainNativeLogs({String? nativeLibraryPath}) {
    final path = nativeLibraryPath;
    if (path != null) {
      _validateFilePath(path, 'nativeLibraryPath');
    }
    final bridge = NativeLlamaBridge.tryOpen(path);
    if (bridge == null) {
      throw UnsupportedFeatureException(
        'Native bridge library was not found or is incompatible: '
        '${path ?? 'the default platform library path'}',
      );
    }
    return bridge.drainLogs();
  }

  static LlamaRuntimeCapabilities currentCapabilities({
    String? nativeLibraryPath,
  }) {
    final path = nativeLibraryPath;
    if (path != null) {
      _validateFilePath(path, 'nativeLibraryPath');
    }
    final NativeLlamaBridge? bridge;
    try {
      bridge = NativeLlamaBridge.tryOpen(nativeLibraryPath);
    } on UnsupportedFeatureException {
      return _unavailableRuntimeCapabilities();
    }
    try {
      if (bridge != null) {
        return bridge.currentCapabilities();
      }
    } on ArgumentError {
      return _unavailableRuntimeCapabilities();
    } on OSError {
      return _unavailableRuntimeCapabilities();
    } on UnsupportedFeatureException {
      return _unavailableRuntimeCapabilities();
    } on NativeBridgeException {
      return _unavailableRuntimeCapabilities();
    }

    return _unavailableRuntimeCapabilities();
  }
}

LlamaRuntimeCapabilities _unavailableRuntimeCapabilities() {
  return const LlamaRuntimeCapabilities(
    nativeBridgeAvailable: false,
    bridgeAbiVersion: LlamaRuntime.bridgeAbiVersion,
    modelLoading: false,
    tokenization: false,
    textGeneration: false,
    structuredOutput: false,
    embeddings: false,
    reranking: false,
    rag: true,
    multimodal: false,
    lora: false,
    speculativeDecoding: false,
    mtp: false,
    metal: false,
    vulkan: false,
    nativeLogging: false,
    prefill: false,
  );
}

abstract final class LlamaModel {
  static Future<LlamaModelFileInfo> validateFile(
    String path, {
    int? expectedSizeBytes,
    String? expectedSha256,
    bool requireGgufMagic = true,
  }) async {
    _validateModelPath(path, 'path');
    final expected = expectedSizeBytes;
    if (expected != null && expected <= 0) {
      throw ArgumentError.value(
        expectedSizeBytes,
        'expectedSizeBytes',
        'must be positive',
      );
    }
    final expectedDigest = expectedSha256 == null
        ? null
        : _normalizeSha256(expectedSha256, 'expectedSha256');

    await _requireRegularModelFile(path);
    final size = await _modelFileLength(path);
    if (size == 0) {
      throw ModelFileException('Model file is empty: $path');
    }
    if (expected != null && size != expected) {
      throw ModelFileException(
        'Model file size mismatch for $path: expected $expected bytes, got $size bytes.',
      );
    }
    if (requireGgufMagic) {
      await _requireGgufMagic(path);
    }

    final digest = expectedDigest == null ? null : await _modelFileSha256(path);
    if (expectedDigest != null && digest != expectedDigest) {
      throw ModelFileException('Model file SHA-256 mismatch: $path');
    }
    return LlamaModelFileInfo(path: path, sizeBytes: size, sha256: digest);
  }

  static Future<String> sha256(String path) async {
    _validateModelPath(path, 'path');
    await _requireRegularModelFile(path);
    return _modelFileSha256(path);
  }

  static Future<LlamaModelFileInfo> deleteFile(
    String path, {
    int? expectedSizeBytes,
    String? expectedSha256,
    bool requireGgufMagic = true,
  }) async {
    final info = await validateFile(
      path,
      expectedSizeBytes: expectedSizeBytes,
      expectedSha256: expectedSha256,
      requireGgufMagic: requireGgufMagic,
    );
    await _rejectModelFileLink(path);
    try {
      await File(path).delete();
    } on FileSystemException catch (error) {
      throw ModelFileException(
        'Model file could not be deleted: $path',
        cause: error,
      );
    }
    return info;
  }

  static Future<LlamaModelInfo> inspect(LlamaModelConfig config) async {
    config.validate();
    return NativeLlamaBridge.inspectModel(config);
  }

  static Future<Map<String, String>> metadata(LlamaModelConfig config) async {
    config.validate();
    return NativeLlamaBridge.modelMetadata(config);
  }

  /// Returns the effective explicit or GGUF-provided chat template.
  ///
  /// Throws [UnsupportedFeatureException] instead of guessing a fallback when
  /// the model has no valid template.
  static Future<String> chatTemplate(LlamaModelConfig config) async {
    config.validate();
    return NativeLlamaBridge.chatTemplate(config);
  }

  static Future<String> architecture(LlamaModelConfig config) async {
    final value = (await metadata(config))['general.architecture'];
    if (value == null || value.trim().isEmpty) {
      throw const ModelLoadException(
        'Model metadata does not define general.architecture.',
      );
    }
    return value;
  }
}

Future<void> _rejectModelFileLink(String path) async {
  final FileSystemEntityType type;
  try {
    type = await FileSystemEntity.type(path, followLinks: false);
  } on FileSystemException catch (error) {
    throw ModelFileException('Model file is not readable: $path', cause: error);
  }
  if (type == FileSystemEntityType.link) {
    throw ModelFileException('Model file path is a symbolic link: $path');
  }
}

Future<void> _rejectStateFileLink(String path) async {
  final FileSystemEntityType type;
  try {
    type = await FileSystemEntity.type(path, followLinks: false);
  } on FileSystemException catch (error) {
    throw StateFileException(
      'Session state path is not readable: $path',
      cause: error,
    );
  }
  if (type == FileSystemEntityType.link) {
    throw StateFileException('Session state path is a symbolic link: $path');
  }
}

abstract final class LlamaTokenizer {
  static Future<List<int>> tokenize(
    LlamaModelConfig config,
    String text, {
    bool addSpecial = false,
    bool parseSpecial = false,
  }) async {
    config.validate();
    _validateNativeText(text, 'text');
    return NativeLlamaBridge.tokenize(
      config,
      text,
      addSpecial: addSpecial,
      parseSpecial: parseSpecial,
    );
  }

  static Future<int> countTokens(
    LlamaModelConfig config,
    String text, {
    bool addSpecial = false,
    bool parseSpecial = false,
  }) async {
    return (await tokenize(
      config,
      text,
      addSpecial: addSpecial,
      parseSpecial: parseSpecial,
    )).length;
  }

  static Future<String> detokenize(
    LlamaModelConfig config,
    List<int> tokens, {
    bool removeSpecial = false,
    bool unparseSpecial = false,
  }) async {
    config.validate();
    final checkedTokens = List<int>.unmodifiable(tokens);
    _validateTokenIds(checkedTokens);
    if (checkedTokens.isEmpty) {
      return '';
    }
    return NativeLlamaBridge.detokenize(
      config,
      checkedTokens,
      removeSpecial: removeSpecial,
      unparseSpecial: unparseSpecial,
    );
  }
}

abstract final class LlamaChatTemplate {
  static Future<LlamaChatTemplateCapabilities> capabilities(
    LlamaModelConfig config,
  ) async {
    config.validate();
    return NativeLlamaBridge.chatTemplateCapabilities(config);
  }

  static Future<String> format(
    LlamaModelConfig config,
    List<ChatMessage> messages, {
    bool addAssistantPrompt = true,
    LlamaToolCallingConfig toolCalling = const LlamaToolCallingConfig(),
  }) async {
    config.validate();
    final checkedMessages = _snapshotChatMessages(messages);
    final checkedToolCalling = _snapshotToolCallingConfig(toolCalling);
    _validateChatMessages(
      checkedMessages,
      allowMultimodal: config.mmprojPath != null,
    );
    checkedToolCalling.validate();
    return NativeLlamaBridge.formatChat(
      config,
      checkedMessages,
      addAssistantPrompt: addAssistantPrompt,
      toolCalling: checkedToolCalling,
    );
  }

  static Future<int> countTokens(
    LlamaModelConfig config,
    List<ChatMessage> messages, {
    bool addAssistantPrompt = true,
    LlamaToolCallingConfig toolCalling = const LlamaToolCallingConfig(),
  }) async {
    if (messages.any((message) => message.hasNonTextParts)) {
      throw const UnsupportedFeatureException(
        'LlamaChatTemplate.countTokens cannot count model-specific media '
        'tokens. Use text-only messages.',
      );
    }
    config.validate();
    final checkedMessages = _snapshotChatMessages(messages);
    final checkedToolCalling = _snapshotToolCallingConfig(toolCalling);
    _validateChatMessages(checkedMessages, allowMultimodal: false);
    checkedToolCalling.validate();
    return NativeLlamaBridge.countChatTokens(
      config,
      checkedMessages,
      addAssistantPrompt: addAssistantPrompt,
      toolCalling: checkedToolCalling,
    );
  }
}

abstract final class LlamaEmbeddings {
  static Future<Float32List> embedText(
    LlamaModelConfig modelConfig,
    String text, {
    EmbeddingConfig config = const EmbeddingConfig(),
  }) async {
    modelConfig.validate();
    _requireEmbeddingContextConfig(modelConfig);
    config.validate();
    if (text.trim().isEmpty) {
      throw ArgumentError.value(text, 'text', 'must not be empty');
    }
    _validateNativeText(text, 'text');
    return NativeLlamaBridge.embedText(modelConfig, text, config);
  }

  static Future<EmbeddingBatch> embedTexts(
    LlamaModelConfig modelConfig,
    List<String> texts, {
    EmbeddingConfig config = const EmbeddingConfig(),
  }) async {
    modelConfig.validate();
    _requireEmbeddingContextConfig(modelConfig);
    config.validate();
    final checkedTexts = List<String>.unmodifiable(texts);
    for (final text in checkedTexts) {
      if (text.trim().isEmpty) {
        throw ArgumentError.value(
          checkedTexts,
          'texts',
          'must not contain empty text',
        );
      }
      if (text.contains('\u0000')) {
        throw ArgumentError.value(
          checkedTexts,
          'texts',
          'must not contain NUL',
        );
      }
    }
    if (checkedTexts.isEmpty) {
      return EmbeddingBatch.empty(
        normalized: config.normalize,
        pooling: config.pooling,
      );
    }
    return NativeLlamaBridge.embedTexts(modelConfig, checkedTexts, config);
  }
}

/// A worker-owned embedding context that keeps its model loaded across calls.
///
/// Operations are serialized by the worker isolate. Call [close] when the
/// owning application no longer needs tokenization or embeddings.
final class LlamaEmbeddingEngine {
  LlamaEmbeddingEngine._(this.modelConfig, this.embeddingConfig, this._native);

  final LlamaModelConfig modelConfig;
  final EmbeddingConfig embeddingConfig;
  final NativeLlamaEngineSession _native;
  bool _closed = false;
  Future<void>? _closeFuture;

  static Future<LlamaEmbeddingEngine> load(
    LlamaModelConfig modelConfig, {
    EmbeddingConfig config = const EmbeddingConfig(),
  }) async {
    modelConfig.validate();
    _requireEmbeddingContextConfig(modelConfig);
    config.validate();
    final native = await NativeLlamaBridge.startEmbeddingEngine(
      modelConfig,
      config.pooling,
    );
    return LlamaEmbeddingEngine._(modelConfig, config, native);
  }

  Future<Float32List> embedText(String text) async {
    final batch = await embedTexts(<String>[text]);
    return Float32List.fromList(batch.single);
  }

  Future<EmbeddingBatch> embedTexts(List<String> texts) {
    _ensureOpen();
    final checkedTexts = List<String>.unmodifiable(texts);
    for (final text in checkedTexts) {
      if (text.trim().isEmpty) {
        throw ArgumentError.value(
          checkedTexts,
          'texts',
          'must not contain empty text',
        );
      }
      if (text.contains('\u0000')) {
        throw ArgumentError.value(
          checkedTexts,
          'texts',
          'must not contain NUL',
        );
      }
    }
    if (checkedTexts.isEmpty) {
      return Future<EmbeddingBatch>.value(
        EmbeddingBatch.empty(
          normalized: embeddingConfig.normalize,
          pooling: embeddingConfig.pooling,
        ),
      );
    }
    return _native.embedTexts(checkedTexts, embeddingConfig);
  }

  Future<LlamaModelInfo> modelInfo() {
    _ensureOpen();
    return _native.modelInfo();
  }

  Future<Map<String, String>> modelMetadata() {
    _ensureOpen();
    return _native.modelMetadata();
  }

  Future<List<int>> tokenize(
    String text, {
    bool addSpecial = false,
    bool parseSpecial = false,
  }) {
    _ensureOpen();
    _validateNativeText(text, 'text');
    return _native.tokenize(
      text,
      addSpecial: addSpecial,
      parseSpecial: parseSpecial,
    );
  }

  Future<int> countTokens(
    String text, {
    bool addSpecial = false,
    bool parseSpecial = false,
  }) {
    _ensureOpen();
    _validateNativeText(text, 'text');
    return _native.countTokens(
      text,
      addSpecial: addSpecial,
      parseSpecial: parseSpecial,
    );
  }

  Future<String> detokenize(
    List<int> tokens, {
    bool removeSpecial = false,
    bool unparseSpecial = false,
  }) {
    _ensureOpen();
    final checkedTokens = List<int>.unmodifiable(tokens);
    _validateTokenIds(checkedTokens);
    if (checkedTokens.isEmpty) {
      return Future<String>.value('');
    }
    return _native.detokenize(
      checkedTokens,
      removeSpecial: removeSpecial,
      unparseSpecial: unparseSpecial,
    );
  }

  Future<void> close() {
    final existing = _closeFuture;
    if (existing != null) {
      return existing;
    }
    _closed = true;
    final closeFuture = _native.close();
    _closeFuture = closeFuture;
    return closeFuture;
  }

  void _ensureOpen() {
    if (_closed) {
      throw const ResourceDisposedException('LlamaEmbeddingEngine is closed.');
    }
  }
}

final class LlamaEmbeddingModel implements EmbeddingModel {
  const LlamaEmbeddingModel(
    this.modelConfig, {
    this.config = const EmbeddingConfig(),
  });

  final LlamaModelConfig modelConfig;
  final EmbeddingConfig config;

  @override
  Future<Float32List> embedText(String text) {
    return LlamaEmbeddings.embedText(modelConfig, text, config: config);
  }

  Future<EmbeddingBatch> embedTexts(List<String> texts) {
    return LlamaEmbeddings.embedTexts(modelConfig, texts, config: config);
  }
}

abstract final class LlamaReranking {
  static Future<double> scorePair(
    LlamaModelConfig modelConfig, {
    required String query,
    required String document,
    RerankingConfig config = const RerankingConfig(),
  }) async {
    final scores = await scoreDocuments(
      modelConfig,
      query: query,
      documents: <String>[document],
      config: config,
    );
    return scores.single;
  }

  static Future<List<double>> scoreDocuments(
    LlamaModelConfig modelConfig, {
    required String query,
    required List<String> documents,
    RerankingConfig config = const RerankingConfig(),
  }) async {
    modelConfig.validate();
    _requireEmbeddingContextConfig(modelConfig);
    _validateRerankText(query, 'query');
    final checkedDocuments = List<String>.unmodifiable(documents);
    for (final document in checkedDocuments) {
      _validateRerankText(document, 'documents');
    }
    if (checkedDocuments.isEmpty) {
      return const <double>[];
    }
    return NativeLlamaBridge.rerankDocuments(
      modelConfig,
      query,
      checkedDocuments,
      config,
    );
  }
}

final class LlamaReranker implements Reranker {
  const LlamaReranker(
    this.modelConfig, {
    this.config = const RerankingConfig(),
  });

  final LlamaModelConfig modelConfig;
  final RerankingConfig config;

  @override
  Future<List<VectorSearchResult>> rerank(
    String query,
    List<VectorSearchResult> results,
  ) async {
    modelConfig.validate();
    _requireEmbeddingContextConfig(modelConfig);
    _validateRerankText(query, 'query');
    if (results.isEmpty) {
      return const <VectorSearchResult>[];
    }
    final checkedResults = _snapshotRerankCandidates(results);
    final scores = await LlamaReranking.scoreDocuments(
      modelConfig,
      query: query,
      documents: <String>[
        for (final result in checkedResults) result.chunk.text,
      ],
      config: config,
    );
    final rescored = <({int index, VectorSearchResult result})>[
      for (var i = 0; i < checkedResults.length; i += 1)
        (
          index: i,
          result: VectorSearchResult(
            chunk: checkedResults[i].chunk,
            score: scores[i],
          ),
        ),
    ];
    rescored.sort((a, b) {
      final byScore = b.result.score.compareTo(a.result.score);
      return byScore == 0 ? a.index.compareTo(b.index) : byScore;
    });
    return List<VectorSearchResult>.unmodifiable(
      rescored.map((item) => item.result),
    );
  }
}

final class GenerationChunk {
  const GenerationChunk({
    required this.text,
    this.isDone = false,
    this.telemetry,
    this.assistantMessage,
    this.stopReason,
  });

  final String text;
  final bool isDone;
  final GenerationTelemetry? telemetry;
  final ChatMessage? assistantMessage;

  /// Why a terminal chunk ended. Non-terminal chunks have no stop reason.
  final GenerationStopReason? stopReason;
}

final class LlamaEngine {
  LlamaEngine._(this.config, this._native);

  final LlamaModelConfig config;
  final NativeLlamaEngineSession _native;
  bool _closed = false;
  Future<void>? _closeFuture;

  static Future<LlamaEngine> load(LlamaModelConfig config) async {
    config.validate();
    _requireSupportedLoadConfig(config);
    final native = await NativeLlamaBridge.startEngine(config);
    return LlamaEngine._(config, native);
  }

  /// Generates an assistant response for [messages].
  ///
  /// When [reusePromptPrefix] is true, the native session keeps its existing
  /// text KV state only if its committed token history is an exact prefix of
  /// the newly rendered and tokenized full prompt. A mismatch resets the
  /// session and evaluates the full prompt. Media prompts are not eligible.
  Stream<GenerationChunk> chat({
    required List<ChatMessage> messages,
    GenerationConfig config = const GenerationConfig(),
    bool reusePromptPrefix = false,
  }) {
    _ensureOpen();
    final checkedMessages = _snapshotChatMessages(messages);
    final checkedConfig = _snapshotGenerationConfig(config);
    _validateChatMessages(
      checkedMessages,
      allowMultimodal: this.config.mmprojPath != null,
    );
    checkedConfig.validate();
    return _native
        .completeChatStream(
          checkedMessages,
          checkedConfig,
          reusePromptPrefix: reusePromptPrefix,
        )
        .map(
          (chunk) => GenerationChunk(
            text: chunk.text,
            isDone: chunk.isDone,
            telemetry: chunk.telemetry,
            assistantMessage: chunk.assistantMessage,
            stopReason: chunk.isDone ? chunk.telemetry?.stopReason : null,
          ),
        );
  }

  Stream<GenerationChunk> complete({
    required String prompt,
    GenerationConfig config = const GenerationConfig(),
  }) {
    _ensureOpen();
    if (prompt.trim().isEmpty) {
      throw ArgumentError.value(prompt, 'prompt', 'must not be blank');
    }
    _validateNativeText(prompt, 'prompt');
    final checkedConfig = _snapshotGenerationConfig(config);
    checkedConfig.validate();
    if (checkedConfig.enableThinking != null ||
        checkedConfig.reasoningBudgetTokens != null) {
      throw const UnsupportedFeatureException(
        'Thinking control requires chat messages and a model chat template.',
      );
    }
    if (checkedConfig.toolCalling.tools.isNotEmpty ||
        checkedConfig.toolCalling.toolChoice is! LlamaAutoToolChoice) {
      throw const UnsupportedFeatureException(
        'Tool calling requires chat messages and a model chat template.',
      );
    }
    return _native
        .completeStream(prompt, checkedConfig)
        .map(
          (chunk) => GenerationChunk(
            text: chunk.text,
            isDone: chunk.isDone,
            telemetry: chunk.telemetry,
            stopReason: chunk.isDone ? chunk.telemetry?.stopReason : null,
          ),
        );
  }

  /// Samples from the current context without evaluating another prompt.
  ///
  /// The context must already contain tokens from [prefill], [complete],
  /// [chat], or a restored state. Cancelling the returned subscription requests
  /// a native decode abort just like [complete].
  Stream<GenerationChunk> continueCompletion({
    GenerationConfig config = const GenerationConfig(),
  }) {
    _ensureOpen();
    final checkedConfig = _snapshotGenerationConfig(config);
    checkedConfig.validate();
    if (checkedConfig.enableThinking != null ||
        checkedConfig.reasoningBudgetTokens != null) {
      throw const UnsupportedFeatureException(
        'Thinking control requires chat messages and a model chat template.',
      );
    }
    if (checkedConfig.toolCalling.tools.isNotEmpty ||
        checkedConfig.toolCalling.toolChoice is! LlamaAutoToolChoice) {
      throw const UnsupportedFeatureException(
        'Tool calling requires chat messages and a model chat template.',
      );
    }
    return _native
        .completeStream('', checkedConfig)
        .map(
          (chunk) => GenerationChunk(
            text: chunk.text,
            isDone: chunk.isDone,
            telemetry: chunk.telemetry,
            stopReason: chunk.isDone ? chunk.telemetry?.stopReason : null,
          ),
        );
  }

  /// Inspects the already-loaded native model without reopening its path.
  Future<LlamaModelInfo> modelInfo() async {
    _ensureOpen();
    return _native.modelInfo();
  }

  /// Reads bounded GGUF metadata from the already-loaded native model.
  Future<Map<String, String>> modelMetadata() async {
    _ensureOpen();
    return _native.modelMetadata();
  }

  /// Returns the loaded model's effective explicit or embedded chat template.
  ///
  /// No generic template is guessed when the model does not provide one.
  Future<String> chatTemplate() async {
    _ensureOpen();
    return _native.chatTemplate();
  }

  /// Tokenizes [text] with the model already owned by this engine.
  ///
  /// The operation runs on the engine worker and is serialized with inference.
  Future<List<int>> tokenize(
    String text, {
    bool addSpecial = false,
    bool parseSpecial = false,
  }) async {
    _ensureOpen();
    _validateNativeText(text, 'text');
    return _native.tokenize(
      text,
      addSpecial: addSpecial,
      parseSpecial: parseSpecial,
    );
  }

  /// Counts [text] tokens without reloading the model or transferring token IDs.
  Future<int> countTokens(
    String text, {
    bool addSpecial = false,
    bool parseSpecial = false,
  }) async {
    _ensureOpen();
    _validateNativeText(text, 'text');
    return _native.countTokens(
      text,
      addSpecial: addSpecial,
      parseSpecial: parseSpecial,
    );
  }

  /// Detokenizes [tokens] with the model already owned by this engine.
  Future<String> detokenize(
    List<int> tokens, {
    bool removeSpecial = false,
    bool unparseSpecial = false,
  }) async {
    _ensureOpen();
    final checkedTokens = List<int>.unmodifiable(tokens);
    _validateTokenIds(checkedTokens);
    if (checkedTokens.isEmpty) {
      return '';
    }
    return _native.detokenize(
      checkedTokens,
      removeSpecial: removeSpecial,
      unparseSpecial: unparseSpecial,
    );
  }

  /// Reports tool-call features supported by the loaded model's chat template.
  Future<LlamaChatTemplateCapabilities> chatTemplateCapabilities() async {
    _ensureOpen();
    return _native.chatTemplateCapabilities();
  }

  /// Applies the loaded model's chat template without starting generation.
  Future<String> formatChat(
    List<ChatMessage> messages, {
    bool addAssistantPrompt = true,
    LlamaToolCallingConfig toolCalling = const LlamaToolCallingConfig(),
  }) async {
    _ensureOpen();
    final checkedMessages = _snapshotChatMessages(messages);
    final checkedToolCalling = _snapshotToolCallingConfig(toolCalling);
    _validateChatMessages(
      checkedMessages,
      allowMultimodal: config.mmprojPath != null,
    );
    checkedToolCalling.validate();
    return _native.formatChat(
      checkedMessages,
      addAssistantPrompt: addAssistantPrompt,
      toolCalling: checkedToolCalling,
    );
  }

  /// Counts a fully templated text-only chat using the already-loaded model.
  Future<int> countChatTokens(
    List<ChatMessage> messages, {
    bool addAssistantPrompt = true,
    LlamaToolCallingConfig toolCalling = const LlamaToolCallingConfig(),
  }) async {
    _ensureOpen();
    if (messages.any((message) => message.hasNonTextParts)) {
      throw const UnsupportedFeatureException(
        'LlamaEngine.countChatTokens cannot count model-specific media '
        'tokens. Use text-only messages.',
      );
    }
    final checkedMessages = _snapshotChatMessages(messages);
    final checkedToolCalling = _snapshotToolCallingConfig(toolCalling);
    _validateChatMessages(checkedMessages, allowMultimodal: false);
    checkedToolCalling.validate();
    return _native.countChatTokens(
      checkedMessages,
      addAssistantPrompt: addAssistantPrompt,
      toolCalling: checkedToolCalling,
    );
  }

  /// Warms the model execution graph without changing session state.
  ///
  /// The context must be empty. Call [reset] first after any inference.
  Future<void> warmUp() async {
    _ensureOpen();
    await _native.warmUp();
  }

  /// Evaluates [prompt] into the current context without sampling a token.
  ///
  /// By default, model special tokens are added only when the context is
  /// empty. Pass [addSpecial] to override that behavior for model-specific
  /// incremental prompting. [parseSpecial] should only be enabled for trusted
  /// model-template text.
  Future<PrefillTelemetry> prefill({
    required String prompt,
    bool? addSpecial,
    bool parseSpecial = false,
  }) async {
    _ensureOpen();
    if (prompt.trim().isEmpty) {
      throw ArgumentError.value(prompt, 'prompt', 'must not be blank');
    }
    _validateNativeText(prompt, 'prompt');
    return _native.prefill(
      prompt,
      addSpecial: addSpecial,
      parseSpecial: parseSpecial,
    );
  }

  Future<void> reset() async {
    _ensureOpen();
    await _native.reset();
  }

  /// Discards old context tokens while preserving a prefix and recent tail.
  ///
  /// When [discardTokens] is omitted, half of the removable suffix is
  /// discarded. The returned value is the number of tokens removed.
  Future<int> shiftContext({int keepTokens = 0, int? discardTokens}) async {
    _ensureOpen();
    if (keepTokens < 0 || keepTokens > 0xFFFFFFFF) {
      throw ArgumentError.value(keepTokens, 'keepTokens', 'must fit uint32');
    }
    if (discardTokens != null &&
        (discardTokens <= 0 || discardTokens > 0xFFFFFFFF)) {
      throw ArgumentError.value(
        discardTokens,
        'discardTokens',
        'must be positive and fit uint32',
      );
    }
    return _native.shiftContext(
      keepTokens: keepTokens,
      discardTokens: discardTokens,
    );
  }

  Future<LlamaContextInfo> contextInfo() async {
    _ensureOpen();
    return _native.contextInfo();
  }

  Future<Uint8List> saveState() async {
    _ensureOpen();
    return _native.saveState();
  }

  Future<void> saveStateToFile(String path) async {
    _ensureOpen();
    _validateFilePath(path, 'path');
    await _rejectStateFileLink(path);
    final state = await saveState();
    try {
      await File(path).writeAsBytes(state, flush: true);
    } on FileSystemException catch (error) {
      throw StateFileException(
        'Session state could not be written: $path',
        cause: error,
      );
    }
  }

  Future<void> restoreState(Uint8List state) async {
    _ensureOpen();
    if (state.isEmpty) {
      throw ArgumentError.value(state, 'state', 'must not be empty');
    }
    await _native.restoreState(state);
  }

  Future<void> restoreStateFromFile(String path) async {
    _ensureOpen();
    _validateFilePath(path, 'path');
    await _rejectStateFileLink(path);
    final state = await _readStateFile(path);
    if (state.isEmpty) {
      throw StateFileException('Session state file is empty: $path');
    }
    await restoreState(state);
  }

  Future<LoraAdapterInfo> loadLora(LoraAdapterConfig config) async {
    _ensureOpen();
    config.validate();
    return _native.loadLora(config);
  }

  Future<List<LoraAdapterInfo>> loraAdapters() async {
    _ensureOpen();
    return _native.loraAdapters();
  }

  Future<void> setLoraScale(int adapterId, double scale) async {
    _ensureOpen();
    if (adapterId <= 0) {
      throw ArgumentError.value(adapterId, 'adapterId', 'must be positive');
    }
    if (!scale.isFinite) {
      throw ArgumentError.value(scale, 'scale', 'must be finite');
    }
    if (scale < 0) {
      throw ArgumentError.value(scale, 'scale', 'must be non-negative');
    }
    await _native.setLoraScale(adapterId, scale);
  }

  Future<void> unloadLora(int adapterId) async {
    _ensureOpen();
    if (adapterId <= 0) {
      throw ArgumentError.value(adapterId, 'adapterId', 'must be positive');
    }
    await _native.unloadLora(adapterId);
  }

  Future<void> close() {
    final existing = _closeFuture;
    if (existing != null) {
      return existing;
    }
    _closed = true;
    final closeFuture = _native.close();
    _closeFuture = closeFuture;
    return closeFuture;
  }

  void _ensureOpen() {
    if (_closed) {
      throw const ResourceDisposedException('LlamaEngine is closed.');
    }
    if (_native.hasActiveGeneration) {
      throw const GenerationException(
        'Another generation is already active on this engine.',
      );
    }
  }
}

List<ChatMessage> _snapshotChatMessages(List<ChatMessage> messages) {
  return List<ChatMessage>.unmodifiable(
    messages.map((message) {
      if (message.toolCalls.isNotEmpty) {
        return ChatMessage.assistantToolCalls(
          text: message.text,
          toolCalls: message.toolCalls.map(_snapshotMessageToolCall).toList(),
        );
      }
      final toolCallId = message.toolCallId;
      final toolName = message.toolName;
      if (toolCallId != null && toolName != null) {
        return ChatMessage.toolResult(
          toolCallId: toolCallId,
          name: toolName,
          text: message.text,
        );
      }
      if (message.parts.isNotEmpty) {
        return ChatMessage.content(role: message.role, parts: message.parts);
      }
      return ChatMessage(role: message.role, text: message.text);
    }),
  );
}

LlamaToolCall _snapshotMessageToolCall(LlamaToolCall call) {
  return LlamaToolCall(
    id: call.id,
    name: call.name,
    arguments: _snapshotJsonObject(
      call.arguments,
      'arguments',
      Set<Object>.identity(),
    ),
  );
}

GenerationConfig _snapshotGenerationConfig(GenerationConfig config) {
  return GenerationConfig(
    maxTokens: config.maxTokens,
    temperature: config.temperature,
    topK: config.topK,
    topP: config.topP,
    minP: config.minP,
    typicalP: config.typicalP,
    penaltyLastN: config.penaltyLastN,
    repeatPenalty: config.repeatPenalty,
    frequencyPenalty: config.frequencyPenalty,
    presencePenalty: config.presencePenalty,
    mirostat: config.mirostat,
    mirostatTau: config.mirostatTau,
    mirostatEta: config.mirostatEta,
    seed: config.seed,
    streamChunkTokens: config.streamChunkTokens,
    stop: List<String>.unmodifiable(config.stop),
    stopTokens: List<int>.unmodifiable(config.stopTokens),
    loraScales: config.loraScales == null
        ? null
        : Map<int, double>.unmodifiable(config.loraScales!),
    grammar: config.grammar,
    jsonSchema: config.jsonSchema == null
        ? null
        : _snapshotJsonObject(
            config.jsonSchema!,
            'jsonSchema',
            Set<Object>.identity(),
          ),
    grammarRoot: config.grammarRoot,
    toolCalling: _snapshotToolCallingConfig(config.toolCalling),
    enableThinking: config.enableThinking,
    reasoningBudgetTokens: config.reasoningBudgetTokens,
  );
}

LlamaToolCallingConfig _snapshotToolCallingConfig(
  LlamaToolCallingConfig config,
) {
  return LlamaToolCallingConfig(
    tools: List<LlamaToolDefinition>.unmodifiable(
      config.tools.map(_snapshotToolDefinition),
    ),
    allowParallelToolCalls: config.allowParallelToolCalls,
    toolChoice: config.toolChoice,
  );
}

LlamaToolDefinition _snapshotToolDefinition(LlamaToolDefinition tool) {
  final json = tool.toJson();
  final function = json['function']! as Map<String, Object?>;
  return LlamaToolDefinition(
    name: function['name']! as String,
    description: function['description']! as String,
    parametersSchema: function['parameters']! as Map<String, Object?>,
  );
}

const _maxMediaInputBytes = 64 * 1024 * 1024;
const _maxMediaInputs = 64;

void _validateChatMessages(
  List<ChatMessage> messages, {
  bool allowMultimodal = false,
}) {
  if (messages.isEmpty) {
    throw ArgumentError.value(messages, 'messages', 'must not be empty');
  }
  var mediaCount = 0;
  final toolCallIds = <String>{};
  final answeredToolCallIds = <String>{};
  for (final message in messages) {
    if (!message.hasNonTextParts &&
        message.toolCalls.isEmpty &&
        message.text.trim().isEmpty) {
      throw ArgumentError.value(
        messages,
        'messages',
        'text-only messages must not be empty',
      );
    }
    if (message.text.contains('\u0000')) {
      throw ArgumentError.value(messages, 'messages', 'must not contain NUL');
    }
    if (message.toolCalls.isNotEmpty) {
      if (message.role != ChatRole.assistant || message.parts.isNotEmpty) {
        throw ArgumentError.value(
          messages,
          'messages',
          'tool calls require a text-only assistant message',
        );
      }
      for (final call in message.toolCalls) {
        call.toJson();
        final id = call.id;
        if (id != null && !toolCallIds.add(id)) {
          throw ArgumentError.value(
            id,
            'messages',
            'tool call ids must be unique',
          );
        }
      }
    }
    final toolCallId = message.toolCallId;
    final toolName = message.toolName;
    if ((toolCallId == null) != (toolName == null) ||
        (toolCallId != null && message.role != ChatRole.tool) ||
        (toolCallId != null && message.parts.isNotEmpty)) {
      throw ArgumentError.value(
        messages,
        'messages',
        'tool results require a text-only tool message with name and call id',
      );
    }
    if (toolCallId != null && toolName != null) {
      _validateToolMessageIdentifier(toolCallId, 'toolCallId');
      _validateToolMessageIdentifier(toolName, 'toolName');
      if (!toolCallIds.contains(toolCallId)) {
        throw ArgumentError.value(
          toolCallId,
          'messages',
          'must reference an earlier assistant tool call',
        );
      }
      if (!answeredToolCallIds.add(toolCallId)) {
        throw ArgumentError.value(
          toolCallId,
          'messages',
          'must have only one tool result',
        );
      }
    }
    for (final part in message.parts) {
      _validateContentPart(part);
      if (part is! TextPart) {
        mediaCount += 1;
      }
    }
    if (message.hasNonTextParts) {
      final kind = _firstNonTextPartKind(message.parts);
      if (kind == 'video') {
        throw const UnsupportedFeatureException(
          'Video chat input requires an external ffmpeg runtime and is not '
          'available in mobile builds.',
        );
      }
      if (!allowMultimodal) {
        throw UnsupportedFeatureException(
          'Multimodal $kind chat parts require LlamaModelConfig.mmprojPath.',
        );
      }
    }
  }
  if (mediaCount > _maxMediaInputs) {
    throw ArgumentError.value(
      messages,
      'messages',
      'must contain at most $_maxMediaInputs media inputs',
    );
  }
}

void _validateToolMessageIdentifier(String value, String name) {
  if (value.trim().isEmpty || value.contains(RegExp(r'\s'))) {
    throw ArgumentError.value(value, name, 'must not be empty or whitespace');
  }
  _validateNativeText(value, name);
}

String _firstNonTextPartKind(List<ChatContentPart> parts) {
  for (final part in parts) {
    if (part is ImagePart) {
      return 'image';
    }
    if (part is AudioPart) {
      return 'audio';
    }
    if (part is VideoPart) {
      return 'video';
    }
  }
  return 'media';
}

void _validateContentPart(ChatContentPart part) {
  switch (part) {
    case TextPart(:final text):
      _validateNativeText(text, 'part.text');
    case ImagePart(:final path, :final bytes, :final mimeType):
      _validateMediaPart(path, bytes, mimeType, 'image');
    case AudioPart(:final path, :final bytes, :final mimeType):
      _validateMediaPart(path, bytes, mimeType, 'audio');
    case VideoPart(:final path, :final bytes, :final mimeType):
      _validateMediaPart(path, bytes, mimeType, 'video');
  }
}

void _validateMediaPart(
  String? path,
  Uint8List? bytes,
  String? mimeType,
  String name,
) {
  if (path == null && bytes == null) {
    throw ArgumentError.value(name, 'part', 'must contain a path or bytes');
  }
  if (path != null) {
    _validateFilePath(path, 'part.path');
  }
  if (bytes != null && bytes.isEmpty) {
    throw ArgumentError.value(bytes, 'part.bytes', 'must not be empty');
  }
  if (bytes != null && bytes.length > _maxMediaInputBytes) {
    throw ArgumentError.value(bytes, 'part.bytes', 'must not exceed 64 MiB');
  }
  final mime = mimeType;
  if (mime != null) {
    if (mime.trim().isEmpty) {
      throw ArgumentError.value(mime, 'part.mimeType', 'must not be empty');
    }
    _validateNativeText(mime, 'part.mimeType');
    if (!mime.startsWith('$name/')) {
      throw ArgumentError.value(
        mime,
        'part.mimeType',
        'must start with $name/',
      );
    }
    final subtype = mime.substring(name.length + 1);
    if (subtype.trim().isEmpty) {
      throw ArgumentError.value(mime, 'part.mimeType', 'must include subtype');
    }
    if (subtype.contains(RegExp(r'\s'))) {
      throw ArgumentError.value(
        mime,
        'part.mimeType',
        'must not contain whitespace',
      );
    }
  }
}

void _validateNativeText(String text, String name) {
  if (text.contains('\u0000')) {
    throw ArgumentError.value(text, name, 'must not contain NUL');
  }
}

void _validateModelPath(String path, String name) {
  _validateFilePath(path, name);
}

void _validateFilePath(String path, String name) {
  if (path.trim().isEmpty) {
    throw ArgumentError.value(path, name, 'must not be empty');
  }
  _validateNativeText(path, name);
  if (path.contains('\n') || path.contains('\r')) {
    throw ArgumentError.value(path, name, 'must not contain line breaks');
  }
}

Future<int> _modelFileLength(String path) async {
  try {
    return await File(path).length();
  } on FileSystemException catch (error) {
    throw ModelFileException('Model file is not readable: $path', cause: error);
  }
}

Future<void> _requireRegularModelFile(String path) async {
  final FileSystemEntityType type;
  try {
    type = await FileSystemEntity.type(path);
  } on FileSystemException catch (error) {
    throw ModelFileException('Model file is not readable: $path', cause: error);
  }
  if (type == FileSystemEntityType.notFound) {
    throw ModelFileException('Model file does not exist: $path');
  }
  if (type != FileSystemEntityType.file) {
    throw ModelFileException('Model path is not a file: $path');
  }
}

Future<void> _requireGgufMagic(String path) async {
  RandomAccessFile? file;
  try {
    file = await File(path).open();
    final bytes = await file.read(4);
    if (bytes.length != 4 ||
        bytes[0] != 0x47 ||
        bytes[1] != 0x47 ||
        bytes[2] != 0x55 ||
        bytes[3] != 0x46) {
      throw ModelFileException('Model file is not a GGUF file: $path');
    }
  } on FileSystemException catch (error) {
    throw ModelFileException('Model file is not readable: $path', cause: error);
  } finally {
    await file?.close();
  }
}

Future<String> _modelFileSha256(String path) async {
  return Isolate.run(() => _modelFileSha256InWorker(path));
}

Future<String> _modelFileSha256InWorker(String path) async {
  try {
    final digest = await crypto.sha256.bind(File(path).openRead()).first;
    return digest.toString();
  } on FileSystemException catch (error) {
    throw ModelFileException('Model file is not readable: $path', cause: error);
  }
}

Future<Uint8List> _readStateFile(String path) async {
  try {
    return await File(path).readAsBytes();
  } on FileSystemException catch (error) {
    throw StateFileException(
      'Session state file is not readable: $path',
      cause: error,
    );
  }
}

String _normalizeSha256(String value, String name) {
  if (!RegExp(r'^[0-9a-fA-F]{64}$').hasMatch(value)) {
    throw ArgumentError.value(value, name, 'must be 64 hex characters');
  }
  return value.toLowerCase();
}

void _validateRerankText(String text, String name) {
  if (text.trim().isEmpty) {
    throw ArgumentError.value(text, name, 'must not be empty');
  }
  _validateNativeText(text, name);
}

List<VectorSearchResult> _snapshotRerankCandidates(
  List<VectorSearchResult> results,
) {
  final snapshot = <VectorSearchResult>[];
  final seen = <String>{};
  for (final result in results) {
    final chunk = result.chunk;
    _validateRagId(chunk.documentId, 'result.chunk.documentId');
    _validateRagId(chunk.id, 'result.chunk.id');
    _validateRerankText(chunk.text, 'result.chunk.text');
    if (chunk.tokenCount <= 0) {
      throw ArgumentError.value(
        chunk.tokenCount,
        'result.chunk.tokenCount',
        'must be positive',
      );
    }
    _validateRagSourceUri(chunk.sourceUri, 'result.chunk.sourceUri');
    if (!result.score.isFinite) {
      throw ArgumentError.value(result.score, 'result.score', 'must be finite');
    }
    final key = '${chunk.documentId}\u0000${chunk.id}';
    if (!seen.add(key)) {
      throw ArgumentError.value(chunk.id, 'result.chunk', 'must be unique');
    }
    snapshot.add(
      VectorSearchResult(
        chunk: TextChunk(
          documentId: chunk.documentId,
          id: chunk.id,
          text: chunk.text,
          tokenCount: chunk.tokenCount,
          metadata: _snapshotRagMetadata(chunk.metadata),
          sourceUri: chunk.sourceUri,
        ),
        score: result.score,
      ),
    );
  }
  return List<VectorSearchResult>.unmodifiable(snapshot);
}

void _validateRagId(String value, String name) {
  if (value.trim().isEmpty) {
    throw ArgumentError.value(value, name, 'must not be empty');
  }
  _validateNativeText(value, name);
  if (value.contains('\n') || value.contains('\r')) {
    throw ArgumentError.value(value, name, 'must not contain line breaks');
  }
}

void _validateRagSourceUri(Uri? value, String name) {
  final text = value?.toString();
  if (text == null) {
    return;
  }
  if (text.trim().isEmpty) {
    throw ArgumentError.value(value, name, 'must not be empty');
  }
  final lower = text.toLowerCase();
  if (text.contains('\u0000') || lower.contains('%00')) {
    throw ArgumentError.value(value, name, 'must not contain NUL');
  }
  if (text.contains('\n') ||
      text.contains('\r') ||
      lower.contains('%0a') ||
      lower.contains('%0d')) {
    throw ArgumentError.value(value, name, 'must not contain line breaks');
  }
}

Map<String, Object?> _snapshotRagMetadata(Map<Object?, Object?> metadata) {
  return _snapshotJsonObject(
    metadata,
    'result.chunk.metadata',
    Set<Object>.identity(),
  );
}

Map<String, Object?> _snapshotJsonObject(
  Map<Object?, Object?> value,
  String name,
  Set<Object> active,
) {
  if (!active.add(value)) {
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
      result[key] = _snapshotJsonValue(entry.value, '$name.$key', active);
    }
    return Map<String, Object?>.unmodifiable(result);
  } finally {
    active.remove(value);
  }
}

Object? _snapshotJsonValue(Object? value, String name, Set<Object> active) {
  if (value == null || value is bool) {
    return value;
  }
  if (value is String) {
    _validateNativeText(value, name);
    return value;
  }
  if (value is num) {
    if (!value.isFinite) {
      throw ArgumentError.value(value, name, 'must be finite');
    }
    return value;
  }
  if (value is List<Object?>) {
    if (!active.add(value)) {
      throw ArgumentError.value(value, name, 'must not contain cycles');
    }
    try {
      return List<Object?>.unmodifiable(<Object?>[
        for (var i = 0; i < value.length; i += 1)
          _snapshotJsonValue(value[i], '$name[$i]', active),
      ]);
    } finally {
      active.remove(value);
    }
  }
  if (value is Map<Object?, Object?>) {
    return _snapshotJsonObject(value, name, active);
  }
  throw ArgumentError.value(value, name, 'must be a JSON value');
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

void _requireSupportedLoadConfig(LlamaModelConfig config) {
  final speculation = config.speculativeDecoding;
  if (config.mmprojPath != null &&
      (speculation is DraftModelSpeculation ||
          speculation is Eagle3Speculation ||
          speculation is DFlashSpeculation ||
          speculation is MtpSpeculation)) {
    throw const UnsupportedFeatureException(
      'Model-backed speculative decoding cannot use an mmproj context.',
    );
  }
  if (speculation is! NGramSpeculation ||
      const <String>{
        'ngram-simple',
        'ngram-map-k',
        'ngram-map-k4v',
      }.contains(speculation.strategy)) {
    return;
  }
  throw const UnsupportedFeatureException(
    'Supported n-gram strategies are ngram-simple, ngram-map-k, and '
    'ngram-map-k4v.',
  );
}

void _requireEmbeddingContextConfig(LlamaModelConfig config) {
  if (config.mmprojPath != null) {
    throw const UnsupportedFeatureException(
      'Embedding and reranking contexts cannot use mmprojPath.',
    );
  }
  if (config.speculativeDecoding is! NoSpeculativeDecoding) {
    throw const UnsupportedFeatureException(
      'Speculative decoding is only supported for text generation contexts.',
    );
  }
}

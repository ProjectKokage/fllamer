part of 'native_bridge.dart';

LlamaModelInfo _inspectModelInWorker(LlamaModelConfig config) {
  final bridge = NativeLlamaBridge.tryOpen(config.nativeLibraryPath);
  if (bridge == null) {
    throw _nativeBridgeUnavailable(config.nativeLibraryPath);
  }
  return bridge._inspectModel(config);
}

Map<String, String> _modelMetadataInWorker(LlamaModelConfig config) {
  final bridge = NativeLlamaBridge.tryOpen(config.nativeLibraryPath);
  if (bridge == null) {
    throw _nativeBridgeUnavailable(config.nativeLibraryPath);
  }
  return bridge._modelMetadata(config);
}

String _chatTemplateInWorker(LlamaModelConfig config) {
  final bridge = NativeLlamaBridge.tryOpen(config.nativeLibraryPath);
  if (bridge == null) {
    throw _nativeBridgeUnavailable(config.nativeLibraryPath);
  }
  return bridge._chatTemplateModel(config);
}

List<int> _tokenizeInWorker(
  LlamaModelConfig config,
  String text, {
  required bool addSpecial,
  required bool parseSpecial,
}) {
  final bridge = NativeLlamaBridge.tryOpen(config.nativeLibraryPath);
  if (bridge == null) {
    throw _nativeBridgeUnavailable(config.nativeLibraryPath);
  }
  return bridge._tokenizeModel(
    config,
    text,
    addSpecial: addSpecial,
    parseSpecial: parseSpecial,
  );
}

String _detokenizeInWorker(
  LlamaModelConfig config,
  List<int> tokens, {
  required bool removeSpecial,
  required bool unparseSpecial,
}) {
  final bridge = NativeLlamaBridge.tryOpen(config.nativeLibraryPath);
  if (bridge == null) {
    throw _nativeBridgeUnavailable(config.nativeLibraryPath);
  }
  return bridge._detokenizeModel(
    config,
    tokens,
    removeSpecial: removeSpecial,
    unparseSpecial: unparseSpecial,
  );
}

Float32List _embedTextInWorker(
  LlamaModelConfig config,
  String text,
  EmbeddingConfig embeddingConfig,
) {
  final bridge = NativeLlamaBridge.tryOpen(config.nativeLibraryPath);
  if (bridge == null) {
    throw _nativeBridgeUnavailable(config.nativeLibraryPath);
  }
  final handles = bridge._openEngine(
    config,
    embeddings: true,
    pooling: embeddingConfig.pooling,
  );
  try {
    final embedding = handles.embedText(text, embeddingConfig);
    return embeddingConfig.normalize ? _normalize(embedding) : embedding;
  } finally {
    handles.close();
  }
}

EmbeddingBatch _embedTextsInWorker(
  LlamaModelConfig config,
  List<String> texts,
  EmbeddingConfig embeddingConfig,
) {
  final bridge = NativeLlamaBridge.tryOpen(config.nativeLibraryPath);
  if (bridge == null) {
    throw _nativeBridgeUnavailable(config.nativeLibraryPath);
  }
  final handles = bridge._openEngine(
    config,
    embeddings: true,
    pooling: embeddingConfig.pooling,
  );
  try {
    return _embedTextsWithHandles(handles, texts, embeddingConfig);
  } finally {
    handles.close();
  }
}

String _formatChatInWorker(
  LlamaModelConfig config,
  List<ChatMessage> messages, {
  required bool addAssistantPrompt,
  required LlamaToolCallingConfig toolCalling,
  bool? enableThinking,
  int? reasoningBudgetTokens,
  int? maximumPromptBytes,
}) {
  final bridge = NativeLlamaBridge.tryOpen(config.nativeLibraryPath);
  if (bridge == null) {
    throw _nativeBridgeUnavailable(config.nativeLibraryPath);
  }
  return bridge._formatChatModel(
    config,
    messages,
    addAssistantPrompt: addAssistantPrompt,
    toolCalling: toolCalling,
    enableThinking: enableThinking,
    reasoningBudgetTokens: reasoningBudgetTokens,
    maximumPromptBytes: maximumPromptBytes,
  );
}

int _countChatTokensInWorker(
  LlamaModelConfig config,
  List<ChatMessage> messages, {
  required bool addAssistantPrompt,
  required LlamaToolCallingConfig toolCalling,
  bool? enableThinking,
  int? reasoningBudgetTokens,
  int? maximumPromptBytes,
}) {
  final bridge = NativeLlamaBridge.tryOpen(config.nativeLibraryPath);
  if (bridge == null) {
    throw _nativeBridgeUnavailable(config.nativeLibraryPath);
  }
  return bridge._countChatTokensModel(
    config,
    messages,
    addAssistantPrompt: addAssistantPrompt,
    toolCalling: toolCalling,
    enableThinking: enableThinking,
    reasoningBudgetTokens: reasoningBudgetTokens,
    maximumPromptBytes: maximumPromptBytes,
  );
}

LlamaChatTemplateCapabilities _chatTemplateCapabilitiesInWorker(
  LlamaModelConfig config,
) {
  final bridge = NativeLlamaBridge.tryOpen(config.nativeLibraryPath);
  if (bridge == null) {
    throw _nativeBridgeUnavailable(config.nativeLibraryPath);
  }
  return bridge._chatTemplateCapabilitiesModel(config);
}

List<double> _rerankDocumentsInWorker(
  LlamaModelConfig config,
  String query,
  List<String> documents,
  RerankingConfig rerankingConfig,
) {
  final bridge = NativeLlamaBridge.tryOpen(config.nativeLibraryPath);
  if (bridge == null) {
    throw _nativeBridgeUnavailable(config.nativeLibraryPath);
  }
  final handles = bridge._openEngine(
    config,
    embeddings: true,
    pooling: EmbeddingPooling.rank,
  );
  try {
    return <double>[
      for (final document in documents)
        handles.rerank(query, document, rerankingConfig),
    ];
  } finally {
    handles.close();
  }
}

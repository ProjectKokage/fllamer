import 'dart:typed_data';

import 'config.dart';
import 'model_info.dart';

import 'native_bridge.dart';
import 'native_chat.dart';
import 'native_engine_handles.dart';
import 'native_model_ops.dart';

LlamaModelInfo inspectModelInWorker(LlamaModelConfig config) {
  final bridge = NativeLlamaBridge.tryOpen(config.nativeLibraryPath);
  if (bridge == null) {
    throw nativeBridgeUnavailable(config.nativeLibraryPath);
  }
  return bridge.loadAndInspectModel(config);
}

Map<String, String> modelMetadataInWorker(LlamaModelConfig config) {
  final bridge = NativeLlamaBridge.tryOpen(config.nativeLibraryPath);
  if (bridge == null) {
    throw nativeBridgeUnavailable(config.nativeLibraryPath);
  }
  return bridge.loadAndReadModelMetadata(config);
}

String chatTemplateInWorker(LlamaModelConfig config) {
  final bridge = NativeLlamaBridge.tryOpen(config.nativeLibraryPath);
  if (bridge == null) {
    throw nativeBridgeUnavailable(config.nativeLibraryPath);
  }
  return bridge.loadAndReadChatTemplate(config);
}

List<int> tokenizeInWorker(
  LlamaModelConfig config,
  String text, {
  required bool addSpecial,
  required bool parseSpecial,
}) {
  final bridge = NativeLlamaBridge.tryOpen(config.nativeLibraryPath);
  if (bridge == null) {
    throw nativeBridgeUnavailable(config.nativeLibraryPath);
  }
  return bridge.loadAndTokenize(
    config,
    text,
    addSpecial: addSpecial,
    parseSpecial: parseSpecial,
  );
}

String detokenizeInWorker(
  LlamaModelConfig config,
  List<int> tokens, {
  required bool removeSpecial,
  required bool unparseSpecial,
}) {
  final bridge = NativeLlamaBridge.tryOpen(config.nativeLibraryPath);
  if (bridge == null) {
    throw nativeBridgeUnavailable(config.nativeLibraryPath);
  }
  return bridge.loadAndDetokenize(
    config,
    tokens,
    removeSpecial: removeSpecial,
    unparseSpecial: unparseSpecial,
  );
}

Float32List embedTextInWorker(
  LlamaModelConfig config,
  String text,
  EmbeddingConfig embeddingConfig,
) {
  final bridge = NativeLlamaBridge.tryOpen(config.nativeLibraryPath);
  if (bridge == null) {
    throw nativeBridgeUnavailable(config.nativeLibraryPath);
  }
  final handles = bridge.openEngine(
    config,
    embeddings: true,
    pooling: embeddingConfig.pooling,
  );
  try {
    final embedding = handles.embedText(text, embeddingConfig);
    return embeddingConfig.normalize ? normalize(embedding) : embedding;
  } finally {
    handles.close();
  }
}

EmbeddingBatch embedTextsInWorker(
  LlamaModelConfig config,
  List<String> texts,
  EmbeddingConfig embeddingConfig,
) {
  final bridge = NativeLlamaBridge.tryOpen(config.nativeLibraryPath);
  if (bridge == null) {
    throw nativeBridgeUnavailable(config.nativeLibraryPath);
  }
  final handles = bridge.openEngine(
    config,
    embeddings: true,
    pooling: embeddingConfig.pooling,
  );
  try {
    return embedTextsWithHandles(handles, texts, embeddingConfig);
  } finally {
    handles.close();
  }
}

String formatChatInWorker(
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
    throw nativeBridgeUnavailable(config.nativeLibraryPath);
  }
  return bridge.loadAndFormatChat(
    config,
    messages,
    addAssistantPrompt: addAssistantPrompt,
    toolCalling: toolCalling,
    enableThinking: enableThinking,
    reasoningBudgetTokens: reasoningBudgetTokens,
    maximumPromptBytes: maximumPromptBytes,
  );
}

int countChatTokensInWorker(
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
    throw nativeBridgeUnavailable(config.nativeLibraryPath);
  }
  return bridge.loadAndCountChatTokens(
    config,
    messages,
    addAssistantPrompt: addAssistantPrompt,
    toolCalling: toolCalling,
    enableThinking: enableThinking,
    reasoningBudgetTokens: reasoningBudgetTokens,
    maximumPromptBytes: maximumPromptBytes,
  );
}

LlamaChatTemplateCapabilities chatTemplateCapabilitiesInWorker(
  LlamaModelConfig config,
) {
  final bridge = NativeLlamaBridge.tryOpen(config.nativeLibraryPath);
  if (bridge == null) {
    throw nativeBridgeUnavailable(config.nativeLibraryPath);
  }
  return bridge.loadAndReadChatTemplateCapabilities(config);
}

List<double> rerankDocumentsInWorker(
  LlamaModelConfig config,
  String query,
  List<String> documents,
  RerankingConfig rerankingConfig,
) {
  final bridge = NativeLlamaBridge.tryOpen(config.nativeLibraryPath);
  if (bridge == null) {
    throw nativeBridgeUnavailable(config.nativeLibraryPath);
  }
  final handles = bridge.openEngine(
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

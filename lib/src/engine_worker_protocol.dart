part of 'native_bridge.dart';

final class _EngineWorkerStart {
  const _EngineWorkerStart(
    this.config,
    this.reply, {
    required this.embeddings,
    required this.pooling,
  });

  final LlamaModelConfig config;
  final SendPort reply;
  final bool embeddings;
  final EmbeddingPooling pooling;
}

final class _EngineWorkerReady {
  const _EngineWorkerReady(this.commands, this.contextAddress);

  final SendPort commands;
  final int contextAddress;
}

final class _EngineWorkerClose {
  const _EngineWorkerClose(this.reply);

  final SendPort reply;
}

final class _EngineWorkerFinalize {
  const _EngineWorkerFinalize();
}

/// A request the worker runs against the open engine and answers once.
abstract base class _EngineRequest<T> {
  const _EngineRequest();

  T run(_NativeEngineHandles handles);
}

/// An [_EngineRequest] with the port its answer goes to.
final class _EngineWorkerRequest {
  const _EngineWorkerRequest(this.request, this.reply);

  final _EngineRequest<Object?> request;
  final SendPort reply;
}

final class _ResetRequest extends _EngineRequest<Null> {
  const _ResetRequest();

  @override
  Null run(_NativeEngineHandles handles) {
    handles.reset();
    return null;
  }
}

final class _WarmUpRequest extends _EngineRequest<Null> {
  const _WarmUpRequest();

  @override
  Null run(_NativeEngineHandles handles) {
    handles.warmUp();
    return null;
  }
}

final class _ModelInfoRequest extends _EngineRequest<LlamaModelInfo> {
  const _ModelInfoRequest();

  @override
  LlamaModelInfo run(_NativeEngineHandles handles) {
    return handles.modelInfo();
  }
}

final class _ModelMetadataRequest extends _EngineRequest<Map<String, String>> {
  const _ModelMetadataRequest();

  @override
  Map<String, String> run(_NativeEngineHandles handles) {
    return handles.modelMetadata();
  }
}

final class _ChatTemplateRequest extends _EngineRequest<String> {
  const _ChatTemplateRequest();

  @override
  String run(_NativeEngineHandles handles) {
    return handles.chatTemplate();
  }
}

final class _TokenizeRequest extends _EngineRequest<List<int>> {
  const _TokenizeRequest(this.text, this.addSpecial, this.parseSpecial);

  final String text;
  final bool addSpecial;
  final bool parseSpecial;

  @override
  List<int> run(_NativeEngineHandles handles) {
    return handles.tokenize(
      text,
      addSpecial: addSpecial,
      parseSpecial: parseSpecial,
    );
  }
}

final class _CountTokensRequest extends _EngineRequest<int> {
  const _CountTokensRequest(this.text, this.addSpecial, this.parseSpecial);

  final String text;
  final bool addSpecial;
  final bool parseSpecial;

  @override
  int run(_NativeEngineHandles handles) {
    return handles.countTokens(
      text,
      addSpecial: addSpecial,
      parseSpecial: parseSpecial,
    );
  }
}

final class _DetokenizeRequest extends _EngineRequest<String> {
  const _DetokenizeRequest(
    this.tokens,
    this.removeSpecial,
    this.unparseSpecial,
  );

  final List<int> tokens;
  final bool removeSpecial;
  final bool unparseSpecial;

  @override
  String run(_NativeEngineHandles handles) {
    return handles.detokenize(
      tokens,
      removeSpecial: removeSpecial,
      unparseSpecial: unparseSpecial,
    );
  }
}

final class _EmbedTextsRequest extends _EngineRequest<EmbeddingBatch> {
  const _EmbedTextsRequest(this.texts, this.config);

  final List<String> texts;
  final EmbeddingConfig config;

  @override
  EmbeddingBatch run(_NativeEngineHandles handles) {
    return _embedTextsWithHandles(handles, texts, config);
  }
}

final class _ChatTemplateCapabilitiesRequest
    extends _EngineRequest<LlamaChatTemplateCapabilities> {
  const _ChatTemplateCapabilitiesRequest();

  @override
  LlamaChatTemplateCapabilities run(_NativeEngineHandles handles) {
    return handles.chatTemplateCapabilities();
  }
}

final class _FormatChatRequest extends _EngineRequest<String> {
  const _FormatChatRequest(
    this.messages,
    this.addAssistantPrompt,
    this.toolCalling,
    this.enableThinking,
    this.reasoningBudgetTokens,
    this.maximumPromptBytes,
  );

  final List<ChatMessage> messages;
  final bool addAssistantPrompt;
  final LlamaToolCallingConfig toolCalling;
  final bool? enableThinking;
  final int? reasoningBudgetTokens;
  final int? maximumPromptBytes;

  @override
  String run(_NativeEngineHandles handles) {
    return handles.formatChat(
      messages,
      addAssistantPrompt: addAssistantPrompt,
      toolCalling: toolCalling,
      enableThinking: enableThinking,
      reasoningBudgetTokens: reasoningBudgetTokens,
      maximumPromptBytes: maximumPromptBytes,
    );
  }
}

final class _CountChatTokensRequest extends _EngineRequest<int> {
  const _CountChatTokensRequest(
    this.messages,
    this.addAssistantPrompt,
    this.toolCalling,
    this.enableThinking,
    this.reasoningBudgetTokens,
    this.maximumPromptBytes,
  );

  final List<ChatMessage> messages;
  final bool addAssistantPrompt;
  final LlamaToolCallingConfig toolCalling;
  final bool? enableThinking;
  final int? reasoningBudgetTokens;
  final int? maximumPromptBytes;

  @override
  int run(_NativeEngineHandles handles) {
    return handles.countChatTokens(
      messages,
      addAssistantPrompt: addAssistantPrompt,
      toolCalling: toolCalling,
      enableThinking: enableThinking,
      reasoningBudgetTokens: reasoningBudgetTokens,
      maximumPromptBytes: maximumPromptBytes,
    );
  }
}

final class _PrefillRequest extends _EngineRequest<PrefillTelemetry> {
  const _PrefillRequest(this.prompt, this.addSpecial, this.parseSpecial);

  final String prompt;
  final bool? addSpecial;
  final bool parseSpecial;

  @override
  PrefillTelemetry run(_NativeEngineHandles handles) {
    return handles.prefill(
      prompt,
      addSpecial: addSpecial,
      parseSpecial: parseSpecial,
    );
  }
}

final class _ShiftContextRequest extends _EngineRequest<int> {
  const _ShiftContextRequest(this.keepTokens, this.discardTokens);

  final int keepTokens;
  final int? discardTokens;

  @override
  int run(_NativeEngineHandles handles) {
    return handles.shiftContext(
      keepTokens: keepTokens,
      discardTokens: discardTokens,
    );
  }
}

final class _ContextInfoRequest extends _EngineRequest<LlamaContextInfo> {
  const _ContextInfoRequest();

  @override
  LlamaContextInfo run(_NativeEngineHandles handles) {
    return handles.contextInfo();
  }
}

final class _SaveStateRequest extends _EngineRequest<Uint8List> {
  const _SaveStateRequest();

  @override
  Uint8List run(_NativeEngineHandles handles) {
    return handles.saveState();
  }
}

final class _RestoreStateRequest extends _EngineRequest<Null> {
  const _RestoreStateRequest(this.state);

  final Uint8List state;

  @override
  Null run(_NativeEngineHandles handles) {
    handles.restoreState(state);
    return null;
  }
}

final class _EngineWorkerStreamComplete {
  const _EngineWorkerStreamComplete(
    this.id,
    this.messages,
    this.config,
    this.reusePromptPrefix,
    this.maximumPromptBytes,
    this.reply,
  );

  final int id;
  final List<ChatMessage> messages;
  final GenerationConfig config;
  final bool reusePromptPrefix;
  final int? maximumPromptBytes;
  final SendPort reply;
}

final class _EngineWorkerStreamPrompt {
  const _EngineWorkerStreamPrompt(
    this.id,
    this.prompt,
    this.config,
    this.reply,
  );

  final int id;
  final String prompt;
  final GenerationConfig config;
  final SendPort reply;
}

final class _EngineWorkerStreamNext {
  const _EngineWorkerStreamNext(this.id);

  final int id;
}

final class _EngineWorkerStreamDispose {
  const _EngineWorkerStreamDispose(this.id, this.reply);

  final int id;
  final SendPort reply;
}

final class _LoadLoraRequest extends _EngineRequest<LoraAdapterInfo> {
  const _LoadLoraRequest(this.config);

  final LoraAdapterConfig config;

  @override
  LoraAdapterInfo run(_NativeEngineHandles handles) {
    return handles.loadLora(config);
  }
}

final class _ListLorasRequest extends _EngineRequest<List<LoraAdapterInfo>> {
  const _ListLorasRequest();

  @override
  List<LoraAdapterInfo> run(_NativeEngineHandles handles) {
    return handles.loraAdapters();
  }
}

final class _SetLoraScaleRequest extends _EngineRequest<Null> {
  const _SetLoraScaleRequest(this.adapterId, this.scale);

  final int adapterId;
  final double scale;

  @override
  Null run(_NativeEngineHandles handles) {
    handles.setLoraScale(adapterId, scale);
    return null;
  }
}

final class _UnloadLoraRequest extends _EngineRequest<Null> {
  const _UnloadLoraRequest(this.adapterId);

  final int adapterId;

  @override
  Null run(_NativeEngineHandles handles) {
    handles.unloadLora(adapterId);
    return null;
  }
}

final class _EngineWorkerStreamProgress {
  const _EngineWorkerStreamProgress(this.generatedTokens);

  final int generatedTokens;
}

final class _EngineWorkerStreamChunk {
  const _EngineWorkerStreamChunk(
    this.text,
    this.isDone,
    this.telemetry,
    this.assistantMessage,
  );

  final String text;
  final bool isDone;
  final GenerationTelemetry? telemetry;
  final ChatMessage? assistantMessage;
}

SendPort? _engineWorkerReply(Object? message) {
  return switch (message) {
    _EngineWorkerRequest(:final reply) => reply,
    _EngineWorkerStreamComplete(:final reply) => reply,
    _EngineWorkerStreamPrompt(:final reply) => reply,
    _ => null,
  };
}

final class _EngineWorkerFailure {
  const _EngineWorkerFailure(this.error);

  final _NativeError error;
}

final class _NativeError {
  const _NativeError(this.type, this.message);

  factory _NativeError.from(Object error) {
    if (error is ModelLoadException) {
      return _NativeError('modelLoad', error.message);
    }
    if (error is ContextCreateException) {
      return _NativeError('contextCreate', error.message);
    }
    if (error is PromptBufferException) {
      return _NativeError('promptBuffer', error.message);
    }
    if (error is GenerationException) {
      return _NativeError('generation', error.message);
    }
    if (error is EmbeddingException) {
      return _NativeError('embedding', error.message);
    }
    if (error is RerankingException) {
      return _NativeError('reranking', error.message);
    }
    if (error is LoraException) {
      return _NativeError('lora', error.message);
    }
    if (error is UnsupportedFeatureException) {
      return _NativeError('unsupported', error.message);
    }
    if (error is NativeOutOfMemoryException) {
      return _NativeError('outOfMemory', error.message);
    }
    if (error is CancelledException) {
      return _NativeError('cancelled', error.message);
    }
    if (error is ResourceDisposedException) {
      return _NativeError('disposed', error.message);
    }
    if (error is NativeBridgeException) {
      return _NativeError('nativeBridge', error.message);
    }
    if (error is LlamaException) {
      return _NativeError('llama', error.message);
    }
    return _NativeError('unknown', error.toString());
  }

  final String type;
  final String message;

  Exception toException() {
    return switch (type) {
      'modelLoad' => ModelLoadException(message),
      'contextCreate' => ContextCreateException(message),
      'generation' => GenerationException(message),
      'promptBuffer' => PromptBufferException(message),
      'embedding' => EmbeddingException(message),
      'reranking' => RerankingException(message),
      'lora' => LoraException(message),
      'unsupported' => UnsupportedFeatureException(message),
      'outOfMemory' => NativeOutOfMemoryException(message),
      'cancelled' => CancelledException(message),
      'disposed' => ResourceDisposedException(message),
      'nativeBridge' => NativeBridgeException(message),
      'llama' => NativeBridgeException(message),
      _ => NativeBridgeException(message),
    };
  }
}

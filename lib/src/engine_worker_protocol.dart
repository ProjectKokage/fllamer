import 'dart:isolate';
import 'dart:typed_data';

import 'config.dart';
import 'errors.dart';
import 'model_info.dart';

import 'native_engine_handles.dart';

final class EngineWorkerStart {
  const EngineWorkerStart(
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

final class EngineWorkerReady {
  const EngineWorkerReady(this.commands, this.contextAddress);

  final SendPort commands;
  final int contextAddress;
}

final class EngineWorkerClose {
  const EngineWorkerClose(this.reply);

  final SendPort reply;
}

final class EngineWorkerFinalize {
  const EngineWorkerFinalize();
}

/// A request the worker runs against the open engine and answers once.
abstract base class EngineRequest<T> {
  const EngineRequest();

  T run(NativeEngineHandles handles);
}

/// An [EngineRequest] with the port its answer goes to.
final class EngineWorkerRequest {
  const EngineWorkerRequest(this.request, this.reply);

  final EngineRequest<Object?> request;
  final SendPort reply;
}

final class ResetRequest extends EngineRequest<Null> {
  const ResetRequest();

  @override
  Null run(NativeEngineHandles handles) {
    handles.reset();
    return null;
  }
}

final class WarmUpRequest extends EngineRequest<Null> {
  const WarmUpRequest();

  @override
  Null run(NativeEngineHandles handles) {
    handles.warmUp();
    return null;
  }
}

final class ModelInfoRequest extends EngineRequest<LlamaModelInfo> {
  const ModelInfoRequest();

  @override
  LlamaModelInfo run(NativeEngineHandles handles) {
    return handles.modelInfo();
  }
}

final class ModelMetadataRequest extends EngineRequest<Map<String, String>> {
  const ModelMetadataRequest();

  @override
  Map<String, String> run(NativeEngineHandles handles) {
    return handles.modelMetadata();
  }
}

final class ChatTemplateRequest extends EngineRequest<String> {
  const ChatTemplateRequest();

  @override
  String run(NativeEngineHandles handles) {
    return handles.chatTemplate();
  }
}

final class TokenizeRequest extends EngineRequest<List<int>> {
  const TokenizeRequest(this.text, this.addSpecial, this.parseSpecial);

  final String text;
  final bool addSpecial;
  final bool parseSpecial;

  @override
  List<int> run(NativeEngineHandles handles) {
    return handles.tokenize(
      text,
      addSpecial: addSpecial,
      parseSpecial: parseSpecial,
    );
  }
}

final class CountTokensRequest extends EngineRequest<int> {
  const CountTokensRequest(this.text, this.addSpecial, this.parseSpecial);

  final String text;
  final bool addSpecial;
  final bool parseSpecial;

  @override
  int run(NativeEngineHandles handles) {
    return handles.countTokens(
      text,
      addSpecial: addSpecial,
      parseSpecial: parseSpecial,
    );
  }
}

final class DetokenizeRequest extends EngineRequest<String> {
  const DetokenizeRequest(this.tokens, this.removeSpecial, this.unparseSpecial);

  final List<int> tokens;
  final bool removeSpecial;
  final bool unparseSpecial;

  @override
  String run(NativeEngineHandles handles) {
    return handles.detokenize(
      tokens,
      removeSpecial: removeSpecial,
      unparseSpecial: unparseSpecial,
    );
  }
}

final class EmbedTextsRequest extends EngineRequest<EmbeddingBatch> {
  const EmbedTextsRequest(this.texts, this.config);

  final List<String> texts;
  final EmbeddingConfig config;

  @override
  EmbeddingBatch run(NativeEngineHandles handles) {
    return embedTextsWithHandles(handles, texts, config);
  }
}

final class ChatTemplateCapabilitiesRequest
    extends EngineRequest<LlamaChatTemplateCapabilities> {
  const ChatTemplateCapabilitiesRequest();

  @override
  LlamaChatTemplateCapabilities run(NativeEngineHandles handles) {
    return handles.chatTemplateCapabilities();
  }
}

final class FormatChatRequest extends EngineRequest<String> {
  const FormatChatRequest(
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
  String run(NativeEngineHandles handles) {
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

final class CountChatTokensRequest extends EngineRequest<int> {
  const CountChatTokensRequest(
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
  int run(NativeEngineHandles handles) {
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

final class PrefillRequest extends EngineRequest<PrefillTelemetry> {
  const PrefillRequest(this.prompt, this.addSpecial, this.parseSpecial);

  final String prompt;
  final bool? addSpecial;
  final bool parseSpecial;

  @override
  PrefillTelemetry run(NativeEngineHandles handles) {
    return handles.prefill(
      prompt,
      addSpecial: addSpecial,
      parseSpecial: parseSpecial,
    );
  }
}

final class ShiftContextRequest extends EngineRequest<int> {
  const ShiftContextRequest(this.keepTokens, this.discardTokens);

  final int keepTokens;
  final int? discardTokens;

  @override
  int run(NativeEngineHandles handles) {
    return handles.shiftContext(
      keepTokens: keepTokens,
      discardTokens: discardTokens,
    );
  }
}

final class ContextInfoRequest extends EngineRequest<LlamaContextInfo> {
  const ContextInfoRequest();

  @override
  LlamaContextInfo run(NativeEngineHandles handles) {
    return handles.contextInfo();
  }
}

final class SaveStateRequest extends EngineRequest<Uint8List> {
  const SaveStateRequest();

  @override
  Uint8List run(NativeEngineHandles handles) {
    return handles.saveState();
  }
}

final class RestoreStateRequest extends EngineRequest<Null> {
  const RestoreStateRequest(this.state);

  final Uint8List state;

  @override
  Null run(NativeEngineHandles handles) {
    handles.restoreState(state);
    return null;
  }
}

final class EngineWorkerStreamComplete {
  const EngineWorkerStreamComplete(
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

final class EngineWorkerStreamPrompt {
  const EngineWorkerStreamPrompt(this.id, this.prompt, this.config, this.reply);

  final int id;
  final String prompt;
  final GenerationConfig config;
  final SendPort reply;
}

final class EngineWorkerStreamNext {
  const EngineWorkerStreamNext(this.id);

  final int id;
}

final class EngineWorkerStreamDispose {
  const EngineWorkerStreamDispose(this.id, this.reply);

  final int id;
  final SendPort reply;
}

final class LoadLoraRequest extends EngineRequest<LoraAdapterInfo> {
  const LoadLoraRequest(this.config);

  final LoraAdapterConfig config;

  @override
  LoraAdapterInfo run(NativeEngineHandles handles) {
    return handles.loadLora(config);
  }
}

final class ListLorasRequest extends EngineRequest<List<LoraAdapterInfo>> {
  const ListLorasRequest();

  @override
  List<LoraAdapterInfo> run(NativeEngineHandles handles) {
    return handles.loraAdapters();
  }
}

final class SetLoraScaleRequest extends EngineRequest<Null> {
  const SetLoraScaleRequest(this.adapterId, this.scale);

  final int adapterId;
  final double scale;

  @override
  Null run(NativeEngineHandles handles) {
    handles.setLoraScale(adapterId, scale);
    return null;
  }
}

final class UnloadLoraRequest extends EngineRequest<Null> {
  const UnloadLoraRequest(this.adapterId);

  final int adapterId;

  @override
  Null run(NativeEngineHandles handles) {
    handles.unloadLora(adapterId);
    return null;
  }
}

final class EngineWorkerStreamProgress {
  const EngineWorkerStreamProgress(this.generatedTokens);

  final int generatedTokens;
}

final class EngineWorkerStreamChunk {
  const EngineWorkerStreamChunk(
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

SendPort? engineWorkerReply(Object? message) {
  return switch (message) {
    EngineWorkerRequest(:final reply) => reply,
    EngineWorkerStreamComplete(:final reply) => reply,
    EngineWorkerStreamPrompt(:final reply) => reply,
    _ => null,
  };
}

final class EngineWorkerFailure {
  const EngineWorkerFailure(this.error);

  final NativeError error;
}

final class NativeError {
  const NativeError(this.type, this.message);

  factory NativeError.from(Object error) {
    if (error is ModelLoadException) {
      return NativeError('modelLoad', error.message);
    }
    if (error is ContextCreateException) {
      return NativeError('contextCreate', error.message);
    }
    if (error is PromptBufferException) {
      return NativeError('promptBuffer', error.message);
    }
    if (error is GenerationException) {
      return NativeError('generation', error.message);
    }
    if (error is EmbeddingException) {
      return NativeError('embedding', error.message);
    }
    if (error is RerankingException) {
      return NativeError('reranking', error.message);
    }
    if (error is LoraException) {
      return NativeError('lora', error.message);
    }
    if (error is UnsupportedFeatureException) {
      return NativeError('unsupported', error.message);
    }
    if (error is NativeOutOfMemoryException) {
      return NativeError('outOfMemory', error.message);
    }
    if (error is CancelledException) {
      return NativeError('cancelled', error.message);
    }
    if (error is ResourceDisposedException) {
      return NativeError('disposed', error.message);
    }
    if (error is NativeBridgeException) {
      return NativeError('nativeBridge', error.message);
    }
    if (error is LlamaException) {
      return NativeError('llama', error.message);
    }
    return NativeError('unknown', error.toString());
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

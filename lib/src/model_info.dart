import 'dart:typed_data';

import 'config.dart';

final class LlamaRuntimeCapabilities {
  const LlamaRuntimeCapabilities({
    required this.nativeBridgeAvailable,
    required this.bridgeAbiVersion,
    required this.modelLoading,
    required this.tokenization,
    required this.textGeneration,
    required this.structuredOutput,
    required this.embeddings,
    required this.reranking,
    required this.rag,
    required this.multimodal,
    required this.lora,
    required this.speculativeDecoding,
    required this.mtp,
    required this.metal,
    required this.vulkan,
    this.toolCalling = false,
    this.nativeLogging = false,
    this.prefill = false,
    this.upstreamCommit,
    this.nativeBuildFlags,
  });

  final bool nativeBridgeAvailable;
  final int bridgeAbiVersion;
  final bool modelLoading;
  final bool tokenization;
  final bool textGeneration;
  final bool structuredOutput;
  final bool embeddings;
  final bool reranking;
  final bool rag;
  final bool multimodal;
  final bool lora;
  final bool speculativeDecoding;
  final bool mtp;
  final bool metal;
  final bool vulkan;
  final bool toolCalling;
  final bool nativeLogging;
  final bool prefill;
  final String? upstreamCommit;
  final String? nativeBuildFlags;
}

enum LlamaLogLevel { disabled, debug, info, warning, error }

final class LlamaLogRecord {
  const LlamaLogRecord({required this.level, required this.message});

  final LlamaLogLevel level;
  final String message;
}

final class LlamaChatTemplateCapabilities {
  const LlamaChatTemplateCapabilities({
    required this.supportsTools,
    required this.supportsToolCalls,
    required this.supportsParallelToolCalls,
  });

  final bool supportsTools;
  final bool supportsToolCalls;
  final bool supportsParallelToolCalls;
}

final class LlamaModelInfo {
  const LlamaModelInfo({
    required this.description,
    this.chatTemplate,
    required this.vocabType,
    required this.vocabSize,
    required this.trainingContextSize,
    required this.embeddingSize,
    required this.inputEmbeddingSize,
    required this.outputEmbeddingSize,
    required this.layerCount,
    required this.nextnLayerCount,
    required this.attentionHeadCount,
    required this.keyValueHeadCount,
    required this.fileType,
    this.fileTypeName,
    required this.sizeBytes,
    required this.parameterCount,
    required this.bosToken,
    required this.eosToken,
    required this.eotToken,
    required this.separatorToken,
    required this.newlineToken,
    required this.paddingToken,
    required this.maskToken,
    required this.addBosToken,
    required this.addEosToken,
    required this.addSeparatorToken,
    required this.hasEncoder,
    required this.hasDecoder,
    required this.isRecurrent,
    required this.isHybrid,
    required this.isDiffusion,
  });

  final String description;

  /// Effective caller-supplied or GGUF-provided chat template, when valid.
  final String? chatTemplate;
  final int vocabType;
  final int vocabSize;
  final int trainingContextSize;
  final int embeddingSize;
  final int inputEmbeddingSize;
  final int outputEmbeddingSize;
  final int layerCount;
  final int nextnLayerCount;
  final int attentionHeadCount;
  final int keyValueHeadCount;
  final int fileType;
  final String? fileTypeName;
  final int sizeBytes;
  final int parameterCount;
  final int bosToken;
  final int eosToken;
  final int eotToken;
  final int separatorToken;
  final int newlineToken;
  final int paddingToken;
  final int maskToken;
  final bool addBosToken;
  final bool addEosToken;
  final bool addSeparatorToken;
  final bool hasEncoder;
  final bool hasDecoder;
  final bool isRecurrent;
  final bool isHybrid;
  final bool isDiffusion;

  bool get hasMtpLayers => nextnLayerCount > 0;
}

final class LlamaModelFileInfo {
  const LlamaModelFileInfo({
    required this.path,
    required this.sizeBytes,
    this.sha256,
  });

  final String path;
  final int sizeBytes;
  final String? sha256;
}

/// A row-major batch of embeddings backed by one flat typed buffer.
///
/// Iteration and [vectorAt] return zero-copy, unmodifiable views. Call
/// [Float32List.fromList] on a view only when an independently owned vector is
/// required. [normalized] and [pooling] describe the request configuration
/// that produced the batch. The constructor takes ownership of [values]; code
/// constructing a batch directly must not mutate another retained view of that
/// buffer.
final class EmbeddingBatch extends Iterable<Float32List> {
  EmbeddingBatch({
    required this.count,
    required this.dimensions,
    required Float32List values,
    this.normalized = true,
    this.pooling = EmbeddingPooling.model,
  }) : values = values.asUnmodifiableView() {
    if (count < 0) {
      throw ArgumentError.value(count, 'count', 'must be non-negative');
    }
    if (dimensions < 0 || count > 0 && dimensions == 0) {
      throw ArgumentError.value(
        dimensions,
        'dimensions',
        count == 0 ? 'must be non-negative' : 'must be positive',
      );
    }
    if (values.length != count * dimensions) {
      throw ArgumentError.value(
        values.length,
        'values',
        'must equal count * dimensions (${count * dimensions})',
      );
    }
  }

  factory EmbeddingBatch.empty({
    bool normalized = true,
    EmbeddingPooling pooling = EmbeddingPooling.model,
  }) => EmbeddingBatch(
    count: 0,
    dimensions: 0,
    values: Float32List(0),
    normalized: normalized,
    pooling: pooling,
  );

  final int count;
  final int dimensions;
  final Float32List values;
  final bool normalized;
  final EmbeddingPooling pooling;

  @override
  int get length => count;

  Float32List vectorAt(int index) {
    if (index < 0 || index >= count) {
      throw RangeError.index(index, this, 'index', null, count);
    }
    final start = index * dimensions;
    return Float32List.sublistView(
      values,
      start,
      start + dimensions,
    ).asUnmodifiableView();
  }

  Float32List operator [](int index) => vectorAt(index);

  @override
  Iterator<Float32List> get iterator =>
      Iterable<Float32List>.generate(count, vectorAt).iterator;
}

final class LlamaContextInfo {
  const LlamaContextInfo({
    required this.contextSize,
    required this.sequenceContextSize,
    required this.batchSize,
    required this.ubatchSize,
    required this.maxSequences,
    this.supportsVision = false,
    this.supportsAudio = false,
    this.gpuBackend = GpuBackend.cpu,
    this.supportsContextShift = false,
    this.usedTokens = 0,
    this.kvCacheKeyType = KvCacheType.f16,
    this.kvCacheValueType = KvCacheType.f16,
    this.flashAttention = FlashAttentionMode.auto,
    this.kvCacheOffload = true,
    this.swaFull = true,
    this.kvUnified = false,
  });

  final int contextSize;
  final int sequenceContextSize;
  final int batchSize;
  final int ubatchSize;
  final int maxSequences;
  final bool supportsVision;
  final bool supportsAudio;
  final GpuBackend gpuBackend;
  final bool supportsContextShift;
  final int usedTokens;
  final KvCacheType kvCacheKeyType;
  final KvCacheType kvCacheValueType;
  final FlashAttentionMode flashAttention;
  final bool kvCacheOffload;
  final bool swaFull;
  final bool kvUnified;
}

final class PrefillTelemetry {
  const PrefillTelemetry({
    required this.promptTokens,
    required this.promptEvalMs,
    required this.totalMs,
  });

  final int promptTokens;
  final double promptEvalMs;
  final double totalMs;

  double get promptEvalTokensPerSecond =>
      _tokensPerSecond(promptTokens, promptEvalMs);
}

enum GenerationStopReason {
  unknown,
  endOfGeneration,
  stopSequence,
  stopToken,
  maxTokens,
}

final class GenerationTelemetry {
  const GenerationTelemetry({
    required this.promptTokens,
    required this.generatedTokens,
    required this.promptEvalMs,
    required this.decodeMs,
    required this.totalMs,
    this.timeToFirstTokenMs = 0.0,
    this.speculativeDraftTokens = 0,
    this.speculativeAcceptedTokens = 0,
    this.speculativeDraftMs = 0.0,
    this.speculativeVerifyMs = 0.0,
    this.stopReason = GenerationStopReason.unknown,
  });

  final int promptTokens;
  final int generatedTokens;
  final double promptEvalMs;
  final double decodeMs;
  final double totalMs;
  final double timeToFirstTokenMs;
  final int speculativeDraftTokens;
  final int speculativeAcceptedTokens;
  final double speculativeDraftMs;
  final double speculativeVerifyMs;
  final GenerationStopReason stopReason;

  double get promptEvalTokensPerSecond =>
      _tokensPerSecond(promptTokens, promptEvalMs);

  double get decodeTokensPerSecond =>
      _tokensPerSecond(generatedTokens, decodeMs);

  double get totalTokensPerSecond =>
      _tokensPerSecond(promptTokens + generatedTokens, totalMs);

  double get speculativeAcceptanceRate {
    if (speculativeDraftTokens <= 0 || speculativeAcceptedTokens <= 0) {
      return 0;
    }
    if (speculativeAcceptedTokens >= speculativeDraftTokens) {
      return 1;
    }
    return speculativeAcceptedTokens / speculativeDraftTokens;
  }
}

double _tokensPerSecond(int tokens, double milliseconds) {
  if (tokens <= 0 || !milliseconds.isFinite || milliseconds <= 0) {
    return 0;
  }
  return tokens * 1000 / milliseconds;
}

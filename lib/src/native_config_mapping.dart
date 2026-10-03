part of 'native_bridge.dart';

String? _readOptionalCString(ffi.Pointer<ffi.Char> pointer) {
  if (pointer == ffi.nullptr) {
    return null;
  }
  final value = pointer.cast<Utf8>().toDartString();
  return value.isEmpty ? null : value;
}

bool _hasFlag(int flags, int flag) => flags & flag != 0;

int _gpuLayers(LlamaModelConfig config) {
  return switch (config.gpu.backend) {
    GpuBackend.cpu => 0,
    GpuBackend.auto ||
    GpuBackend.metal ||
    GpuBackend.vulkan => config.gpu.layers ?? -1,
  };
}

int _gpuBackend(LlamaModelConfig config) {
  return switch (config.gpu.backend) {
    GpuBackend.auto => llama_dart_gpu_backend.LLAMA_DART_GPU_BACKEND_AUTO.value,
    GpuBackend.cpu => llama_dart_gpu_backend.LLAMA_DART_GPU_BACKEND_CPU.value,
    GpuBackend.metal =>
      llama_dart_gpu_backend.LLAMA_DART_GPU_BACKEND_METAL.value,
    GpuBackend.vulkan =>
      llama_dart_gpu_backend.LLAMA_DART_GPU_BACKEND_VULKAN.value,
  };
}

GpuBackend _gpuBackendFromNative(int value) {
  return switch (llama_dart_gpu_backend.fromValue(value)) {
    llama_dart_gpu_backend.LLAMA_DART_GPU_BACKEND_AUTO => GpuBackend.auto,
    llama_dart_gpu_backend.LLAMA_DART_GPU_BACKEND_CPU => GpuBackend.cpu,
    llama_dart_gpu_backend.LLAMA_DART_GPU_BACKEND_METAL => GpuBackend.metal,
    llama_dart_gpu_backend.LLAMA_DART_GPU_BACKEND_VULKAN => GpuBackend.vulkan,
  };
}

int _kvCacheType(KvCacheType type) {
  return switch (type) {
    KvCacheType.f32 => llama_dart_kv_cache_type.LLAMA_DART_KV_CACHE_F32.value,
    KvCacheType.f16 => llama_dart_kv_cache_type.LLAMA_DART_KV_CACHE_F16.value,
    KvCacheType.bf16 => llama_dart_kv_cache_type.LLAMA_DART_KV_CACHE_BF16.value,
    KvCacheType.q8Zero =>
      llama_dart_kv_cache_type.LLAMA_DART_KV_CACHE_Q8_0.value,
    KvCacheType.q4Zero =>
      llama_dart_kv_cache_type.LLAMA_DART_KV_CACHE_Q4_0.value,
    KvCacheType.q4One =>
      llama_dart_kv_cache_type.LLAMA_DART_KV_CACHE_Q4_1.value,
    KvCacheType.iq4Nl =>
      llama_dart_kv_cache_type.LLAMA_DART_KV_CACHE_IQ4_NL.value,
    KvCacheType.q5Zero =>
      llama_dart_kv_cache_type.LLAMA_DART_KV_CACHE_Q5_0.value,
    KvCacheType.q5One =>
      llama_dart_kv_cache_type.LLAMA_DART_KV_CACHE_Q5_1.value,
  };
}

KvCacheType _kvCacheTypeFromNative(int value) {
  return switch (llama_dart_kv_cache_type.fromValue(value)) {
    llama_dart_kv_cache_type.LLAMA_DART_KV_CACHE_DEFAULT ||
    llama_dart_kv_cache_type.LLAMA_DART_KV_CACHE_F16 => KvCacheType.f16,
    llama_dart_kv_cache_type.LLAMA_DART_KV_CACHE_F32 => KvCacheType.f32,
    llama_dart_kv_cache_type.LLAMA_DART_KV_CACHE_BF16 => KvCacheType.bf16,
    llama_dart_kv_cache_type.LLAMA_DART_KV_CACHE_Q8_0 => KvCacheType.q8Zero,
    llama_dart_kv_cache_type.LLAMA_DART_KV_CACHE_Q4_0 => KvCacheType.q4Zero,
    llama_dart_kv_cache_type.LLAMA_DART_KV_CACHE_Q4_1 => KvCacheType.q4One,
    llama_dart_kv_cache_type.LLAMA_DART_KV_CACHE_IQ4_NL => KvCacheType.iq4Nl,
    llama_dart_kv_cache_type.LLAMA_DART_KV_CACHE_Q5_0 => KvCacheType.q5Zero,
    llama_dart_kv_cache_type.LLAMA_DART_KV_CACHE_Q5_1 => KvCacheType.q5One,
  };
}

int _flashAttention(FlashAttentionMode mode) {
  return switch (mode) {
    FlashAttentionMode.auto =>
      llama_dart_flash_attention_mode.LLAMA_DART_FLASH_ATTENTION_AUTO.value,
    FlashAttentionMode.disabled =>
      llama_dart_flash_attention_mode.LLAMA_DART_FLASH_ATTENTION_DISABLED.value,
    FlashAttentionMode.enabled =>
      llama_dart_flash_attention_mode.LLAMA_DART_FLASH_ATTENTION_ENABLED.value,
  };
}

FlashAttentionMode _flashAttentionFromNative(int value) {
  return switch (llama_dart_flash_attention_mode.fromValue(value)) {
    llama_dart_flash_attention_mode.LLAMA_DART_FLASH_ATTENTION_AUTO =>
      FlashAttentionMode.auto,
    llama_dart_flash_attention_mode.LLAMA_DART_FLASH_ATTENTION_DISABLED =>
      FlashAttentionMode.disabled,
    llama_dart_flash_attention_mode.LLAMA_DART_FLASH_ATTENTION_ENABLED =>
      FlashAttentionMode.enabled,
  };
}

LlamaLogLevel _nativeLogLevelFromValue(int value) {
  return switch (llama_dart_log_level.fromValue(value)) {
    llama_dart_log_level.LLAMA_DART_LOG_DISABLED =>
      throw const NativeBridgeException(
        'Native log drain returned a disabled level for a message.',
      ),
    llama_dart_log_level.LLAMA_DART_LOG_DEBUG => LlamaLogLevel.debug,
    llama_dart_log_level.LLAMA_DART_LOG_INFO => LlamaLogLevel.info,
    llama_dart_log_level.LLAMA_DART_LOG_WARNING => LlamaLogLevel.warning,
    llama_dart_log_level.LLAMA_DART_LOG_ERROR => LlamaLogLevel.error,
  };
}

int _poolingType(EmbeddingPooling pooling) {
  return switch (pooling) {
    EmbeddingPooling.model => -1,
    EmbeddingPooling.mean => 1,
    EmbeddingPooling.cls => 2,
    EmbeddingPooling.last => 3,
    EmbeddingPooling.rank => 4,
  };
}

int _ngramN(LlamaModelConfig config) {
  final speculation = config.speculativeDecoding;
  if (speculation is NGramSpeculation) {
    return speculation.ngramSize;
  }
  if (speculation is NGramModSpeculation) {
    return speculation.matchLength;
  }
  return 0;
}

int _ngramM(LlamaModelConfig config) {
  final speculation = config.speculativeDecoding;
  if (speculation is NGramSpeculation) {
    return speculation.draftLength;
  }
  if (speculation is NGramModSpeculation) {
    return speculation.maximumDraftLength;
  }
  return 0;
}

int _ngramMinDraft(LlamaModelConfig config) {
  final speculation = config.speculativeDecoding;
  return speculation is NGramModSpeculation
      ? speculation.minimumDraftLength
      : 0;
}

String? _speculativeModelPath(LlamaModelConfig config) {
  return switch (config.speculativeDecoding) {
    DraftModelSpeculation(:final draftModelPath) => draftModelPath,
    Eagle3Speculation(:final draftModelPath) => draftModelPath,
    DFlashSpeculation(:final draftModelPath) => draftModelPath,
    MtpSpeculation(:final mtpModelPath) => mtpModelPath,
    NoSpeculativeDecoding() ||
    NGramSpeculation() ||
    NGramModSpeculation() ||
    NGramCacheSpeculation() => null,
  };
}

bool _loadsEmbeddedMtp(LlamaModelConfig config) {
  return switch (config.speculativeDecoding) {
    MtpSpeculation(mtpModelPath: null) => true,
    _ => false,
  };
}

int _speculativeType(LlamaModelConfig config) {
  return switch (config.speculativeDecoding) {
    NoSpeculativeDecoding() =>
      llama_dart_speculative_type.LLAMA_DART_SPECULATIVE_NONE.value,
    NGramSpeculation(strategy: 'ngram-simple') =>
      llama_dart_speculative_type.LLAMA_DART_SPECULATIVE_NGRAM_SIMPLE.value,
    NGramSpeculation(strategy: 'ngram-map-k') =>
      llama_dart_speculative_type.LLAMA_DART_SPECULATIVE_NGRAM_MAP_K.value,
    NGramSpeculation(strategy: 'ngram-map-k4v') =>
      llama_dart_speculative_type.LLAMA_DART_SPECULATIVE_NGRAM_MAP_K4V.value,
    NGramModSpeculation() =>
      llama_dart_speculative_type.LLAMA_DART_SPECULATIVE_NGRAM_MOD.value,
    NGramCacheSpeculation() =>
      llama_dart_speculative_type.LLAMA_DART_SPECULATIVE_NGRAM_CACHE.value,
    NGramSpeculation() => throw const UnsupportedFeatureException(
      'Supported n-gram strategies are ngram-simple, ngram-map-k, and '
      'ngram-map-k4v.',
    ),
    DraftModelSpeculation() =>
      llama_dart_speculative_type.LLAMA_DART_SPECULATIVE_DRAFT_MODEL.value,
    Eagle3Speculation() =>
      llama_dart_speculative_type.LLAMA_DART_SPECULATIVE_EAGLE3.value,
    DFlashSpeculation() =>
      llama_dart_speculative_type.LLAMA_DART_SPECULATIVE_DFLASH.value,
    MtpSpeculation() =>
      llama_dart_speculative_type.LLAMA_DART_SPECULATIVE_MTP.value,
  };
}

int _speculativeDraftMax(LlamaModelConfig config) {
  return switch (config.speculativeDecoding) {
    DraftModelSpeculation(:final draftLength) ||
    Eagle3Speculation(:final draftLength) ||
    DFlashSpeculation(:final draftLength) ||
    MtpSpeculation(:final draftLength) => draftLength,
    NoSpeculativeDecoding() ||
    NGramSpeculation() ||
    NGramModSpeculation() ||
    NGramCacheSpeculation() => 0,
  };
}

int _mirostatMode(MirostatMode? mode) {
  return switch (mode) {
    null => 0,
    MirostatMode.v1 => 1,
    MirostatMode.v2 => 2,
  };
}

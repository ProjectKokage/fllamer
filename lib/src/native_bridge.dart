import 'dart:async';
import 'dart:convert';
import 'dart:ffi' as ffi;
import 'dart:io';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import 'config.dart';
import 'errors.dart';
import 'ffi/generated_bindings.dart';
import 'ffi/native_asset_lookup.dart';
import 'input_validation.dart';
import 'model_info.dart';

import 'engine_session.dart';
import 'native_config_mapping.dart';
import 'native_one_shot.dart';

const _maxJsonSchemaGrammarBytes = 16 * 1024 * 1024;

final class NativeLlamaBridge {
  NativeLlamaBridge._(this.bindings);

  static const expectedAbiVersion = LLAMA_DART_ABI_VERSION;

  final LlamaDartBridgeBindings bindings;

  int nextToolCallId = 1;

  static NativeLlamaBridge? tryOpen(String? nativeLibraryPath) {
    final explicitPath = nativeLibraryPath;
    if (explicitPath != null) {
      validateSingleLineText(explicitPath, 'nativeLibraryPath');
    }
    if (explicitPath == null) {
      nativeLibraryPathFromEnvironment(Platform.environment);
    }

    try {
      final bridge = NativeLlamaBridge._(_openBindings(nativeLibraryPath));
      bridge._ensureAbiVersion();
      return bridge;
    } on ArgumentError {
      return null;
    } on OSError {
      return null;
    }
  }

  static Future<LlamaModelInfo> inspectModel(LlamaModelConfig config) {
    return Isolate.run(() => inspectModelInWorker(config));
  }

  static Future<Map<String, String>> modelMetadata(LlamaModelConfig config) {
    return Isolate.run(() => modelMetadataInWorker(config));
  }

  static Future<String> chatTemplate(LlamaModelConfig config) {
    return Isolate.run(() => chatTemplateInWorker(config));
  }

  static String jsonSchemaGrammar(Map<String, Object?> schema) {
    final bridge = tryOpen(null);
    if (bridge == null) {
      throw const NativeBridgeException(
        'The native bridge is unavailable for JSON Schema conversion.',
      );
    }
    return bridge._jsonSchemaGrammar(schema);
  }

  static Future<List<int>> tokenize(
    LlamaModelConfig config,
    String text, {
    required bool addSpecial,
    required bool parseSpecial,
  }) {
    return Isolate.run(
      () => tokenizeInWorker(
        config,
        text,
        addSpecial: addSpecial,
        parseSpecial: parseSpecial,
      ),
    );
  }

  static Future<String> detokenize(
    LlamaModelConfig config,
    List<int> tokens, {
    required bool removeSpecial,
    required bool unparseSpecial,
  }) {
    return Isolate.run(
      () => detokenizeInWorker(
        config,
        tokens,
        removeSpecial: removeSpecial,
        unparseSpecial: unparseSpecial,
      ),
    );
  }

  static Future<Float32List> embedText(
    LlamaModelConfig config,
    String text,
    EmbeddingConfig embeddingConfig,
  ) {
    return Isolate.run(() => embedTextInWorker(config, text, embeddingConfig));
  }

  static Future<EmbeddingBatch> embedTexts(
    LlamaModelConfig config,
    List<String> texts,
    EmbeddingConfig embeddingConfig,
  ) {
    return Isolate.run(
      () => embedTextsInWorker(config, texts, embeddingConfig),
    );
  }

  static Future<List<double>> rerankDocuments(
    LlamaModelConfig config,
    String query,
    List<String> documents,
    RerankingConfig rerankingConfig,
  ) {
    return Isolate.run(
      () => rerankDocumentsInWorker(config, query, documents, rerankingConfig),
    );
  }

  static Future<String> formatChat(
    LlamaModelConfig config,
    List<ChatMessage> messages, {
    required bool addAssistantPrompt,
    required LlamaToolCallingConfig toolCalling,
    bool? enableThinking,
    int? reasoningBudgetTokens,
    int? maximumPromptBytes,
  }) {
    return Isolate.run(
      () => formatChatInWorker(
        config,
        messages,
        addAssistantPrompt: addAssistantPrompt,
        toolCalling: toolCalling,
        enableThinking: enableThinking,
        reasoningBudgetTokens: reasoningBudgetTokens,
        maximumPromptBytes: maximumPromptBytes,
      ),
    );
  }

  static Future<int> countChatTokens(
    LlamaModelConfig config,
    List<ChatMessage> messages, {
    required bool addAssistantPrompt,
    required LlamaToolCallingConfig toolCalling,
    bool? enableThinking,
    int? reasoningBudgetTokens,
    int? maximumPromptBytes,
  }) {
    return Isolate.run(
      () => countChatTokensInWorker(
        config,
        messages,
        addAssistantPrompt: addAssistantPrompt,
        toolCalling: toolCalling,
        enableThinking: enableThinking,
        reasoningBudgetTokens: reasoningBudgetTokens,
        maximumPromptBytes: maximumPromptBytes,
      ),
    );
  }

  static Future<LlamaChatTemplateCapabilities> chatTemplateCapabilities(
    LlamaModelConfig config,
  ) {
    return Isolate.run(() => chatTemplateCapabilitiesInWorker(config));
  }

  static Future<NativeLlamaEngineSession> startEngine(
    LlamaModelConfig config,
  ) => spawnEngineSession(
    config,
    embeddings: false,
    pooling: EmbeddingPooling.model,
  );

  static Future<NativeLlamaEngineSession> startEmbeddingEngine(
    LlamaModelConfig config,
    EmbeddingPooling pooling,
  ) => spawnEngineSession(config, embeddings: true, pooling: pooling);

  LlamaRuntimeCapabilities currentCapabilities() {
    final capabilities = calloc<llama_dart_capabilities>();
    try {
      capabilities.ref.struct_size = ffi.sizeOf<llama_dart_capabilities>();
      check(bindings.llama_dart_get_capabilities(capabilities));
      final flags = capabilities.ref.flags;
      return LlamaRuntimeCapabilities(
        nativeBridgeAvailable: true,
        bridgeAbiVersion: capabilities.ref.abi_version,
        modelLoading: hasFlag(
          flags,
          llama_dart_capability_flags.LLAMA_DART_CAP_MODEL_LOADING.value,
        ),
        tokenization: hasFlag(
          flags,
          llama_dart_capability_flags.LLAMA_DART_CAP_TOKENIZATION.value,
        ),
        textGeneration: hasFlag(
          flags,
          llama_dart_capability_flags.LLAMA_DART_CAP_TEXT_GENERATION.value,
        ),
        structuredOutput: hasFlag(
          flags,
          llama_dart_capability_flags.LLAMA_DART_CAP_STRUCTURED_OUTPUT.value,
        ),
        embeddings: hasFlag(
          flags,
          llama_dart_capability_flags.LLAMA_DART_CAP_EMBEDDINGS.value,
        ),
        reranking: hasFlag(
          flags,
          llama_dart_capability_flags.LLAMA_DART_CAP_RERANKING.value,
        ),
        rag: true,
        multimodal: hasFlag(
          flags,
          llama_dart_capability_flags.LLAMA_DART_CAP_MULTIMODAL.value,
        ),
        lora: hasFlag(
          flags,
          llama_dart_capability_flags.LLAMA_DART_CAP_LORA.value,
        ),
        speculativeDecoding: hasFlag(
          flags,
          llama_dart_capability_flags.LLAMA_DART_CAP_SPECULATIVE_DECODING.value,
        ),
        mtp: hasFlag(
          flags,
          llama_dart_capability_flags.LLAMA_DART_CAP_MTP.value,
        ),
        metal: hasFlag(
          flags,
          llama_dart_capability_flags.LLAMA_DART_CAP_METAL.value,
        ),
        vulkan: hasFlag(
          flags,
          llama_dart_capability_flags.LLAMA_DART_CAP_VULKAN.value,
        ),
        toolCalling: hasFlag(
          flags,
          llama_dart_capability_flags.LLAMA_DART_CAP_TOOL_CALLING.value,
        ),
        nativeLogging: hasFlag(
          flags,
          llama_dart_capability_flags.LLAMA_DART_CAP_LOGGING.value,
        ),
        prefill: hasFlag(
          flags,
          llama_dart_capability_flags.LLAMA_DART_CAP_PREFILL.value,
        ),
        upstreamCommit: readOptionalCString(
          bindings.llama_dart_upstream_commit(),
        ),
        nativeBuildFlags: readOptionalCString(
          bindings.llama_dart_build_flags(),
        ),
      );
    } finally {
      calloc.free(capabilities);
    }
  }

  void configureLogging(LlamaLogLevel minimumLevel) {
    final nativeLevel = switch (minimumLevel) {
      LlamaLogLevel.disabled => llama_dart_log_level.LLAMA_DART_LOG_DISABLED,
      LlamaLogLevel.debug => llama_dart_log_level.LLAMA_DART_LOG_DEBUG,
      LlamaLogLevel.info => llama_dart_log_level.LLAMA_DART_LOG_INFO,
      LlamaLogLevel.warning => llama_dart_log_level.LLAMA_DART_LOG_WARNING,
      LlamaLogLevel.error => llama_dart_log_level.LLAMA_DART_LOG_ERROR,
    };
    check(bindings.llama_dart_log_set_level(nativeLevel.value));
  }

  List<LlamaLogRecord> drainLogs() {
    final level = calloc<ffi.Uint32>();
    final message = calloc<llama_dart_buffer>();
    final records = <LlamaLogRecord>[];
    try {
      while (true) {
        level.value = 0;
        message.ref
          ..data = ffi.nullptr
          ..size = 0;
        check(bindings.llama_dart_log_next(level, message));
        final data = message.ref.data;
        final size = message.ref.size;
        if (data == ffi.nullptr) {
          if (size != 0) {
            throw const NativeBridgeException(
              'Native log drain returned a null message with non-zero size.',
            );
          }
          break;
        }
        if (size == 0) {
          throw const NativeBridgeException(
            'Native log drain returned an empty allocated message.',
          );
        }
        records.add(
          LlamaLogRecord(
            level: nativeLogLevelFromValue(level.value),
            message: utf8.decode(data.asTypedList(size), allowMalformed: true),
          ),
        );
        bindings.llama_dart_buffer_free(data);
        message.ref.data = ffi.nullptr;
        message.ref.size = 0;
      }
      return List<LlamaLogRecord>.unmodifiable(records);
    } finally {
      bindings.llama_dart_buffer_free(message.ref.data);
      calloc.free(message);
      calloc.free(level);
    }
  }

  String get multimodalMarker {
    final marker = readOptionalCString(bindings.llama_dart_multimodal_marker());
    if (marker == null || marker.isEmpty) {
      throw const NativeBridgeException(
        'Native bridge returned an empty multimodal marker.',
      );
    }
    return marker;
  }

  void _ensureAbiVersion() {
    final actual = bindings.llama_dart_abi_version();
    if (actual != LLAMA_DART_ABI_VERSION) {
      throw UnsupportedFeatureException(
        'Native bridge ABI version $actual is incompatible with Dart bindings '
        'ABI version $LLAMA_DART_ABI_VERSION.',
      );
    }
  }

  String _jsonSchemaGrammar(Map<String, Object?> schema) {
    final bytes = utf8.encode(jsonEncode(schema));
    if (bytes.length > _maxJsonSchemaGrammarBytes) {
      throw ArgumentError.value(
        schema,
        'schema',
        'encoded JSON must not exceed 16 MiB',
      );
    }

    final input = calloc<ffi.Uint8>(bytes.length);
    final out = calloc<llama_dart_buffer>();
    try {
      input.asTypedList(bytes.length).setAll(0, bytes);
      final result = bindings.llama_dart_json_schema_to_grammar(
        input,
        bytes.length,
        out,
      );
      if (result == llama_dart_result.LLAMA_DART_ERROR_INVALID_ARGUMENT) {
        throw ArgumentError.value(schema, 'schema', _lastError());
      }
      check(result);
      final data = out.ref.data;
      final size = out.ref.size;
      if (data == ffi.nullptr || size == 0) {
        throw const NativeBridgeException(
          'Native JSON Schema conversion returned an empty grammar.',
        );
      }
      if (size > _maxJsonSchemaGrammarBytes) {
        throw const UnsupportedFeatureException(
          'Generated JSON Schema grammar exceeds the 16 MiB safety limit.',
        );
      }
      try {
        return utf8.decode(data.asTypedList(size));
      } on FormatException catch (error) {
        throw UnsupportedFeatureException(
          'Generated JSON Schema grammar is not valid UTF-8.',
          cause: error,
        );
      }
    } finally {
      bindings.llama_dart_buffer_free(out.ref.data);
      calloc.free(out);
      calloc.free(input);
    }
  }

  void check(llama_dart_result result) {
    if (result == llama_dart_result.LLAMA_DART_SUCCESS) {
      return;
    }
    final message = _lastError();
    switch (result) {
      case llama_dart_result.LLAMA_DART_SUCCESS:
        return;
      case llama_dart_result.LLAMA_DART_ERROR_MODEL_LOAD:
        throw ModelLoadException(message);
      case llama_dart_result.LLAMA_DART_ERROR_CONTEXT_CREATE:
        throw ContextCreateException(message);
      case llama_dart_result.LLAMA_DART_ERROR_GENERATION:
        throw GenerationException(message);
      case llama_dart_result.LLAMA_DART_ERROR_EMBEDDING:
        throw EmbeddingException(message);
      case llama_dart_result.LLAMA_DART_ERROR_RERANKING:
        throw RerankingException(message);
      case llama_dart_result.LLAMA_DART_ERROR_LORA:
        throw LoraException(message);
      case llama_dart_result.LLAMA_DART_ERROR_CANCELLED:
        throw CancelledException(message);
      case llama_dart_result.LLAMA_DART_ERROR_UNSUPPORTED:
        throw UnsupportedFeatureException(message);
      case llama_dart_result.LLAMA_DART_ERROR_INTERNAL:
        if (message == 'native allocation failed') {
          throw NativeOutOfMemoryException(message);
        }
        throw NativeBridgeException(message);
      case llama_dart_result.LLAMA_DART_ERROR_INVALID_ARGUMENT:
      case llama_dart_result.LLAMA_DART_ERROR_BUFFER_TOO_SMALL:
        throw NativeBridgeException(message);
    }
  }

  String _lastError() {
    final pointer = bindings.llama_dart_last_error_message();
    if (pointer == ffi.nullptr) {
      return 'Native bridge failed without an error message.';
    }
    final message = _decodeNativeLastErrorMessage(pointer);
    if (message.isEmpty) {
      return 'Native bridge failed without an error message.';
    }
    return message;
  }

  void throwIfLastError(String operation) {
    final pointer = bindings.llama_dart_last_error_message();
    if (pointer == ffi.nullptr) {
      throw NativeBridgeException(
        '$operation failed without an error message.',
      );
    }
    final message = _decodeNativeLastErrorMessage(pointer);
    if (message.isNotEmpty) {
      throw NativeBridgeException('$operation failed: $message');
    }
  }

  void cancelContextAddress(int contextAddress) {
    if (contextAddress == 0) {
      return;
    }
    check(
      bindings.llama_dart_context_cancel(
        ffi.Pointer<llama_dart_context>.fromAddress(contextAddress),
      ),
    );
  }

  static LlamaDartBridgeBindings _openBindings(String? nativeLibraryPath) {
    final explicitPath = nativeLibraryPath;
    if (explicitPath != null && explicitPath.trim().isNotEmpty) {
      return LlamaDartBridgeBindings(ffi.DynamicLibrary.open(explicitPath));
    }

    final envPath = nativeLibraryPathFromEnvironment(Platform.environment);
    if (envPath != null) {
      return LlamaDartBridgeBindings(ffi.DynamicLibrary.open(envPath));
    }
    return LlamaDartBridgeBindings.fromLookup(lookupLlamaDartNativeAssetSymbol);
  }
}

UnsupportedFeatureException nativeBridgeUnavailable(String? path) {
  final location = path ?? 'the default platform library path';
  return UnsupportedFeatureException(
    'Native bridge library was not found or is incompatible: $location',
  );
}

String? nativeLibraryPathFromEnvironment(Map<String, String> environment) {
  final path = environment['FLLAMER_NATIVE_LIBRARY'];
  if (path == null || path.trim().isEmpty) {
    return null;
  }
  validateSingleLineText(path, 'FLLAMER_NATIVE_LIBRARY');
  return path;
}

String _decodeNativeLastErrorMessage(ffi.Pointer<ffi.Char> pointer) {
  const capacity = 4096;
  final bytes = pointer.cast<ffi.Uint8>();
  var length = 0;
  while (length < capacity && bytes[length] != 0) {
    length += 1;
  }
  return utf8.decode(bytes.asTypedList(length), allowMalformed: true);
}

import 'dart:async';
import 'dart:convert';
import 'dart:ffi' as ffi;
import 'dart:io';
import 'dart:isolate';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import 'config.dart';
import 'errors.dart';
import 'ffi/generated_bindings.dart';
import 'ffi/native_asset_lookup.dart';
import 'model_info.dart';

const _addSpecialNever = 0;
const _addSpecialAlways = 1;
const _addSpecialIfContextEmpty = 2;
const _maxModelDescriptionBytes = 1024 * 1024;
const _maxChatTemplateBytes = 16 * 1024 * 1024;
const _maxModelMetadataEntries = 65536;
const _maxModelMetadataKeyBytes = 4096;
const _maxModelMetadataValueBytes = 16 * 1024 * 1024;
const _maxModelMetadataTotalBytes = 64 * 1024 * 1024;

final class NativeLlamaBridge {
  NativeLlamaBridge._(this._bindings);

  static const expectedAbiVersion = LLAMA_DART_ABI_VERSION;

  final LlamaDartBridgeBindings _bindings;
  int _nextToolCallId = 1;

  static NativeLlamaBridge? tryOpen(String? nativeLibraryPath) {
    final explicitPath = nativeLibraryPath;
    if (explicitPath != null) {
      _validateLibraryPathText(explicitPath, 'nativeLibraryPath');
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
    return Isolate.run(() => _inspectModelInWorker(config));
  }

  static Future<Map<String, String>> modelMetadata(LlamaModelConfig config) {
    return Isolate.run(() => _modelMetadataInWorker(config));
  }

  static Future<String> chatTemplate(LlamaModelConfig config) {
    return Isolate.run(() => _chatTemplateInWorker(config));
  }

  static Future<List<int>> tokenize(
    LlamaModelConfig config,
    String text, {
    required bool addSpecial,
    required bool parseSpecial,
  }) {
    return Isolate.run(
      () => _tokenizeInWorker(
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
      () => _detokenizeInWorker(
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
    return Isolate.run(() => _embedTextInWorker(config, text, embeddingConfig));
  }

  static Future<EmbeddingBatch> embedTexts(
    LlamaModelConfig config,
    List<String> texts,
    EmbeddingConfig embeddingConfig,
  ) {
    return Isolate.run(
      () => _embedTextsInWorker(config, texts, embeddingConfig),
    );
  }

  static Future<List<double>> rerankDocuments(
    LlamaModelConfig config,
    String query,
    List<String> documents,
    RerankingConfig rerankingConfig,
  ) {
    return Isolate.run(
      () => _rerankDocumentsInWorker(config, query, documents, rerankingConfig),
    );
  }

  static Future<String> formatChat(
    LlamaModelConfig config,
    List<ChatMessage> messages, {
    required bool addAssistantPrompt,
    required LlamaToolCallingConfig toolCalling,
  }) {
    return Isolate.run(
      () => _formatChatInWorker(
        config,
        messages,
        addAssistantPrompt: addAssistantPrompt,
        toolCalling: toolCalling,
      ),
    );
  }

  static Future<int> countChatTokens(
    LlamaModelConfig config,
    List<ChatMessage> messages, {
    required bool addAssistantPrompt,
    required LlamaToolCallingConfig toolCalling,
  }) {
    return Isolate.run(
      () => _countChatTokensInWorker(
        config,
        messages,
        addAssistantPrompt: addAssistantPrompt,
        toolCalling: toolCalling,
      ),
    );
  }

  static Future<LlamaChatTemplateCapabilities> chatTemplateCapabilities(
    LlamaModelConfig config,
  ) {
    return Isolate.run(() => _chatTemplateCapabilitiesInWorker(config));
  }

  static Future<NativeLlamaEngineSession> startEngine(
    LlamaModelConfig config,
  ) => _startEngine(config, embeddings: false, pooling: EmbeddingPooling.model);

  static Future<NativeLlamaEngineSession> startEmbeddingEngine(
    LlamaModelConfig config,
    EmbeddingPooling pooling,
  ) => _startEngine(config, embeddings: true, pooling: pooling);

  static Future<NativeLlamaEngineSession> _startEngine(
    LlamaModelConfig config, {
    required bool embeddings,
    required EmbeddingPooling pooling,
  }) async {
    final ready = ReceivePort();
    final lifecyclePort = ReceivePort();
    final lifecycle = _EngineWorkerLifecycle(lifecyclePort);
    final Isolate isolate;
    try {
      isolate = await Isolate.spawn(
        _engineWorkerMain,
        _EngineWorkerStart(
          config,
          ready.sendPort,
          embeddings: embeddings,
          pooling: pooling,
        ),
        errorsAreFatal: true,
        onError: lifecyclePort.sendPort,
        onExit: lifecyclePort.sendPort,
      );
    } catch (_) {
      ready.close();
      await lifecycle.dispose(expected: true);
      rethrow;
    }
    final message = await lifecycle.receive(ready);

    if (message is _EngineWorkerReady) {
      return NativeLlamaEngineSession._(
        isolate,
        message.commands,
        config.nativeLibraryPath,
        message.contextAddress,
        lifecycle,
      );
    }
    isolate.kill(priority: Isolate.immediate);
    await lifecycle.dispose(expected: true);
    if (message is _EngineWorkerFailure) {
      throw message.error.toException();
    }
    throw NativeBridgeException('Unexpected engine worker response: $message');
  }

  LlamaRuntimeCapabilities currentCapabilities() {
    final capabilities = calloc<llama_dart_capabilities>();
    try {
      capabilities.ref.struct_size = ffi.sizeOf<llama_dart_capabilities>();
      _check(_bindings.llama_dart_get_capabilities(capabilities));
      final flags = capabilities.ref.flags;
      return LlamaRuntimeCapabilities(
        nativeBridgeAvailable: true,
        bridgeAbiVersion: capabilities.ref.abi_version,
        modelLoading: _hasFlag(
          flags,
          llama_dart_capability_flags.LLAMA_DART_CAP_MODEL_LOADING.value,
        ),
        tokenization: _hasFlag(
          flags,
          llama_dart_capability_flags.LLAMA_DART_CAP_TOKENIZATION.value,
        ),
        textGeneration: _hasFlag(
          flags,
          llama_dart_capability_flags.LLAMA_DART_CAP_TEXT_GENERATION.value,
        ),
        structuredOutput: _hasFlag(
          flags,
          llama_dart_capability_flags.LLAMA_DART_CAP_STRUCTURED_OUTPUT.value,
        ),
        embeddings: _hasFlag(
          flags,
          llama_dart_capability_flags.LLAMA_DART_CAP_EMBEDDINGS.value,
        ),
        reranking: _hasFlag(
          flags,
          llama_dart_capability_flags.LLAMA_DART_CAP_RERANKING.value,
        ),
        rag: true,
        multimodal: _hasFlag(
          flags,
          llama_dart_capability_flags.LLAMA_DART_CAP_MULTIMODAL.value,
        ),
        lora: _hasFlag(
          flags,
          llama_dart_capability_flags.LLAMA_DART_CAP_LORA.value,
        ),
        speculativeDecoding: _hasFlag(
          flags,
          llama_dart_capability_flags.LLAMA_DART_CAP_SPECULATIVE_DECODING.value,
        ),
        mtp: _hasFlag(
          flags,
          llama_dart_capability_flags.LLAMA_DART_CAP_MTP.value,
        ),
        metal: _hasFlag(
          flags,
          llama_dart_capability_flags.LLAMA_DART_CAP_METAL.value,
        ),
        vulkan: _hasFlag(
          flags,
          llama_dart_capability_flags.LLAMA_DART_CAP_VULKAN.value,
        ),
        toolCalling: _hasFlag(
          flags,
          llama_dart_capability_flags.LLAMA_DART_CAP_TOOL_CALLING.value,
        ),
        nativeLogging: _hasFlag(
          flags,
          llama_dart_capability_flags.LLAMA_DART_CAP_LOGGING.value,
        ),
        prefill: _hasFlag(
          flags,
          llama_dart_capability_flags.LLAMA_DART_CAP_PREFILL.value,
        ),
        upstreamCommit: _readOptionalCString(
          _bindings.llama_dart_upstream_commit(),
        ),
        nativeBuildFlags: _readOptionalCString(
          _bindings.llama_dart_build_flags(),
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
    _check(_bindings.llama_dart_log_set_level(nativeLevel.value));
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
        _check(_bindings.llama_dart_log_next(level, message));
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
            level: _nativeLogLevelFromValue(level.value),
            message: utf8.decode(data.asTypedList(size), allowMalformed: true),
          ),
        );
        _bindings.llama_dart_buffer_free(data);
        message.ref.data = ffi.nullptr;
        message.ref.size = 0;
      }
      return List<LlamaLogRecord>.unmodifiable(records);
    } finally {
      _bindings.llama_dart_buffer_free(message.ref.data);
      calloc.free(message);
      calloc.free(level);
    }
  }

  String get _multimodalMarker {
    final marker = _readOptionalCString(
      _bindings.llama_dart_multimodal_marker(),
    );
    if (marker == null || marker.isEmpty) {
      throw const NativeBridgeException(
        'Native bridge returned an empty multimodal marker.',
      );
    }
    return marker;
  }

  void _ensureAbiVersion() {
    final actual = _bindings.llama_dart_abi_version();
    if (actual != LLAMA_DART_ABI_VERSION) {
      throw UnsupportedFeatureException(
        'Native bridge ABI version $actual is incompatible with Dart bindings '
        'ABI version $LLAMA_DART_ABI_VERSION.',
      );
    }
  }

  LlamaModelInfo _inspectModel(LlamaModelConfig config) {
    return _withLoadedModel(config, _readModelInfo);
  }

  Map<String, String> _modelMetadata(LlamaModelConfig config) {
    return _withLoadedModel(config, _readModelMetadata);
  }

  String _chatTemplateModel(LlamaModelConfig config) {
    return _withLoadedModel(config, _readChatTemplate);
  }

  List<int> _tokenizeModel(
    LlamaModelConfig config,
    String text, {
    required bool addSpecial,
    required bool parseSpecial,
  }) {
    return _withLoadedModel(
      config,
      (model) => _tokenize(
        model,
        text,
        addSpecial: addSpecial,
        parseSpecial: parseSpecial,
      ),
    );
  }

  List<int> _tokenize(
    ffi.Pointer<llama_dart_model> model,
    String text, {
    required bool addSpecial,
    required bool parseSpecial,
  }) {
    final textBytes = utf8.encode(text);
    final textPointer = calloc<ffi.Uint8>(textBytes.length);
    final outCount = calloc<ffi.Size>();
    try {
      textPointer.asTypedList(textBytes.length).setAll(0, textBytes);
      final firstResult = _bindings.llama_dart_model_tokenize(
        model,
        textPointer,
        textBytes.length,
        ffi.nullptr,
        0,
        outCount,
        addSpecial ? 1 : 0,
        parseSpecial ? 1 : 0,
      );
      if (firstResult != llama_dart_result.LLAMA_DART_ERROR_BUFFER_TOO_SMALL) {
        _check(firstResult);
      }

      final tokensPointer = calloc<ffi.Int32>(outCount.value);
      try {
        _check(
          _bindings.llama_dart_model_tokenize(
            model,
            textPointer,
            textBytes.length,
            tokensPointer,
            outCount.value,
            outCount,
            addSpecial ? 1 : 0,
            parseSpecial ? 1 : 0,
          ),
        );
        return Int32List.fromList(tokensPointer.asTypedList(outCount.value));
      } finally {
        calloc.free(tokensPointer);
      }
    } finally {
      calloc.free(outCount);
      calloc.free(textPointer);
    }
  }

  String _detokenizeModel(
    LlamaModelConfig config,
    List<int> tokens, {
    required bool removeSpecial,
    required bool unparseSpecial,
  }) {
    return _withLoadedModel(
      config,
      (model) => _detokenize(
        model,
        tokens,
        removeSpecial: removeSpecial,
        unparseSpecial: unparseSpecial,
      ),
    );
  }

  String _detokenize(
    ffi.Pointer<llama_dart_model> model,
    List<int> tokens, {
    required bool removeSpecial,
    required bool unparseSpecial,
  }) {
    final tokensPointer = calloc<ffi.Int32>(tokens.length);
    final outSize = calloc<ffi.Size>();
    try {
      tokensPointer.asTypedList(tokens.length).setAll(0, tokens);
      final firstResult = _bindings.llama_dart_model_detokenize(
        model,
        tokensPointer,
        tokens.length,
        ffi.nullptr,
        0,
        outSize,
        removeSpecial ? 1 : 0,
        unparseSpecial ? 1 : 0,
      );
      if (firstResult != llama_dart_result.LLAMA_DART_ERROR_BUFFER_TOO_SMALL) {
        _check(firstResult);
      }

      final textPointer = calloc<ffi.Uint8>(outSize.value);
      try {
        _check(
          _bindings.llama_dart_model_detokenize(
            model,
            tokensPointer,
            tokens.length,
            textPointer,
            outSize.value,
            outSize,
            removeSpecial ? 1 : 0,
            unparseSpecial ? 1 : 0,
          ),
        );
        return utf8.decode(
          textPointer.asTypedList(outSize.value),
          allowMalformed: true,
        );
      } finally {
        calloc.free(textPointer);
      }
    } finally {
      calloc.free(outSize);
      calloc.free(tokensPointer);
    }
  }

  String _formatChatModel(
    LlamaModelConfig config,
    List<ChatMessage> messages, {
    required bool addAssistantPrompt,
    required LlamaToolCallingConfig toolCalling,
  }) {
    return _withLoadedModel(
      config,
      (model) => _formatChat(
        model,
        messages,
        addAssistantPrompt: addAssistantPrompt,
        toolCalling: toolCalling,
      ),
    );
  }

  String _formatChat(
    ffi.Pointer<llama_dart_model> model,
    List<ChatMessage> messages, {
    required bool addAssistantPrompt,
    required LlamaToolCallingConfig toolCalling,
  }) {
    final prepared = _prepareMultimodalChat(messages, _multimodalMarker);
    return _renderPreparedChat(
      model,
      prepared.messages,
      addAssistantPrompt: addAssistantPrompt,
      toolCalling: toolCalling,
    ).prompt;
  }

  _NativeRenderedChat _renderPreparedChat(
    ffi.Pointer<llama_dart_model> model,
    List<ChatMessage> messages, {
    required bool addAssistantPrompt,
    required LlamaToolCallingConfig toolCalling,
    String? grammar,
    Map<String, Object?>? jsonSchema,
    bool? enableThinking,
  }) {
    final parseOutput = _requiresChatPlan(messages, toolCalling);
    if (parseOutput || enableThinking != null) {
      final plan = _createChatPlan(
        model,
        messages,
        toolCalling,
        addAssistantPrompt: addAssistantPrompt,
        grammar: grammar,
        jsonSchema: jsonSchema,
        enableThinking: enableThinking,
        parseOutput: parseOutput,
      );
      return _NativeRenderedChat(prompt: plan.prompt, plan: plan);
    }

    try {
      return _NativeRenderedChat(
        prompt: _applyChatTemplate(
          model,
          messages,
          addAssistantPrompt: addAssistantPrompt,
        ),
      );
    } on UnsupportedFeatureException {
      // Keep the plan for its prompt, grammar metadata, and stop strings. Plain
      // bounded output stays raw because the upstream terminal parser is
      // strict and can reject an otherwise valid truncated response.
      final plan = _createChatPlan(
        model,
        messages,
        toolCalling,
        addAssistantPrompt: addAssistantPrompt,
        grammar: grammar,
        jsonSchema: jsonSchema,
        enableThinking: enableThinking,
        parseOutput: false,
      );
      return _NativeRenderedChat(prompt: plan.prompt, plan: plan);
    }
  }

  LlamaChatTemplateCapabilities _chatTemplateCapabilitiesModel(
    LlamaModelConfig config,
  ) {
    return _withLoadedModel(config, _chatTemplateCapabilities);
  }

  LlamaChatTemplateCapabilities _chatTemplateCapabilities(
    ffi.Pointer<llama_dart_model> model,
  ) {
    final out = calloc<llama_dart_chat_template_capabilities>();
    try {
      out.ref.struct_size = ffi.sizeOf<llama_dart_chat_template_capabilities>();
      _check(
        _bindings.llama_dart_model_get_chat_template_capabilities(model, out),
      );
      return LlamaChatTemplateCapabilities(
        supportsTools: out.ref.supports_tools != 0,
        supportsToolCalls: out.ref.supports_tool_calls != 0,
        supportsParallelToolCalls: out.ref.supports_parallel_tool_calls != 0,
      );
    } finally {
      calloc.free(out);
    }
  }

  int _countChatTokensModel(
    LlamaModelConfig config,
    List<ChatMessage> messages, {
    required bool addAssistantPrompt,
    required LlamaToolCallingConfig toolCalling,
  }) {
    return _withLoadedModel(
      config,
      (model) => _countChatTokens(
        model,
        messages,
        addAssistantPrompt: addAssistantPrompt,
        toolCalling: toolCalling,
      ),
    );
  }

  int _countChatTokens(
    ffi.Pointer<llama_dart_model> model,
    List<ChatMessage> messages, {
    required bool addAssistantPrompt,
    required LlamaToolCallingConfig toolCalling,
  }) {
    final prompt = _formatChat(
      model,
      messages,
      addAssistantPrompt: addAssistantPrompt,
      toolCalling: toolCalling,
    );
    return _tokenize(
      model,
      prompt,
      addSpecial: true,
      parseSpecial: true,
    ).length;
  }

  T _withLoadedModel<T>(
    LlamaModelConfig config,
    T Function(ffi.Pointer<llama_dart_model> model) useModel, {
    bool vocabOnly = true,
  }) {
    final modelPath = utf8.encode(config.modelPath);
    final chatTemplate = config.chatTemplate == null
        ? const <int>[]
        : utf8.encode(config.chatTemplate!);
    final pathPointer = calloc<ffi.Uint8>(modelPath.length);
    ffi.Pointer<ffi.Uint8> chatTemplatePointer = ffi.nullptr;
    final loadConfig = calloc<llama_dart_model_load_config>();
    final outModel = calloc<ffi.Pointer<llama_dart_model>>();

    ffi.Pointer<llama_dart_model> model = ffi.nullptr;
    try {
      pathPointer.asTypedList(modelPath.length).setAll(0, modelPath);
      if (chatTemplate.isNotEmpty) {
        chatTemplatePointer = calloc<ffi.Uint8>(chatTemplate.length);
        chatTemplatePointer
            .asTypedList(chatTemplate.length)
            .setAll(0, chatTemplate);
      }

      loadConfig.ref
        ..struct_size = ffi.sizeOf<llama_dart_model_load_config>()
        ..model_path_data = pathPointer
        ..model_path_size = modelPath.length
        ..n_gpu_layers = _gpuLayers(config)
        ..vocab_only = vocabOnly ? 1 : 0
        ..use_mmap = config.useMmap ? 1 : 0
        ..use_mlock = config.useMlock ? 1 : 0
        ..check_tensors = config.checkTensors ? 1 : 0
        ..gpu_backend = _gpuBackend(config)
        ..chat_template_data = chatTemplatePointer
        ..chat_template_size = chatTemplate.length;

      _check(_bindings.llama_dart_model_load(loadConfig, outModel));
      model = outModel.value;
      if (model == ffi.nullptr) {
        throw const ModelLoadException('Native bridge returned a null model.');
      }
      return useModel(model);
    } finally {
      if (model != ffi.nullptr) {
        _bindings.llama_dart_model_free(model);
        _throwIfLastError('Native model free');
      }
      calloc.free(outModel);
      calloc.free(loadConfig);
      if (chatTemplatePointer != ffi.nullptr) {
        calloc.free(chatTemplatePointer);
      }
      calloc.free(pathPointer);
    }
  }

  LlamaModelInfo _readModelInfo(ffi.Pointer<llama_dart_model> model) {
    final info = calloc<llama_dart_model_info>();
    try {
      info.ref.struct_size = ffi.sizeOf<llama_dart_model_info>();
      _check(_bindings.llama_dart_model_get_info(model, info));
      return LlamaModelInfo(
        description: _readDescription(model),
        chatTemplate: _tryReadChatTemplate(model),
        vocabType: info.ref.vocab_type,
        vocabSize: info.ref.n_vocab,
        trainingContextSize: info.ref.n_ctx_train,
        embeddingSize: info.ref.n_embd,
        inputEmbeddingSize: info.ref.n_embd_inp,
        outputEmbeddingSize: info.ref.n_embd_out,
        layerCount: info.ref.n_layer,
        nextnLayerCount: info.ref.n_layer_nextn,
        attentionHeadCount: info.ref.n_head,
        keyValueHeadCount: info.ref.n_head_kv,
        fileType: info.ref.ftype,
        fileTypeName: _readOptionalCString(
          _bindings.llama_dart_model_file_type_name(info.ref.ftype),
        ),
        sizeBytes: info.ref.size_bytes,
        parameterCount: info.ref.n_params,
        bosToken: info.ref.token_bos,
        eosToken: info.ref.token_eos,
        eotToken: info.ref.token_eot,
        separatorToken: info.ref.token_sep,
        newlineToken: info.ref.token_nl,
        paddingToken: info.ref.token_pad,
        maskToken: info.ref.token_mask,
        addBosToken: info.ref.add_bos != 0,
        addEosToken: info.ref.add_eos != 0,
        addSeparatorToken: info.ref.add_sep != 0,
        hasEncoder: info.ref.has_encoder != 0,
        hasDecoder: info.ref.has_decoder != 0,
        isRecurrent: info.ref.is_recurrent != 0,
        isHybrid: info.ref.is_hybrid != 0,
        isDiffusion: info.ref.is_diffusion != 0,
      );
    } finally {
      calloc.free(info);
    }
  }

  String _readDescription(ffi.Pointer<llama_dart_model> model) {
    var size = 256;
    while (true) {
      final buffer = calloc<ffi.Char>(size);
      final outSize = calloc<ffi.Size>();
      try {
        final result = _bindings.llama_dart_model_get_description(
          model,
          buffer,
          size,
          outSize,
        );
        if (result == llama_dart_result.LLAMA_DART_ERROR_BUFFER_TOO_SMALL) {
          if (outSize.value > _maxModelDescriptionBytes) {
            throw const ModelLoadException(
              'Model description exceeds the 1 MiB safety limit.',
            );
          }
          size = outSize.value + 1;
          continue;
        }
        _check(result);
        return _decodeModelUtf8(buffer, outSize.value, 'description');
      } finally {
        calloc.free(outSize);
        calloc.free(buffer);
      }
    }
  }

  String _readChatTemplate(ffi.Pointer<llama_dart_model> model) {
    final out = calloc<llama_dart_buffer>();
    try {
      _check(_bindings.llama_dart_model_get_chat_template(model, out));
      final data = out.ref.data;
      final size = out.ref.size;
      if (data == ffi.nullptr || size == 0) {
        throw const NativeBridgeException(
          'Native bridge returned an empty chat template.',
        );
      }
      if (size > _maxChatTemplateBytes) {
        throw const UnsupportedFeatureException(
          'Model chat template exceeds the 16 MiB safety limit.',
        );
      }
      try {
        return utf8.decode(data.asTypedList(size));
      } on FormatException catch (error) {
        throw UnsupportedFeatureException(
          'Model chat template is not valid UTF-8.',
          cause: error,
        );
      }
    } finally {
      _bindings.llama_dart_buffer_free(out.ref.data);
      calloc.free(out);
    }
  }

  String? _tryReadChatTemplate(ffi.Pointer<llama_dart_model> model) {
    try {
      return _readChatTemplate(model);
    } on UnsupportedFeatureException {
      return null;
    }
  }

  Map<String, String> _readModelMetadata(ffi.Pointer<llama_dart_model> model) {
    final outCount = calloc<ffi.Size>();
    try {
      _check(_bindings.llama_dart_model_metadata_count(model, outCount));
      if (outCount.value > _maxModelMetadataEntries) {
        throw const ModelLoadException(
          'Model metadata exceeds the 65536-entry safety limit.',
        );
      }
      final metadata = <String, String>{};
      var totalBytes = 0;
      for (var i = 0; i < outCount.value; i += 1) {
        final keySize = calloc<ffi.Size>();
        final valueSize = calloc<ffi.Size>();
        try {
          final firstResult = _bindings.llama_dart_model_metadata_get(
            model,
            i,
            ffi.nullptr,
            0,
            keySize,
            ffi.nullptr,
            0,
            valueSize,
          );
          if (firstResult !=
              llama_dart_result.LLAMA_DART_ERROR_BUFFER_TOO_SMALL) {
            _check(firstResult);
          }
          if (keySize.value > _maxModelMetadataKeyBytes) {
            throw ModelLoadException(
              'Model metadata key $i exceeds the 4 KiB safety limit.',
            );
          }
          if (valueSize.value > _maxModelMetadataValueBytes) {
            throw ModelLoadException(
              'Model metadata value $i exceeds the 16 MiB safety limit.',
            );
          }
          totalBytes += keySize.value + valueSize.value;
          if (totalBytes > _maxModelMetadataTotalBytes) {
            throw const ModelLoadException(
              'Model metadata exceeds the 64 MiB total safety limit.',
            );
          }

          final key = calloc<ffi.Char>(keySize.value + 1);
          final value = calloc<ffi.Char>(valueSize.value + 1);
          try {
            _check(
              _bindings.llama_dart_model_metadata_get(
                model,
                i,
                key,
                keySize.value + 1,
                keySize,
                value,
                valueSize.value + 1,
                valueSize,
              ),
            );
            final decodedKey = _decodeModelUtf8(
              key,
              keySize.value,
              'metadata key at index $i',
            );
            if (decodedKey.trim().isEmpty ||
                decodedKey.contains('\u0000') ||
                decodedKey.contains('\n') ||
                decodedKey.contains('\r')) {
              throw ModelLoadException(
                'Model metadata key $i is empty or contains control bytes.',
              );
            }
            if (metadata.containsKey(decodedKey)) {
              throw ModelLoadException(
                'Model metadata contains a duplicate key at index $i.',
              );
            }
            metadata[decodedKey] = _decodeModelUtf8(
              value,
              valueSize.value,
              'metadata value at index $i',
            );
          } finally {
            calloc.free(value);
            calloc.free(key);
          }
        } finally {
          calloc.free(valueSize);
          calloc.free(keySize);
        }
      }
      return Map<String, String>.unmodifiable(metadata);
    } finally {
      calloc.free(outCount);
    }
  }

  void _check(llama_dart_result result) {
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
    final pointer = _bindings.llama_dart_last_error_message();
    if (pointer == ffi.nullptr) {
      return 'Native bridge failed without an error message.';
    }
    final message = _decodeNativeLastErrorMessage(pointer);
    if (message.isEmpty) {
      return 'Native bridge failed without an error message.';
    }
    return message;
  }

  void _throwIfLastError(String operation) {
    final pointer = _bindings.llama_dart_last_error_message();
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

  void _cancelContextAddress(int contextAddress) {
    if (contextAddress == 0) {
      return;
    }
    _check(
      _bindings.llama_dart_context_cancel(
        ffi.Pointer<llama_dart_context>.fromAddress(contextAddress),
      ),
    );
  }

  static String? _readOptionalCString(ffi.Pointer<ffi.Char> pointer) {
    if (pointer == ffi.nullptr) {
      return null;
    }
    final value = pointer.cast<Utf8>().toDartString();
    return value.isEmpty ? null : value;
  }

  static bool _hasFlag(int flags, int flag) => flags & flag != 0;

  static LlamaModelInfo _inspectModelInWorker(LlamaModelConfig config) {
    final bridge = tryOpen(config.nativeLibraryPath);
    if (bridge == null) {
      throw _nativeBridgeUnavailable(config.nativeLibraryPath);
    }
    return bridge._inspectModel(config);
  }

  static Map<String, String> _modelMetadataInWorker(LlamaModelConfig config) {
    final bridge = tryOpen(config.nativeLibraryPath);
    if (bridge == null) {
      throw _nativeBridgeUnavailable(config.nativeLibraryPath);
    }
    return bridge._modelMetadata(config);
  }

  static String _chatTemplateInWorker(LlamaModelConfig config) {
    final bridge = tryOpen(config.nativeLibraryPath);
    if (bridge == null) {
      throw _nativeBridgeUnavailable(config.nativeLibraryPath);
    }
    return bridge._chatTemplateModel(config);
  }

  static List<int> _tokenizeInWorker(
    LlamaModelConfig config,
    String text, {
    required bool addSpecial,
    required bool parseSpecial,
  }) {
    final bridge = tryOpen(config.nativeLibraryPath);
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

  static String _detokenizeInWorker(
    LlamaModelConfig config,
    List<int> tokens, {
    required bool removeSpecial,
    required bool unparseSpecial,
  }) {
    final bridge = tryOpen(config.nativeLibraryPath);
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

  static Float32List _embedTextInWorker(
    LlamaModelConfig config,
    String text,
    EmbeddingConfig embeddingConfig,
  ) {
    final bridge = tryOpen(config.nativeLibraryPath);
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

  static EmbeddingBatch _embedTextsInWorker(
    LlamaModelConfig config,
    List<String> texts,
    EmbeddingConfig embeddingConfig,
  ) {
    final bridge = tryOpen(config.nativeLibraryPath);
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

  static EmbeddingBatch _embedTextsWithHandles(
    _NativeEngineHandles handles,
    List<String> texts,
    EmbeddingConfig embeddingConfig,
  ) {
    if (texts.isEmpty) {
      return EmbeddingBatch.empty(
        normalized: embeddingConfig.normalize,
        pooling: embeddingConfig.pooling,
      );
    }
    Float32List embed(String text) {
      final values = handles.embedText(text, embeddingConfig);
      return embeddingConfig.normalize ? _normalize(values) : values;
    }

    final first = embed(texts.first);
    if (first.isEmpty) {
      throw const EmbeddingException(
        'Native bridge returned an empty embedding.',
      );
    }
    final dimensions = first.length;
    final values = Float32List(texts.length * dimensions)
      ..setRange(0, dimensions, first);
    for (var i = 1; i < texts.length; i += 1) {
      final vector = embed(texts[i]);
      if (vector.length != dimensions) {
        throw EmbeddingException(
          'Embedding dimension changed within one batch: expected '
          '$dimensions, got ${vector.length}.',
        );
      }
      final start = i * dimensions;
      values.setRange(start, start + dimensions, vector);
    }
    return EmbeddingBatch(
      count: texts.length,
      dimensions: dimensions,
      values: values,
      normalized: embeddingConfig.normalize,
      pooling: embeddingConfig.pooling,
    );
  }

  static String _formatChatInWorker(
    LlamaModelConfig config,
    List<ChatMessage> messages, {
    required bool addAssistantPrompt,
    required LlamaToolCallingConfig toolCalling,
  }) {
    final bridge = tryOpen(config.nativeLibraryPath);
    if (bridge == null) {
      throw _nativeBridgeUnavailable(config.nativeLibraryPath);
    }
    return bridge._formatChatModel(
      config,
      messages,
      addAssistantPrompt: addAssistantPrompt,
      toolCalling: toolCalling,
    );
  }

  static int _countChatTokensInWorker(
    LlamaModelConfig config,
    List<ChatMessage> messages, {
    required bool addAssistantPrompt,
    required LlamaToolCallingConfig toolCalling,
  }) {
    final bridge = tryOpen(config.nativeLibraryPath);
    if (bridge == null) {
      throw _nativeBridgeUnavailable(config.nativeLibraryPath);
    }
    return bridge._countChatTokensModel(
      config,
      messages,
      addAssistantPrompt: addAssistantPrompt,
      toolCalling: toolCalling,
    );
  }

  static LlamaChatTemplateCapabilities _chatTemplateCapabilitiesInWorker(
    LlamaModelConfig config,
  ) {
    final bridge = tryOpen(config.nativeLibraryPath);
    if (bridge == null) {
      throw _nativeBridgeUnavailable(config.nativeLibraryPath);
    }
    return bridge._chatTemplateCapabilitiesModel(config);
  }

  static List<double> _rerankDocumentsInWorker(
    LlamaModelConfig config,
    String query,
    List<String> documents,
    RerankingConfig rerankingConfig,
  ) {
    final bridge = tryOpen(config.nativeLibraryPath);
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

  static void _validateLibraryPathText(String path, String name) {
    if (path.trim().isEmpty) {
      throw ArgumentError.value(path, name, 'must not be empty');
    }
    if (path.contains('\u0000')) {
      throw ArgumentError.value(path, name, 'must not contain NUL');
    }
    if (path.contains('\n') || path.contains('\r')) {
      throw ArgumentError.value(path, name, 'must not contain line breaks');
    }
  }

  _NativeEngineHandles _openEngine(
    LlamaModelConfig config, {
    bool embeddings = false,
    EmbeddingPooling pooling = EmbeddingPooling.model,
  }) {
    final modelPath = utf8.encode(config.modelPath);
    final chatTemplate = config.chatTemplate == null
        ? const <int>[]
        : utf8.encode(config.chatTemplate!);
    final mmprojPath = config.mmprojPath == null
        ? const <int>[]
        : utf8.encode(config.mmprojPath!);
    final speculativeModelPath = embeddings
        ? const <int>[]
        : utf8.encode(_speculativeModelPath(config) ?? '');
    final pathPointer = calloc<ffi.Uint8>(modelPath.length);
    ffi.Pointer<ffi.Uint8> chatTemplatePointer = ffi.nullptr;
    ffi.Pointer<ffi.Uint8> mmprojPathPointer = ffi.nullptr;
    ffi.Pointer<ffi.Uint8> speculativeModelPathPointer = ffi.nullptr;
    final loadConfig = calloc<llama_dart_model_load_config>();
    final outModel = calloc<ffi.Pointer<llama_dart_model>>();
    final contextConfig = calloc<llama_dart_context_config>();
    final outContext = calloc<ffi.Pointer<llama_dart_context>>();

    ffi.Pointer<llama_dart_model> model = ffi.nullptr;
    ffi.Pointer<llama_dart_context> context = ffi.nullptr;
    try {
      pathPointer.asTypedList(modelPath.length).setAll(0, modelPath);
      if (chatTemplate.isNotEmpty) {
        chatTemplatePointer = calloc<ffi.Uint8>(chatTemplate.length);
        chatTemplatePointer
            .asTypedList(chatTemplate.length)
            .setAll(0, chatTemplate);
      }
      if (mmprojPath.isNotEmpty) {
        mmprojPathPointer = calloc<ffi.Uint8>(mmprojPath.length);
        mmprojPathPointer.asTypedList(mmprojPath.length).setAll(0, mmprojPath);
      }
      if (speculativeModelPath.isNotEmpty) {
        speculativeModelPathPointer = calloc<ffi.Uint8>(
          speculativeModelPath.length,
        );
        speculativeModelPathPointer
            .asTypedList(speculativeModelPath.length)
            .setAll(0, speculativeModelPath);
      }
      loadConfig.ref
        ..struct_size = ffi.sizeOf<llama_dart_model_load_config>()
        ..model_path_data = pathPointer
        ..model_path_size = modelPath.length
        ..n_gpu_layers = _gpuLayers(config)
        ..vocab_only = 0
        ..use_mmap = config.useMmap ? 1 : 0
        ..use_mlock = config.useMlock ? 1 : 0
        ..check_tensors = config.checkTensors ? 1 : 0
        ..gpu_backend = _gpuBackend(config)
        ..chat_template_data = chatTemplatePointer
        ..chat_template_size = chatTemplate.length;

      _check(_bindings.llama_dart_model_load(loadConfig, outModel));
      model = outModel.value;
      if (model == ffi.nullptr) {
        throw const ModelLoadException('Native bridge returned a null model.');
      }

      contextConfig.ref
        ..struct_size = ffi.sizeOf<llama_dart_context_config>()
        ..context_size = config.contextSize
        ..batch_size = config.batchSize
        ..ubatch_size = config.ubatchSize ?? config.batchSize
        ..threads = config.threads ?? 0
        ..batch_threads = config.batchThreads ?? config.threads ?? 0
        ..embeddings = embeddings ? 1 : 0
        ..pooling_type = embeddings ? _poolingType(pooling) : -1
        ..attention_type = -1
        ..speculative_ngram_n = embeddings ? 0 : _ngramN(config)
        ..speculative_ngram_m = embeddings ? 0 : _ngramM(config)
        ..mmproj_path_data = mmprojPathPointer
        ..mmproj_path_size = mmprojPath.length
        ..mmproj_use_gpu = mmprojPath.isNotEmpty && _gpuLayers(config) != 0
            ? 1
            : 0
        ..speculative_type = embeddings
            ? llama_dart_speculative_type.LLAMA_DART_SPECULATIVE_NONE.value
            : _speculativeType(config)
        ..speculative_model_path_data = speculativeModelPathPointer
        ..speculative_model_path_size = speculativeModelPath.length
        ..speculative_draft_max = embeddings ? 0 : _speculativeDraftMax(config)
        ..kv_cache_key_type = _kvCacheType(config.kvCache.keyType)
        ..kv_cache_value_type = _kvCacheType(config.kvCache.valueType)
        ..flash_attention = _flashAttention(config.kvCache.flashAttention)
        ..kv_cache_offload = config.kvCache.offload ? 1 : 0
        ..swa_full = config.kvCache.swaFull ? 1 : 0
        ..kv_unified = config.kvCache.unified ? 1 : 0
        ..speculative_ngram_min_draft = embeddings ? 0 : _ngramMinDraft(config);

      _check(
        _bindings.llama_dart_context_create(model, contextConfig, outContext),
      );
      context = outContext.value;
      if (context == ffi.nullptr) {
        throw const ContextCreateException(
          'Native bridge returned a null context.',
        );
      }

      return _NativeEngineHandles(bridge: this, model: model, context: context);
    } catch (_) {
      if (context != ffi.nullptr) {
        _bindings.llama_dart_context_free(context);
      }
      if (model != ffi.nullptr) {
        _bindings.llama_dart_model_free(model);
      }
      rethrow;
    } finally {
      calloc.free(outContext);
      calloc.free(contextConfig);
      calloc.free(outModel);
      calloc.free(loadConfig);
      if (chatTemplatePointer != ffi.nullptr) {
        calloc.free(chatTemplatePointer);
      }
      if (mmprojPathPointer != ffi.nullptr) {
        calloc.free(mmprojPathPointer);
      }
      if (speculativeModelPathPointer != ffi.nullptr) {
        calloc.free(speculativeModelPathPointer);
      }
      calloc.free(pathPointer);
    }
  }

  static int _gpuLayers(LlamaModelConfig config) {
    return switch (config.gpu.backend) {
      GpuBackend.cpu => 0,
      GpuBackend.auto ||
      GpuBackend.metal ||
      GpuBackend.vulkan => config.gpu.layers ?? -1,
    };
  }

  static int _gpuBackend(LlamaModelConfig config) {
    return switch (config.gpu.backend) {
      GpuBackend.auto =>
        llama_dart_gpu_backend.LLAMA_DART_GPU_BACKEND_AUTO.value,
      GpuBackend.cpu => llama_dart_gpu_backend.LLAMA_DART_GPU_BACKEND_CPU.value,
      GpuBackend.metal =>
        llama_dart_gpu_backend.LLAMA_DART_GPU_BACKEND_METAL.value,
      GpuBackend.vulkan =>
        llama_dart_gpu_backend.LLAMA_DART_GPU_BACKEND_VULKAN.value,
    };
  }

  static GpuBackend _gpuBackendFromNative(int value) {
    return switch (llama_dart_gpu_backend.fromValue(value)) {
      llama_dart_gpu_backend.LLAMA_DART_GPU_BACKEND_AUTO => GpuBackend.auto,
      llama_dart_gpu_backend.LLAMA_DART_GPU_BACKEND_CPU => GpuBackend.cpu,
      llama_dart_gpu_backend.LLAMA_DART_GPU_BACKEND_METAL => GpuBackend.metal,
      llama_dart_gpu_backend.LLAMA_DART_GPU_BACKEND_VULKAN => GpuBackend.vulkan,
    };
  }

  static int _kvCacheType(KvCacheType type) {
    return switch (type) {
      KvCacheType.f32 => llama_dart_kv_cache_type.LLAMA_DART_KV_CACHE_F32.value,
      KvCacheType.f16 => llama_dart_kv_cache_type.LLAMA_DART_KV_CACHE_F16.value,
      KvCacheType.bf16 =>
        llama_dart_kv_cache_type.LLAMA_DART_KV_CACHE_BF16.value,
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

  static KvCacheType _kvCacheTypeFromNative(int value) {
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

  static int _flashAttention(FlashAttentionMode mode) {
    return switch (mode) {
      FlashAttentionMode.auto =>
        llama_dart_flash_attention_mode.LLAMA_DART_FLASH_ATTENTION_AUTO.value,
      FlashAttentionMode.disabled =>
        llama_dart_flash_attention_mode
            .LLAMA_DART_FLASH_ATTENTION_DISABLED
            .value,
      FlashAttentionMode.enabled =>
        llama_dart_flash_attention_mode
            .LLAMA_DART_FLASH_ATTENTION_ENABLED
            .value,
    };
  }

  static FlashAttentionMode _flashAttentionFromNative(int value) {
    return switch (llama_dart_flash_attention_mode.fromValue(value)) {
      llama_dart_flash_attention_mode.LLAMA_DART_FLASH_ATTENTION_AUTO =>
        FlashAttentionMode.auto,
      llama_dart_flash_attention_mode.LLAMA_DART_FLASH_ATTENTION_DISABLED =>
        FlashAttentionMode.disabled,
      llama_dart_flash_attention_mode.LLAMA_DART_FLASH_ATTENTION_ENABLED =>
        FlashAttentionMode.enabled,
    };
  }

  static LlamaLogLevel _nativeLogLevelFromValue(int value) {
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

  static int _poolingType(EmbeddingPooling pooling) {
    return switch (pooling) {
      EmbeddingPooling.model => -1,
      EmbeddingPooling.mean => 1,
      EmbeddingPooling.cls => 2,
      EmbeddingPooling.last => 3,
      EmbeddingPooling.rank => 4,
    };
  }

  static int _ngramN(LlamaModelConfig config) {
    final speculation = config.speculativeDecoding;
    if (speculation is NGramSpeculation) {
      return speculation.ngramSize;
    }
    if (speculation is NGramModSpeculation) {
      return speculation.matchLength;
    }
    return 0;
  }

  static int _ngramM(LlamaModelConfig config) {
    final speculation = config.speculativeDecoding;
    if (speculation is NGramSpeculation) {
      return speculation.draftLength;
    }
    if (speculation is NGramModSpeculation) {
      return speculation.maximumDraftLength;
    }
    return 0;
  }

  static int _ngramMinDraft(LlamaModelConfig config) {
    final speculation = config.speculativeDecoding;
    return speculation is NGramModSpeculation
        ? speculation.minimumDraftLength
        : 0;
  }

  static String? _speculativeModelPath(LlamaModelConfig config) {
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

  static int _speculativeType(LlamaModelConfig config) {
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

  static int _speculativeDraftMax(LlamaModelConfig config) {
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

  static int _mirostatMode(MirostatMode? mode) {
    return switch (mode) {
      null => 0,
      MirostatMode.v1 => 1,
      MirostatMode.v2 => 2,
    };
  }

  _NativeChatPlan _createChatPlan(
    ffi.Pointer<llama_dart_model> model,
    List<ChatMessage> messages,
    LlamaToolCallingConfig toolCalling, {
    required bool addAssistantPrompt,
    String? grammar,
    Map<String, Object?>? jsonSchema,
    bool? enableThinking,
    required bool parseOutput,
  }) {
    var tools = toolCalling.toJson();
    final String toolChoice;
    switch (toolCalling.toolChoice) {
      case LlamaAutoToolChoice():
        toolChoice = 'auto';
      case LlamaNoToolChoice():
        toolChoice = 'none';
      case LlamaRequiredToolChoice():
        toolChoice = 'required';
      case LlamaNamedToolChoice(:final name):
        tools = List<Map<String, Object?>>.unmodifiable(
          tools.where((tool) {
            final function = tool['function']! as Map<String, Object?>;
            return function['name'] == name;
          }),
        );
        toolChoice = 'required';
    }
    final request = <String, Object?>{
      'messages': <Map<String, Object?>>[
        for (final message in messages) _chatMessageToOpenAiJson(message),
      ],
      'tools': tools,
      'tool_choice': toolChoice,
      'parallel_tool_calls': toolCalling.allowParallelToolCalls,
      'add_generation_prompt': addAssistantPrompt,
      'grammar': ?grammar,
      'json_schema': ?jsonSchema,
      'enable_thinking': ?enableThinking,
    };
    final requestBytes = utf8.encode(jsonEncode(request));
    final requestPointer = calloc<ffi.Uint8>(requestBytes.length);
    final out = calloc<llama_dart_buffer>();
    try {
      requestPointer.asTypedList(requestBytes.length).setAll(0, requestBytes);
      _check(
        _bindings.llama_dart_model_create_chat_plan(
          model,
          requestPointer,
          requestBytes.length,
          out,
        ),
      );
      final data = out.ref.data;
      final size = out.ref.size;
      if (data == ffi.nullptr || size == 0) {
        throw const NativeBridgeException(
          'Native bridge returned an empty chat plan.',
        );
      }
      final planJson = utf8.decode(data.asTypedList(size));
      final decoded = jsonDecode(planJson);
      if (decoded is! Map<Object?, Object?> || decoded['prompt'] is! String) {
        throw const NativeBridgeException(
          'Native bridge returned an invalid chat plan.',
        );
      }
      return _NativeChatPlan(
        json: planJson,
        prompt: decoded['prompt'] as String,
        parseOutput: parseOutput,
        usedToolCallIds: <String>{
          for (final message in messages) ...<String>{
            for (final call in message.toolCalls)
              if (call.id != null) call.id!,
            if (message.toolCallId != null) message.toolCallId!,
          },
        },
      );
    } on FormatException catch (error) {
      throw NativeBridgeException(
        'Native bridge returned invalid chat plan JSON.',
        cause: error,
      );
    } finally {
      _bindings.llama_dart_buffer_free(out.ref.data);
      calloc.free(out);
      calloc.free(requestPointer);
    }
  }

  ChatMessage _parseChatOutput(
    _NativeChatPlan plan,
    String output,
    LlamaToolCallingConfig toolCalling,
  ) {
    final planBytes = utf8.encode(plan.json);
    final outputBytes = utf8.encode(output);
    final planPointer = calloc<ffi.Uint8>(planBytes.length);
    final outputPointer = outputBytes.isEmpty
        ? ffi.nullptr
        : calloc<ffi.Uint8>(outputBytes.length);
    final out = calloc<llama_dart_buffer>();
    try {
      planPointer.asTypedList(planBytes.length).setAll(0, planBytes);
      if (outputBytes.isNotEmpty) {
        outputPointer.asTypedList(outputBytes.length).setAll(0, outputBytes);
      }
      _check(
        _bindings.llama_dart_chat_parse_output(
          planPointer,
          planBytes.length,
          outputPointer,
          outputBytes.length,
          out,
        ),
      );
      final data = out.ref.data;
      final size = out.ref.size;
      if (data == ffi.nullptr || size == 0) {
        throw const GenerationException(
          'Native chat parser returned an empty message.',
        );
      }
      final Object? decoded;
      try {
        decoded = jsonDecode(utf8.decode(data.asTypedList(size)));
      } on FormatException catch (error) {
        throw GenerationException(
          'Native chat parser returned malformed UTF-8 or JSON.',
          cause: error,
        );
      }
      if (decoded is! Map<Object?, Object?> || decoded['role'] != 'assistant') {
        throw const GenerationException(
          'Native chat parser returned an invalid assistant message.',
        );
      }
      final contentValue = decoded['content'];
      final content = switch (contentValue) {
        null => '',
        String value => value,
        _ => throw const GenerationException(
          'Native chat parser returned invalid assistant content.',
        ),
      };
      final callsValue = decoded['tool_calls'];
      if (callsValue == null) {
        return ChatMessage.assistant(content);
      }
      final calls = LlamaToolCalls.fromJson(
        callsValue,
        allowParallelToolCalls: toolCalling.allowParallelToolCalls,
      );
      if (toolCalling.toolChoice is LlamaNoToolChoice) {
        throw const GenerationException(
          'Model returned tool calls when tool choice was none.',
        );
      }
      final allowed = <String, LlamaToolDefinition>{
        for (final tool in toolCalling.tools) tool.name: tool,
      };
      final usedIds = <String>{...plan.usedToolCallIds};
      for (final call in calls) {
        final id = call.id;
        if (id != null && !usedIds.add(id)) {
          throw GenerationException(
            'Model reused tool call id $id from chat history.',
          );
        }
      }
      final normalizedCalls = <LlamaToolCall>[];
      for (final call in calls) {
        final tool = allowed[call.name];
        if (tool == null) {
          throw GenerationException(
            'Model returned an unknown tool call: ${call.name}.',
          );
        }
        if (toolCalling.toolChoice case LlamaNamedToolChoice(:final name)) {
          if (call.name != name) {
            throw GenerationException(
              'Model returned ${call.name} when $name was required.',
            );
          }
        }
        try {
          tool.validateArguments(call.arguments);
        } on ArgumentError catch (error) {
          throw GenerationException(
            'Model returned invalid arguments for ${call.name}: '
            '${error.message ?? error}.',
            cause: error,
          );
        }
        var id = call.id;
        if (id == null) {
          do {
            id = 'call_${_nextToolCallId++}';
          } while (usedIds.contains(id));
          usedIds.add(id);
        }
        normalizedCalls.add(
          LlamaToolCall(id: id, name: call.name, arguments: call.arguments),
        );
      }
      return ChatMessage.assistantToolCalls(
        text: content,
        toolCalls: normalizedCalls,
      );
    } on FormatException catch (error) {
      throw GenerationException(
        'Native chat parser returned invalid JSON.',
        cause: error,
      );
    } on ArgumentError catch (error) {
      throw GenerationException(
        'Native chat parser returned invalid tool calls.',
        cause: error,
      );
    } finally {
      _bindings.llama_dart_buffer_free(out.ref.data);
      calloc.free(out);
      if (outputPointer != ffi.nullptr) {
        calloc.free(outputPointer);
      }
      calloc.free(planPointer);
    }
  }

  String _applyChatTemplate(
    ffi.Pointer<llama_dart_model> model,
    List<ChatMessage> messages, {
    required bool addAssistantPrompt,
  }) {
    final nativeMessages = calloc<llama_dart_chat_message>(messages.length);
    final allocated = <ffi.Pointer<ffi.Uint8>>[];
    final out = calloc<llama_dart_buffer>();
    try {
      for (var i = 0; i < messages.length; i += 1) {
        final message = messages[i];
        final role = utf8.encode(_chatRoleName(message.role));
        final content = utf8.encode(message.text);
        final rolePointer = calloc<ffi.Uint8>(role.length);
        rolePointer.asTypedList(role.length).setAll(0, role);
        allocated.add(rolePointer);

        ffi.Pointer<ffi.Uint8> contentPointer = ffi.nullptr;
        if (content.isNotEmpty) {
          contentPointer = calloc<ffi.Uint8>(content.length);
          contentPointer.asTypedList(content.length).setAll(0, content);
          allocated.add(contentPointer);
        }

        nativeMessages[i]
          ..struct_size = ffi.sizeOf<llama_dart_chat_message>()
          ..role_data = rolePointer
          ..role_size = role.length
          ..content_data = contentPointer
          ..content_size = content.length;
      }

      _check(
        _bindings.llama_dart_model_apply_chat_template(
          model,
          nativeMessages,
          messages.length,
          addAssistantPrompt ? 1 : 0,
          out,
        ),
      );

      final data = out.ref.data;
      final size = out.ref.size;
      if (data == ffi.nullptr || size == 0) {
        return '';
      }
      try {
        return utf8.decode(data.asTypedList(size));
      } on FormatException catch (error) {
        throw NativeBridgeException(
          'Model chat template produced invalid UTF-8.',
          cause: error,
        );
      }
    } finally {
      _bindings.llama_dart_buffer_free(out.ref.data);
      calloc.free(out);
      for (final pointer in allocated) {
        calloc.free(pointer);
      }
      calloc.free(nativeMessages);
    }
  }
}

final class _EngineWorkerLifecycle {
  _EngineWorkerLifecycle(this._port) {
    _port.listen(_handleEvent);
  }

  final ReceivePort _port;
  final StreamController<_EngineWorkerFailure> _failures =
      StreamController<_EngineWorkerFailure>.broadcast(sync: true);
  _EngineWorkerFailure? _failure;
  bool _expected = false;

  _EngineWorkerFailure? get failure => _failure;
  Stream<_EngineWorkerFailure> get failures => _failures.stream;

  Future<Object?> receive(ReceivePort reply) {
    final existingFailure = _failure;
    if (existingFailure != null) {
      reply.close();
      return Future<Object?>.value(existingFailure);
    }

    final completer = Completer<Object?>();
    late final StreamSubscription<Object?> replySubscription;
    late final StreamSubscription<_EngineWorkerFailure> failureSubscription;

    void complete(Object? value) {
      if (!completer.isCompleted) {
        completer.complete(value);
      }
    }

    replySubscription = reply.listen(complete);
    failureSubscription = failures.listen(complete);
    final racedFailure = _failure;
    if (racedFailure != null) {
      complete(racedFailure);
    }

    return completer.future.whenComplete(() async {
      await replySubscription.cancel();
      await failureSubscription.cancel();
      reply.close();
    });
  }

  Future<void> dispose({required bool expected}) async {
    _expected = _expected || expected;
    _port.close();
    if (!_failures.isClosed) {
      await _failures.close();
    }
  }

  void _handleEvent(Object? event) {
    if (!_expected && _failure == null) {
      final details = event is List<Object?> && event.isNotEmpty
          ? ': ${event.first}'
          : '';
      final failure = _EngineWorkerFailure(
        _NativeError(
          'nativeBridge',
          'Inference worker exited unexpectedly$details',
        ),
      );
      _failure = failure;
      _failures.add(failure);
    }
    _port.close();
    if (!_failures.isClosed) {
      unawaited(_failures.close());
    }
  }
}

final class NativeLlamaEngineSession {
  NativeLlamaEngineSession._(
    this._isolate,
    this._commands,
    this._nativeLibraryPath,
    this._contextAddress,
    this._lifecycle,
  ) {
    _finalizer.attach(
      this,
      _EngineFinalizerToken(_commands, _nativeLibraryPath, _contextAddress),
      detach: _finalizerDetach,
    );
  }

  static final Finalizer<_EngineFinalizerToken> _finalizer = Finalizer(
    (token) => token.release(),
  );

  final Isolate _isolate;
  final SendPort _commands;
  final String? _nativeLibraryPath;
  final int _contextAddress;
  final _EngineWorkerLifecycle _lifecycle;
  final Object _finalizerDetach = Object();
  int _nextStreamId = 1;
  int? _activeStreamId;
  bool _closed = false;

  bool get hasActiveGeneration => _activeStreamId != null;

  Future<
    ({
      String text,
      GenerationTelemetry telemetry,
      ChatMessage? assistantMessage,
    })
  >
  completeChat(List<ChatMessage> messages, GenerationConfig config) async {
    if (_closed) {
      throw const ResourceDisposedException('LlamaEngine is closed.');
    }
    final reply = ReceivePort();
    _commands.send(_EngineWorkerComplete(messages, config, reply.sendPort));
    final message = await _lifecycle.receive(reply);
    if (message is _EngineWorkerText) {
      return (
        text: message.text,
        telemetry: message.telemetry,
        assistantMessage: message.assistantMessage,
      );
    }
    if (message is _EngineWorkerFailure) {
      throw message.error.toException();
    }
    throw NativeBridgeException('Unexpected engine worker response: $message');
  }

  Future<
    ({
      String text,
      GenerationTelemetry telemetry,
      ChatMessage? assistantMessage,
    })
  >
  complete(String prompt, GenerationConfig config) async {
    if (_closed) {
      throw const ResourceDisposedException('LlamaEngine is closed.');
    }
    final reply = ReceivePort();
    _commands.send(_EngineWorkerCompletePrompt(prompt, config, reply.sendPort));
    final message = await _lifecycle.receive(reply);
    if (message is _EngineWorkerText) {
      return (
        text: message.text,
        telemetry: message.telemetry,
        assistantMessage: message.assistantMessage,
      );
    }
    if (message is _EngineWorkerFailure) {
      throw message.error.toException();
    }
    throw NativeBridgeException('Unexpected engine worker response: $message');
  }

  Stream<
    ({
      String text,
      bool isDone,
      GenerationTelemetry? telemetry,
      ChatMessage? assistantMessage,
    })
  >
  completeChatStream(List<ChatMessage> messages, GenerationConfig config) {
    if (_closed) {
      throw const ResourceDisposedException('LlamaEngine is closed.');
    }
    return _streamGeneration(
      (id, reply) => _EngineWorkerStreamComplete(id, messages, config, reply),
    );
  }

  Stream<
    ({
      String text,
      bool isDone,
      GenerationTelemetry? telemetry,
      ChatMessage? assistantMessage,
    })
  >
  completeStream(String prompt, GenerationConfig config) {
    if (_closed) {
      throw const ResourceDisposedException('LlamaEngine is closed.');
    }
    return _streamGeneration(
      (id, reply) => _EngineWorkerStreamPrompt(id, prompt, config, reply),
    );
  }

  Stream<
    ({
      String text,
      bool isDone,
      GenerationTelemetry? telemetry,
      ChatMessage? assistantMessage,
    })
  >
  _streamGeneration(Object Function(int id, SendPort reply) createMessage) {
    late final StreamController<
      ({
        String text,
        bool isDone,
        GenerationTelemetry? telemetry,
        ChatMessage? assistantMessage,
      })
    >
    controller;
    ReceivePort? reply;
    StreamSubscription<_EngineWorkerFailure>? lifecycleSubscription;
    var streamId = 0;
    var paused = false;
    var waitingForWorker = false;
    var terminalResponseReceived = false;
    var slotClaimed = false;

    void releaseSlot() {
      if (slotClaimed && _activeStreamId == streamId) {
        _activeStreamId = null;
      }
      slotClaimed = false;
    }

    void stopLifecycleListening() {
      final subscription = lifecycleSubscription;
      lifecycleSubscription = null;
      if (subscription != null) {
        unawaited(subscription.cancel());
      }
    }

    void handleWorkerFailure(_EngineWorkerFailure failure) {
      if (terminalResponseReceived) {
        return;
      }
      terminalResponseReceived = true;
      waitingForWorker = false;
      reply?.close();
      stopLifecycleListening();
      releaseSlot();
      if (!controller.isClosed) {
        controller.addError(failure.error.toException());
        unawaited(controller.close());
      }
    }

    void requestNext() {
      if (streamId == 0 ||
          paused ||
          waitingForWorker ||
          terminalResponseReceived ||
          controller.isClosed) {
        return;
      }
      waitingForWorker = true;
      _commands.send(_EngineWorkerStreamNext(streamId));
    }

    controller =
        StreamController<
          ({
            String text,
            bool isDone,
            GenerationTelemetry? telemetry,
            ChatMessage? assistantMessage,
          })
        >(
          onListen: () {
            if (_closed) {
              terminalResponseReceived = true;
              controller.addError(
                const ResourceDisposedException('LlamaEngine is closed.'),
              );
              controller.close();
              return;
            }
            final existingFailure = _lifecycle.failure;
            if (existingFailure != null) {
              handleWorkerFailure(existingFailure);
              return;
            }
            if (_activeStreamId != null) {
              terminalResponseReceived = true;
              controller.addError(
                const GenerationException(
                  'Another generation is already active on this engine.',
                ),
              );
              controller.close();
              return;
            }
            streamId = _nextStreamId;
            _nextStreamId += 1;
            _activeStreamId = streamId;
            slotClaimed = true;
            reply = ReceivePort();
            lifecycleSubscription = _lifecycle.failures.listen(
              handleWorkerFailure,
            );
            reply!.listen((message) {
              waitingForWorker = false;
              if (message is _EngineWorkerStreamChunk) {
                if (message.isDone) {
                  terminalResponseReceived = true;
                  releaseSlot();
                }
                if (!controller.isClosed &&
                    (message.text.isNotEmpty || message.isDone)) {
                  controller.add((
                    text: message.text,
                    isDone: message.isDone,
                    telemetry: message.telemetry,
                    assistantMessage: message.assistantMessage,
                  ));
                }
                if (message.isDone) {
                  reply?.close();
                  stopLifecycleListening();
                  if (!controller.isClosed) {
                    controller.close();
                  }
                } else {
                  scheduleMicrotask(requestNext);
                }
              } else if (message is _EngineWorkerFailure) {
                terminalResponseReceived = true;
                reply?.close();
                stopLifecycleListening();
                releaseSlot();
                if (!controller.isClosed) {
                  controller.addError(message.error.toException());
                  controller.close();
                }
              } else {
                terminalResponseReceived = true;
                reply?.close();
                stopLifecycleListening();
                releaseSlot();
                if (!controller.isClosed) {
                  controller.addError(
                    NativeBridgeException(
                      'Unexpected engine worker response: $message',
                    ),
                  );
                  controller.close();
                }
              }
            });
            waitingForWorker = true;
            _commands.send(createMessage(streamId, reply!.sendPort));
          },
          onPause: () {
            paused = true;
          },
          onResume: () {
            paused = false;
            requestNext();
          },
          onCancel: () async {
            if (terminalResponseReceived || !slotClaimed || streamId == 0) {
              reply?.close();
              stopLifecycleListening();
              return;
            }

            terminalResponseReceived = true;
            waitingForWorker = false;
            reply?.close();
            stopLifecycleListening();

            (Object, StackTrace)? cancellationFailure;
            try {
              requestCancel();
            } catch (error, stackTrace) {
              cancellationFailure = (error, stackTrace);
            }

            final disposeReply = ReceivePort();
            try {
              _commands.send(
                _EngineWorkerStreamDispose(streamId, disposeReply.sendPort),
              );
              final message = await _lifecycle.receive(disposeReply);
              if (message is _EngineWorkerFailure) {
                throw message.error.toException();
              }
              if (message != null) {
                throw NativeBridgeException(
                  'Unexpected engine worker stream-dispose response: $message',
                );
              }
            } finally {
              releaseSlot();
            }

            final failure = cancellationFailure;
            if (failure != null) {
              Error.throwWithStackTrace(failure.$1, failure.$2);
            }
          },
        );
    return controller.stream;
  }

  Future<void> reset() async {
    if (_closed) {
      throw const ResourceDisposedException('LlamaEngine is closed.');
    }
    final reply = ReceivePort();
    _commands.send(_EngineWorkerReset(reply.sendPort));
    final message = await _lifecycle.receive(reply);
    if (message == null) {
      return;
    }
    if (message is _EngineWorkerFailure) {
      throw message.error.toException();
    }
    throw NativeBridgeException('Unexpected engine worker response: $message');
  }

  Future<void> warmUp() async {
    if (_closed) {
      throw const ResourceDisposedException('LlamaEngine is closed.');
    }
    final reply = ReceivePort();
    _commands.send(_EngineWorkerWarmUp(reply.sendPort));
    final message = await _lifecycle.receive(reply);
    if (message == null) {
      return;
    }
    if (message is _EngineWorkerFailure) {
      throw message.error.toException();
    }
    throw NativeBridgeException('Unexpected engine worker response: $message');
  }

  Future<LlamaModelInfo> modelInfo() async {
    if (_closed) {
      throw const ResourceDisposedException('LlamaEngine is closed.');
    }
    final reply = ReceivePort();
    _commands.send(_EngineWorkerModelInfo(reply.sendPort));
    final message = await _lifecycle.receive(reply);
    if (message is LlamaModelInfo) {
      return message;
    }
    if (message is _EngineWorkerFailure) {
      throw message.error.toException();
    }
    throw NativeBridgeException('Unexpected engine worker response: $message');
  }

  Future<Map<String, String>> modelMetadata() async {
    if (_closed) {
      throw const ResourceDisposedException('LlamaEngine is closed.');
    }
    final reply = ReceivePort();
    _commands.send(_EngineWorkerModelMetadata(reply.sendPort));
    final message = await _lifecycle.receive(reply);
    if (message is Map<String, String>) {
      return Map<String, String>.unmodifiable(message);
    }
    if (message is _EngineWorkerFailure) {
      throw message.error.toException();
    }
    throw NativeBridgeException('Unexpected engine worker response: $message');
  }

  Future<String> chatTemplate() async {
    if (_closed) {
      throw const ResourceDisposedException('LlamaEngine is closed.');
    }
    final reply = ReceivePort();
    _commands.send(_EngineWorkerChatTemplate(reply.sendPort));
    final message = await _lifecycle.receive(reply);
    if (message is String) {
      return message;
    }
    if (message is _EngineWorkerFailure) {
      throw message.error.toException();
    }
    throw NativeBridgeException('Unexpected engine worker response: $message');
  }

  Future<List<int>> tokenize(
    String text, {
    required bool addSpecial,
    required bool parseSpecial,
  }) async {
    if (_closed) {
      throw const ResourceDisposedException('LlamaEngine is closed.');
    }
    final reply = ReceivePort();
    _commands.send(
      _EngineWorkerTokenize(text, addSpecial, parseSpecial, reply.sendPort),
    );
    final message = await _lifecycle.receive(reply);
    if (message is List<int>) {
      return message;
    }
    if (message is _EngineWorkerFailure) {
      throw message.error.toException();
    }
    throw NativeBridgeException('Unexpected engine worker response: $message');
  }

  Future<int> countTokens(
    String text, {
    required bool addSpecial,
    required bool parseSpecial,
  }) async {
    if (_closed) {
      throw const ResourceDisposedException('LlamaEngine is closed.');
    }
    final reply = ReceivePort();
    _commands.send(
      _EngineWorkerCountTokens(text, addSpecial, parseSpecial, reply.sendPort),
    );
    final message = await _lifecycle.receive(reply);
    if (message is int) {
      return message;
    }
    if (message is _EngineWorkerFailure) {
      throw message.error.toException();
    }
    throw NativeBridgeException('Unexpected engine worker response: $message');
  }

  Future<String> detokenize(
    List<int> tokens, {
    required bool removeSpecial,
    required bool unparseSpecial,
  }) async {
    if (_closed) {
      throw const ResourceDisposedException('LlamaEngine is closed.');
    }
    final reply = ReceivePort();
    _commands.send(
      _EngineWorkerDetokenize(
        tokens,
        removeSpecial,
        unparseSpecial,
        reply.sendPort,
      ),
    );
    final message = await _lifecycle.receive(reply);
    if (message is String) {
      return message;
    }
    if (message is _EngineWorkerFailure) {
      throw message.error.toException();
    }
    throw NativeBridgeException('Unexpected engine worker response: $message');
  }

  Future<EmbeddingBatch> embedTexts(
    List<String> texts,
    EmbeddingConfig config,
  ) async {
    if (_closed) {
      throw const ResourceDisposedException('LlamaEngine is closed.');
    }
    final reply = ReceivePort();
    _commands.send(_EngineWorkerEmbedTexts(texts, config, reply.sendPort));
    final message = await _lifecycle.receive(reply);
    if (message is EmbeddingBatch) {
      return message;
    }
    if (message is _EngineWorkerFailure) {
      throw message.error.toException();
    }
    throw NativeBridgeException('Unexpected engine worker response: $message');
  }

  Future<LlamaChatTemplateCapabilities> chatTemplateCapabilities() async {
    if (_closed) {
      throw const ResourceDisposedException('LlamaEngine is closed.');
    }
    final reply = ReceivePort();
    _commands.send(_EngineWorkerChatTemplateCapabilities(reply.sendPort));
    final message = await _lifecycle.receive(reply);
    if (message is LlamaChatTemplateCapabilities) {
      return message;
    }
    if (message is _EngineWorkerFailure) {
      throw message.error.toException();
    }
    throw NativeBridgeException('Unexpected engine worker response: $message');
  }

  Future<String> formatChat(
    List<ChatMessage> messages, {
    required bool addAssistantPrompt,
    required LlamaToolCallingConfig toolCalling,
  }) async {
    if (_closed) {
      throw const ResourceDisposedException('LlamaEngine is closed.');
    }
    final reply = ReceivePort();
    _commands.send(
      _EngineWorkerFormatChat(
        messages,
        addAssistantPrompt,
        toolCalling,
        reply.sendPort,
      ),
    );
    final message = await _lifecycle.receive(reply);
    if (message is String) {
      return message;
    }
    if (message is _EngineWorkerFailure) {
      throw message.error.toException();
    }
    throw NativeBridgeException('Unexpected engine worker response: $message');
  }

  Future<int> countChatTokens(
    List<ChatMessage> messages, {
    required bool addAssistantPrompt,
    required LlamaToolCallingConfig toolCalling,
  }) async {
    if (_closed) {
      throw const ResourceDisposedException('LlamaEngine is closed.');
    }
    final reply = ReceivePort();
    _commands.send(
      _EngineWorkerCountChatTokens(
        messages,
        addAssistantPrompt,
        toolCalling,
        reply.sendPort,
      ),
    );
    final message = await _lifecycle.receive(reply);
    if (message is int) {
      return message;
    }
    if (message is _EngineWorkerFailure) {
      throw message.error.toException();
    }
    throw NativeBridgeException('Unexpected engine worker response: $message');
  }

  Future<PrefillTelemetry> prefill(
    String prompt, {
    required bool? addSpecial,
    required bool parseSpecial,
  }) async {
    if (_closed) {
      throw const ResourceDisposedException('LlamaEngine is closed.');
    }
    final reply = ReceivePort();
    _commands.send(
      _EngineWorkerPrefill(prompt, addSpecial, parseSpecial, reply.sendPort),
    );
    final message = await _lifecycle.receive(reply);
    if (message is PrefillTelemetry) {
      return message;
    }
    if (message is _EngineWorkerFailure) {
      throw message.error.toException();
    }
    throw NativeBridgeException('Unexpected engine worker response: $message');
  }

  Future<int> shiftContext({
    required int keepTokens,
    required int? discardTokens,
  }) async {
    if (_closed) {
      throw const ResourceDisposedException('LlamaEngine is closed.');
    }
    final reply = ReceivePort();
    _commands.send(
      _EngineWorkerShiftContext(keepTokens, discardTokens, reply.sendPort),
    );
    final message = await _lifecycle.receive(reply);
    if (message is int) {
      return message;
    }
    if (message is _EngineWorkerFailure) {
      throw message.error.toException();
    }
    throw NativeBridgeException('Unexpected engine worker response: $message');
  }

  Future<LlamaContextInfo> contextInfo() async {
    if (_closed) {
      throw const ResourceDisposedException('LlamaEngine is closed.');
    }
    final reply = ReceivePort();
    _commands.send(_EngineWorkerContextInfo(reply.sendPort));
    final message = await _lifecycle.receive(reply);
    if (message is LlamaContextInfo) {
      return message;
    }
    if (message is _EngineWorkerFailure) {
      throw message.error.toException();
    }
    throw NativeBridgeException('Unexpected engine worker response: $message');
  }

  Future<Uint8List> saveState() async {
    if (_closed) {
      throw const ResourceDisposedException('LlamaEngine is closed.');
    }
    final reply = ReceivePort();
    _commands.send(_EngineWorkerSaveState(reply.sendPort));
    final message = await _lifecycle.receive(reply);
    if (message is Uint8List) {
      return message;
    }
    if (message is _EngineWorkerFailure) {
      throw message.error.toException();
    }
    throw NativeBridgeException('Unexpected engine worker response: $message');
  }

  Future<void> restoreState(Uint8List state) async {
    if (_closed) {
      throw const ResourceDisposedException('LlamaEngine is closed.');
    }
    final reply = ReceivePort();
    _commands.send(_EngineWorkerRestoreState(state, reply.sendPort));
    final message = await _lifecycle.receive(reply);
    if (message == null) {
      return;
    }
    if (message is _EngineWorkerFailure) {
      throw message.error.toException();
    }
    throw NativeBridgeException('Unexpected engine worker response: $message');
  }

  Future<LoraAdapterInfo> loadLora(LoraAdapterConfig config) async {
    if (_closed) {
      throw const ResourceDisposedException('LlamaEngine is closed.');
    }
    final reply = ReceivePort();
    _commands.send(_EngineWorkerLoadLora(config, reply.sendPort));
    final message = await _lifecycle.receive(reply);
    if (message is LoraAdapterInfo) {
      return message;
    }
    if (message is _EngineWorkerFailure) {
      throw message.error.toException();
    }
    throw NativeBridgeException('Unexpected engine worker response: $message');
  }

  Future<List<LoraAdapterInfo>> loraAdapters() async {
    if (_closed) {
      throw const ResourceDisposedException('LlamaEngine is closed.');
    }
    final reply = ReceivePort();
    _commands.send(_EngineWorkerListLoras(reply.sendPort));
    final message = await _lifecycle.receive(reply);
    if (message is List<LoraAdapterInfo>) {
      return List<LoraAdapterInfo>.unmodifiable(message);
    }
    if (message is _EngineWorkerFailure) {
      throw message.error.toException();
    }
    throw NativeBridgeException('Unexpected engine worker response: $message');
  }

  Future<void> setLoraScale(int adapterId, double scale) async {
    if (_closed) {
      throw const ResourceDisposedException('LlamaEngine is closed.');
    }
    final reply = ReceivePort();
    _commands.send(_EngineWorkerSetLoraScale(adapterId, scale, reply.sendPort));
    final message = await _lifecycle.receive(reply);
    if (message == null) {
      return;
    }
    if (message is _EngineWorkerFailure) {
      throw message.error.toException();
    }
    throw NativeBridgeException('Unexpected engine worker response: $message');
  }

  Future<void> unloadLora(int adapterId) async {
    if (_closed) {
      throw const ResourceDisposedException('LlamaEngine is closed.');
    }
    final reply = ReceivePort();
    _commands.send(_EngineWorkerUnloadLora(adapterId, reply.sendPort));
    final message = await _lifecycle.receive(reply);
    if (message == null) {
      return;
    }
    if (message is _EngineWorkerFailure) {
      throw message.error.toException();
    }
    throw NativeBridgeException('Unexpected engine worker response: $message');
  }

  Future<void> close() async {
    if (_closed) {
      return;
    }
    _finalizer.detach(_finalizerDetach);
    (Object, StackTrace)? cancellationFailure;
    if (_lifecycle.failure == null) {
      try {
        _requestCancel();
      } catch (error, stackTrace) {
        cancellationFailure = (error, stackTrace);
      }
    }
    _closed = true;
    final reply = ReceivePort();
    _commands.send(_EngineWorkerClose(reply.sendPort));
    final Object? message;
    try {
      message = await _lifecycle.receive(reply);
    } finally {
      _isolate.kill(priority: Isolate.immediate);
      await _lifecycle.dispose(expected: true);
    }
    if (message is _EngineWorkerFailure) {
      throw message.error.toException();
    }
    if (message != null) {
      throw NativeBridgeException(
        'Unexpected engine worker close response: $message',
      );
    }
    final failure = cancellationFailure;
    if (failure != null) {
      Error.throwWithStackTrace(failure.$1, failure.$2);
    }
  }

  void requestCancel() {
    if (_closed || _lifecycle.failure != null) {
      return;
    }
    _requestCancel();
  }

  void _requestCancel() {
    final bridge = NativeLlamaBridge.tryOpen(_nativeLibraryPath);
    bridge?._cancelContextAddress(_contextAddress);
  }
}

({List<ChatMessage> messages, List<_NativeMediaInput> media})
_prepareMultimodalChat(List<ChatMessage> messages, String marker) {
  final formatted = <ChatMessage>[];
  final media = <_NativeMediaInput>[];
  for (final message in messages) {
    if (message.parts.isEmpty) {
      formatted.add(message);
      continue;
    }
    final content = StringBuffer();
    for (final part in message.parts) {
      switch (part) {
        case TextPart(:final text):
          content.write(text);
        case ImagePart(:final path, :final bytes):
          content.write(marker);
          media.add(
            _NativeMediaInput(
              type: llama_dart_media_type.LLAMA_DART_MEDIA_IMAGE.value,
              path: path,
              bytes: bytes,
            ),
          );
        case AudioPart(:final path, :final bytes):
          content.write(marker);
          media.add(
            _NativeMediaInput(
              type: llama_dart_media_type.LLAMA_DART_MEDIA_AUDIO.value,
              path: path,
              bytes: bytes,
            ),
          );
        case VideoPart():
          throw const UnsupportedFeatureException(
            'Video chat input is not available in mobile builds.',
          );
      }
    }
    formatted.add(ChatMessage(role: message.role, text: content.toString()));
  }
  return (
    messages: List<ChatMessage>.unmodifiable(formatted),
    media: List<_NativeMediaInput>.unmodifiable(media),
  );
}

bool _requiresChatPlan(
  List<ChatMessage> messages,
  LlamaToolCallingConfig toolCalling,
) {
  return toolCalling.tools.isNotEmpty ||
      messages.any((message) => message.hasToolData);
}

Map<String, Object?> _chatMessageToOpenAiJson(ChatMessage message) {
  final json = <String, Object?>{
    'role': _chatRoleName(message.role),
    'content': message.toolCalls.isNotEmpty && message.text.isEmpty
        ? null
        : message.text,
  };
  if (message.toolCalls.isNotEmpty) {
    json['tool_calls'] = <Map<String, Object?>>[
      for (final call in message.toolCalls) call.toOpenAiJson(),
    ];
  }
  final toolName = message.toolName;
  final toolCallId = message.toolCallId;
  if (toolName != null) {
    json['name'] = toolName;
  }
  if (toolCallId != null) {
    json['tool_call_id'] = toolCallId;
  }
  return json;
}

final class _NativeChatPlan {
  _NativeChatPlan({
    required this.json,
    required this.prompt,
    required this.parseOutput,
    required Set<String> usedToolCallIds,
  }) : usedToolCallIds = Set<String>.unmodifiable(usedToolCallIds);

  final String json;
  final String prompt;
  final bool parseOutput;
  final Set<String> usedToolCallIds;
}

final class _NativeRenderedChat {
  const _NativeRenderedChat({required this.prompt, this.plan});

  final String prompt;
  final _NativeChatPlan? plan;
}

final class _NativeMediaInput {
  const _NativeMediaInput({
    required this.type,
    required this.path,
    required this.bytes,
  });

  final int type;
  final String? path;
  final Uint8List? bytes;
}

final class _NativeEngineHandles {
  _NativeEngineHandles({
    required this.bridge,
    required this.model,
    required this.context,
  });

  final NativeLlamaBridge bridge;
  ffi.Pointer<llama_dart_model> model;
  ffi.Pointer<llama_dart_context> context;
  final Map<int, _NativeLoraAdapter> _loraAdapters =
      <int, _NativeLoraAdapter>{};
  int _nextLoraAdapterId = 1;

  void close() {
    if (context != ffi.nullptr) {
      bridge._bindings.llama_dart_context_free(context);
      bridge._throwIfLastError('Native context free');
      context = ffi.nullptr;
    }
    for (final id in _loraAdapters.keys.toList(growable: false)) {
      final adapter = _loraAdapters[id]!;
      bridge._bindings.llama_dart_lora_free(adapter.pointer);
      bridge._throwIfLastError('Native LoRA adapter free');
      _loraAdapters.remove(id);
    }
    if (model != ffi.nullptr) {
      bridge._bindings.llama_dart_model_free(model);
      model = ffi.nullptr;
      bridge._throwIfLastError('Native model free');
    }
  }

  void reset() {
    bridge._check(bridge._bindings.llama_dart_context_reset(context));
  }

  void warmUp() {
    bridge._check(bridge._bindings.llama_dart_context_warm_up(context));
  }

  LlamaModelInfo modelInfo() => bridge._readModelInfo(model);

  Map<String, String> modelMetadata() => bridge._readModelMetadata(model);

  String chatTemplate() => bridge._readChatTemplate(model);

  List<int> tokenize(
    String text, {
    required bool addSpecial,
    required bool parseSpecial,
  }) {
    return bridge._tokenize(
      model,
      text,
      addSpecial: addSpecial,
      parseSpecial: parseSpecial,
    );
  }

  int countTokens(
    String text, {
    required bool addSpecial,
    required bool parseSpecial,
  }) {
    return tokenize(
      text,
      addSpecial: addSpecial,
      parseSpecial: parseSpecial,
    ).length;
  }

  String detokenize(
    List<int> tokens, {
    required bool removeSpecial,
    required bool unparseSpecial,
  }) {
    return bridge._detokenize(
      model,
      tokens,
      removeSpecial: removeSpecial,
      unparseSpecial: unparseSpecial,
    );
  }

  LlamaChatTemplateCapabilities chatTemplateCapabilities() {
    return bridge._chatTemplateCapabilities(model);
  }

  String formatChat(
    List<ChatMessage> messages, {
    required bool addAssistantPrompt,
    required LlamaToolCallingConfig toolCalling,
  }) {
    return bridge._formatChat(
      model,
      messages,
      addAssistantPrompt: addAssistantPrompt,
      toolCalling: toolCalling,
    );
  }

  int countChatTokens(
    List<ChatMessage> messages, {
    required bool addAssistantPrompt,
    required LlamaToolCallingConfig toolCalling,
  }) {
    return bridge._countChatTokens(
      model,
      messages,
      addAssistantPrompt: addAssistantPrompt,
      toolCalling: toolCalling,
    );
  }

  int shiftContext({required int keepTokens, required int? discardTokens}) {
    final discarded = calloc<ffi.Uint32>();
    try {
      bridge._check(
        bridge._bindings.llama_dart_context_shift(
          context,
          keepTokens,
          discardTokens ?? 0,
          discarded,
        ),
      );
      return discarded.value;
    } finally {
      calloc.free(discarded);
    }
  }

  LlamaContextInfo contextInfo() {
    final info = calloc<llama_dart_context_info>();
    try {
      info.ref.struct_size = ffi.sizeOf<llama_dart_context_info>();
      bridge._check(
        bridge._bindings.llama_dart_context_get_info(context, info),
      );
      return LlamaContextInfo(
        contextSize: info.ref.context_size,
        sequenceContextSize: info.ref.sequence_context_size,
        batchSize: info.ref.batch_size,
        ubatchSize: info.ref.ubatch_size,
        maxSequences: info.ref.max_sequences,
        supportsVision: info.ref.supports_vision != 0,
        supportsAudio: info.ref.supports_audio != 0,
        gpuBackend: NativeLlamaBridge._gpuBackendFromNative(
          info.ref.gpu_backend,
        ),
        supportsContextShift: info.ref.supports_context_shift != 0,
        usedTokens: info.ref.used_tokens,
        kvCacheKeyType: NativeLlamaBridge._kvCacheTypeFromNative(
          info.ref.kv_cache_key_type,
        ),
        kvCacheValueType: NativeLlamaBridge._kvCacheTypeFromNative(
          info.ref.kv_cache_value_type,
        ),
        flashAttention: NativeLlamaBridge._flashAttentionFromNative(
          info.ref.flash_attention,
        ),
        kvCacheOffload: info.ref.kv_cache_offload != 0,
        swaFull: info.ref.swa_full != 0,
        kvUnified: info.ref.kv_unified != 0,
      );
    } finally {
      calloc.free(info);
    }
  }

  Uint8List saveState() {
    final out = calloc<llama_dart_buffer>();
    try {
      bridge._check(
        bridge._bindings.llama_dart_context_state_get(context, out),
      );
      final data = out.ref.data;
      final size = out.ref.size;
      if (data == ffi.nullptr || size == 0) {
        return Uint8List(0);
      }
      return Uint8List.fromList(data.asTypedList(size));
    } finally {
      bridge._bindings.llama_dart_buffer_free(out.ref.data);
      calloc.free(out);
    }
  }

  void restoreState(Uint8List state) {
    final statePointer = calloc<ffi.Uint8>(state.length);
    try {
      statePointer.asTypedList(state.length).setAll(0, state);
      bridge._check(
        bridge._bindings.llama_dart_context_state_set(
          context,
          statePointer,
          state.length,
        ),
      );
    } finally {
      calloc.free(statePointer);
    }
  }

  PrefillTelemetry prefill(
    String prompt, {
    required bool? addSpecial,
    required bool parseSpecial,
  }) {
    final addSpecialMode = switch (addSpecial) {
      true => _addSpecialAlways,
      false => _addSpecialNever,
      null => _addSpecialIfContextEmpty,
    };
    return _withCompletionConfig(
      prompt,
      const GenerationConfig(maxTokens: 0, temperature: 0),
      (completionConfig) {
        final out = calloc<llama_dart_buffer>();
        final stats = calloc<llama_dart_completion_stats>();
        try {
          stats.ref.struct_size = ffi.sizeOf<llama_dart_completion_stats>();
          bridge._check(
            bridge._bindings.llama_dart_context_complete(
              context,
              completionConfig,
              out,
              stats,
            ),
          );
          if (out.ref.size != 0 || stats.ref.generated_tokens != 0) {
            throw const NativeBridgeException(
              'Native prefill unexpectedly generated output.',
            );
          }
          return _prefillTelemetryFromStats(stats.ref);
        } finally {
          bridge._bindings.llama_dart_buffer_free(out.ref.data);
          calloc.free(stats);
          calloc.free(out);
        }
      },
      addSpecialMode: addSpecialMode,
      parseSpecial: parseSpecial,
    );
  }

  R _withCompletionConfig<R>(
    String prompt,
    GenerationConfig config,
    R Function(ffi.Pointer<llama_dart_completion_config> config) run, {
    List<_NativeMediaInput> media = const <_NativeMediaInput>[],
    _NativeChatPlan? chatPlan,
    int addSpecialMode = _addSpecialIfContextEmpty,
    bool parseSpecial = false,
    bool manageRequestLoras = true,
  }) {
    final promptBytes = utf8.encode(prompt);
    final grammarBytes = config.grammar == null
        ? const <int>[]
        : utf8.encode(config.grammar!);
    final grammarRootBytes = grammarBytes.isEmpty
        ? const <int>[]
        : utf8.encode(config.grammarRoot);
    final jsonSchemaBytes = config.jsonSchema == null
        ? const <int>[]
        : utf8.encode(jsonEncode(config.jsonSchema));
    final chatPlanBytes = chatPlan == null
        ? const <int>[]
        : utf8.encode(chatPlan.json);
    final stopBytes = <List<int>>[
      for (final marker in config.stop) utf8.encode(marker),
    ];
    final promptPointer = calloc<ffi.Uint8>(promptBytes.length);
    ffi.Pointer<ffi.Uint8> grammarPointer = ffi.nullptr;
    ffi.Pointer<ffi.Uint8> grammarRootPointer = ffi.nullptr;
    ffi.Pointer<ffi.Uint8> jsonSchemaPointer = ffi.nullptr;
    ffi.Pointer<ffi.Uint8> chatPlanPointer = ffi.nullptr;
    ffi.Pointer<llama_dart_string_view> stopSequences = ffi.nullptr;
    ffi.Pointer<ffi.Int32> stopTokens = ffi.nullptr;
    ffi.Pointer<llama_dart_media_input> mediaInputs = ffi.nullptr;
    final stopPointers = <ffi.Pointer<ffi.Uint8>>[];
    final mediaPointers = <ffi.Pointer<ffi.Uint8>>[];
    final completionConfig = calloc<llama_dart_completion_config>();
    final requestLoraScales = manageRequestLoras ? config.loraScales : null;
    var restoreLoras = false;
    try {
      if (requestLoraScales != null) {
        restoreLoras = true;
        _applyLoras(requestLoraScales);
      }
      promptPointer.asTypedList(promptBytes.length).setAll(0, promptBytes);
      if (grammarBytes.isNotEmpty) {
        grammarPointer = calloc<ffi.Uint8>(grammarBytes.length);
        grammarPointer.asTypedList(grammarBytes.length).setAll(0, grammarBytes);
        grammarRootPointer = calloc<ffi.Uint8>(grammarRootBytes.length);
        grammarRootPointer
            .asTypedList(grammarRootBytes.length)
            .setAll(0, grammarRootBytes);
      }
      if (jsonSchemaBytes.isNotEmpty) {
        jsonSchemaPointer = calloc<ffi.Uint8>(jsonSchemaBytes.length);
        jsonSchemaPointer
            .asTypedList(jsonSchemaBytes.length)
            .setAll(0, jsonSchemaBytes);
      }
      if (chatPlanBytes.isNotEmpty) {
        chatPlanPointer = calloc<ffi.Uint8>(chatPlanBytes.length);
        chatPlanPointer
            .asTypedList(chatPlanBytes.length)
            .setAll(0, chatPlanBytes);
      }
      if (stopBytes.isNotEmpty) {
        stopSequences = calloc<llama_dart_string_view>(stopBytes.length);
        for (var i = 0; i < stopBytes.length; i += 1) {
          final bytes = stopBytes[i];
          final pointer = calloc<ffi.Uint8>(bytes.length);
          pointer.asTypedList(bytes.length).setAll(0, bytes);
          stopPointers.add(pointer);
          stopSequences[i]
            ..data = pointer
            ..size = bytes.length;
        }
      }
      if (config.stopTokens.isNotEmpty) {
        stopTokens = calloc<ffi.Int32>(config.stopTokens.length);
        stopTokens
            .asTypedList(config.stopTokens.length)
            .setAll(0, config.stopTokens);
      }
      if (media.isNotEmpty) {
        mediaInputs = calloc<llama_dart_media_input>(media.length);
        for (var i = 0; i < media.length; i += 1) {
          final input = media[i];
          ffi.Pointer<ffi.Uint8> pathPointer = ffi.nullptr;
          ffi.Pointer<ffi.Uint8> contentPointer = ffi.nullptr;
          var pathSize = 0;
          var contentSize = 0;
          final path = input.path;
          if (path != null) {
            final bytes = utf8.encode(path);
            pathPointer = calloc<ffi.Uint8>(bytes.length);
            pathPointer.asTypedList(bytes.length).setAll(0, bytes);
            mediaPointers.add(pathPointer);
            pathSize = bytes.length;
          } else {
            final bytes = input.bytes!;
            contentPointer = calloc<ffi.Uint8>(bytes.length);
            contentPointer.asTypedList(bytes.length).setAll(0, bytes);
            mediaPointers.add(contentPointer);
            contentSize = bytes.length;
          }
          mediaInputs[i]
            ..struct_size = ffi.sizeOf<llama_dart_media_input>()
            ..type = input.type
            ..path_data = pathPointer
            ..path_size = pathSize
            ..content_data = contentPointer
            ..content_size = contentSize;
        }
      }
      completionConfig.ref
        ..struct_size = ffi.sizeOf<llama_dart_completion_config>()
        ..prompt_data = promptPointer
        ..prompt_size = promptBytes.length
        ..max_tokens = config.maxTokens
        ..temperature = config.temperature
        ..top_k = config.topK
        ..top_p = config.topP
        ..min_p = config.minP
        ..typical_p = config.typicalP
        ..penalty_last_n = config.penaltyLastN
        ..repeat_penalty = config.repeatPenalty
        ..frequency_penalty = config.frequencyPenalty
        ..presence_penalty = config.presencePenalty
        ..mirostat = NativeLlamaBridge._mirostatMode(config.mirostat)
        ..mirostat_tau = config.mirostatTau
        ..mirostat_eta = config.mirostatEta
        ..grammar_data = grammarPointer
        ..grammar_size = grammarBytes.length
        ..grammar_root_data = grammarRootPointer
        ..grammar_root_size = grammarRootBytes.length
        ..stop_sequences = stopSequences
        ..stop_sequence_count = stopBytes.length
        ..seed = config.seed ?? 0xFFFFFFFF
        ..add_special = addSpecialMode
        ..parse_special = parseSpecial ? 1 : 0
        ..media_inputs = mediaInputs
        ..media_input_count = media.length
        ..json_schema_data = jsonSchemaPointer
        ..json_schema_size = jsonSchemaBytes.length
        ..chat_plan_data = chatPlanPointer
        ..chat_plan_size = chatPlanBytes.length
        ..stop_tokens = stopTokens
        ..stop_token_count = config.stopTokens.length;
      return run(completionConfig);
    } finally {
      calloc.free(completionConfig);
      if (grammarRootPointer != ffi.nullptr) {
        calloc.free(grammarRootPointer);
      }
      if (grammarPointer != ffi.nullptr) {
        calloc.free(grammarPointer);
      }
      if (jsonSchemaPointer != ffi.nullptr) {
        calloc.free(jsonSchemaPointer);
      }
      if (chatPlanPointer != ffi.nullptr) {
        calloc.free(chatPlanPointer);
      }
      for (final pointer in stopPointers) {
        calloc.free(pointer);
      }
      if (stopSequences != ffi.nullptr) {
        calloc.free(stopSequences);
      }
      if (stopTokens != ffi.nullptr) {
        calloc.free(stopTokens);
      }
      for (final pointer in mediaPointers) {
        calloc.free(pointer);
      }
      if (mediaInputs != ffi.nullptr) {
        calloc.free(mediaInputs);
      }
      calloc.free(promptPointer);
      if (restoreLoras) {
        _applyLoras();
      }
    }
  }

  ({String text, GenerationTelemetry telemetry, ChatMessage? assistantMessage})
  complete(
    String prompt,
    GenerationConfig config, {
    List<_NativeMediaInput> media = const <_NativeMediaInput>[],
    _NativeChatPlan? chatPlan,
    bool parseSpecial = false,
  }) {
    return _withCompletionConfig(
      prompt,
      config,
      (completionConfig) {
        final out = calloc<llama_dart_buffer>();
        final stats = calloc<llama_dart_completion_stats>();
        try {
          stats.ref.struct_size = ffi.sizeOf<llama_dart_completion_stats>();
          bridge._check(
            bridge._bindings.llama_dart_context_complete(
              context,
              completionConfig,
              out,
              stats,
            ),
          );

          final data = out.ref.data;
          final size = out.ref.size;
          final telemetry = _telemetryFromStats(stats.ref);
          final text = data == ffi.nullptr || size == 0
              ? ''
              : utf8.decode(data.asTypedList(size), allowMalformed: true);
          final assistantMessage = chatPlan == null || !chatPlan.parseOutput
              ? null
              : _parseTerminalChatOutput(
                  bridge,
                  chatPlan,
                  text,
                  config.toolCalling,
                  telemetry.stopReason,
                );
          return (
            text: text,
            telemetry: telemetry,
            assistantMessage: assistantMessage,
          );
        } finally {
          bridge._bindings.llama_dart_buffer_free(out.ref.data);
          calloc.free(stats);
          calloc.free(out);
        }
      },
      media: media,
      chatPlan: chatPlan,
      parseSpecial: parseSpecial,
    );
  }

  ({String text, GenerationTelemetry telemetry, ChatMessage? assistantMessage})
  completeChat(List<ChatMessage> messages, GenerationConfig config) {
    final prepared = _prepareMultimodalChat(messages, bridge._multimodalMarker);
    final rendered = bridge._renderPreparedChat(
      model,
      prepared.messages,
      addAssistantPrompt: true,
      toolCalling: config.toolCalling,
      grammar: config.grammar,
      jsonSchema: config.jsonSchema,
      enableThinking: config.enableThinking,
    );
    return complete(
      rendered.prompt,
      config,
      media: prepared.media,
      chatPlan: rendered.plan,
      parseSpecial: true,
    );
  }

  _NativeStreamingGeneration startCompletionStream(
    String prompt,
    GenerationConfig config, {
    List<_NativeMediaInput> media = const <_NativeMediaInput>[],
    _NativeChatPlan? chatPlan,
    bool parseSpecial = false,
  }) {
    final restoreLoras = config.loraScales != null;
    if (restoreLoras) {
      _applyLoras(config.loraScales!);
    }
    try {
      return _withCompletionConfig(
        prompt,
        config,
        (completionConfig) {
          final generationOut = calloc<ffi.Pointer<llama_dart_generation>>();
          ffi.Pointer<llama_dart_generation> generation = ffi.nullptr;
          try {
            bridge._check(
              bridge._bindings.llama_dart_generation_start(
                context,
                completionConfig,
                generationOut,
              ),
            );
            generation = generationOut.value;
            if (generation == ffi.nullptr) {
              throw const GenerationException(
                'Native bridge returned a null generation.',
              );
            }
            return _NativeStreamingGeneration(
              bridge: bridge,
              generation: generation,
              chunkTokens: config.streamChunkTokens,
              chatPlan: chatPlan,
              toolCalling: config.toolCalling,
              onClose: restoreLoras ? _applyLoras : null,
            );
          } catch (_) {
            if (generation != ffi.nullptr) {
              bridge._bindings.llama_dart_generation_free(generation);
            }
            rethrow;
          } finally {
            calloc.free(generationOut);
          }
        },
        media: media,
        chatPlan: chatPlan,
        parseSpecial: parseSpecial,
        manageRequestLoras: false,
      );
    } catch (_) {
      if (restoreLoras) {
        _applyLoras();
      }
      rethrow;
    }
  }

  _NativeStreamingGeneration startChatStream(
    List<ChatMessage> messages,
    GenerationConfig config,
  ) {
    final prepared = _prepareMultimodalChat(messages, bridge._multimodalMarker);
    final rendered = bridge._renderPreparedChat(
      model,
      prepared.messages,
      addAssistantPrompt: true,
      toolCalling: config.toolCalling,
      grammar: config.grammar,
      jsonSchema: config.jsonSchema,
      enableThinking: config.enableThinking,
    );
    return startCompletionStream(
      rendered.prompt,
      config,
      media: prepared.media,
      chatPlan: rendered.plan,
      parseSpecial: true,
    );
  }

  Float32List embedText(String text, EmbeddingConfig config) {
    final textBytes = utf8.encode(text);
    final textPointer = calloc<ffi.Uint8>(textBytes.length);
    final embeddingConfig = calloc<llama_dart_embedding_config>();
    final out = calloc<llama_dart_float_buffer>();
    try {
      textPointer.asTypedList(textBytes.length).setAll(0, textBytes);
      embeddingConfig.ref
        ..struct_size = ffi.sizeOf<llama_dart_embedding_config>()
        ..text_data = textPointer
        ..text_size = textBytes.length
        ..add_special = config.addSpecial ? 1 : 0
        ..parse_special = config.parseSpecial ? 1 : 0;

      bridge._check(
        bridge._bindings.llama_dart_context_embed(
          context,
          embeddingConfig,
          out,
        ),
      );

      final data = out.ref.data;
      final length = out.ref.length;
      if (data == ffi.nullptr || length == 0) {
        return Float32List(0);
      }
      return Float32List.fromList(data.asTypedList(length));
    } finally {
      bridge._bindings.llama_dart_float_buffer_free(out.ref.data);
      calloc.free(out);
      calloc.free(embeddingConfig);
      calloc.free(textPointer);
    }
  }

  double rerank(String query, String document, RerankingConfig config) {
    final queryBytes = utf8.encode(query);
    final documentBytes = utf8.encode(document);
    final queryPointer = calloc<ffi.Uint8>(queryBytes.length);
    final documentPointer = calloc<ffi.Uint8>(documentBytes.length);
    final rerankConfig = calloc<llama_dart_rerank_config>();
    final outScore = calloc<ffi.Float>();
    try {
      queryPointer.asTypedList(queryBytes.length).setAll(0, queryBytes);
      documentPointer
          .asTypedList(documentBytes.length)
          .setAll(0, documentBytes);
      rerankConfig.ref
        ..struct_size = ffi.sizeOf<llama_dart_rerank_config>()
        ..query_data = queryPointer
        ..query_size = queryBytes.length
        ..document_data = documentPointer
        ..document_size = documentBytes.length
        ..add_special = config.addSpecial ? 1 : 0
        ..parse_special = config.parseSpecial ? 1 : 0;

      bridge._check(
        bridge._bindings.llama_dart_context_rerank(
          context,
          rerankConfig,
          outScore,
        ),
      );
      return outScore.value;
    } finally {
      calloc.free(outScore);
      calloc.free(rerankConfig);
      calloc.free(documentPointer);
      calloc.free(queryPointer);
    }
  }

  LoraAdapterInfo loadLora(LoraAdapterConfig config) {
    final pathBytes = utf8.encode(config.path);
    final pathPointer = calloc<ffi.Uint8>(pathBytes.length);
    final loadConfig = calloc<llama_dart_lora_load_config>();
    final outAdapter = calloc<ffi.Pointer<llama_dart_lora_adapter>>();
    try {
      pathPointer.asTypedList(pathBytes.length).setAll(0, pathBytes);
      loadConfig.ref
        ..struct_size = ffi.sizeOf<llama_dart_lora_load_config>()
        ..path_data = pathPointer
        ..path_size = pathBytes.length;

      bridge._check(
        bridge._bindings.llama_dart_lora_load(model, loadConfig, outAdapter),
      );
      final pointer = outAdapter.value;
      if (pointer == ffi.nullptr) {
        throw const LoraException(
          'Native bridge returned a null LoRA adapter.',
        );
      }

      final id = _nextLoraAdapterId;
      _nextLoraAdapterId += 1;
      final adapter = _NativeLoraAdapter(
        id: id,
        path: config.path,
        scale: config.scale,
        pointer: pointer,
      );
      _loraAdapters[id] = adapter;
      try {
        _applyLoras();
      } catch (_) {
        _loraAdapters.remove(id);
        bridge._bindings.llama_dart_lora_free(pointer);
        rethrow;
      }
      return adapter.info;
    } finally {
      calloc.free(outAdapter);
      calloc.free(loadConfig);
      calloc.free(pathPointer);
    }
  }

  List<LoraAdapterInfo> loraAdapters() {
    return List<LoraAdapterInfo>.unmodifiable(
      _loraAdapters.values.map((adapter) => adapter.info),
    );
  }

  void setLoraScale(int adapterId, double scale) {
    final adapter = _loraAdapters[adapterId];
    if (adapter == null) {
      throw LoraException('LoRA adapter $adapterId is not loaded.');
    }

    final previousScale = adapter.scale;
    adapter.scale = scale;
    try {
      _applyLoras();
    } catch (_) {
      adapter.scale = previousScale;
      rethrow;
    }
  }

  void unloadLora(int adapterId) {
    final adapter = _loraAdapters.remove(adapterId);
    if (adapter == null) {
      throw LoraException('LoRA adapter $adapterId is not loaded.');
    }

    try {
      _applyLoras();
    } catch (_) {
      _loraAdapters[adapterId] = adapter;
      rethrow;
    }
    bridge._bindings.llama_dart_lora_free(adapter.pointer);
    try {
      bridge._throwIfLastError('Native LoRA adapter free');
    } catch (_) {
      _loraAdapters[adapterId] = adapter;
      rethrow;
    }
  }

  void _applyLoras([Map<int, double>? requestedScales]) {
    final selections = <({_NativeLoraAdapter adapter, double scale})>[];
    if (requestedScales == null) {
      for (final adapter in _loraAdapters.values) {
        selections.add((adapter: adapter, scale: adapter.scale));
      }
    } else {
      for (final entry in requestedScales.entries) {
        final adapter = _loraAdapters[entry.key];
        if (adapter == null) {
          throw LoraException('LoRA adapter ${entry.key} is not loaded.');
        }
        selections.add((adapter: adapter, scale: entry.value));
      }
    }
    selections.sort((a, b) => a.adapter.id.compareTo(b.adapter.id));
    if (selections.isEmpty) {
      bridge._check(
        bridge._bindings.llama_dart_context_set_lora_adapters(
          context,
          ffi.nullptr,
          ffi.nullptr,
          0,
        ),
      );
      return;
    }

    final adapterPointers = calloc<ffi.Pointer<llama_dart_lora_adapter>>(
      selections.length,
    );
    final nativeScales = calloc<ffi.Float>(selections.length);
    try {
      for (var i = 0; i < selections.length; i += 1) {
        adapterPointers[i] = selections[i].adapter.pointer;
        nativeScales[i] = selections[i].scale;
      }
      bridge._check(
        bridge._bindings.llama_dart_context_set_lora_adapters(
          context,
          adapterPointers,
          nativeScales,
          selections.length,
        ),
      );
    } finally {
      calloc.free(nativeScales);
      calloc.free(adapterPointers);
    }
  }
}

final class _NativeStreamingGeneration {
  _NativeStreamingGeneration({
    required this.bridge,
    required ffi.Pointer<llama_dart_generation> generation,
    required this.chunkTokens,
    required this.chatPlan,
    required this.toolCalling,
    required void Function()? onClose,
  }) : _generation = generation,
       _onClose = onClose;

  final NativeLlamaBridge bridge;
  final int chunkTokens;
  final _NativeChatPlan? chatPlan;
  final LlamaToolCallingConfig toolCalling;
  final void Function()? _onClose;
  final ffi.Pointer<llama_dart_buffer> _out = calloc<llama_dart_buffer>();
  final ffi.Pointer<llama_dart_completion_stats> _stats =
      calloc<llama_dart_completion_stats>();
  final ffi.Pointer<ffi.Uint8> _done = calloc<ffi.Uint8>();
  final _Utf8ChunkDecoder _decoder = _Utf8ChunkDecoder();
  final StringBuffer _generated = StringBuffer();
  ffi.Pointer<llama_dart_generation> _generation;
  int _generatedTokens = 0;
  bool _terminal = false;
  bool _closed = false;

  ({
    String text,
    bool isDone,
    GenerationTelemetry? telemetry,
    ChatMessage? assistantMessage,
  })
  next() {
    if (_closed || _generation == ffi.nullptr) {
      throw const ResourceDisposedException('Native generation is closed.');
    }
    if (_terminal) {
      throw const GenerationException('Native generation is already done.');
    }

    final initialGeneratedTokens = _generatedTokens;
    final chunk = StringBuffer();
    var nativeSteps = 0;
    while (nativeSteps < chunkTokens) {
      _stats.ref.struct_size = ffi.sizeOf<llama_dart_completion_stats>();
      _done.value = 0;
      bridge._check(
        bridge._bindings.llama_dart_generation_next(
          _generation,
          _out,
          _stats,
          _done,
        ),
      );
      var text = '';
      try {
        final data = _out.ref.data;
        final size = _out.ref.size;
        if (data != ffi.nullptr && size > 0) {
          text = _decoder.add(data.asTypedList(size));
        }
      } finally {
        bridge._bindings.llama_dart_buffer_free(_out.ref.data);
        _out.ref.data = ffi.nullptr;
        _out.ref.size = 0;
      }

      nativeSteps += 1;
      _generatedTokens = _stats.ref.generated_tokens;
      final isDone = _done.value != 0;
      if (isDone) {
        text += _decoder.close();
      }
      if (text.isNotEmpty) {
        chunk.write(text);
        _generated.write(text);
      }
      if (isDone) {
        _terminal = true;
        final telemetry = _telemetryFromStats(_stats.ref);
        final assistantMessage = chatPlan == null || !chatPlan!.parseOutput
            ? null
            : _parseTerminalChatOutput(
                bridge,
                chatPlan!,
                _generated.toString(),
                toolCalling,
                telemetry.stopReason,
              );
        return (
          text: chunk.toString(),
          isDone: true,
          telemetry: telemetry,
          assistantMessage: assistantMessage,
        );
      }
      if (_generatedTokens - initialGeneratedTokens >= chunkTokens) {
        break;
      }
    }
    return (
      text: chunk.toString(),
      isDone: false,
      telemetry: null,
      assistantMessage: null,
    );
  }

  void close() {
    if (_closed) {
      return;
    }
    _closed = true;
    try {
      bridge._bindings.llama_dart_buffer_free(_out.ref.data);
      _out.ref.data = ffi.nullptr;
      _out.ref.size = 0;
      if (_generation != ffi.nullptr) {
        bridge._bindings.llama_dart_generation_free(_generation);
        _generation = ffi.nullptr;
      }
      calloc.free(_done);
      calloc.free(_stats);
      calloc.free(_out);
    } finally {
      _onClose?.call();
    }
  }
}

final class _NativeLoraAdapter {
  _NativeLoraAdapter({
    required this.id,
    required this.path,
    required this.scale,
    required this.pointer,
  });

  final int id;
  final String path;
  double scale;
  final ffi.Pointer<llama_dart_lora_adapter> pointer;

  LoraAdapterInfo get info {
    return LoraAdapterInfo(id: id, path: path, scale: scale);
  }
}

final class _EngineFinalizerToken {
  const _EngineFinalizerToken(
    this.commands,
    this.nativeLibraryPath,
    this.contextAddress,
  );

  final SendPort commands;
  final String? nativeLibraryPath;
  final int contextAddress;

  void release() {
    try {
      NativeLlamaBridge.tryOpen(
        nativeLibraryPath,
      )?._cancelContextAddress(contextAddress);
    } catch (_) {
      // Finalizers cannot report cleanup failures to an owning caller.
    }
    try {
      commands.send(const _EngineWorkerFinalize());
    } catch (_) {
      // The worker may already have terminated during process shutdown.
    }
  }
}

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

final class _EngineWorkerReset {
  const _EngineWorkerReset(this.reply);

  final SendPort reply;
}

final class _EngineWorkerWarmUp {
  const _EngineWorkerWarmUp(this.reply);

  final SendPort reply;
}

final class _EngineWorkerModelInfo {
  const _EngineWorkerModelInfo(this.reply);

  final SendPort reply;
}

final class _EngineWorkerModelMetadata {
  const _EngineWorkerModelMetadata(this.reply);

  final SendPort reply;
}

final class _EngineWorkerChatTemplate {
  const _EngineWorkerChatTemplate(this.reply);

  final SendPort reply;
}

final class _EngineWorkerTokenize {
  const _EngineWorkerTokenize(
    this.text,
    this.addSpecial,
    this.parseSpecial,
    this.reply,
  );

  final String text;
  final bool addSpecial;
  final bool parseSpecial;
  final SendPort reply;
}

final class _EngineWorkerCountTokens {
  const _EngineWorkerCountTokens(
    this.text,
    this.addSpecial,
    this.parseSpecial,
    this.reply,
  );

  final String text;
  final bool addSpecial;
  final bool parseSpecial;
  final SendPort reply;
}

final class _EngineWorkerDetokenize {
  const _EngineWorkerDetokenize(
    this.tokens,
    this.removeSpecial,
    this.unparseSpecial,
    this.reply,
  );

  final List<int> tokens;
  final bool removeSpecial;
  final bool unparseSpecial;
  final SendPort reply;
}

final class _EngineWorkerEmbedTexts {
  const _EngineWorkerEmbedTexts(this.texts, this.config, this.reply);

  final List<String> texts;
  final EmbeddingConfig config;
  final SendPort reply;
}

final class _EngineWorkerChatTemplateCapabilities {
  const _EngineWorkerChatTemplateCapabilities(this.reply);

  final SendPort reply;
}

final class _EngineWorkerFormatChat {
  const _EngineWorkerFormatChat(
    this.messages,
    this.addAssistantPrompt,
    this.toolCalling,
    this.reply,
  );

  final List<ChatMessage> messages;
  final bool addAssistantPrompt;
  final LlamaToolCallingConfig toolCalling;
  final SendPort reply;
}

final class _EngineWorkerCountChatTokens {
  const _EngineWorkerCountChatTokens(
    this.messages,
    this.addAssistantPrompt,
    this.toolCalling,
    this.reply,
  );

  final List<ChatMessage> messages;
  final bool addAssistantPrompt;
  final LlamaToolCallingConfig toolCalling;
  final SendPort reply;
}

final class _EngineWorkerPrefill {
  const _EngineWorkerPrefill(
    this.prompt,
    this.addSpecial,
    this.parseSpecial,
    this.reply,
  );

  final String prompt;
  final bool? addSpecial;
  final bool parseSpecial;
  final SendPort reply;
}

final class _EngineWorkerShiftContext {
  const _EngineWorkerShiftContext(
    this.keepTokens,
    this.discardTokens,
    this.reply,
  );

  final int keepTokens;
  final int? discardTokens;
  final SendPort reply;
}

final class _EngineWorkerContextInfo {
  const _EngineWorkerContextInfo(this.reply);

  final SendPort reply;
}

final class _EngineWorkerSaveState {
  const _EngineWorkerSaveState(this.reply);

  final SendPort reply;
}

final class _EngineWorkerRestoreState {
  const _EngineWorkerRestoreState(this.state, this.reply);

  final Uint8List state;
  final SendPort reply;
}

final class _EngineWorkerComplete {
  const _EngineWorkerComplete(this.messages, this.config, this.reply);

  final List<ChatMessage> messages;
  final GenerationConfig config;
  final SendPort reply;
}

final class _EngineWorkerCompletePrompt {
  const _EngineWorkerCompletePrompt(this.prompt, this.config, this.reply);

  final String prompt;
  final GenerationConfig config;
  final SendPort reply;
}

final class _EngineWorkerStreamComplete {
  const _EngineWorkerStreamComplete(
    this.id,
    this.messages,
    this.config,
    this.reply,
  );

  final int id;
  final List<ChatMessage> messages;
  final GenerationConfig config;
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

final class _EngineWorkerLoadLora {
  const _EngineWorkerLoadLora(this.config, this.reply);

  final LoraAdapterConfig config;
  final SendPort reply;
}

final class _EngineWorkerListLoras {
  const _EngineWorkerListLoras(this.reply);

  final SendPort reply;
}

final class _EngineWorkerSetLoraScale {
  const _EngineWorkerSetLoraScale(this.adapterId, this.scale, this.reply);

  final int adapterId;
  final double scale;
  final SendPort reply;
}

final class _EngineWorkerUnloadLora {
  const _EngineWorkerUnloadLora(this.adapterId, this.reply);

  final int adapterId;
  final SendPort reply;
}

final class _EngineWorkerText {
  const _EngineWorkerText(this.text, this.telemetry, this.assistantMessage);

  final String text;
  final GenerationTelemetry telemetry;
  final ChatMessage? assistantMessage;
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
    _EngineWorkerReset(:final reply) => reply,
    _EngineWorkerWarmUp(:final reply) => reply,
    _EngineWorkerModelInfo(:final reply) => reply,
    _EngineWorkerModelMetadata(:final reply) => reply,
    _EngineWorkerChatTemplate(:final reply) => reply,
    _EngineWorkerTokenize(:final reply) => reply,
    _EngineWorkerCountTokens(:final reply) => reply,
    _EngineWorkerDetokenize(:final reply) => reply,
    _EngineWorkerEmbedTexts(:final reply) => reply,
    _EngineWorkerChatTemplateCapabilities(:final reply) => reply,
    _EngineWorkerFormatChat(:final reply) => reply,
    _EngineWorkerCountChatTokens(:final reply) => reply,
    _EngineWorkerPrefill(:final reply) => reply,
    _EngineWorkerShiftContext(:final reply) => reply,
    _EngineWorkerContextInfo(:final reply) => reply,
    _EngineWorkerSaveState(:final reply) => reply,
    _EngineWorkerRestoreState(:final reply) => reply,
    _EngineWorkerComplete(:final reply) => reply,
    _EngineWorkerCompletePrompt(:final reply) => reply,
    _EngineWorkerStreamComplete(:final reply) => reply,
    _EngineWorkerStreamPrompt(:final reply) => reply,
    _EngineWorkerLoadLora(:final reply) => reply,
    _EngineWorkerListLoras(:final reply) => reply,
    _EngineWorkerSetLoraScale(:final reply) => reply,
    _EngineWorkerUnloadLora(:final reply) => reply,
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

PrefillTelemetry _prefillTelemetryFromStats(llama_dart_completion_stats stats) {
  return PrefillTelemetry(
    promptTokens: stats.prompt_tokens,
    promptEvalMs: stats.prompt_eval_ms,
    totalMs: stats.total_ms,
  );
}

GenerationTelemetry _telemetryFromStats(llama_dart_completion_stats stats) {
  return GenerationTelemetry(
    promptTokens: stats.prompt_tokens,
    generatedTokens: stats.generated_tokens,
    promptEvalMs: stats.prompt_eval_ms,
    decodeMs: stats.decode_ms,
    totalMs: stats.total_ms,
    timeToFirstTokenMs: stats.time_to_first_token_ms,
    speculativeDraftTokens: stats.speculative_draft_tokens,
    speculativeAcceptedTokens: stats.speculative_accepted_tokens,
    speculativeDraftMs: stats.speculative_draft_ms,
    speculativeVerifyMs: stats.speculative_verify_ms,
    stopReason: switch (stats.stop_reason) {
      1 => GenerationStopReason.endOfGeneration,
      2 => GenerationStopReason.stopSequence,
      3 => GenerationStopReason.stopToken,
      4 => GenerationStopReason.maxTokens,
      _ => GenerationStopReason.unknown,
    },
  );
}

ChatMessage _parseTerminalChatOutput(
  NativeLlamaBridge bridge,
  _NativeChatPlan plan,
  String output,
  LlamaToolCallingConfig toolCalling,
  GenerationStopReason stopReason,
) {
  try {
    return bridge._parseChatOutput(plan, output, toolCalling);
  } on GenerationException catch (error) {
    if (stopReason == GenerationStopReason.maxTokens &&
        error.message.startsWith('failed to parse chat output:')) {
      throw GenerationException(
        'Maximum token limit was reached before the chat or tool call was complete.',
        cause: error,
      );
    }
    rethrow;
  }
}

final class _Utf8ChunkDecoder {
  _Utf8ChunkDecoder();

  final _StringChunkSink _text = _StringChunkSink();

  late final ByteConversionSink _sink = const Utf8Decoder(
    allowMalformed: true,
  ).startChunkedConversion(StringConversionSink.fromStringSink(_text));

  String add(Uint8List bytes) {
    _sink.add(bytes);
    return _text.take();
  }

  String close() {
    _sink.close();
    return _text.take();
  }
}

final class _StringChunkSink implements StringSink {
  final List<String> _chunks = <String>[];

  @override
  void write(Object? object) {
    _chunks.add(object.toString());
  }

  @override
  void writeAll(Iterable<dynamic> objects, [String separator = '']) {
    _chunks.add(objects.join(separator));
  }

  @override
  void writeCharCode(int charCode) {
    _chunks.add(String.fromCharCode(charCode));
  }

  @override
  void writeln([Object? object = '']) {
    _chunks.add('$object\n');
  }

  String take() {
    final text = _chunks.join();
    _chunks.clear();
    return text;
  }
}

Float32List _normalize(Float32List values) {
  var sumSquares = 0.0;
  for (final value in values) {
    sumSquares += value * value;
  }
  if (sumSquares == 0) {
    return values;
  }
  final scale = 1 / math.sqrt(sumSquares);
  for (var i = 0; i < values.length; i += 1) {
    values[i] *= scale;
  }
  return values;
}

UnsupportedFeatureException _nativeBridgeUnavailable(String? path) {
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
  NativeLlamaBridge._validateLibraryPathText(path, 'FLLAMER_NATIVE_LIBRARY');
  return path;
}

String _decodeModelUtf8(
  ffi.Pointer<ffi.Char> pointer,
  int length,
  String field,
) {
  try {
    return utf8.decode(pointer.cast<ffi.Uint8>().asTypedList(length));
  } on FormatException catch (error) {
    throw ModelLoadException('Model $field is not valid UTF-8.', cause: error);
  }
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

void _engineWorkerMain(_EngineWorkerStart start) {
  _NativeEngineHandles? handles;
  final commands = ReceivePort();
  try {
    final bridge = NativeLlamaBridge.tryOpen(start.config.nativeLibraryPath);
    if (bridge == null) {
      throw _nativeBridgeUnavailable(start.config.nativeLibraryPath);
    }
    handles = bridge._openEngine(
      start.config,
      embeddings: start.embeddings,
      pooling: start.pooling,
    );
    start.reply.send(
      _EngineWorkerReady(commands.sendPort, handles.context.address),
    );
  } catch (error) {
    start.reply.send(_EngineWorkerFailure(_NativeError.from(error)));
    return;
  }

  _NativeStreamingGeneration? streamingGeneration;
  SendPort? streamingReply;
  var streamingId = 0;
  late void Function(Object? message) handleMessage;

  Object? closeStreamingGeneration({bool resetContext = false}) {
    final generation = streamingGeneration;
    streamingGeneration = null;
    streamingReply = null;
    streamingId = 0;
    Object? cleanupError;
    if (generation != null) {
      try {
        generation.close();
      } catch (error) {
        cleanupError = error;
      }
    }
    if (resetContext) {
      try {
        final active = handles;
        if (active == null) {
          throw const ResourceDisposedException('LlamaEngine is closed.');
        }
        active.reset();
      } catch (error) {
        cleanupError ??= error;
      }
    }
    return cleanupError;
  }

  void stepStreamingGeneration(int id) {
    final generation = streamingGeneration;
    final reply = streamingReply;
    if (generation == null || reply == null || streamingId != id) {
      return;
    }
    try {
      final chunk = generation.next();
      if (chunk.isDone) {
        Object? closeError = closeStreamingGeneration();
        if (closeError != null) {
          closeError =
              closeStreamingGeneration(resetContext: true) ?? closeError;
          reply.send(_EngineWorkerFailure(_NativeError.from(closeError)));
        } else {
          reply.send(
            _EngineWorkerStreamChunk(
              chunk.text,
              true,
              chunk.telemetry,
              chunk.assistantMessage,
            ),
          );
        }
      } else {
        reply.send(_EngineWorkerStreamChunk(chunk.text, false, null, null));
      }
    } catch (error) {
      final closeError = closeStreamingGeneration(resetContext: true);
      reply.send(_EngineWorkerFailure(_NativeError.from(closeError ?? error)));
    }
  }

  handleMessage = (message) {
    if (message is _EngineWorkerFinalize) {
      closeStreamingGeneration();
      try {
        handles?.close();
      } catch (_) {
        // No caller remains to receive a finalizer cleanup error.
      } finally {
        handles = null;
        commands.close();
      }
    } else if (message is _EngineWorkerClose) {
      final activeReply = streamingReply;
      Object? cleanupError = closeStreamingGeneration();
      activeReply?.send(
        const _EngineWorkerFailure(
          _NativeError('cancelled', 'generation cancelled'),
        ),
      );
      try {
        handles?.close();
      } catch (error) {
        cleanupError ??= error;
      } finally {
        handles = null;
        if (cleanupError == null) {
          message.reply.send(null);
        } else {
          message.reply.send(
            _EngineWorkerFailure(_NativeError.from(cleanupError)),
          );
        }
      }
    } else if (message is _EngineWorkerReset) {
      try {
        final active = handles;
        if (active == null) {
          throw const ResourceDisposedException('LlamaEngine is closed.');
        }
        active.reset();
        message.reply.send(null);
      } catch (error) {
        message.reply.send(_EngineWorkerFailure(_NativeError.from(error)));
      }
    } else if (message is _EngineWorkerWarmUp) {
      try {
        final active = handles;
        if (active == null) {
          throw const ResourceDisposedException('LlamaEngine is closed.');
        }
        active.warmUp();
        message.reply.send(null);
      } catch (error) {
        message.reply.send(_EngineWorkerFailure(_NativeError.from(error)));
      }
    } else if (message is _EngineWorkerModelInfo) {
      try {
        final active = handles;
        if (active == null) {
          throw const ResourceDisposedException('LlamaEngine is closed.');
        }
        message.reply.send(active.modelInfo());
      } catch (error) {
        message.reply.send(_EngineWorkerFailure(_NativeError.from(error)));
      }
    } else if (message is _EngineWorkerModelMetadata) {
      try {
        final active = handles;
        if (active == null) {
          throw const ResourceDisposedException('LlamaEngine is closed.');
        }
        message.reply.send(active.modelMetadata());
      } catch (error) {
        message.reply.send(_EngineWorkerFailure(_NativeError.from(error)));
      }
    } else if (message is _EngineWorkerChatTemplate) {
      try {
        final active = handles;
        if (active == null) {
          throw const ResourceDisposedException('LlamaEngine is closed.');
        }
        message.reply.send(active.chatTemplate());
      } catch (error) {
        message.reply.send(_EngineWorkerFailure(_NativeError.from(error)));
      }
    } else if (message is _EngineWorkerTokenize) {
      try {
        final active = handles;
        if (active == null) {
          throw const ResourceDisposedException('LlamaEngine is closed.');
        }
        message.reply.send(
          active.tokenize(
            message.text,
            addSpecial: message.addSpecial,
            parseSpecial: message.parseSpecial,
          ),
        );
      } catch (error) {
        message.reply.send(_EngineWorkerFailure(_NativeError.from(error)));
      }
    } else if (message is _EngineWorkerCountTokens) {
      try {
        final active = handles;
        if (active == null) {
          throw const ResourceDisposedException('LlamaEngine is closed.');
        }
        message.reply.send(
          active.countTokens(
            message.text,
            addSpecial: message.addSpecial,
            parseSpecial: message.parseSpecial,
          ),
        );
      } catch (error) {
        message.reply.send(_EngineWorkerFailure(_NativeError.from(error)));
      }
    } else if (message is _EngineWorkerDetokenize) {
      try {
        final active = handles;
        if (active == null) {
          throw const ResourceDisposedException('LlamaEngine is closed.');
        }
        message.reply.send(
          active.detokenize(
            message.tokens,
            removeSpecial: message.removeSpecial,
            unparseSpecial: message.unparseSpecial,
          ),
        );
      } catch (error) {
        message.reply.send(_EngineWorkerFailure(_NativeError.from(error)));
      }
    } else if (message is _EngineWorkerEmbedTexts) {
      try {
        final active = handles;
        if (active == null) {
          throw const ResourceDisposedException('LlamaEngine is closed.');
        }
        message.reply.send(
          NativeLlamaBridge._embedTextsWithHandles(
            active,
            message.texts,
            message.config,
          ),
        );
      } catch (error) {
        message.reply.send(_EngineWorkerFailure(_NativeError.from(error)));
      }
    } else if (message is _EngineWorkerChatTemplateCapabilities) {
      try {
        final active = handles;
        if (active == null) {
          throw const ResourceDisposedException('LlamaEngine is closed.');
        }
        message.reply.send(active.chatTemplateCapabilities());
      } catch (error) {
        message.reply.send(_EngineWorkerFailure(_NativeError.from(error)));
      }
    } else if (message is _EngineWorkerFormatChat) {
      try {
        final active = handles;
        if (active == null) {
          throw const ResourceDisposedException('LlamaEngine is closed.');
        }
        message.reply.send(
          active.formatChat(
            message.messages,
            addAssistantPrompt: message.addAssistantPrompt,
            toolCalling: message.toolCalling,
          ),
        );
      } catch (error) {
        message.reply.send(_EngineWorkerFailure(_NativeError.from(error)));
      }
    } else if (message is _EngineWorkerCountChatTokens) {
      try {
        final active = handles;
        if (active == null) {
          throw const ResourceDisposedException('LlamaEngine is closed.');
        }
        message.reply.send(
          active.countChatTokens(
            message.messages,
            addAssistantPrompt: message.addAssistantPrompt,
            toolCalling: message.toolCalling,
          ),
        );
      } catch (error) {
        message.reply.send(_EngineWorkerFailure(_NativeError.from(error)));
      }
    } else if (message is _EngineWorkerPrefill) {
      try {
        final active = handles;
        if (active == null) {
          throw const ResourceDisposedException('LlamaEngine is closed.');
        }
        message.reply.send(
          active.prefill(
            message.prompt,
            addSpecial: message.addSpecial,
            parseSpecial: message.parseSpecial,
          ),
        );
      } catch (error) {
        message.reply.send(_EngineWorkerFailure(_NativeError.from(error)));
      }
    } else if (message is _EngineWorkerShiftContext) {
      try {
        final active = handles;
        if (active == null) {
          throw const ResourceDisposedException('LlamaEngine is closed.');
        }
        message.reply.send(
          active.shiftContext(
            keepTokens: message.keepTokens,
            discardTokens: message.discardTokens,
          ),
        );
      } catch (error) {
        message.reply.send(_EngineWorkerFailure(_NativeError.from(error)));
      }
    } else if (message is _EngineWorkerContextInfo) {
      try {
        final active = handles;
        if (active == null) {
          throw const ResourceDisposedException('LlamaEngine is closed.');
        }
        message.reply.send(active.contextInfo());
      } catch (error) {
        message.reply.send(_EngineWorkerFailure(_NativeError.from(error)));
      }
    } else if (message is _EngineWorkerSaveState) {
      try {
        final active = handles;
        if (active == null) {
          throw const ResourceDisposedException('LlamaEngine is closed.');
        }
        message.reply.send(active.saveState());
      } catch (error) {
        message.reply.send(_EngineWorkerFailure(_NativeError.from(error)));
      }
    } else if (message is _EngineWorkerRestoreState) {
      try {
        final active = handles;
        if (active == null) {
          throw const ResourceDisposedException('LlamaEngine is closed.');
        }
        active.restoreState(message.state);
        message.reply.send(null);
      } catch (error) {
        message.reply.send(_EngineWorkerFailure(_NativeError.from(error)));
      }
    } else if (message is _EngineWorkerComplete) {
      try {
        final active = handles;
        if (active == null) {
          throw const ResourceDisposedException('LlamaEngine is closed.');
        }
        final result = active.completeChat(message.messages, message.config);
        message.reply.send(
          _EngineWorkerText(
            result.text,
            result.telemetry,
            result.assistantMessage,
          ),
        );
      } catch (error) {
        message.reply.send(_EngineWorkerFailure(_NativeError.from(error)));
      }
    } else if (message is _EngineWorkerCompletePrompt) {
      try {
        final active = handles;
        if (active == null) {
          throw const ResourceDisposedException('LlamaEngine is closed.');
        }
        final result = active.complete(message.prompt, message.config);
        message.reply.send(
          _EngineWorkerText(
            result.text,
            result.telemetry,
            result.assistantMessage,
          ),
        );
      } catch (error) {
        message.reply.send(_EngineWorkerFailure(_NativeError.from(error)));
      }
    } else if (message is _EngineWorkerStreamComplete) {
      try {
        final active = handles;
        if (active == null) {
          throw const ResourceDisposedException('LlamaEngine is closed.');
        }
        streamingGeneration = active.startChatStream(
          message.messages,
          message.config,
        );
        streamingId = message.id;
        streamingReply = message.reply;
        stepStreamingGeneration(message.id);
      } catch (error) {
        final closeError = closeStreamingGeneration(resetContext: true);
        message.reply.send(
          _EngineWorkerFailure(_NativeError.from(closeError ?? error)),
        );
      }
    } else if (message is _EngineWorkerStreamPrompt) {
      try {
        final active = handles;
        if (active == null) {
          throw const ResourceDisposedException('LlamaEngine is closed.');
        }
        streamingGeneration = active.startCompletionStream(
          message.prompt,
          message.config,
        );
        streamingId = message.id;
        streamingReply = message.reply;
        stepStreamingGeneration(message.id);
      } catch (error) {
        final closeError = closeStreamingGeneration(resetContext: true);
        message.reply.send(
          _EngineWorkerFailure(_NativeError.from(closeError ?? error)),
        );
      }
    } else if (message is _EngineWorkerStreamNext) {
      stepStreamingGeneration(message.id);
    } else if (message is _EngineWorkerStreamDispose) {
      if (streamingGeneration != null && streamingId != message.id) {
        message.reply.send(
          const _EngineWorkerFailure(
            _NativeError(
              'generation',
              'Cannot dispose a generation owned by another stream.',
            ),
          ),
        );
      } else {
        final cleanupError = closeStreamingGeneration(resetContext: true);
        if (cleanupError == null) {
          message.reply.send(null);
        } else {
          message.reply.send(
            _EngineWorkerFailure(_NativeError.from(cleanupError)),
          );
        }
      }
    } else if (message is _EngineWorkerLoadLora) {
      try {
        final active = handles;
        if (active == null) {
          throw const ResourceDisposedException('LlamaEngine is closed.');
        }
        message.reply.send(active.loadLora(message.config));
      } catch (error) {
        message.reply.send(_EngineWorkerFailure(_NativeError.from(error)));
      }
    } else if (message is _EngineWorkerListLoras) {
      try {
        final active = handles;
        if (active == null) {
          throw const ResourceDisposedException('LlamaEngine is closed.');
        }
        message.reply.send(active.loraAdapters());
      } catch (error) {
        message.reply.send(_EngineWorkerFailure(_NativeError.from(error)));
      }
    } else if (message is _EngineWorkerSetLoraScale) {
      try {
        final active = handles;
        if (active == null) {
          throw const ResourceDisposedException('LlamaEngine is closed.');
        }
        active.setLoraScale(message.adapterId, message.scale);
        message.reply.send(null);
      } catch (error) {
        message.reply.send(_EngineWorkerFailure(_NativeError.from(error)));
      }
    } else if (message is _EngineWorkerUnloadLora) {
      try {
        final active = handles;
        if (active == null) {
          throw const ResourceDisposedException('LlamaEngine is closed.');
        }
        active.unloadLora(message.adapterId);
        message.reply.send(null);
      } catch (error) {
        message.reply.send(_EngineWorkerFailure(_NativeError.from(error)));
      }
    }
  };

  commands.listen((message) {
    if (streamingGeneration != null &&
        message is! _EngineWorkerStreamNext &&
        message is! _EngineWorkerStreamDispose &&
        message is! _EngineWorkerClose &&
        message is! _EngineWorkerFinalize) {
      _engineWorkerReply(message)?.send(
        const _EngineWorkerFailure(
          _NativeError(
            'generation',
            'Another generation is already active on this engine.',
          ),
        ),
      );
      return;
    }
    handleMessage(message);
  });
}

String _chatRoleName(ChatRole role) {
  return switch (role) {
    ChatRole.system => 'system',
    ChatRole.user => 'user',
    ChatRole.assistant => 'assistant',
    ChatRole.tool => 'tool',
  };
}

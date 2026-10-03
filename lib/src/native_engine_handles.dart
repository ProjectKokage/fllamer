part of 'native_bridge.dart';

extension NativeEngineOpening on NativeLlamaBridge {
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
        ..chat_template_size = chatTemplate.length
        ..load_mtp = _loadsEmbeddedMtp(config) ? 1 : 0;

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
}

EmbeddingBatch _embedTextsWithHandles(
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
    bool? enableThinking,
    int? reasoningBudgetTokens,
    int? maximumPromptBytes,
  }) {
    return bridge._formatChat(
      model,
      messages,
      addAssistantPrompt: addAssistantPrompt,
      toolCalling: toolCalling,
      enableThinking: enableThinking,
      reasoningBudgetTokens: reasoningBudgetTokens,
      maximumPromptBytes: maximumPromptBytes,
    );
  }

  int countChatTokens(
    List<ChatMessage> messages, {
    required bool addAssistantPrompt,
    required LlamaToolCallingConfig toolCalling,
    bool? enableThinking,
    int? reasoningBudgetTokens,
    int? maximumPromptBytes,
  }) {
    return bridge._countChatTokens(
      model,
      messages,
      addAssistantPrompt: addAssistantPrompt,
      toolCalling: toolCalling,
      enableThinking: enableThinking,
      reasoningBudgetTokens: reasoningBudgetTokens,
      maximumPromptBytes: maximumPromptBytes,
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
        gpuBackend: _gpuBackendFromNative(info.ref.gpu_backend),
        supportsContextShift: info.ref.supports_context_shift != 0,
        usedTokens: info.ref.used_tokens,
        kvCacheKeyType: _kvCacheTypeFromNative(info.ref.kv_cache_key_type),
        kvCacheValueType: _kvCacheTypeFromNative(info.ref.kv_cache_value_type),
        flashAttention: _flashAttentionFromNative(info.ref.flash_attention),
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
      true => llama_dart_add_special_mode.LLAMA_DART_ADD_SPECIAL_ALWAYS,
      false => llama_dart_add_special_mode.LLAMA_DART_ADD_SPECIAL_NEVER,
      null =>
        llama_dart_add_special_mode.LLAMA_DART_ADD_SPECIAL_IF_CONTEXT_EMPTY,
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
    llama_dart_add_special_mode addSpecialMode =
        llama_dart_add_special_mode.LLAMA_DART_ADD_SPECIAL_IF_CONTEXT_EMPTY,
    bool parseSpecial = false,
    bool manageRequestLoras = true,
    bool reusePromptPrefix = false,
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
        ..mirostat = _mirostatMode(config.mirostat)
        ..mirostat_tau = config.mirostatTau
        ..mirostat_eta = config.mirostatEta
        ..grammar_data = grammarPointer
        ..grammar_size = grammarBytes.length
        ..grammar_root_data = grammarRootPointer
        ..grammar_root_size = grammarRootBytes.length
        ..stop_sequences = stopSequences
        ..stop_sequence_count = stopBytes.length
        ..seed = config.seed ?? 0xFFFFFFFF
        ..add_special = addSpecialMode.value
        ..parse_special = parseSpecial ? 1 : 0
        ..media_inputs = mediaInputs
        ..media_input_count = media.length
        ..json_schema_data = jsonSchemaPointer
        ..json_schema_size = jsonSchemaBytes.length
        ..chat_plan_data = chatPlanPointer
        ..chat_plan_size = chatPlanBytes.length
        ..stop_tokens = stopTokens
        ..stop_token_count = config.stopTokens.length
        ..reuse_prompt_prefix = reusePromptPrefix ? 1 : 0;
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

  _NativeStreamingGeneration startCompletionStream(
    String prompt,
    GenerationConfig config, {
    List<_NativeMediaInput> media = const <_NativeMediaInput>[],
    _NativeChatPlan? chatPlan,
    bool parseSpecial = false,
    bool reusePromptPrefix = false,
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
        reusePromptPrefix: reusePromptPrefix,
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
    GenerationConfig config, {
    required bool reusePromptPrefix,
    int? maximumPromptBytes,
  }) {
    final prepared = _prepareMultimodalChat(
      messages,
      bridge._multimodalMarker,
      maximumPromptBytes: maximumPromptBytes,
    );
    final rendered = bridge._renderPreparedChat(
      model,
      prepared.messages,
      addAssistantPrompt: true,
      toolCalling: config.toolCalling,
      grammar: config.grammar,
      jsonSchema: config.jsonSchema,
      enableThinking: config.enableThinking,
      reasoningBudgetTokens: config.reasoningBudgetTokens,
      maximumPromptBytes: maximumPromptBytes,
    );
    return startCompletionStream(
      rendered.prompt,
      config,
      media: prepared.media,
      chatPlan: rendered.plan,
      parseSpecial: true,
      reusePromptPrefix: reusePromptPrefix,
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
  next({required void Function(int generatedTokens) onProgress}) {
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
      final generatedTokens = _stats.ref.generated_tokens;
      if (generatedTokens > _generatedTokens) {
        _generatedTokens = generatedTokens;
        onProgress(generatedTokens);
      }
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
    stopReason: _stopReason(stats.stop_reason),
  );
}

GenerationStopReason _stopReason(int nativeValue) {
  for (final reason in llama_dart_stop_reason.values) {
    if (reason.value == nativeValue) {
      return switch (reason) {
        llama_dart_stop_reason.LLAMA_DART_STOP_REASON_UNKNOWN =>
          GenerationStopReason.unknown,
        llama_dart_stop_reason.LLAMA_DART_STOP_REASON_END_OF_GENERATION =>
          GenerationStopReason.endOfGeneration,
        llama_dart_stop_reason.LLAMA_DART_STOP_REASON_STOP_SEQUENCE =>
          GenerationStopReason.stopSequence,
        llama_dart_stop_reason.LLAMA_DART_STOP_REASON_STOP_TOKEN =>
          GenerationStopReason.stopToken,
        llama_dart_stop_reason.LLAMA_DART_STOP_REASON_MAX_TOKENS =>
          GenerationStopReason.maxTokens,
      };
    }
  }
  return GenerationStopReason.unknown;
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

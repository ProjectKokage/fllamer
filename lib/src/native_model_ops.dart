import 'dart:convert';
import 'dart:ffi' as ffi;
import 'dart:typed_data';

import 'package:ffi/ffi.dart';
import 'config.dart';
import 'errors.dart';
import 'ffi/generated_bindings.dart';
import 'model_info.dart';

import 'native_bridge.dart';
import 'native_config_mapping.dart';

const _maxModelDescriptionBytes = 1024 * 1024;

const _maxChatTemplateBytes = 16 * 1024 * 1024;

const _maxModelMetadataEntries = 65536;

const _maxModelMetadataKeyBytes = 4096;

const _maxModelMetadataValueBytes = 16 * 1024 * 1024;

const _maxModelMetadataTotalBytes = 64 * 1024 * 1024;

extension NativeModelOps on NativeLlamaBridge {
  LlamaModelInfo loadAndInspectModel(LlamaModelConfig config) {
    return withLoadedModel(config, readModelInfo);
  }

  Map<String, String> loadAndReadModelMetadata(LlamaModelConfig config) {
    return withLoadedModel(config, readModelMetadata);
  }

  String loadAndReadChatTemplate(LlamaModelConfig config) {
    return withLoadedModel(config, readChatTemplate);
  }

  List<int> loadAndTokenize(
    LlamaModelConfig config,
    String text, {
    required bool addSpecial,
    required bool parseSpecial,
  }) {
    return withLoadedModel(
      config,
      (model) => tokenizeWithModel(
        model,
        text,
        addSpecial: addSpecial,
        parseSpecial: parseSpecial,
      ),
    );
  }

  List<int> tokenizeWithModel(
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
      final firstResult = bindings.llama_dart_model_tokenize(
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
        check(firstResult);
      }

      final tokensPointer = calloc<ffi.Int32>(outCount.value);
      try {
        check(
          bindings.llama_dart_model_tokenize(
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

  String loadAndDetokenize(
    LlamaModelConfig config,
    List<int> tokens, {
    required bool removeSpecial,
    required bool unparseSpecial,
  }) {
    return withLoadedModel(
      config,
      (model) => detokenizeWithModel(
        model,
        tokens,
        removeSpecial: removeSpecial,
        unparseSpecial: unparseSpecial,
      ),
    );
  }

  String detokenizeWithModel(
    ffi.Pointer<llama_dart_model> model,
    List<int> tokens, {
    required bool removeSpecial,
    required bool unparseSpecial,
  }) {
    final tokensPointer = calloc<ffi.Int32>(tokens.length);
    final outSize = calloc<ffi.Size>();
    try {
      tokensPointer.asTypedList(tokens.length).setAll(0, tokens);
      final firstResult = bindings.llama_dart_model_detokenize(
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
        check(firstResult);
      }

      final textPointer = calloc<ffi.Uint8>(outSize.value);
      try {
        check(
          bindings.llama_dart_model_detokenize(
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

  T withLoadedModel<T>(
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
        ..n_gpu_layers = gpuLayers(config)
        ..vocab_only = vocabOnly ? 1 : 0
        ..use_mmap = config.useMmap ? 1 : 0
        ..use_mlock = config.useMlock ? 1 : 0
        ..check_tensors = config.checkTensors ? 1 : 0
        ..gpu_backend = gpuBackend(config)
        ..chat_template_data = chatTemplatePointer
        ..chat_template_size = chatTemplate.length
        ..load_mtp = 0;

      check(bindings.llama_dart_model_load(loadConfig, outModel));
      model = outModel.value;
      if (model == ffi.nullptr) {
        throw const ModelLoadException('Native bridge returned a null model.');
      }
      return useModel(model);
    } finally {
      if (model != ffi.nullptr) {
        bindings.llama_dart_model_free(model);
        throwIfLastError('Native model free');
      }
      calloc.free(outModel);
      calloc.free(loadConfig);
      if (chatTemplatePointer != ffi.nullptr) {
        calloc.free(chatTemplatePointer);
      }
      calloc.free(pathPointer);
    }
  }

  LlamaModelInfo readModelInfo(ffi.Pointer<llama_dart_model> model) {
    final info = calloc<llama_dart_model_info>();
    try {
      info.ref.struct_size = ffi.sizeOf<llama_dart_model_info>();
      check(bindings.llama_dart_model_get_info(model, info));
      return LlamaModelInfo(
        description: _readDescription(model),
        chatTemplate: _tryReadChatTemplate(model),
        vocabType: info.ref.vocab_type,
        vocabSize: info.ref.n_vocab,
        maximumTokenPieceBytes: info.ref.maximum_token_piece_bytes,
        trainingContextSize: info.ref.n_ctx_train,
        embeddingSize: info.ref.n_embd,
        inputEmbeddingSize: info.ref.n_embd_inp,
        outputEmbeddingSize: info.ref.n_embd_out,
        layerCount: info.ref.n_layer,
        nextnLayerCount: info.ref.n_layer_nextn,
        attentionHeadCount: info.ref.n_head,
        keyValueHeadCount: info.ref.n_head_kv,
        fileType: info.ref.ftype,
        fileTypeName: readOptionalCString(
          bindings.llama_dart_model_file_type_name(info.ref.ftype),
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
        final result = bindings.llama_dart_model_get_description(
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
        check(result);
        return _decodeModelUtf8(buffer, outSize.value, 'description');
      } finally {
        calloc.free(outSize);
        calloc.free(buffer);
      }
    }
  }

  String readChatTemplate(ffi.Pointer<llama_dart_model> model) {
    final out = calloc<llama_dart_buffer>();
    try {
      check(bindings.llama_dart_model_get_chat_template(model, out));
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
      bindings.llama_dart_buffer_free(out.ref.data);
      calloc.free(out);
    }
  }

  String? _tryReadChatTemplate(ffi.Pointer<llama_dart_model> model) {
    try {
      return readChatTemplate(model);
    } on UnsupportedFeatureException {
      return null;
    }
  }

  Map<String, String> readModelMetadata(ffi.Pointer<llama_dart_model> model) {
    final outCount = calloc<ffi.Size>();
    try {
      check(bindings.llama_dart_model_metadata_count(model, outCount));
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
          final firstResult = bindings.llama_dart_model_metadata_get(
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
            check(firstResult);
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
            check(
              bindings.llama_dart_model_metadata_get(
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

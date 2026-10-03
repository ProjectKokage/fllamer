import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:fllamer/fllamer.dart';
import 'package:test/test.dart';

import 'src/native_test_support.dart';

void main() {
  group('config validation', () {
    test('rejects empty model paths', () {
      expect(
        () => const LlamaModelConfig(modelPath: '').validate(),
        throwsArgumentError,
      );
      expect(
        () => const LlamaModelConfig(
          modelPath: 'model.gguf',
          nativeLibraryPath: ' ',
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => const LlamaModelConfig(
          modelPath: 'model.gguf',
          speculativeDecoding: DraftModelSpeculation(
            draftModelPath: 'draft.gguf',
            draftLength: 0,
          ),
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => const LlamaModelConfig(
          modelPath: 'model.gguf',
          speculativeDecoding: MtpSpeculation(draftLength: 1025),
        ).validate(),
        throwsArgumentError,
      );
      const draft = DraftModelSpeculation(draftModelPath: 'draft.gguf');
      const eagle = Eagle3Speculation(draftModelPath: 'eagle.gguf');
      const dflash = DFlashSpeculation(draftModelPath: 'dflash.gguf');
      const mtp = MtpSpeculation();
      const ngramMod = NGramModSpeculation();
      expect(draft.draftLength, 3);
      expect(eagle.draftLength, 3);
      expect(dflash.draftLength, 15);
      expect(mtp.draftLength, 3);
      expect(ngramMod.matchLength, 24);
      expect(ngramMod.minimumDraftLength, 48);
      expect(ngramMod.maximumDraftLength, 64);
      expect(
        () => const LlamaModelConfig(
          modelPath: 'model.gguf',
          speculativeDecoding: NGramCacheSpeculation(),
        ).validate(),
        returnsNormally,
      );
    });

    test('accepts model loading flags', () {
      const defaults = LlamaModelConfig(modelPath: 'model.gguf');
      expect(defaults.useMmap, isTrue);
      expect(defaults.useMlock, isFalse);
      expect(defaults.checkTensors, isTrue);

      expect(
        () => const LlamaModelConfig(
          modelPath: 'model.gguf',
          useMmap: false,
          useMlock: true,
          checkTensors: false,
        ).validate(),
        returnsNormally,
      );
    });

    test('validates explicit chat template overrides', () {
      expect(
        () => const LlamaModelConfig(
          modelPath: 'model.gguf',
          chatTemplate: 'chatml',
        ).validate(),
        returnsNormally,
      );
      expect(
        () => const LlamaModelConfig(
          modelPath: 'model.gguf',
          chatTemplate: ' \n\t',
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => const LlamaModelConfig(
          modelPath: 'model.gguf',
          chatTemplate: 'bad\u0000template',
        ).validate(),
        throwsArgumentError,
      );
    });

    test('validates typed KV cache tuning', () {
      const defaults = LlamaModelConfig(modelPath: 'model.gguf');
      expect(defaults.kvCache.keyType, KvCacheType.f16);
      expect(defaults.kvCache.valueType, KvCacheType.f16);
      expect(defaults.kvCache.offload, isTrue);
      expect(defaults.kvCache.flashAttention, FlashAttentionMode.auto);
      expect(defaults.kvCache.swaFull, isTrue);
      expect(defaults.kvCache.unified, isFalse);

      for (final type in KvCacheType.values) {
        expect(
          () => LlamaModelConfig(
            modelPath: 'model.gguf',
            kvCache: KvCacheConfig(
              keyType: type,
              valueType: type,
              flashAttention: FlashAttentionMode.enabled,
            ),
          ).validate(),
          returnsNormally,
        );
      }
      expect(
        () => const LlamaModelConfig(
          modelPath: 'model.gguf',
          kvCache: KvCacheConfig(
            keyType: KvCacheType.q8Zero,
            flashAttention: FlashAttentionMode.disabled,
          ),
        ).validate(),
        returnsNormally,
      );
      expect(
        () => const LlamaModelConfig(
          modelPath: 'model.gguf',
          kvCache: KvCacheConfig(
            valueType: KvCacheType.q8Zero,
            flashAttention: FlashAttentionMode.disabled,
          ),
        ).validate(),
        throwsArgumentError,
      );
    });

    test('rejects native integer overflow config', () {
      expect(
        () => const GpuConfig.auto(layers: -1).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GpuConfig.auto(layers: 0x80000000).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GpuConfig.metal(layers: 0).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GpuConfig.vulkan(layers: 0).validate(),
        throwsArgumentError,
      );
      expect(
        () => const LlamaModelConfig(
          modelPath: 'model.gguf',
          contextSize: 0x100000000,
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => const LlamaModelConfig(
          modelPath: 'model.gguf',
          batchSize: 0x100000000,
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => const LlamaModelConfig(
          modelPath: 'model.gguf',
          ubatchSize: 0x100000000,
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => const LlamaModelConfig(
          modelPath: 'model.gguf',
          threads: 0x80000000,
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => const LlamaModelConfig(
          modelPath: 'model.gguf',
          batchThreads: 0x80000000,
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => const LlamaModelConfig(
          modelPath: 'model.gguf',
          ubatchSize: 0,
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => const LlamaModelConfig(
          modelPath: 'model.gguf',
          batchThreads: 0,
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => const LlamaModelConfig(
          modelPath: 'model.gguf',
          batchSize: 16,
          ubatchSize: 32,
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => const LlamaModelConfig(
          modelPath: 'model.gguf',
          contextSize: 0x80000000,
          speculativeDecoding: MtpSpeculation(),
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => const LlamaModelConfig(
          modelPath: 'model.gguf',
          batchSize: 0x80000000,
          speculativeDecoding: DraftModelSpeculation(
            draftModelPath: 'draft.gguf',
          ),
        ).validate(),
        throwsArgumentError,
      );
    });

    test('rejects unsafe model paths', () {
      expect(
        () => const LlamaModelConfig(modelPath: 'bad\u0000path').validate(),
        throwsArgumentError,
      );
      expect(
        () => const LlamaModelConfig(modelPath: 'bad\npath').validate(),
        throwsArgumentError,
      );
      expect(
        () => const LlamaModelConfig(
          modelPath: 'model.gguf',
          nativeLibraryPath: 'bridge\u0000',
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => const LlamaModelConfig(
          modelPath: 'model.gguf',
          nativeLibraryPath: 'bad\nbridge',
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => const LlamaModelConfig(
          modelPath: 'model.gguf',
          mmprojPath: 'bad\u0000mmproj.gguf',
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => const LlamaModelConfig(
          modelPath: 'model.gguf',
          mmprojPath: 'bad\nmmproj.gguf',
        ).validate(),
        throwsArgumentError,
      );
    });

    test('rejects invalid speculative decoding config', () {
      expect(
        () => const LlamaModelConfig(
          modelPath: 'model.gguf',
          speculativeDecoding: DraftModelSpeculation(draftModelPath: ''),
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => const LlamaModelConfig(
          modelPath: 'model.gguf',
          speculativeDecoding: Eagle3Speculation(
            draftModelPath: 'draft.gguf\u0000',
          ),
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => const LlamaModelConfig(
          modelPath: 'model.gguf',
          speculativeDecoding: DFlashSpeculation(draftModelPath: ''),
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => const LlamaModelConfig(
          modelPath: 'model.gguf',
          speculativeDecoding: NGramModSpeculation(
            minimumDraftLength: 9,
            maximumDraftLength: 8,
          ),
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => const LlamaModelConfig(
          modelPath: 'model.gguf',
          speculativeDecoding: MtpSpeculation(mtpModelPath: ''),
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => const LlamaModelConfig(
          modelPath: 'model.gguf',
          speculativeDecoding: NGramSpeculation(strategy: ''),
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => const LlamaModelConfig(
          modelPath: 'model.gguf',
          speculativeDecoding: NGramSpeculation(strategy: 'ngram-simple\n'),
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => const LlamaModelConfig(
          modelPath: 'model.gguf',
          speculativeDecoding: NGramSpeculation(strategy: ' ngram-simple '),
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => const LlamaModelConfig(
          modelPath: 'model.gguf',
          speculativeDecoding: NGramSpeculation(
            strategy: 'ngram-simple',
            ngramSize: 0,
          ),
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => const LlamaModelConfig(
          modelPath: 'model.gguf',
          speculativeDecoding: NGramSpeculation(
            strategy: 'ngram-simple',
            ngramSize: 16,
            draftLength: 8,
          ),
        ).validate(),
        throwsArgumentError,
      );
      expect(
        LlamaEngine.load(
          const LlamaModelConfig(
            modelPath: 'model.gguf',
            speculativeDecoding: NGramSpeculation(strategy: 'unknown-ngram'),
          ),
        ),
        throwsA(isA<UnsupportedFeatureException>()),
      );
      expect(
        LlamaEngine.load(
          const LlamaModelConfig(
            modelPath: 'model.gguf',
            mmprojPath: 'vision-mmproj.gguf',
          ),
        ),
        throwsA(isA<ModelLoadException>()),
      );
    });

    test('rejects empty raw completion prompts', () async {
      final bridgePath = nativeBridgePath;
      if (!File(bridgePath).existsSync()) {
        markTestSkipped('native bridge has not been built at $bridgePath');
      }
      const modelPath = 'third_party/llama.cpp/models/ggml-vocab-gpt-2.gguf';
      if (!File(modelPath).existsSync()) {
        markTestSkipped('vocab fixture is missing at $modelPath');
      }

      try {
        final engine = await LlamaEngine.load(
          LlamaModelConfig(
            modelPath: modelPath,
            nativeLibraryPath: bridgePath,
            contextSize: 128,
            batchSize: 16,
            threads: 1,
          ),
        );
        try {
          await expectLater(
            engine.complete(prompt: '').toList(),
            throwsArgumentError,
          );
          await expectLater(
            engine.complete(prompt: '  \n\t').toList(),
            throwsArgumentError,
          );
        } finally {
          await engine.close();
        }
      } on ModelLoadException {
        // Vocab-only GGUF fixtures may not contain weights needed for full load.
      } on ContextCreateException {
        // Some vocab fixtures load but cannot create a usable context.
      }
    });

    test('rejects invalid sampling config', () {
      expect(
        () => const GenerationConfig(maxTokens: 0x100000000).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(maxTokens: 0x80000000).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(streamChunkTokens: 0).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(streamChunkTokens: 1025).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(temperature: double.nan).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(topK: -1).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(topK: 0x80000000).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(topP: 1.1).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(topP: double.nan).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(minP: -0.1).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(typicalP: 1.1).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(penaltyLastN: -1).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(penaltyLastN: 0x80000000).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(repeatPenalty: -0.1).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(
          frequencyPenalty: double.infinity,
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(mirostatTau: 0).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(mirostatEta: double.nan).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(seed: -1).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(seed: 0x100000000).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(stop: <String>['']).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(stop: <String>['bad\u0000']).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(stopTokens: <int>[-1]).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(stopTokens: <int>[0x80000000]).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(stopTokens: <int>[1, 1]).validate(),
        throwsArgumentError,
      );
      expect(
        () => GenerationConfig(
          stopTokens: List<int>.generate(1025, (index) => index),
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () =>
            const GenerationConfig(loraScales: <int, double>{0: 1}).validate(),
        throwsArgumentError,
      );
      expect(
        () =>
            const GenerationConfig(loraScales: <int, double>{1: -1}).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(
          loraScales: <int, double>{1: double.nan},
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(grammar: '').validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(grammar: '  \n\t').validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(grammar: 'root ::= "x"\u0000').validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(grammarRoot: 'custom').validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(
          grammar: 'root ::= "x"',
          enableThinking: true,
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(
          maxTokens: 16,
          enableThinking: true,
          reasoningBudgetTokens: -1,
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(
          maxTokens: 16,
          reasoningBudgetTokens: 8,
        ).validate(),
        returnsNormally,
      );
      expect(
        () => const GenerationConfig(
          maxTokens: 16,
          enableThinking: false,
          reasoningBudgetTokens: 8,
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(
          maxTokens: 16,
          enableThinking: true,
          reasoningBudgetTokens: 16,
        ).validate(),
        returnsNormally,
      );
      expect(
        () => const GenerationConfig(
          maxTokens: 16,
          enableThinking: true,
          reasoningBudgetTokens: 8,
        ).validate(),
        returnsNormally,
      );
      expect(
        () => GenerationConfig.jsonSchema(
          schema: const <String, Object?>{'type': 'string'},
          enableThinking: true,
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(
          grammar: 'root ::= "x"',
          grammarRoot: '',
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(
          grammar: 'root ::= "x"',
          grammarRoot: '  \t',
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(
          grammar: 'root ::= "x"',
          grammarRoot: 'root\u0000',
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(
          grammar: 'root ::= "x"',
          grammarRoot: 'root\nother',
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(
          grammar: 'root ::= "x"',
          grammarRoot: 'root other',
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(
          mirostat: MirostatMode.v1,
          mirostatTau: 0,
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(
          mirostat: MirostatMode.v2,
          mirostatEta: double.nan,
        ).validate(),
        throwsArgumentError,
      );
    });

    test('json schema generation config snapshots mutable inputs', () {
      final stops = <String>['</json>'];
      final stopTokens = <int>[1, 2];
      final loraScales = <int, double>{1: 0.5};
      final properties = <String, Object?>{
        'query': <String, Object?>{'type': 'string'},
      };
      final toolSchema = <String, Object?>{
        'type': 'object',
        'additionalProperties': false,
        'required': <Object?>['query'],
        'properties': properties,
      };
      final tool = LlamaToolDefinition(
        name: 'search_local',
        description: 'Search local notes.',
        parametersSchema: toolSchema,
      );
      final tools = <LlamaToolDefinition>[tool];
      final outputSchema = <String, Object?>{
        'type': 'object',
        'properties': <String, Object?>{
          'answer': <String, Object?>{'type': 'string'},
        },
      };
      final config = GenerationConfig.jsonSchema(
        schema: outputSchema,
        streamChunkTokens: 7,
        stop: stops,
        stopTokens: stopTokens,
        loraScales: loraScales,
        toolCalling: LlamaToolCallingConfig(tools: tools),
      );

      stops[0] = 'changed';
      stopTokens[0] = 99;
      loraScales[1] = 2;
      tools.clear();
      properties['query'] = <String, Object?>{'type': 'number'};
      (outputSchema['properties']! as Map<String, Object?>)['answer'] =
          <String, Object?>{'type': 'number'};

      expect(config.stop, const <String>['</json>']);
      expect(config.streamChunkTokens, 7);
      expect(() => config.stop[0] = 'changed', throwsUnsupportedError);
      expect(config.stopTokens, const <int>[1, 2]);
      expect(() => config.stopTokens[0] = 99, throwsUnsupportedError);
      expect(config.loraScales, const <int, double>{1: 0.5});
      expect(() => config.loraScales![1] = 2, throwsUnsupportedError);
      expect(config.toolCalling.tools.single.name, 'search_local');
      expect(() => config.toolCalling.tools.clear(), throwsUnsupportedError);
      final exportedTool = config.toolCalling.toJson().single;
      final exportedFunction =
          exportedTool['function']! as Map<String, Object?>;
      final exportedSchema =
          exportedFunction['parameters']! as Map<String, Object?>;
      final exportedProperties =
          exportedSchema['properties']! as Map<String, Object?>;
      expect(exportedProperties['query'], <String, Object?>{'type': 'string'});
      final savedOutputProperties =
          config.jsonSchema!['properties']! as Map<String, Object?>;
      expect(savedOutputProperties['answer'], <String, Object?>{
        'type': 'string',
      });
      expect(
        () => savedOutputProperties['answer'] = <String, Object?>{},
        throwsUnsupportedError,
      );
    });

    test('rejects invalid LoRA adapter config', () {
      expect(
        () => const LoraAdapterConfig(path: '').validate(),
        throwsArgumentError,
      );
      expect(
        () => const LoraAdapterConfig(path: 'adapter.gguf\u0000').validate(),
        throwsArgumentError,
      );
      expect(
        () => const LoraAdapterConfig(path: 'adapter\n.gguf').validate(),
        throwsArgumentError,
      );
      expect(
        () => const LoraAdapterConfig(
          path: 'adapter.gguf',
          scale: double.nan,
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => const LoraAdapterConfig(
          path: 'adapter.gguf',
          scale: -0.1,
        ).validate(),
        throwsArgumentError,
      );
    });

    test('rejects rank pooling through embedding config validation', () {
      expect(
        () => const EmbeddingConfig(pooling: EmbeddingPooling.rank).validate(),
        throwsUnsupportedFeature,
      );
      expect(
        () => const EmbeddingConfig(pooling: EmbeddingPooling.mean).validate(),
        returnsNormally,
      );
    });

    test('json mode uses the built-in JSON grammar', () {
      const config = GenerationConfig.jsonMode(maxTokens: 32);

      expect(config.maxTokens, 32);
      expect(config.grammar, llamaJsonGrammar);
      expect(config.grammar, contains('[0-9] [0-9]{0,15})? ws'));
      expect(config.grammarRoot, 'root');
      expect(() => config.validate(), returnsNormally);
    });

    test('json schema mode preserves schema for native conversion', () {
      final config = GenerationConfig.jsonSchema(
        schema: <String, Object?>{
          'type': 'object',
          'additionalProperties': false,
          'required': <Object?>['answer', 'score', 'tags'],
          'properties': <String, Object?>{
            'answer': <String, Object?>{'type': 'string'},
            'score': <String, Object?>{'type': 'integer'},
            'tags': <String, Object?>{
              'type': 'array',
              'items': <String, Object?>{
                'enum': <Object?>['a', 'b'],
              },
            },
          },
        },
      );

      expect(config.grammarRoot, 'root');
      expect(config.grammar, isNull);
      expect(config.jsonSchema?['type'], 'object');
      expect(
        config.jsonSchema?['properties'],
        containsPair('answer', <String, Object?>{'type': 'string'}),
      );
      expect(() => config.validate(), returnsNormally);
    });

    test('json schema mode delegates broader schemas to upstream', () {
      final config = GenerationConfig.jsonSchema(
        schema: <String, Object?>{
          r'$defs': <String, Object?>{
            'code': <String, Object?>{
              'type': 'string',
              'pattern': r'^[A-Z]{2}$',
            },
          },
          r'$ref': r'#/$defs/code',
        },
      );

      expect(config.grammar, isNull);
      expect(config.jsonSchema?[r'$ref'], r'#/$defs/code');
      expect(() => config.validate(), returnsNormally);
      expect(
        () => GenerationConfig(
          grammar: llamaJsonGrammar,
          jsonSchema: const <String, Object?>{'type': 'string'},
        ).validate(),
        throwsArgumentError,
      );
    });

    test('json schema helper delegates to the pinned native converter', () {
      final grammar = llamaJsonSchemaGrammar(<String, Object?>{
        'type': 'object',
        'additionalProperties': false,
        'required': <Object?>['answer'],
        'properties': <String, Object?>{
          'answer': <String, Object?>{'type': 'string'},
        },
      });

      expect(grammar, contains('root ::='));
      expect(grammar, contains('answer'));
    });

    test('generation chunks can carry telemetry', () {
      const telemetry = GenerationTelemetry(
        promptTokens: 2,
        generatedTokens: 3,
        promptEvalMs: 1.5,
        decodeMs: 2.5,
        totalMs: 4.0,
        timeToFirstTokenMs: 2.0,
        speculativeDraftTokens: 4,
        speculativeAcceptedTokens: 3,
        speculativeDraftMs: 1.25,
        speculativeVerifyMs: 2.5,
      );
      const chunk = GenerationChunk(
        text: 'ok',
        isDone: true,
        telemetry: telemetry,
      );

      expect(chunk.telemetry?.generatedTokens, 3);
      expect(chunk.telemetry?.totalMs, 4.0);
      expect(chunk.telemetry?.timeToFirstTokenMs, 2.0);
      expect(
        chunk.telemetry?.promptEvalTokensPerSecond,
        closeTo(1333.333, 0.001),
      );
      expect(chunk.telemetry?.decodeTokensPerSecond, 1200.0);
      expect(chunk.telemetry?.totalTokensPerSecond, 1250.0);
      expect(chunk.telemetry?.speculativeAcceptanceRate, 0.75);
      expect(chunk.telemetry?.speculativeDraftMs, 1.25);
      expect(chunk.telemetry?.speculativeVerifyMs, 2.5);
      expect(
        const GenerationTelemetry(
          promptTokens: 0,
          generatedTokens: 0,
          promptEvalMs: 0,
          decodeMs: double.nan,
          totalMs: 0,
        ).decodeTokensPerSecond,
        0,
      );
      expect(
        const GenerationTelemetry(
          promptTokens: 0,
          generatedTokens: 0,
          promptEvalMs: 0,
          decodeMs: 0,
          totalMs: 0,
          speculativeDraftTokens: 2,
          speculativeAcceptedTokens: 3,
        ).speculativeAcceptanceRate,
        1,
      );
      expect(
        const GenerationTelemetry(
          promptTokens: 0,
          generatedTokens: 0,
          promptEvalMs: 0,
          decodeMs: 0,
          totalMs: 0,
          speculativeDraftTokens: 2,
          speculativeAcceptedTokens: -1,
        ).speculativeAcceptanceRate,
        0,
      );
    });

    test('rejects empty embedding text', () {
      expect(
        LlamaEmbeddings.embedText(
          const LlamaModelConfig(
            modelPath: 'model.gguf',
            nativeLibraryPath: 'bridge',
          ),
          '',
        ),
        throwsArgumentError,
      );
      expect(
        LlamaEmbeddings.embedText(
          const LlamaModelConfig(modelPath: 'model.gguf'),
          ' ',
        ),
        throwsArgumentError,
      );
    });

    test('rejects rank pooling through embedding APIs before native work', () {
      expect(
        LlamaEmbeddings.embedText(
          const LlamaModelConfig(modelPath: 'model.gguf'),
          'query',
          config: const EmbeddingConfig(pooling: EmbeddingPooling.rank),
        ),
        throwsA(isA<UnsupportedFeatureException>()),
      );
      expect(
        LlamaEmbeddings.embedTexts(
          const LlamaModelConfig(modelPath: 'model.gguf'),
          const <String>['query'],
          config: const EmbeddingConfig(pooling: EmbeddingPooling.rank),
        ),
        throwsA(isA<UnsupportedFeatureException>()),
      );
      expect(
        LlamaEmbeddingEngine.load(
          const LlamaModelConfig(modelPath: 'model.gguf'),
          config: const EmbeddingConfig(pooling: EmbeddingPooling.rank),
        ),
        throwsA(isA<UnsupportedFeatureException>()),
      );
    });

    test('embedding contexts reject generation-only model settings', () {
      const speculative = LlamaModelConfig(
        modelPath: 'model.gguf',
        speculativeDecoding: NGramSpeculation(strategy: 'ngram-simple'),
      );
      final throwsSpeculative = throwsA(
        isA<UnsupportedFeatureException>().having(
          (error) => error.message,
          'message',
          contains('Speculative decoding'),
        ),
      );

      expect(
        LlamaEmbeddings.embedText(speculative, 'query'),
        throwsSpeculative,
      );
      expect(
        LlamaEmbeddings.embedTexts(speculative, const <String>['query']),
        throwsSpeculative,
      );
      expect(LlamaEmbeddingEngine.load(speculative), throwsSpeculative);
      expect(
        LlamaReranking.scorePair(
          speculative,
          query: 'query',
          document: 'document',
        ),
        throwsSpeculative,
      );
      expect(
        const LlamaReranker(
          speculative,
        ).rerank('query', const <VectorSearchResult>[]),
        throwsSpeculative,
      );
      expect(
        LlamaEmbeddings.embedText(
          const LlamaModelConfig(
            modelPath: 'model.gguf',
            mmprojPath: 'vision-mmproj.gguf',
          ),
          'query',
        ),
        throwsA(
          isA<UnsupportedFeatureException>().having(
            (error) => error.message,
            'message',
            contains('mmprojPath'),
          ),
        ),
      );
    });

    test('rejects NUL bytes in native text inputs before native work', () {
      expect(
        LlamaTokenizer.tokenize(
          const LlamaModelConfig(modelPath: 'model.gguf'),
          'bad\u0000',
        ),
        throwsArgumentError,
      );
      expect(
        LlamaEmbeddings.embedText(
          const LlamaModelConfig(modelPath: 'model.gguf'),
          'bad\u0000',
        ),
        throwsArgumentError,
      );
      expect(
        LlamaEmbeddings.embedTexts(
          const LlamaModelConfig(modelPath: 'model.gguf'),
          const <String>['ok', 'bad\u0000'],
        ),
        throwsArgumentError,
      );
    });

    test('rejects invalid detokenize token ids before native work', () {
      expect(
        LlamaTokenizer.detokenize(
          const LlamaModelConfig(modelPath: 'model.gguf'),
          const <int>[-1],
        ),
        throwsArgumentError,
      );
      expect(
        LlamaTokenizer.detokenize(
          const LlamaModelConfig(modelPath: 'model.gguf'),
          const <int>[0x80000000],
        ),
        throwsArgumentError,
      );
    });

    test('detokenize allows empty token lists without native work', () async {
      final text = await LlamaTokenizer.detokenize(
        const LlamaModelConfig(modelPath: 'model.gguf'),
        const <int>[],
      );

      expect(text, isEmpty);
    });

    test('batch embeddings allow empty batches without native work', () async {
      final embeddings = await LlamaEmbeddings.embedTexts(
        const LlamaModelConfig(modelPath: 'model.gguf'),
        const <String>[],
        config: const EmbeddingConfig(
          normalize: false,
          pooling: EmbeddingPooling.mean,
        ),
      );

      expect(embeddings, isEmpty);
      expect(embeddings.count, 0);
      expect(embeddings.dimensions, 0);
      expect(embeddings.values, isEmpty);
      expect(embeddings.normalized, isFalse);
      expect(embeddings.pooling, EmbeddingPooling.mean);
    });

    test('embedding batches expose flat storage and vector views', () {
      final batch = EmbeddingBatch(
        count: 2,
        dimensions: 3,
        values: Float32List.fromList(<double>[1, 2, 3, 4, 5, 6]),
        normalized: false,
        pooling: EmbeddingPooling.mean,
      );

      expect(batch.values, <double>[1, 2, 3, 4, 5, 6]);
      expect(batch[0], <double>[1, 2, 3]);
      expect(batch.vectorAt(1), <double>[4, 5, 6]);
      expect(batch.normalized, isFalse);
      expect(batch.pooling, EmbeddingPooling.mean);
      expect(batch.toList(), <List<double>>[
        <double>[1, 2, 3],
        <double>[4, 5, 6],
      ]);
      expect(() => batch[2], throwsRangeError);
      expect(() => batch.values[0] = 0, throwsUnsupportedError);
      expect(
        () => EmbeddingBatch(count: 2, dimensions: 3, values: Float32List(5)),
        throwsArgumentError,
      );
    });

    test(
      'reranking allows empty document batches without native work',
      () async {
        final scores = await LlamaReranking.scoreDocuments(
          const LlamaModelConfig(modelPath: 'model.gguf'),
          query: 'query',
          documents: const <String>[],
          config: const RerankingConfig(addSpecial: false, parseSpecial: true),
        );

        expect(scores, isEmpty);
        const reranker = LlamaReranker(
          LlamaModelConfig(modelPath: 'model.gguf'),
          config: RerankingConfig(addSpecial: false, parseSpecial: true),
        );
        expect(reranker.config.addSpecial, isFalse);
        expect(reranker.config.parseSpecial, isTrue);
      },
    );

    test('reranker rejects invalid query before empty-candidate return', () {
      const reranker = LlamaReranker(LlamaModelConfig(modelPath: 'model.gguf'));

      expect(
        reranker.rerank('', const <VectorSearchResult>[]),
        throwsArgumentError,
      );
      expect(
        reranker.rerank('bad\u0000query', const <VectorSearchResult>[]),
        throwsArgumentError,
      );
    });

    test('reranker rejects malformed candidates before native work', () {
      const reranker = LlamaReranker(LlamaModelConfig(modelPath: 'model.gguf'));

      expect(
        reranker.rerank('query', const <VectorSearchResult>[
          VectorSearchResult(
            chunk: TextChunk(
              documentId: 'doc',
              id: '0',
              text: 'candidate',
              tokenCount: 1,
              metadata: <String, Object?>{'bad': Object()},
            ),
            score: 1,
          ),
        ]),
        throwsArgumentError,
      );
    });

    test('reranking rejects empty or NUL text before native work', () {
      expect(
        LlamaReranking.scorePair(
          const LlamaModelConfig(modelPath: 'model.gguf'),
          query: '',
          document: 'document',
        ),
        throwsArgumentError,
      );
      expect(
        LlamaReranking.scorePair(
          const LlamaModelConfig(modelPath: 'model.gguf'),
          query: ' ',
          document: 'document',
        ),
        throwsArgumentError,
      );
      expect(
        LlamaReranking.scorePair(
          const LlamaModelConfig(modelPath: 'model.gguf'),
          query: 'query',
          document: 'bad\u0000document',
        ),
        throwsArgumentError,
      );
      expect(
        LlamaReranking.scorePair(
          const LlamaModelConfig(modelPath: 'model.gguf'),
          query: 'query',
          document: ' ',
        ),
        throwsArgumentError,
      );
    });

    test('rejects empty batch embedding text', () {
      expect(
        LlamaEmbeddings.embedTexts(
          const LlamaModelConfig(modelPath: 'model.gguf'),
          const <String>['ok', ''],
        ),
        throwsArgumentError,
      );
      expect(
        LlamaEmbeddings.embedTexts(
          const LlamaModelConfig(modelPath: 'model.gguf'),
          const <String>['ok', ' '],
        ),
        throwsArgumentError,
      );
    });

    test('rejects unsupported or invalid chat template inputs', () {
      expect(
        LlamaChatTemplate.format(
          const LlamaModelConfig(
            modelPath: 'model.gguf',
            nativeLibraryPath: 'bridge',
          ),
          const <ChatMessage>[],
        ),
        throwsArgumentError,
      );
      expect(
        LlamaChatTemplate.format(
          const LlamaModelConfig(modelPath: 'model.gguf'),
          <ChatMessage>[ChatMessage.user(' ')],
        ),
        throwsArgumentError,
      );
      expect(
        LlamaChatTemplate.format(
          const LlamaModelConfig(modelPath: 'model.gguf'),
          <ChatMessage>[
            ChatMessage.content(
              role: ChatRole.user,
              parts: const <ChatContentPart>[TextPart('')],
            ),
          ],
        ),
        throwsArgumentError,
      );
      expect(
        LlamaChatTemplate.countTokens(
          const LlamaModelConfig(
            modelPath: 'model.gguf',
            mmprojPath: 'mmproj.gguf',
          ),
          <ChatMessage>[
            ChatMessage.content(
              role: ChatRole.user,
              parts: const <ChatContentPart>[ImagePart.fromFile('image.png')],
            ),
          ],
        ),
        throwsUnsupportedFeature,
      );
      expect(
        LlamaChatTemplate.format(
          const LlamaModelConfig(
            modelPath: 'model.gguf',
            mmprojPath: 'mmproj.gguf',
          ),
          <ChatMessage>[
            ChatMessage.content(
              role: ChatRole.user,
              parts: const <ChatContentPart>[VideoPart.fromFile('video.mp4')],
            ),
          ],
        ),
        throwsUnsupportedFeature,
      );
      expect(
        LlamaChatTemplate.format(
          const LlamaModelConfig(
            modelPath: 'model.gguf',
            mmprojPath: 'mmproj.gguf',
          ),
          <ChatMessage>[
            ChatMessage.content(
              role: ChatRole.user,
              parts: <ChatContentPart>[
                for (var i = 0; i < 65; i += 1)
                  ImagePart.fromFile('image-$i.png'),
              ],
            ),
          ],
        ),
        throwsArgumentError,
      );
    });

    test('rejects NUL bytes in chat messages', () {
      expect(
        LlamaChatTemplate.format(
          const LlamaModelConfig(
            modelPath: 'model.gguf',
            nativeLibraryPath: 'bridge',
          ),
          <ChatMessage>[ChatMessage.user('bad\u0000')],
        ),
        throwsArgumentError,
      );
    });

    test('chat content parts aggregate text and gate media', () {
      final textOnly = ChatMessage.content(
        role: ChatRole.user,
        parts: const <ChatContentPart>[TextPart('hello'), TextPart(' world')],
      );

      expect(textOnly.text, 'hello world');
      expect(textOnly.hasNonTextParts, isFalse);
      expect(
        () => ChatMessage.content(
          role: ChatRole.user,
          parts: const <ChatContentPart>[],
        ),
        throwsArgumentError,
      );
      final imageBytes = Uint8List.fromList(<int>[1]);
      final imagePart = ImagePart.fromBytes(imageBytes);
      imageBytes[0] = 2;
      expect(imagePart.bytes!.single, 1);
      expect(() => imagePart.bytes![0] = 3, throwsUnsupportedError);
      expect(() => ImagePart.fromBytes(Uint8List(0)), throwsArgumentError);
      expect(() => AudioPart.fromBytes(Uint8List(0)), throwsArgumentError);
      expect(() => VideoPart.fromBytes(Uint8List(0)), throwsArgumentError);
      expect(
        LlamaChatTemplate.format(
          const LlamaModelConfig(
            modelPath: 'model.gguf',
            nativeLibraryPath: 'bridge',
          ),
          <ChatMessage>[
            ChatMessage.content(
              role: ChatRole.user,
              parts: const <ChatContentPart>[
                ImagePart.fromFile('image.png', mimeType: 'image/ '),
              ],
            ),
          ],
        ),
        throwsArgumentError,
      );
      expect(
        LlamaChatTemplate.format(
          const LlamaModelConfig(
            modelPath: 'model.gguf',
            nativeLibraryPath: 'bridge',
          ),
          <ChatMessage>[
            ChatMessage.content(
              role: ChatRole.user,
              parts: const <ChatContentPart>[
                ImagePart.fromFile('image.png', mimeType: 'image/png '),
              ],
            ),
          ],
        ),
        throwsArgumentError,
      );
      expect(
        LlamaChatTemplate.format(
          const LlamaModelConfig(
            modelPath: 'model.gguf',
            nativeLibraryPath: 'bridge',
          ),
          <ChatMessage>[
            ChatMessage.content(
              role: ChatRole.user,
              parts: const <ChatContentPart>[
                TextPart('look'),
                ImagePart.fromFile('image.png'),
              ],
            ),
          ],
        ),
        throwsA(
          isA<UnsupportedFeatureException>().having(
            (error) => error.message,
            'message',
            allOf(contains('image'), contains('mmproj')),
          ),
        ),
      );
      expect(
        LlamaChatTemplate.format(
          const LlamaModelConfig(
            modelPath: 'model.gguf',
            nativeLibraryPath: 'bridge',
          ),
          <ChatMessage>[
            ChatMessage.content(
              role: ChatRole.user,
              parts: const <ChatContentPart>[
                ImagePart.fromFile('image.png', mimeType: 'text/plain'),
              ],
            ),
          ],
        ),
        throwsArgumentError,
      );
      expect(
        LlamaChatTemplate.format(
          const LlamaModelConfig(
            modelPath: 'model.gguf',
            nativeLibraryPath: 'bridge',
          ),
          <ChatMessage>[
            ChatMessage.content(
              role: ChatRole.user,
              parts: const <ChatContentPart>[
                ImagePart.fromFile('image.png', mimeType: 'image/'),
              ],
            ),
          ],
        ),
        throwsArgumentError,
      );
    });

    test('tool definitions validate and serialize', () {
      const tool = LlamaToolDefinition(
        name: 'search_local',
        description: 'Search local notes.',
        parametersSchema: <String, Object?>{
          'type': 'object',
          'additionalProperties': false,
          'required': <Object?>['query'],
          'properties': <String, Object?>{
            'query': <String, Object?>{'type': 'string'},
          },
        },
      );

      final exportedTool = tool.toJson();
      expect(exportedTool['type'], 'function');
      expect(() => exportedTool['type'] = 'bad', throwsUnsupportedError);
      final exportedToolFunction =
          exportedTool['function']! as Map<String, Object?>;
      expect(
        () => exportedToolFunction['name'] = 'changed',
        throwsUnsupportedError,
      );
      final exportedSchema =
          exportedToolFunction['parameters']! as Map<String, Object?>;
      final exportedProperties =
          exportedSchema['properties']! as Map<String, Object?>;
      expect(
        () => exportedProperties['query'] = <String, Object?>{'type': 'number'},
        throwsUnsupportedError,
      );
      expect(
        const LlamaToolCallingConfig(
          tools: <LlamaToolDefinition>[tool],
        ).toJson().single['type'],
        'function',
      );
      expect(
        const LlamaToolCallingConfig(
          tools: <LlamaToolDefinition>[tool],
          allowParallelToolCalls: false,
        ).toOpenAiJson()['parallel_tool_calls'],
        isFalse,
      );
      expect(
        const LlamaToolCallingConfig(
          tools: <LlamaToolDefinition>[tool],
          toolChoice: LlamaToolChoice.required(),
        ).toOpenAiJson()['tool_choice'],
        'required',
      );
      final namedToolChoice = const LlamaToolCallingConfig(
        tools: <LlamaToolDefinition>[tool],
        toolChoice: LlamaToolChoice.named('search_local'),
      ).toOpenAiJson();
      expect(namedToolChoice['tool_choice'], <String, Object?>{
        'type': 'function',
        'function': <String, Object?>{'name': 'search_local'},
      });
      expect(
        () => namedToolChoice['tool_choice'] = 'none',
        throwsUnsupportedError,
      );
      final namedToolChoiceValue =
          namedToolChoice['tool_choice']! as Map<String, Object?>;
      expect(
        () => namedToolChoiceValue['type'] = 'changed',
        throwsUnsupportedError,
      );
      final namedToolChoiceFunction =
          namedToolChoiceValue['function']! as Map<String, Object?>;
      expect(
        () => namedToolChoiceFunction['name'] = 'changed',
        throwsUnsupportedError,
      );
      expect(
        const LlamaToolCallingConfig(
          tools: <LlamaToolDefinition>[tool],
        ).validate,
        returnsNormally,
      );
      expect(
        const LlamaToolCallingConfig(
          tools: <LlamaToolDefinition>[tool, tool],
        ).validate,
        throwsArgumentError,
      );
      expect(
        const GenerationConfig(
          toolCalling: LlamaToolCallingConfig(
            tools: <LlamaToolDefinition>[tool],
            allowParallelToolCalls: false,
          ),
        ).validate,
        returnsNormally,
      );
      expect(
        const LlamaToolDefinition(
          name: '',
          description: 'bad',
          parametersSchema: <String, Object?>{},
        ).validate,
        throwsArgumentError,
      );
      expect(
        const LlamaToolDefinition(
          name: 'bad\nname',
          description: 'bad',
          parametersSchema: <String, Object?>{},
        ).validate,
        throwsArgumentError,
      );
      expect(
        const LlamaToolDefinition(
          name: 'bad name',
          description: 'bad',
          parametersSchema: <String, Object?>{},
        ).validate,
        throwsArgumentError,
      );
      expect(
        const LlamaToolDefinition(
          name: 'bad_description',
          description: ' ',
          parametersSchema: <String, Object?>{},
        ).validate,
        throwsArgumentError,
      );
      expect(
        const LlamaToolDefinition(
          name: 'bad_schema',
          description: 'bad',
          parametersSchema: <String, Object?>{
            'type': 'object',
            'properties': <String, Object?>{
              'bad': <String, Object?>{
                'enum': <Object?>[double.nan],
              },
            },
          },
        ).validate,
        throwsArgumentError,
      );
      expect(
        const LlamaToolCallingConfig(
          tools: <LlamaToolDefinition>[tool],
          toolChoice: LlamaToolChoice.named('missing'),
        ).validate,
        throwsArgumentError,
      );
      expect(
        const LlamaToolCallingConfig(
          tools: <LlamaToolDefinition>[tool],
          toolChoice: LlamaToolChoice.named('bad\tname'),
        ).validate,
        throwsArgumentError,
      );
      expect(
        const LlamaToolCallingConfig(
          toolChoice: LlamaToolChoice.required(),
        ).validate,
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(
          grammar: 'root ::= "x"',
          toolCalling: LlamaToolCallingConfig(
            tools: <LlamaToolDefinition>[tool],
          ),
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => GenerationConfig.jsonSchema(
          schema: const <String, Object?>{'type': 'object'},
          toolCalling: const LlamaToolCallingConfig(
            tools: <LlamaToolDefinition>[tool],
          ),
        ).validate(),
        throwsArgumentError,
      );
    });

    test('tool definitions validate parsed arguments against their schema', () {
      const tool = LlamaToolDefinition(
        name: 'record_probe',
        description: 'Record a local probe.',
        parametersSchema: <String, Object?>{
          'type': 'object',
          'properties': <String, Object?>{
            'sentinel': <String, Object?>{
              'type': 'string',
              'enum': <Object?>['EXPECTED_SENTINEL'],
            },
            'samples': <String, Object?>{
              'type': 'array',
              'items': <String, Object?>{'\$ref': '#/\$defs/sample'},
              'minItems': 1,
            },
          },
          'required': <Object?>['sentinel'],
          'additionalProperties': false,
          '\$defs': <String, Object?>{
            'sample': <String, Object?>{
              'type': 'integer',
              'minimum': 0,
              'maximum': 10,
            },
          },
        },
      );

      expect(
        () => tool.validateArguments(<String, Object?>{
          'sentinel': 'EXPECTED_SENTINEL',
          'samples': <Object?>[0, 10],
        }),
        returnsNormally,
      );
      expect(
        () => tool.validateArguments(<String, Object?>{
          'sentinel': 'EXPECTED_SENTINEL',
          'markdown': '**reasoning**',
        }),
        throwsArgumentError,
      );
      expect(
        () => tool.validateArguments(<String, Object?>{}),
        throwsArgumentError,
      );
      expect(
        () => tool.validateArguments(<String, Object?>{'sentinel': 'wrong'}),
        throwsArgumentError,
      );
      expect(
        () => tool.validateArguments(<String, Object?>{
          'sentinel': 'EXPECTED_SENTINEL',
          'samples': <Object?>[11],
        }),
        throwsArgumentError,
      );
    });

    test('typed tool messages snapshot calls and expose result metadata', () {
      final arguments = <String, Object?>{'query': 'alpha'};
      final calls = <LlamaToolCall>[
        LlamaToolCall(id: 'call_1', name: 'lookup', arguments: arguments),
      ];
      final assistant = ChatMessage.assistantToolCalls(toolCalls: calls);
      arguments['query'] = 'changed';
      calls.clear();

      expect(assistant.role, ChatRole.assistant);
      expect(assistant.text, isEmpty);
      expect(assistant.toolCalls.single.arguments['query'], 'alpha');
      expect(() => assistant.toolCalls.clear(), throwsUnsupportedError);

      const result = ChatMessage.toolResult(
        toolCallId: 'call_1',
        name: 'lookup',
        text: 'local result',
      );
      expect(result.role, ChatRole.tool);
      expect(result.toolCallId, 'call_1');
      expect(result.toolName, 'lookup');
      expect(
        () => ChatMessage.assistantToolCalls(toolCalls: const []),
        throwsArgumentError,
      );

      final finalChunk = GenerationChunk(
        text: '',
        isDone: true,
        assistantMessage: assistant,
      );
      expect(finalChunk.assistantMessage, same(assistant));
    });

    test('rejects malformed typed tool history before native work', () {
      const config = LlamaModelConfig(
        modelPath: 'model.gguf',
        nativeLibraryPath: 'bridge',
      );
      expect(
        LlamaChatTemplate.format(config, <ChatMessage>[
          ChatMessage.assistantToolCalls(
            toolCalls: const <LlamaToolCall>[
              LlamaToolCall(
                id: 'bad id',
                name: 'lookup',
                arguments: <String, Object?>{},
              ),
            ],
          ),
        ]),
        throwsArgumentError,
      );
      expect(
        LlamaChatTemplate.format(config, const <ChatMessage>[
          ChatMessage.toolResult(
            toolCallId: 'bad id',
            name: 'lookup',
            text: 'result',
          ),
        ]),
        throwsArgumentError,
      );
    });

    test('tool call parser handles direct and OpenAI-shaped JSON', () {
      final direct = LlamaToolCalls.parse(
        '{"name":"search_local","arguments":{"query":"llama"}}',
      );

      expect(direct.single.name, 'search_local');
      expect(direct.single.arguments['query'], 'llama');
      final typedAsObject = LlamaToolCalls.fromJson(<Object?, Object?>{
        'name': 'search_local',
        'arguments': <Object?, Object?>{'query': 'typed'},
      });

      expect(typedAsObject.single.arguments['query'], 'typed');

      final openAi = LlamaToolCalls.parse(
        '{"tool_calls":[{"id":"call_1","function":{"name":"search_local",'
        '"arguments":"{\\"query\\":\\"rag\\"}"}}]}',
      );

      expect(openAi.single.id, 'call_1');
      expect(openAi.single.arguments['query'], 'rag');
      final nestedArguments = <Object?>['original'];
      final arguments = <String, Object?>{'query': nestedArguments};
      final exported = LlamaToolCall(
        name: 'search_local',
        arguments: arguments,
      ).toJson();
      final exportedOpenAi = LlamaToolCall(
        name: 'search_local',
        arguments: arguments,
      ).toOpenAiJson();
      nestedArguments[0] = 'changed';

      expect(openAi.single.toJson()['name'], 'search_local');
      expect(() => exported['name'] = 'changed', throwsUnsupportedError);
      final exportedArguments = exported['arguments']! as Map<String, Object?>;
      final exportedQuery = exportedArguments['query']! as List<Object?>;
      expect(exportedQuery.single, 'original');
      expect(() => exportedQuery[0] = 'changed', throwsUnsupportedError);
      final exportedOpenAiFunction =
          exportedOpenAi['function']! as Map<String, Object?>;
      expect(() => exportedOpenAi['type'] = 'bad', throwsUnsupportedError);
      expect(
        () => exportedOpenAiFunction['name'] = 'changed',
        throwsUnsupportedError,
      );
      expect(
        jsonDecode(exportedOpenAiFunction['arguments']! as String),
        <String, Object?>{
          'query': <Object?>['original'],
        },
      );
      final exportedOpenAiCalls = LlamaToolCalls.toOpenAiJson(openAi);
      final exportedOpenAiCallsList =
          exportedOpenAiCalls['tool_calls']! as List<Map<String, Object?>>;
      expect(
        () => exportedOpenAiCalls['tool_calls'] = <Object?>[],
        throwsUnsupportedError,
      );
      expect(
        () => exportedOpenAiCallsList.add(exportedOpenAi),
        throwsUnsupportedError,
      );
      expect(
        () => LlamaToolCalls.toOpenAiJson(const <LlamaToolCall>[]),
        throwsArgumentError,
      );
      expect(
        () => LlamaToolCalls.toOpenAiJson(const <LlamaToolCall>[
          LlamaToolCall(
            id: 'call_1',
            name: 'search_local',
            arguments: <String, Object?>{},
          ),
          LlamaToolCall(
            id: 'call_1',
            name: 'lookup',
            arguments: <String, Object?>{},
          ),
        ]),
        throwsArgumentError,
      );
      final roundTrip = LlamaToolCalls.fromJson(exportedOpenAiCalls);
      expect(roundTrip.single.id, 'call_1');
      expect(roundTrip.single.arguments['query'], 'rag');
      expect(
        () => LlamaToolCalls.parse(
          '[{"name":"a","arguments":{}},{"name":"b","arguments":{}}]',
          allowParallelToolCalls: false,
        ),
        throwsUnsupportedFeature,
      );
      expect(() => LlamaToolCalls.parse('{'), throwsArgumentError);
      expect(() => LlamaToolCalls.parse('[]'), throwsArgumentError);
      expect(
        () => LlamaToolCalls.parse('{"name":"","arguments":{}}'),
        throwsArgumentError,
      );
      expect(
        () => LlamaToolCalls.parse('{"tool_calls":null}'),
        throwsArgumentError,
      );
      expect(
        () => LlamaToolCalls.parse('{"tool_calls":[]}'),
        throwsArgumentError,
      );
      expect(
        () => LlamaToolCalls.parse('{"name":"search_local","arguments":null}'),
        throwsArgumentError,
      );
      expect(
        () => LlamaToolCalls.parse('{"name":"search_local","arguments":""}'),
        throwsArgumentError,
      );
      expect(
        () => LlamaToolCalls.parse('{"tool_calls":[],"extra":true}'),
        throwsArgumentError,
      );
      expect(
        () => LlamaToolCalls.parse(
          '{"type":"custom","name":"search_local","arguments":{}}',
        ),
        throwsArgumentError,
      );
      expect(
        () => LlamaToolCalls.parse(
          '{"name":"search_local","arguments":{},"extra":true}',
        ),
        throwsArgumentError,
      );
      expect(
        () => LlamaToolCalls.parse(
          '{"tool_calls":[{"id":"","function":{"name":"search_local",'
          '"arguments":"{}"}}]}',
        ),
        throwsArgumentError,
      );
      expect(
        () => LlamaToolCalls.parse('{"name":"bad\\nname","arguments":{}}'),
        throwsArgumentError,
      );
      expect(
        () => LlamaToolCalls.parse('{"name":"bad name","arguments":{}}'),
        throwsArgumentError,
      );
      expect(
        () => LlamaToolCalls.parse(
          '{"tool_calls":[{"id":"bad\\nid","function":{"name":"search_local",'
          '"arguments":"{}"}}]}',
        ),
        throwsArgumentError,
      );
      expect(
        () => LlamaToolCalls.parse(
          '{"tool_calls":[{"type":"custom","function":{"name":"search_local",'
          '"arguments":"{}"}}]}',
        ),
        throwsArgumentError,
      );
      expect(
        () => LlamaToolCalls.parse(
          '{"tool_calls":[{"id":"call_1","extra":true,'
          '"function":{"name":"search_local","arguments":"{}"}}]}',
        ),
        throwsArgumentError,
      );
      expect(
        () => LlamaToolCalls.parse(
          '{"tool_calls":[{"id":"call_1","function":{"name":"search_local",'
          '"arguments":"{}","extra":true}}]}',
        ),
        throwsArgumentError,
      );
      expect(
        () => LlamaToolCalls.parse(
          '{"tool_calls":[{"id":"call_1","function":{"name":"search_local",'
          '"arguments":null}}]}',
        ),
        throwsArgumentError,
      );
      expect(
        () => LlamaToolCalls.parse(
          '{"tool_calls":[{"id":"call_1","function":{"name":"search_local",'
          '"arguments":""}}]}',
        ),
        throwsArgumentError,
      );
      expect(
        () => LlamaToolCalls.parse(
          '{"tool_calls":[{"id":"call_1","function":{"name":"search_local",'
          '"arguments":"{"}}]}',
        ),
        throwsArgumentError,
      );
      expect(
        () => LlamaToolCalls.parse(
          '{"tool_calls":[{"id":"call_1","function":{"name":"search_local",'
          '"arguments":"{}"}},{"id":"call_1","function":{"name":"lookup",'
          '"arguments":"{}"}}]}',
        ),
        throwsArgumentError,
      );
      expect(
        () => const LlamaToolCall(
          name: 'search local',
          arguments: <String, Object?>{},
        ).toJson(),
        throwsArgumentError,
      );
      expect(
        () => LlamaToolCalls.fromJson(<String, Object?>{
          'name': 'search_local',
          'arguments': <String, Object?>{'bad': Object()},
        }),
        throwsArgumentError,
      );
      expect(
        () => const LlamaToolCall(
          name: 'search_local',
          arguments: <String, Object?>{'bad': Object()},
        ).toOpenAiJson(),
        throwsArgumentError,
      );
      expect(
        () => const LlamaToolCall(
          name: 'search_local',
          arguments: <String, Object?>{'bad': 'value\u0000'},
        ).toOpenAiJson(),
        throwsArgumentError,
      );
    });
  });
}

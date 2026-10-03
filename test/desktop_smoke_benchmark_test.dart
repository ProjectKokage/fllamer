import 'dart:async';
import 'dart:io';

import 'package:fllamer/fllamer.dart';
import 'package:test/test.dart';

import '../benchmark/desktop_smoke/benchmark.dart' as desktop_benchmark;
import '../example/dart_cli/chat.dart' as chat_cli;
import '../example/dart_cli/local_rag.dart' as local_rag_cli;

void main() {
  group('desktop smoke benchmark', () {
    test('reports Dart build mode metadata', () {
      expect(desktop_benchmark.benchmarkBuildMode(), 'jit');
      expect(Platform.version, isNotEmpty);
    });

    test('reports process memory metadata', () {
      final memory = desktop_benchmark.benchmarkMemorySnapshot();

      expect(memory['currentRssBytes'], greaterThan(0));
      expect(memory['maxRssBytes'], greaterThan(0));
    });

    test('reports stable benchmark model file names', () {
      expect(
        desktop_benchmark.benchmarkModelFileName('/models/tiny-q4.gguf'),
        'tiny-q4.gguf',
      );
      expect(
        desktop_benchmark.benchmarkModelFileName(r'C:\models\tiny-q4.gguf'),
        'tiny-q4.gguf',
      );
      expect(
        desktop_benchmark.benchmarkKvCacheTypeName(KvCacheType.iq4Nl),
        'iq4_nl',
      );
      expect(
        desktop_benchmark.benchmarkFlashAttentionName(
          FlashAttentionMode.disabled,
        ),
        'disabled',
      );
    });

    test('parses benchmark CLI options', () {
      final options = desktop_benchmark.parseBenchmarkArgs(<String>[
        '--model',
        'model.gguf',
        '--native-library=build/native/libllama_dart_bridge.dylib',
        '--device-model',
        'MacBookPro18,3',
        '--prompt',
        'hello',
        '--max-tokens',
        '8',
        '--iterations',
        '5',
        '--no-warm-up',
        '--context-size',
        '1024',
        '--batch-size',
        '128',
        '--ubatch-size',
        '64',
        '--threads',
        '2',
        '--batch-threads',
        '3',
        '--gpu-layers',
        '0',
        '--kv-cache-key',
        'q4_0',
        '--kv-cache-value',
        'q8_0',
        '--no-kv-offload',
        '--flash-attention',
        'enabled',
        '--no-swa-full',
        '--kv-unified',
        '--seed',
        '42',
        '--spec-ngram-simple',
        '--spec-ngram-size',
        '8',
        '--spec-min-draft-length',
        '12',
        '--spec-draft-length',
        '16',
        '--json-out',
        'benchmark.json',
      ]);

      expect(options.modelPath, 'model.gguf');
      expect(
        options.nativeLibraryPath,
        'build/native/libllama_dart_bridge.dylib',
      );
      expect(options.deviceModel, 'MacBookPro18,3');
      expect(options.prompt, 'hello');
      expect(options.maxTokens, 8);
      expect(options.iterations, 5);
      expect(options.warmUp, isFalse);
      expect(options.contextSize, 1024);
      expect(options.batchSize, 128);
      expect(options.ubatchSize, 64);
      expect(options.threads, 2);
      expect(options.batchThreads, 3);
      expect(options.gpuLayers, 0);
      expect(options.kvCacheKeyType, KvCacheType.q4Zero);
      expect(options.kvCacheValueType, KvCacheType.q8Zero);
      expect(options.kvCacheOffload, isFalse);
      expect(options.flashAttention, FlashAttentionMode.enabled);
      expect(options.swaFull, isFalse);
      expect(options.kvUnified, isTrue);
      expect(options.seed, 42);
      expect(options.speculativeNgramSimple, isTrue);
      expect(options.speculativeNgramStrategy, 'ngram-simple');
      expect(options.speculativeNgramSize, 8);
      expect(options.speculativeMinimumDraftLength, 12);
      expect(options.speculativeDraftLength, 16);
      expect(options.jsonOut, 'benchmark.json');
      expect(
        desktop_benchmark.parseBenchmarkArgs(<String>[
          '--help',
          '--max-tokens',
        ]).help,
        isTrue,
      );
      expect(
        desktop_benchmark.parseBenchmarkArgs(<String>[
          '--spec-ngram',
          'ngram-map-k4v',
        ]).speculativeNgramStrategy,
        'ngram-map-k4v',
      );
      final modOptions = desktop_benchmark.parseBenchmarkArgs(<String>[
        '--spec-ngram',
        'ngram-mod',
        '--spec-ngram-size',
        '24',
        '--spec-min-draft-length',
        '48',
        '--spec-draft-length',
        '64',
      ]);
      expect(
        desktop_benchmark.benchmarkSpeculativeDecodingConfig(modOptions),
        isA<NGramModSpeculation>()
            .having((config) => config.matchLength, 'matchLength', 24)
            .having(
              (config) => config.minimumDraftLength,
              'minimumDraftLength',
              48,
            )
            .having(
              (config) => config.maximumDraftLength,
              'maximumDraftLength',
              64,
            ),
      );
      final cacheConfig = desktop_benchmark.benchmarkSpeculativeDecodingConfig(
        desktop_benchmark.parseBenchmarkArgs(<String>[
          '--spec-ngram',
          'ngram-cache',
        ]),
      );
      expect(cacheConfig, isA<NGramCacheSpeculation>());
      expect(
        desktop_benchmark.benchmarkSpeculativeDecodingRecord(cacheConfig),
        <String, Object?>{'strategy': 'ngram-cache'},
      );
    });

    test('rejects invalid benchmark CLI options', () {
      expect(
        () => desktop_benchmark.parseBenchmarkArgs(<String>['--max-tokens']),
        throwsFormatException,
      );
      expect(
        () => desktop_benchmark.parseBenchmarkArgs(<String>[
          '--model',
          '--max-tokens',
          '8',
        ]),
        throwsA(
          isA<FormatException>().having(
            (error) => error.message,
            'message',
            'Missing value for --model.',
          ),
        ),
      );
      expect(
        () => desktop_benchmark.parseBenchmarkArgs(<String>[
          '--gpu-layers',
          '-1',
        ]),
        throwsFormatException,
      );
      expect(
        () => desktop_benchmark.parseBenchmarkArgs(<String>['--model=']),
        throwsFormatException,
      );
      expect(
        () => desktop_benchmark.parseBenchmarkArgs(<String>[
          '--model',
          'bad\nmodel.gguf',
        ]),
        throwsFormatException,
      );
      expect(
        () => desktop_benchmark.parseBenchmarkArgs(<String>[
          '--native-library',
          'bad\rbridge',
        ]),
        throwsFormatException,
      );
      expect(
        () => desktop_benchmark.parseBenchmarkArgs(<String>[
          '--prompt',
          'bad\u0000',
        ]),
        throwsFormatException,
      );
      expect(
        () => desktop_benchmark.parseBenchmarkArgs(<String>['--json-out=']),
        throwsFormatException,
      );
      expect(
        () => desktop_benchmark.parseBenchmarkArgs(<String>[
          '--json-out',
          'bad\nbenchmark.json',
        ]),
        throwsFormatException,
      );
      expect(
        () => desktop_benchmark.parseBenchmarkArgs(<String>['--unknown']),
        throwsFormatException,
      );
      expect(
        () => desktop_benchmark.parseBenchmarkArgs(<String>[
          '--spec-ngram-simple=true',
        ]),
        throwsFormatException,
      );
      expect(
        () => desktop_benchmark.parseBenchmarkArgs(<String>[
          '--spec-ngram',
          'ngram-unknown',
        ]),
        throwsFormatException,
      );
      expect(
        () =>
            desktop_benchmark.parseBenchmarkArgs(<String>['--iterations', '0']),
        throwsFormatException,
      );
      expect(
        () => desktop_benchmark.parseBenchmarkArgs(<String>[
          '--iterations',
          '1001',
        ]),
        throwsFormatException,
      );
      expect(
        () =>
            desktop_benchmark.parseBenchmarkArgs(<String>['--no-warm-up=true']),
        throwsFormatException,
      );
      expect(
        () => desktop_benchmark.parseBenchmarkArgs(<String>[
          '--kv-cache-key',
          'q2_bad',
        ]),
        throwsFormatException,
      );
      expect(
        () => desktop_benchmark.parseBenchmarkArgs(<String>[
          '--flash-attention',
          'sometimes',
        ]),
        throwsFormatException,
      );
      expect(
        () => desktop_benchmark.parseBenchmarkArgs(<String>[
          '--no-kv-offload=true',
        ]),
        throwsFormatException,
      );
      expect(
        () => desktop_benchmark.parseBenchmarkArgs(<String>[
          '--no-swa-full=true',
        ]),
        throwsFormatException,
      );
      expect(
        () =>
            desktop_benchmark.parseBenchmarkArgs(<String>['--kv-unified=true']),
        throwsFormatException,
      );
      expect(
        () => desktop_benchmark.parseBenchmarkArgs(<String>[
          '--kv-cache-value',
          'q8_0',
          '--flash-attention',
          'disabled',
        ]),
        throwsFormatException,
      );
      expect(
        () => desktop_benchmark.parseBenchmarkArgs(<String>[
          '--batch-size',
          '8',
          '--ubatch-size',
          '16',
        ]),
        throwsFormatException,
      );
      expect(
        () => desktop_benchmark.parseBenchmarkArgs(<String>[
          '--spec-ngram-size',
          '16',
          '--spec-draft-length',
          '8',
        ]),
        throwsFormatException,
      );
      expect(
        () => desktop_benchmark.parseBenchmarkArgs(<String>[
          '--spec-ngram',
          'ngram-mod',
          '--spec-min-draft-length',
          '65',
          '--spec-draft-length',
          '64',
        ]),
        throwsFormatException,
      );
    });

    test('benchmark CLI reports usage for invalid options', () async {
      final output = StringBuffer();
      final errors = StringBuffer();
      final status = await desktop_benchmark.runBenchmarkCli(
        const <String>['--max-tokens'],
        output: output,
        errorOutput: errors,
      );

      expect(status, 64);
      expect(output.toString(), isEmpty);
      expect(errors.toString(), contains('Missing value for --max-tokens.'));
      expect(errors.toString(), contains('Usage: dart run'));
      expect(errors.toString(), isNot(contains('Unhandled exception')));
    });

    test('benchmark CLI reports help in process', () async {
      final output = StringBuffer();
      final errors = StringBuffer();
      final status = await desktop_benchmark.runBenchmarkCli(
        const <String>['--help'],
        output: output,
        errorOutput: errors,
      );

      expect(status, 0);
      expect(output.toString(), contains('Usage: dart run'));
      expect(errors.toString(), isEmpty);
    });

    test('chat CLI reports usage for invalid options', () async {
      Future<(int, String, String)> invoke(List<String> args) async {
        final output = StringBuffer();
        final errors = StringBuffer();
        final status = await chat_cli.runChatCli(
          args,
          output: output,
          errorOutput: errors,
        );
        return (status, output.toString(), errors.toString());
      }

      final result = await invoke(const <String>['--model', 'bad\nmodel.gguf']);
      expect(result.$1, 64);
      expect(result.$2, isEmpty);
      expect(result.$3, contains('--model must not contain line breaks.'));
      expect(result.$3, contains('Usage: dart run'));
      expect(result.$3, isNot(contains('Unhandled exception')));

      final tooLarge = await invoke(const <String>[
        '--model',
        'model.gguf',
        '--max-tokens',
        '2147483648',
      ]);
      expect(tooLarge.$1, 64);
      expect(tooLarge.$2, isEmpty);
      expect(tooLarge.$3, contains('--max-tokens must be between 1'));
      expect(tooLarge.$3, contains('Usage: dart run'));
      expect(tooLarge.$3, isNot(contains('Unhandled exception')));

      final missingBeforeFlag = await invoke(const <String>[
        '--model',
        '--max-tokens',
        '8',
      ]);
      expect(missingBeforeFlag.$1, 64);
      expect(missingBeforeFlag.$2, isEmpty);
      expect(missingBeforeFlag.$3, contains('Missing value for --model.'));
      expect(missingBeforeFlag.$3, contains('Usage: dart run'));
      expect(missingBeforeFlag.$3, isNot(contains('Unhandled exception')));
    });

    test('local RAG CLI reports usage for invalid query text', () async {
      final output = StringBuffer();
      final errors = StringBuffer();
      final status = await local_rag_cli.runLocalRagCli(
        const <String>[''],
        output: output,
        errorOutput: errors,
      );

      expect(status, 64);
      expect(output.toString(), isEmpty);
      expect(errors.toString(), contains('Query must not be empty.'));
      expect(errors.toString(), contains('Usage: dart run'));
      expect(errors.toString(), isNot(contains('Unhandled exception')));
    });

    test('local RAG CLI runs in process', () async {
      final output = StringBuffer();
      final errors = StringBuffer();
      final status = await local_rag_cli.runLocalRagCli(
        const <String>[],
        output: output,
        errorOutput: errors,
      );

      expect(status, 0);
      expect(output.toString(), contains('Citations:'));
      expect(output.toString(), contains('privacy#privacy:0'));
      expect(errors.toString(), isEmpty);
    });

    test('rejects invalid direct benchmark options before native work', () {
      expect(
        desktop_benchmark.runDesktopSmokeBenchmark(
          const desktop_benchmark.DesktopSmokeBenchmarkOptions(
            modelPath: 'model.gguf',
            prompt: '',
          ),
        ),
        throwsFormatException,
      );
      expect(
        desktop_benchmark.runDesktopSmokeBenchmark(
          const desktop_benchmark.DesktopSmokeBenchmarkOptions(
            modelPath: 'model.gguf',
            kvCacheValueType: KvCacheType.q8Zero,
            flashAttention: FlashAttentionMode.disabled,
          ),
        ),
        throwsFormatException,
      );
      expect(
        desktop_benchmark.runDesktopSmokeBenchmark(
          const desktop_benchmark.DesktopSmokeBenchmarkOptions(
            modelPath: 'model.gguf',
            batchSize: 8,
            ubatchSize: 16,
          ),
        ),
        throwsFormatException,
      );
      expect(
        desktop_benchmark.runDesktopSmokeBenchmark(
          const desktop_benchmark.DesktopSmokeBenchmarkOptions(
            modelPath: 'model.gguf',
            maxTokens: 0,
          ),
        ),
        throwsFormatException,
      );
      expect(
        desktop_benchmark.runDesktopSmokeBenchmark(
          const desktop_benchmark.DesktopSmokeBenchmarkOptions(
            modelPath: 'model.gguf',
            gpuLayers: -1,
          ),
        ),
        throwsFormatException,
      );
      expect(
        desktop_benchmark.runDesktopSmokeBenchmark(
          const desktop_benchmark.DesktopSmokeBenchmarkOptions(
            modelPath: 'model.gguf',
            seed: 0x100000000,
          ),
        ),
        throwsFormatException,
      );
      expect(
        desktop_benchmark.runDesktopSmokeBenchmark(
          const desktop_benchmark.DesktopSmokeBenchmarkOptions(
            modelPath: 'model.gguf',
            jsonOut: 'bad\nbenchmark.json',
          ),
        ),
        throwsFormatException,
      );
      expect(
        desktop_benchmark.runDesktopSmokeBenchmark(
          const desktop_benchmark.DesktopSmokeBenchmarkOptions(
            modelPath: 'model.gguf',
            iterations: 0,
          ),
        ),
        throwsFormatException,
      );
      expect(
        desktop_benchmark.runDesktopSmokeBenchmark(
          const desktop_benchmark.DesktopSmokeBenchmarkOptions(
            modelPath: 'model.gguf',
            speculativeNgramSize: 1025,
          ),
        ),
        throwsFormatException,
      );
    });

    test('aggregates sustained benchmark telemetry', () {
      const first = GenerationTelemetry(
        promptTokens: 10,
        generatedTokens: 20,
        promptEvalMs: 100,
        decodeMs: 200,
        totalMs: 300,
        timeToFirstTokenMs: 40,
        speculativeDraftTokens: 8,
        speculativeAcceptedTokens: 4,
        speculativeDraftMs: 12,
        speculativeVerifyMs: 15,
      );
      const second = GenerationTelemetry(
        promptTokens: 10,
        generatedTokens: 20,
        promptEvalMs: 100,
        decodeMs: 200,
        totalMs: 300,
        timeToFirstTokenMs: 60,
        speculativeDraftTokens: 12,
        speculativeAcceptedTokens: 6,
        speculativeDraftMs: 18,
        speculativeVerifyMs: 25,
      );

      final aggregate = desktop_benchmark.benchmarkAggregateTelemetry(
        const <GenerationTelemetry>[first, second],
      );

      expect(aggregate['iterations'], 2);
      expect(aggregate['promptTokens'], 20);
      expect(aggregate['generatedTokens'], 40);
      expect(aggregate['timeToFirstTokenMsMean'], 50);
      expect(aggregate['timeToFirstTokenMsMin'], 40);
      expect(aggregate['timeToFirstTokenMsMax'], 60);
      expect(aggregate['promptEvalTokensPerSecond'], 100);
      expect(aggregate['decodeTokensPerSecond'], 100);
      expect(aggregate['speculativeDraftTokens'], 20);
      expect(aggregate['speculativeAcceptedTokens'], 10);
      expect(aggregate['speculativeAcceptanceRate'], 0.5);
      expect(aggregate['speculativeDraftMs'], 30);
      expect(aggregate['speculativeVerifyMs'], 40);
      expect(
        () => desktop_benchmark.benchmarkAggregateTelemetry(
          const <GenerationTelemetry>[],
        ),
        throwsArgumentError,
      );
    });
  });
}

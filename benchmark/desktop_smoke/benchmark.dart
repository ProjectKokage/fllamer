import 'dart:convert';
import 'dart:io';

import 'package:fllamer/fllamer.dart';

const _defaultPrompt = 'Write one sentence about local inference.';
const _maxInt32 = 0x7FFFFFFF;
const _maxUint32 = 0xFFFFFFFF;
const _maxSpeculativeNgramSize = 1024;
const _maxBenchmarkIterations = 1000;

Future<void> main(List<String> args) async {
  final status = await runBenchmarkCli(args);
  if (status != 0) {
    exitCode = status;
  }
}

Future<int> runBenchmarkCli(
  List<String> args, {
  StringSink? output,
  StringSink? errorOutput,
}) async {
  final out = output ?? stdout;
  final errors = errorOutput ?? stderr;
  final DesktopSmokeBenchmarkOptions options;
  try {
    options = parseBenchmarkArgs(args);
  } on FormatException catch (error) {
    errors.writeln(error.message);
    errors.writeln();
    errors.writeln(benchmarkUsage);
    return 64;
  }
  if (options.help) {
    out.writeln(benchmarkUsage);
    return 0;
  }
  if (options.modelPath == null) {
    errors.writeln('Missing required --model path.\n');
    errors.writeln(benchmarkUsage);
    return 64;
  }

  final result = await runDesktopSmokeBenchmark(options);
  final json = const JsonEncoder.withIndent('  ').convert(result);
  final jsonOut = options.jsonOut;
  if (jsonOut == null) {
    out.writeln(json);
  } else {
    await File(jsonOut).writeAsString('$json\n', flush: true);
  }
  return 0;
}

const benchmarkUsage = '''
Usage: dart run benchmark/desktop_smoke/benchmark.dart --model MODEL.gguf [options]

Options:
  --native-library PATH   Native bridge library path.
  --device-model TEXT     Device/host model recorded in JSON.
  --prompt TEXT           Prompt text.
  --max-tokens N          Generated-token limit. Default: 128.
  --iterations N          Reset and generate N times. Default: 3.
  --no-warm-up            Skip the empty-context native warm-up.
  --context-size N        Context size. Default: 4096.
  --batch-size N          Batch size. Default: 512.
  --ubatch-size N         Micro-batch size.
  --threads N             Decode thread count.
  --batch-threads N       Prompt/batch thread count.
  --gpu-layers N          GPU layer count. Default: 0.
  --kv-cache-key TYPE     K cache type. Default: f16.
  --kv-cache-value TYPE   V cache type. Default: f16.
  --no-kv-offload         Keep KV cache and KQV operations off accelerators.
  --flash-attention MODE  auto, disabled, or enabled. Default: auto.
  --no-swa-full           Disable the full-size sliding-window cache.
  --kv-unified            Use one unified KV buffer across sequences.
  --seed N                Generation seed.
  --spec-ngram STRATEGY   ngram-simple, ngram-map-k, ngram-map-k4v,
                          ngram-mod, or ngram-cache.
  --spec-ngram-simple     Alias for --spec-ngram ngram-simple.
  --spec-ngram-size N     Lookup/match length. Default: 12.
  --spec-min-draft-length N
                          ngram-mod minimum draft length. Default: 48.
  --spec-draft-length N   Draft/maximum draft length. Default: 48.
  --json-out PATH         Write JSON output to a file.
  --help                  Show this help.
''';

final class DesktopSmokeBenchmarkOptions {
  const DesktopSmokeBenchmarkOptions({
    this.help = false,
    this.modelPath,
    this.nativeLibraryPath,
    this.deviceModel,
    this.prompt = _defaultPrompt,
    this.maxTokens = 128,
    this.iterations = 3,
    this.warmUp = true,
    this.contextSize = 4096,
    this.batchSize = 512,
    this.ubatchSize,
    this.threads,
    this.batchThreads,
    this.gpuLayers = 0,
    this.kvCacheKeyType = KvCacheType.f16,
    this.kvCacheValueType = KvCacheType.f16,
    this.kvCacheOffload = true,
    this.flashAttention = FlashAttentionMode.auto,
    this.swaFull = true,
    this.kvUnified = false,
    this.seed,
    this.speculativeNgramStrategy,
    this.speculativeNgramSize = 12,
    this.speculativeMinimumDraftLength = 48,
    this.speculativeDraftLength = 48,
    this.jsonOut,
  });

  final bool help;
  final String? modelPath;
  final String? nativeLibraryPath;
  final String? deviceModel;
  final String prompt;
  final int maxTokens;
  final int iterations;
  final bool warmUp;
  final int contextSize;
  final int batchSize;
  final int? ubatchSize;
  final int? threads;
  final int? batchThreads;
  final int gpuLayers;
  final KvCacheType kvCacheKeyType;
  final KvCacheType kvCacheValueType;
  final bool kvCacheOffload;
  final FlashAttentionMode flashAttention;
  final bool swaFull;
  final bool kvUnified;
  final int? seed;
  final String? speculativeNgramStrategy;
  bool get speculativeNgramSimple => speculativeNgramStrategy == 'ngram-simple';
  final int speculativeNgramSize;
  final int speculativeMinimumDraftLength;
  final int speculativeDraftLength;
  final String? jsonOut;
}

DesktopSmokeBenchmarkOptions parseBenchmarkArgs(List<String> args) {
  var options = const DesktopSmokeBenchmarkOptions();
  for (var i = 0; i < args.length; i += 1) {
    final arg = args[i];
    if (arg == '--help' || arg == '-h') {
      return const DesktopSmokeBenchmarkOptions(help: true);
    }

    String value() {
      final equals = arg.indexOf('=');
      if (equals != -1) {
        return arg.substring(equals + 1);
      }
      if (i + 1 >= args.length) {
        throw FormatException('Missing value for $arg.');
      }
      i += 1;
      if (args[i].startsWith('--')) {
        throw FormatException('Missing value for $arg.');
      }
      return args[i];
    }

    final flag = arg.split('=').first;
    switch (flag) {
      case '--model':
        options = options._copy(modelPath: _pathValue(value(), flag));
      case '--native-library':
        options = options._copy(nativeLibraryPath: _pathValue(value(), flag));
      case '--device-model':
        options = options._copy(
          deviceModel: _singleLineTextValue(value(), flag),
        );
      case '--prompt':
        options = options._copy(prompt: _textValue(value(), flag));
      case '--max-tokens':
        options = options._copy(maxTokens: _positiveInt(value(), flag));
      case '--iterations':
        options = options._copy(iterations: _positiveInt(value(), flag));
      case '--no-warm-up':
        if (arg.contains('=')) {
          throw const FormatException('--no-warm-up does not take a value.');
        }
        options = options._copy(warmUp: false);
      case '--context-size':
        options = options._copy(contextSize: _positiveInt(value(), flag));
      case '--batch-size':
        options = options._copy(batchSize: _positiveInt(value(), flag));
      case '--ubatch-size':
        options = options._copy(ubatchSize: _positiveInt(value(), flag));
      case '--threads':
        options = options._copy(threads: _positiveInt(value(), flag));
      case '--batch-threads':
        options = options._copy(batchThreads: _positiveInt(value(), flag));
      case '--gpu-layers':
        options = options._copy(gpuLayers: _nonNegativeInt(value(), flag));
      case '--kv-cache-key':
        options = options._copy(kvCacheKeyType: _kvCacheType(value(), flag));
      case '--kv-cache-value':
        options = options._copy(kvCacheValueType: _kvCacheType(value(), flag));
      case '--no-kv-offload':
        if (arg.contains('=')) {
          throw const FormatException('--no-kv-offload does not take a value.');
        }
        options = options._copy(kvCacheOffload: false);
      case '--flash-attention':
        options = options._copy(
          flashAttention: _flashAttentionMode(value(), flag),
        );
      case '--no-swa-full':
        if (arg.contains('=')) {
          throw const FormatException('--no-swa-full does not take a value.');
        }
        options = options._copy(swaFull: false);
      case '--kv-unified':
        if (arg.contains('=')) {
          throw const FormatException('--kv-unified does not take a value.');
        }
        options = options._copy(kvUnified: true);
      case '--seed':
        options = options._copy(seed: _nonNegativeInt(value(), flag));
      case '--spec-ngram-simple':
        if (arg.contains('=')) {
          throw const FormatException(
            '--spec-ngram-simple does not take a value.',
          );
        }
        options = options._copy(speculativeNgramStrategy: 'ngram-simple');
      case '--spec-ngram':
        options = options._copy(
          speculativeNgramStrategy: _ngramStrategy(value(), flag),
        );
      case '--spec-ngram-size':
        options = options._copy(
          speculativeNgramSize: _positiveInt(value(), flag),
        );
      case '--spec-min-draft-length':
        options = options._copy(
          speculativeMinimumDraftLength: _positiveInt(value(), flag),
        );
      case '--spec-draft-length':
        options = options._copy(
          speculativeDraftLength: _positiveInt(value(), flag),
        );
      case '--json-out':
        options = options._copy(jsonOut: _pathValue(value(), flag));
      default:
        throw FormatException('Unknown option: $arg.');
    }
  }
  _validateBenchmarkOptions(options);
  return options;
}

void _validateBenchmarkOptions(DesktopSmokeBenchmarkOptions options) {
  final modelPath = options.modelPath;
  if (modelPath != null) {
    _pathValue(modelPath, '--model');
  }
  final nativeLibraryPath = options.nativeLibraryPath;
  if (nativeLibraryPath != null) {
    _pathValue(nativeLibraryPath, '--native-library');
  }
  final deviceModel = options.deviceModel;
  if (deviceModel != null) {
    _singleLineTextValue(deviceModel, '--device-model');
  }
  _textValue(options.prompt, '--prompt');
  final jsonOut = options.jsonOut;
  if (jsonOut != null) {
    _pathValue(jsonOut, '--json-out');
  }
  _positiveOption(options.maxTokens, '--max-tokens', _maxInt32);
  _positiveOption(options.iterations, '--iterations', _maxBenchmarkIterations);
  _positiveOption(options.contextSize, '--context-size', _maxUint32);
  _positiveOption(options.batchSize, '--batch-size', _maxUint32);
  final threads = options.threads;
  if (threads != null) {
    _positiveOption(threads, '--threads', _maxInt32);
  }
  final batchThreads = options.batchThreads;
  if (batchThreads != null) {
    _positiveOption(batchThreads, '--batch-threads', _maxInt32);
  }
  _nonNegativeOption(options.gpuLayers, '--gpu-layers', _maxInt32);
  final seed = options.seed;
  if (seed != null) {
    _nonNegativeOption(seed, '--seed', _maxUint32);
  }
  final speculativeNgramStrategy = options.speculativeNgramStrategy;
  if (speculativeNgramStrategy != null) {
    _ngramStrategy(speculativeNgramStrategy, '--spec-ngram');
  }
  _positiveOption(
    options.speculativeNgramSize,
    '--spec-ngram-size',
    _maxSpeculativeNgramSize,
  );
  _positiveOption(
    options.speculativeMinimumDraftLength,
    '--spec-min-draft-length',
    _maxSpeculativeNgramSize,
  );
  _positiveOption(
    options.speculativeDraftLength,
    '--spec-draft-length',
    _maxSpeculativeNgramSize,
  );
  final ubatchSize = options.ubatchSize;
  if (ubatchSize != null) {
    _positiveOption(ubatchSize, '--ubatch-size', _maxUint32);
    if (ubatchSize > options.batchSize) {
      throw const FormatException(
        '--ubatch-size must not exceed --batch-size.',
      );
    }
  }
  if (speculativeNgramStrategy == 'ngram-mod' &&
      options.speculativeMinimumDraftLength > options.speculativeDraftLength) {
    throw const FormatException(
      '--spec-min-draft-length must not exceed --spec-draft-length.',
    );
  }
  if (speculativeNgramStrategy != 'ngram-mod' &&
      speculativeNgramStrategy != 'ngram-cache' &&
      options.speculativeDraftLength < options.speculativeNgramSize) {
    throw const FormatException(
      '--spec-draft-length must be greater than or equal to --spec-ngram-size.',
    );
  }
  if (options.flashAttention == FlashAttentionMode.disabled &&
      _quantizedKvCacheType(options.kvCacheValueType)) {
    throw const FormatException(
      '--kv-cache-value cannot be quantized when --flash-attention is disabled.',
    );
  }
}

Future<Map<String, Object?>> runDesktopSmokeBenchmark(
  DesktopSmokeBenchmarkOptions options,
) async {
  _validateBenchmarkOptions(options);
  final modelPath = options.modelPath;
  if (modelPath == null) {
    throw const FormatException('--model is required.');
  }
  final modelConfig = LlamaModelConfig(
    modelPath: modelPath,
    nativeLibraryPath: options.nativeLibraryPath,
    contextSize: options.contextSize,
    batchSize: options.batchSize,
    ubatchSize: options.ubatchSize,
    threads: options.threads,
    batchThreads: options.batchThreads,
    gpu: GpuConfig.auto(layers: options.gpuLayers),
    kvCache: KvCacheConfig(
      keyType: options.kvCacheKeyType,
      valueType: options.kvCacheValueType,
      offload: options.kvCacheOffload,
      flashAttention: options.flashAttention,
      swaFull: options.swaFull,
      unified: options.kvUnified,
    ),
    speculativeDecoding: benchmarkSpeculativeDecodingConfig(options),
  );
  final generationConfig = GenerationConfig(
    maxTokens: options.maxTokens,
    seed: options.seed,
  );

  final capabilities = LlamaRuntime.currentCapabilities(
    nativeLibraryPath: options.nativeLibraryPath,
  );
  final memoryBeforeLoad = benchmarkMemorySnapshot();
  final inspectWatch = Stopwatch()..start();
  final modelInfo = await LlamaModel.inspect(modelConfig);
  inspectWatch.stop();

  final loadWatch = Stopwatch()..start();
  final engine = await LlamaEngine.load(modelConfig);
  loadWatch.stop();
  try {
    final memoryAfterLoad = benchmarkMemorySnapshot();
    final contextInfo = await engine.contextInfo();
    final warmUpWatch = Stopwatch();
    if (options.warmUp) {
      warmUpWatch.start();
      await engine.warmUp();
      warmUpWatch.stop();
    }
    final memoryAfterWarmUp = benchmarkMemorySnapshot();
    final runs = <Map<String, Object?>>[];
    final runTelemetry = <GenerationTelemetry>[];
    var outputBytes = 0;
    final sustainedWatch = Stopwatch()..start();
    for (var iteration = 0; iteration < options.iterations; iteration += 1) {
      if (iteration > 0) {
        await engine.reset();
      }
      GenerationTelemetry? telemetry;
      var runOutputBytes = 0;
      final runWatch = Stopwatch()..start();
      await for (final chunk in engine.complete(
        prompt: options.prompt,
        config: generationConfig,
      )) {
        runOutputBytes += utf8.encode(chunk.text).length;
        telemetry = chunk.telemetry ?? telemetry;
      }
      runWatch.stop();
      final finalTelemetry = telemetry;
      if (finalTelemetry == null) {
        throw StateError(
          'Generation ${iteration + 1} completed without final telemetry.',
        );
      }
      outputBytes += runOutputBytes;
      runTelemetry.add(finalTelemetry);
      runs.add(<String, Object?>{
        'iteration': iteration + 1,
        'wallMs': runWatch.elapsedMicroseconds / 1000.0,
        'outputBytes': runOutputBytes,
        'memoryAfter': benchmarkMemorySnapshot(),
        'telemetry': benchmarkTelemetryRecord(finalTelemetry),
      });
    }
    sustainedWatch.stop();
    final memoryAfterGeneration = benchmarkMemorySnapshot();
    return <String, Object?>{
      'schemaVersion': 3,
      'timestampUtc': DateTime.now().toUtc().toIso8601String(),
      'host': <String, Object?>{
        'os': Platform.operatingSystem,
        'osVersion': Platform.operatingSystemVersion,
        'deviceModel': options.deviceModel ?? 'unspecified',
        'localHostname': _localHostname(),
        'numberOfProcessors': Platform.numberOfProcessors,
      },
      'app': <String, Object?>{
        'buildMode': benchmarkBuildMode(),
        'dartVersion': Platform.version,
      },
      'runtime': <String, Object?>{
        'nativeBridgeAvailable': capabilities.nativeBridgeAvailable,
        'bridgeAbiVersion': capabilities.bridgeAbiVersion,
        'upstreamCommit': capabilities.upstreamCommit,
        'nativeBuildFlags': capabilities.nativeBuildFlags,
      },
      'model': <String, Object?>{
        'path': modelPath,
        'fileName': benchmarkModelFileName(modelPath),
        'description': modelInfo.description,
        'sizeBytes': modelInfo.sizeBytes,
        'parameterCount': modelInfo.parameterCount,
        'trainingContextSize': modelInfo.trainingContextSize,
        'nextnLayerCount': modelInfo.nextnLayerCount,
        'hasMtpLayers': modelInfo.hasMtpLayers,
        'fileType': modelInfo.fileType,
        'fileTypeName': modelInfo.fileTypeName,
        'quantization': modelInfo.fileTypeName,
      },
      'config': <String, Object?>{
        'contextSize': options.contextSize,
        'batchSize': options.batchSize,
        'ubatchSize': options.ubatchSize,
        'threads': options.threads,
        'batchThreads': options.batchThreads,
        'gpuLayers': options.gpuLayers,
        'kvCacheKeyType': benchmarkKvCacheTypeName(options.kvCacheKeyType),
        'kvCacheValueType': benchmarkKvCacheTypeName(options.kvCacheValueType),
        'kvCacheOffload': options.kvCacheOffload,
        'flashAttention': benchmarkFlashAttentionName(options.flashAttention),
        'swaFull': options.swaFull,
        'kvUnified': options.kvUnified,
        'maxTokens': options.maxTokens,
        'iterations': options.iterations,
        'warmUp': options.warmUp,
        'seed': options.seed,
        'speculativeDecoding': benchmarkSpeculativeDecodingRecord(
          modelConfig.speculativeDecoding,
        ),
      },
      'context': <String, Object?>{
        'contextSize': contextInfo.contextSize,
        'sequenceContextSize': contextInfo.sequenceContextSize,
        'batchSize': contextInfo.batchSize,
        'ubatchSize': contextInfo.ubatchSize,
        'maxSequences': contextInfo.maxSequences,
        'gpuBackend': contextInfo.gpuBackend.name,
        'kvCacheKeyType': benchmarkKvCacheTypeName(contextInfo.kvCacheKeyType),
        'kvCacheValueType': benchmarkKvCacheTypeName(
          contextInfo.kvCacheValueType,
        ),
        'kvCacheOffload': contextInfo.kvCacheOffload,
        'flashAttention': benchmarkFlashAttentionName(
          contextInfo.flashAttention,
        ),
        'swaFull': contextInfo.swaFull,
        'kvUnified': contextInfo.kvUnified,
      },
      'timing': <String, Object?>{
        'modelInspectMs': inspectWatch.elapsedMilliseconds,
        'engineLoadMs': loadWatch.elapsedMilliseconds,
        'warmUpMs': warmUpWatch.elapsedMicroseconds / 1000.0,
        'sustainedGenerationMs': sustainedWatch.elapsedMicroseconds / 1000.0,
      },
      'memory': <String, Object?>{
        'beforeLoad': memoryBeforeLoad,
        'afterLoad': memoryAfterLoad,
        'afterWarmUp': memoryAfterWarmUp,
        'afterGeneration': memoryAfterGeneration,
      },
      'prompt': <String, Object?>{
        'bytesPerRun': utf8.encode(options.prompt).length,
      },
      'output': <String, Object?>{'bytes': outputBytes},
      'telemetry': benchmarkAggregateTelemetry(runTelemetry),
      'runs': runs,
    };
  } finally {
    await engine.close();
  }
}

Map<String, Object?> benchmarkTelemetryRecord(GenerationTelemetry telemetry) {
  return <String, Object?>{
    'promptTokens': telemetry.promptTokens,
    'generatedTokens': telemetry.generatedTokens,
    'promptEvalMs': telemetry.promptEvalMs,
    'decodeMs': telemetry.decodeMs,
    'totalMs': telemetry.totalMs,
    'timeToFirstTokenMs': telemetry.timeToFirstTokenMs,
    'promptEvalTokensPerSecond': telemetry.promptEvalTokensPerSecond,
    'decodeTokensPerSecond': telemetry.decodeTokensPerSecond,
    'totalTokensPerSecond': telemetry.totalTokensPerSecond,
    'speculativeDraftTokens': telemetry.speculativeDraftTokens,
    'speculativeAcceptedTokens': telemetry.speculativeAcceptedTokens,
    'speculativeAcceptanceRate': telemetry.speculativeAcceptanceRate,
    'speculativeDraftMs': telemetry.speculativeDraftMs,
    'speculativeVerifyMs': telemetry.speculativeVerifyMs,
  };
}

Map<String, Object?> benchmarkAggregateTelemetry(
  List<GenerationTelemetry> runs,
) {
  if (runs.isEmpty) {
    throw ArgumentError.value(runs, 'runs', 'must not be empty');
  }
  var promptTokens = 0;
  var generatedTokens = 0;
  var promptEvalMs = 0.0;
  var decodeMs = 0.0;
  var totalMs = 0.0;
  var timeToFirstTokenMs = 0.0;
  var minTimeToFirstTokenMs = double.infinity;
  var maxTimeToFirstTokenMs = 0.0;
  var speculativeDraftTokens = 0;
  var speculativeAcceptedTokens = 0;
  var speculativeDraftMs = 0.0;
  var speculativeVerifyMs = 0.0;
  for (final telemetry in runs) {
    promptTokens += telemetry.promptTokens;
    generatedTokens += telemetry.generatedTokens;
    promptEvalMs += telemetry.promptEvalMs;
    decodeMs += telemetry.decodeMs;
    totalMs += telemetry.totalMs;
    timeToFirstTokenMs += telemetry.timeToFirstTokenMs;
    if (telemetry.timeToFirstTokenMs < minTimeToFirstTokenMs) {
      minTimeToFirstTokenMs = telemetry.timeToFirstTokenMs;
    }
    if (telemetry.timeToFirstTokenMs > maxTimeToFirstTokenMs) {
      maxTimeToFirstTokenMs = telemetry.timeToFirstTokenMs;
    }
    speculativeDraftTokens += telemetry.speculativeDraftTokens;
    speculativeAcceptedTokens += telemetry.speculativeAcceptedTokens;
    speculativeDraftMs += telemetry.speculativeDraftMs;
    speculativeVerifyMs += telemetry.speculativeVerifyMs;
  }
  return <String, Object?>{
    'iterations': runs.length,
    'promptTokens': promptTokens,
    'generatedTokens': generatedTokens,
    'promptEvalMs': promptEvalMs,
    'decodeMs': decodeMs,
    'totalMs': totalMs,
    'timeToFirstTokenMsMean': timeToFirstTokenMs / runs.length,
    'timeToFirstTokenMsMin': minTimeToFirstTokenMs,
    'timeToFirstTokenMsMax': maxTimeToFirstTokenMs,
    'promptEvalTokensPerSecond': _tokensPerSecond(promptTokens, promptEvalMs),
    'decodeTokensPerSecond': _tokensPerSecond(generatedTokens, decodeMs),
    'totalTokensPerSecond': _tokensPerSecond(
      promptTokens + generatedTokens,
      totalMs,
    ),
    'speculativeDraftTokens': speculativeDraftTokens,
    'speculativeAcceptedTokens': speculativeAcceptedTokens,
    'speculativeAcceptanceRate': speculativeDraftTokens == 0
        ? 0.0
        : speculativeAcceptedTokens / speculativeDraftTokens,
    'speculativeDraftMs': speculativeDraftMs,
    'speculativeVerifyMs': speculativeVerifyMs,
  };
}

double _tokensPerSecond(int tokens, double milliseconds) {
  return milliseconds <= 0 ? 0.0 : tokens * 1000.0 / milliseconds;
}

String _textValue(String value, String name) {
  if (value.trim().isEmpty || value.contains('\u0000')) {
    throw FormatException('$name must not be empty or contain NUL.');
  }
  return value;
}

String _pathValue(String value, String name) {
  return _singleLineTextValue(value, name);
}

String _singleLineTextValue(String value, String name) {
  _textValue(value, name);
  if (value.contains('\n') || value.contains('\r')) {
    throw FormatException('$name must not contain line breaks.');
  }
  return value;
}

int _positiveInt(String value, String name) {
  final parsed = int.tryParse(value);
  if (parsed == null || parsed <= 0) {
    throw FormatException('$name must be a positive integer.');
  }
  return parsed;
}

int _nonNegativeInt(String value, String name) {
  final parsed = int.tryParse(value);
  if (parsed == null || parsed < 0) {
    throw FormatException('$name must be a non-negative integer.');
  }
  return parsed;
}

String _ngramStrategy(String value, String name) {
  return switch (value.toLowerCase()) {
    'ngram-simple' => 'ngram-simple',
    'ngram-map-k' => 'ngram-map-k',
    'ngram-map-k4v' => 'ngram-map-k4v',
    'ngram-mod' => 'ngram-mod',
    'ngram-cache' => 'ngram-cache',
    _ => throw FormatException(
      '$name must be ngram-simple, ngram-map-k, ngram-map-k4v, '
      'ngram-mod, or ngram-cache.',
    ),
  };
}

SpeculativeDecodingConfig benchmarkSpeculativeDecodingConfig(
  DesktopSmokeBenchmarkOptions options,
) {
  return switch (options.speculativeNgramStrategy) {
    null => const NoSpeculativeDecoding(),
    'ngram-mod' => NGramModSpeculation(
      matchLength: options.speculativeNgramSize,
      minimumDraftLength: options.speculativeMinimumDraftLength,
      maximumDraftLength: options.speculativeDraftLength,
    ),
    'ngram-cache' => const NGramCacheSpeculation(),
    final strategy => NGramSpeculation(
      strategy: strategy,
      ngramSize: options.speculativeNgramSize,
      draftLength: options.speculativeDraftLength,
    ),
  };
}

Object benchmarkSpeculativeDecodingRecord(SpeculativeDecodingConfig config) {
  return switch (config) {
    NoSpeculativeDecoding() => 'none',
    NGramSpeculation(:final strategy, :final ngramSize, :final draftLength) =>
      <String, Object?>{
        'strategy': strategy,
        'ngramSize': ngramSize,
        'draftLength': draftLength,
      },
    NGramModSpeculation(
      :final matchLength,
      :final minimumDraftLength,
      :final maximumDraftLength,
    ) =>
      <String, Object?>{
        'strategy': 'ngram-mod',
        'matchLength': matchLength,
        'minimumDraftLength': minimumDraftLength,
        'maximumDraftLength': maximumDraftLength,
      },
    NGramCacheSpeculation() => <String, Object?>{'strategy': 'ngram-cache'},
    DraftModelSpeculation() ||
    Eagle3Speculation() ||
    DFlashSpeculation() ||
    MtpSpeculation() => throw ArgumentError.value(
      config,
      'config',
      'desktop smoke benchmark accepts only draftless speculation',
    ),
  };
}

KvCacheType _kvCacheType(String value, String name) {
  return switch (value.toLowerCase()) {
    'f32' => KvCacheType.f32,
    'f16' => KvCacheType.f16,
    'bf16' => KvCacheType.bf16,
    'q8_0' => KvCacheType.q8Zero,
    'q4_0' => KvCacheType.q4Zero,
    'q4_1' => KvCacheType.q4One,
    'iq4_nl' => KvCacheType.iq4Nl,
    'q5_0' => KvCacheType.q5Zero,
    'q5_1' => KvCacheType.q5One,
    _ => throw FormatException(
      '$name must be one of f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, or q5_1.',
    ),
  };
}

FlashAttentionMode _flashAttentionMode(String value, String name) {
  return switch (value.toLowerCase()) {
    'auto' => FlashAttentionMode.auto,
    'disabled' => FlashAttentionMode.disabled,
    'enabled' => FlashAttentionMode.enabled,
    _ => throw FormatException('$name must be auto, disabled, or enabled.'),
  };
}

bool _quantizedKvCacheType(KvCacheType type) {
  return switch (type) {
    KvCacheType.f32 || KvCacheType.f16 || KvCacheType.bf16 => false,
    KvCacheType.q8Zero ||
    KvCacheType.q4Zero ||
    KvCacheType.q4One ||
    KvCacheType.iq4Nl ||
    KvCacheType.q5Zero ||
    KvCacheType.q5One => true,
  };
}

String benchmarkKvCacheTypeName(KvCacheType type) {
  return switch (type) {
    KvCacheType.f32 => 'f32',
    KvCacheType.f16 => 'f16',
    KvCacheType.bf16 => 'bf16',
    KvCacheType.q8Zero => 'q8_0',
    KvCacheType.q4Zero => 'q4_0',
    KvCacheType.q4One => 'q4_1',
    KvCacheType.iq4Nl => 'iq4_nl',
    KvCacheType.q5Zero => 'q5_0',
    KvCacheType.q5One => 'q5_1',
  };
}

String benchmarkFlashAttentionName(FlashAttentionMode mode) => mode.name;

void _positiveOption(int value, String name, int max) {
  if (value <= 0 || value > max) {
    throw FormatException('$name must be between 1 and $max.');
  }
}

void _nonNegativeOption(int value, String name, int max) {
  if (value < 0 || value > max) {
    throw FormatException('$name must be between 0 and $max.');
  }
}

String? _localHostname() {
  try {
    return Platform.localHostname;
  } on Object {
    return null;
  }
}

String benchmarkBuildMode() {
  if (const bool.fromEnvironment('dart.vm.product')) {
    return 'aot-release';
  }
  return 'jit';
}

Map<String, int> benchmarkMemorySnapshot() {
  return <String, int>{
    'currentRssBytes': ProcessInfo.currentRss,
    'maxRssBytes': ProcessInfo.maxRss,
  };
}

String benchmarkModelFileName(String path) {
  final parts = path.replaceAll(r'\', '/').split('/');
  for (var i = parts.length - 1; i >= 0; i -= 1) {
    if (parts[i].isNotEmpty) {
      return parts[i];
    }
  }
  return path;
}

extension on DesktopSmokeBenchmarkOptions {
  DesktopSmokeBenchmarkOptions _copy({
    bool? help,
    String? modelPath,
    String? nativeLibraryPath,
    String? deviceModel,
    String? prompt,
    int? maxTokens,
    int? iterations,
    bool? warmUp,
    int? contextSize,
    int? batchSize,
    int? ubatchSize,
    int? threads,
    int? batchThreads,
    int? gpuLayers,
    KvCacheType? kvCacheKeyType,
    KvCacheType? kvCacheValueType,
    bool? kvCacheOffload,
    FlashAttentionMode? flashAttention,
    bool? swaFull,
    bool? kvUnified,
    int? seed,
    String? speculativeNgramStrategy,
    int? speculativeNgramSize,
    int? speculativeMinimumDraftLength,
    int? speculativeDraftLength,
    String? jsonOut,
  }) {
    return DesktopSmokeBenchmarkOptions(
      help: help ?? this.help,
      modelPath: modelPath ?? this.modelPath,
      nativeLibraryPath: nativeLibraryPath ?? this.nativeLibraryPath,
      deviceModel: deviceModel ?? this.deviceModel,
      prompt: prompt ?? this.prompt,
      maxTokens: maxTokens ?? this.maxTokens,
      iterations: iterations ?? this.iterations,
      warmUp: warmUp ?? this.warmUp,
      contextSize: contextSize ?? this.contextSize,
      batchSize: batchSize ?? this.batchSize,
      ubatchSize: ubatchSize ?? this.ubatchSize,
      threads: threads ?? this.threads,
      batchThreads: batchThreads ?? this.batchThreads,
      gpuLayers: gpuLayers ?? this.gpuLayers,
      kvCacheKeyType: kvCacheKeyType ?? this.kvCacheKeyType,
      kvCacheValueType: kvCacheValueType ?? this.kvCacheValueType,
      kvCacheOffload: kvCacheOffload ?? this.kvCacheOffload,
      flashAttention: flashAttention ?? this.flashAttention,
      swaFull: swaFull ?? this.swaFull,
      kvUnified: kvUnified ?? this.kvUnified,
      seed: seed ?? this.seed,
      speculativeNgramStrategy:
          speculativeNgramStrategy ?? this.speculativeNgramStrategy,
      speculativeNgramSize: speculativeNgramSize ?? this.speculativeNgramSize,
      speculativeMinimumDraftLength:
          speculativeMinimumDraftLength ?? this.speculativeMinimumDraftLength,
      speculativeDraftLength:
          speculativeDraftLength ?? this.speculativeDraftLength,
      jsonOut: jsonOut ?? this.jsonOut,
    );
  }
}

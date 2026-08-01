import 'dart:convert';
import 'dart:typed_data';

import 'errors.dart';

enum GpuBackend { auto, cpu, metal, vulkan }

enum MirostatMode { v1, v2 }

enum KvCacheType { f32, f16, bf16, q8Zero, q4Zero, q4One, iq4Nl, q5Zero, q5One }

enum FlashAttentionMode { auto, disabled, enabled }

const _maxInt32 = 0x7FFFFFFF;
const _maxUint32 = 0xFFFFFFFF;
const _maxChatTemplateBytes = 16 * 1024 * 1024;

const llamaJsonGrammar = r'''
root   ::= object
value  ::= object | array | string | number | ("true" | "false" | "null") ws

object ::=
  "{" ws (
            string ":" ws value
    ("," ws string ":" ws value)*
  )? "}" ws

array  ::=
  "[" ws (
            value
    ("," ws value)*
  )? "]" ws

string ::=
  "\"" (
    [^"\\\x7F\x00-\x1F] |
    "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4})
  )* "\"" ws

number ::= ("-"? ([0-9] | [1-9] [0-9]{0,15})) ("." [0-9]+)? ([eE] [-+]? [0-9] [0-9]{0,15})? ws

ws ::= | " " | "\n" [ \t]{0,20}
''';

final class GpuConfig {
  const GpuConfig.auto({this.layers}) : backend = GpuBackend.auto;

  const GpuConfig.cpu() : backend = GpuBackend.cpu, layers = 0;

  const GpuConfig.metal({this.layers}) : backend = GpuBackend.metal;

  const GpuConfig.vulkan({this.layers}) : backend = GpuBackend.vulkan;

  final GpuBackend backend;
  final int? layers;

  void validate() {
    final layers = this.layers;
    if (layers != null && layers < 0) {
      throw ArgumentError.value(layers, 'layers', 'must be non-negative');
    }
    if (layers != null && layers > _maxInt32) {
      throw ArgumentError.value(layers, 'layers', 'must fit int32');
    }
    if ((backend == GpuBackend.metal || backend == GpuBackend.vulkan) &&
        layers == 0) {
      throw ArgumentError.value(
        layers,
        'layers',
        'must be positive for an explicit GPU backend',
      );
    }
  }
}

final class KvCacheConfig {
  const KvCacheConfig({
    this.keyType = KvCacheType.f16,
    this.valueType = KvCacheType.f16,
    this.offload = true,
    this.flashAttention = FlashAttentionMode.auto,
    this.swaFull = true,
    this.unified = false,
  });

  final KvCacheType keyType;
  final KvCacheType valueType;
  final bool offload;
  final FlashAttentionMode flashAttention;
  final bool swaFull;
  final bool unified;

  void validate() {
    if (flashAttention == FlashAttentionMode.disabled &&
        _isQuantizedKvCacheType(valueType)) {
      throw ArgumentError.value(
        valueType,
        'valueType',
        'quantized V cache types require Flash Attention',
      );
    }
  }
}

bool _isQuantizedKvCacheType(KvCacheType type) {
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

final class LlamaModelConfig {
  const LlamaModelConfig({
    required this.modelPath,
    this.nativeLibraryPath,
    this.mmprojPath,
    this.chatTemplate,
    this.contextSize = 4096,
    this.batchSize = 512,
    this.ubatchSize,
    this.threads,
    this.batchThreads,
    this.useMmap = true,
    this.useMlock = false,
    this.checkTensors = true,
    this.gpu = const GpuConfig.auto(),
    this.kvCache = const KvCacheConfig(),
    this.speculativeDecoding = const NoSpeculativeDecoding(),
  });

  final String modelPath;
  final String? nativeLibraryPath;
  final String? mmprojPath;

  /// Explicit llama.cpp chat template used instead of GGUF metadata.
  ///
  /// Leave this null to require the model's embedded default template. Chat
  /// APIs fail with [UnsupportedFeatureException] when neither is available.
  final String? chatTemplate;
  final int contextSize;
  final int batchSize;
  final int? ubatchSize;
  final int? threads;
  final int? batchThreads;
  final bool useMmap;
  final bool useMlock;
  final bool checkTensors;
  final GpuConfig gpu;
  final KvCacheConfig kvCache;
  final SpeculativeDecodingConfig speculativeDecoding;

  void validate() {
    _validatePathText(modelPath, 'modelPath');
    final nativeLibraryPath = this.nativeLibraryPath;
    if (nativeLibraryPath != null) {
      _validatePathText(nativeLibraryPath, 'nativeLibraryPath');
    }
    final mmprojPath = this.mmprojPath;
    if (mmprojPath != null) {
      _validatePathText(mmprojPath, 'mmprojPath');
    }
    final chatTemplate = this.chatTemplate;
    if (chatTemplate != null) {
      if (chatTemplate.trim().isEmpty) {
        throw ArgumentError.value(
          chatTemplate,
          'chatTemplate',
          'must not be blank',
        );
      }
      if (chatTemplate.contains('\u0000')) {
        throw ArgumentError.value(
          chatTemplate,
          'chatTemplate',
          'must not contain NUL',
        );
      }
      if (utf8.encode(chatTemplate).length > _maxChatTemplateBytes) {
        throw ArgumentError.value(
          chatTemplate,
          'chatTemplate',
          'must not exceed 16 MiB as UTF-8',
        );
      }
    }
    if (contextSize <= 0) {
      throw ArgumentError.value(contextSize, 'contextSize', 'must be positive');
    }
    if (contextSize > _maxUint32) {
      throw ArgumentError.value(contextSize, 'contextSize', 'must fit uint32');
    }
    if (batchSize <= 0) {
      throw ArgumentError.value(batchSize, 'batchSize', 'must be positive');
    }
    if (batchSize > _maxUint32) {
      throw ArgumentError.value(batchSize, 'batchSize', 'must fit uint32');
    }
    final ubatchSize = this.ubatchSize;
    if (ubatchSize != null && ubatchSize <= 0) {
      throw ArgumentError.value(ubatchSize, 'ubatchSize', 'must be positive');
    }
    if (ubatchSize != null && ubatchSize > _maxUint32) {
      throw ArgumentError.value(ubatchSize, 'ubatchSize', 'must fit uint32');
    }
    if (ubatchSize != null && ubatchSize > batchSize) {
      throw ArgumentError.value(
        ubatchSize,
        'ubatchSize',
        'must not exceed batchSize',
      );
    }
    final threads = this.threads;
    if (threads != null && threads <= 0) {
      throw ArgumentError.value(threads, 'threads', 'must be positive');
    }
    if (threads != null && threads > _maxInt32) {
      throw ArgumentError.value(threads, 'threads', 'must fit int32');
    }
    final batchThreads = this.batchThreads;
    if (batchThreads != null && batchThreads <= 0) {
      throw ArgumentError.value(
        batchThreads,
        'batchThreads',
        'must be positive',
      );
    }
    if (batchThreads != null && batchThreads > _maxInt32) {
      throw ArgumentError.value(batchThreads, 'batchThreads', 'must fit int32');
    }
    gpu.validate();
    kvCache.validate();
    _validateSpeculativeDecoding(speculativeDecoding);
    if (_isModelBackedSpeculation(speculativeDecoding)) {
      if (contextSize > _maxInt32) {
        throw ArgumentError.value(
          contextSize,
          'contextSize',
          'must fit int32 for model-backed speculation',
        );
      }
      if (batchSize > _maxInt32) {
        throw ArgumentError.value(
          batchSize,
          'batchSize',
          'must fit int32 for model-backed speculation',
        );
      }
      if (ubatchSize != null && ubatchSize > _maxInt32) {
        throw ArgumentError.value(
          ubatchSize,
          'ubatchSize',
          'must fit int32 for model-backed speculation',
        );
      }
    }
  }
}

final class GenerationConfig {
  const GenerationConfig({
    this.maxTokens = 128,
    this.temperature = 0.8,
    this.topK = 40,
    this.topP = 0.95,
    this.minP = 0.0,
    this.typicalP = 1.0,
    this.penaltyLastN = 64,
    this.repeatPenalty = 1.0,
    this.frequencyPenalty = 0.0,
    this.presencePenalty = 0.0,
    this.mirostat,
    this.mirostatTau = 5.0,
    this.mirostatEta = 0.1,
    this.seed,
    this.streamChunkTokens = 4,
    this.stop = const <String>[],
    this.stopTokens = const <int>[],
    this.loraScales,
    this.grammar,
    this.jsonSchema,
    this.grammarRoot = 'root',
    this.toolCalling = const LlamaToolCallingConfig(),
    this.enableThinking,
    this.reasoningBudgetTokens,
  });

  const GenerationConfig.jsonMode({
    this.maxTokens = 128,
    this.temperature = 0.8,
    this.topK = 40,
    this.topP = 0.95,
    this.minP = 0.0,
    this.typicalP = 1.0,
    this.penaltyLastN = 64,
    this.repeatPenalty = 1.0,
    this.frequencyPenalty = 0.0,
    this.presencePenalty = 0.0,
    this.mirostat,
    this.mirostatTau = 5.0,
    this.mirostatEta = 0.1,
    this.seed,
    this.streamChunkTokens = 4,
    this.stop = const <String>[],
    this.stopTokens = const <int>[],
    this.loraScales,
    this.toolCalling = const LlamaToolCallingConfig(),
    this.enableThinking,
  }) : grammar = llamaJsonGrammar,
       jsonSchema = null,
       grammarRoot = 'root',
       reasoningBudgetTokens = null;

  factory GenerationConfig.jsonSchema({
    required Map<String, Object?> schema,
    int maxTokens = 128,
    double temperature = 0.8,
    int topK = 40,
    double topP = 0.95,
    double minP = 0.0,
    double typicalP = 1.0,
    int penaltyLastN = 64,
    double repeatPenalty = 1.0,
    double frequencyPenalty = 0.0,
    double presencePenalty = 0.0,
    MirostatMode? mirostat,
    double mirostatTau = 5.0,
    double mirostatEta = 0.1,
    int? seed,
    int streamChunkTokens = 4,
    List<String> stop = const <String>[],
    List<int> stopTokens = const <int>[],
    Map<int, double>? loraScales,
    LlamaToolCallingConfig toolCalling = const LlamaToolCallingConfig(),
    bool? enableThinking,
  }) {
    return GenerationConfig(
      maxTokens: maxTokens,
      temperature: temperature,
      topK: topK,
      topP: topP,
      minP: minP,
      typicalP: typicalP,
      penaltyLastN: penaltyLastN,
      repeatPenalty: repeatPenalty,
      frequencyPenalty: frequencyPenalty,
      presencePenalty: presencePenalty,
      mirostat: mirostat,
      mirostatTau: mirostatTau,
      mirostatEta: mirostatEta,
      seed: seed,
      streamChunkTokens: streamChunkTokens,
      stop: List<String>.unmodifiable(stop),
      stopTokens: List<int>.unmodifiable(stopTokens),
      loraScales: loraScales == null
          ? null
          : Map<int, double>.unmodifiable(loraScales),
      jsonSchema: _jsonObjectSnapshot(schema, 'schema'),
      grammarRoot: 'root',
      toolCalling: _snapshotToolCallingConfig(toolCalling),
      enableThinking: enableThinking,
    );
  }

  final int maxTokens;
  final double temperature;
  final int topK;
  final double topP;
  final double minP;
  final double typicalP;
  final int penaltyLastN;
  final double repeatPenalty;
  final double frequencyPenalty;
  final double presencePenalty;
  final MirostatMode? mirostat;
  final double mirostatTau;
  final double mirostatEta;
  final int? seed;

  /// Maximum generated tokens decoded into one streamed chunk.
  final int streamChunkTokens;
  final List<String> stop;
  final List<int> stopTokens;

  /// Per-request adapter scales keyed by loaded adapter ID.
  ///
  /// `null` uses global scales; an empty map disables all adapters.
  final Map<int, double>? loraScales;
  final String? grammar;
  final Map<String, Object?>? jsonSchema;
  final String grammarRoot;
  final LlamaToolCallingConfig toolCalling;

  /// Optional typed input for chat templates that expose `enable_thinking`.
  ///
  /// `null` preserves the legacy-first formatting path unless
  /// [reasoningBudgetTokens] requests native chat planning. On that planning
  /// path, llama.cpp supplies its default `enable_thinking` value (`true`).
  /// Non-null values are applied only by [LlamaEngine.chat].
  final bool? enableThinking;

  /// Maximum tokens counted inside each chat-template reasoning block.
  ///
  /// Reasoning already present in a template's generation prefill also counts.
  /// This is separate from [maxTokens], which remains the total output limit
  /// and must leave room for model-specific thinking markers and public output.
  /// A non-null value leaves thinking at llama.cpp's default (`true`) when
  /// [enableThinking] is `null`, and is invalid when it is explicitly `false`.
  final int? reasoningBudgetTokens;

  void validate() {
    if (maxTokens <= 0) {
      throw ArgumentError.value(maxTokens, 'maxTokens', 'must be positive');
    }
    if (maxTokens > _maxInt32) {
      throw ArgumentError.value(maxTokens, 'maxTokens', 'must fit int32');
    }
    if (temperature < 0) {
      throw ArgumentError.value(
        temperature,
        'temperature',
        'must be non-negative',
      );
    }
    _validateFinite(temperature, 'temperature');
    if (topK < 0) {
      throw ArgumentError.value(topK, 'topK', 'must be non-negative');
    }
    if (topK > _maxInt32) {
      throw ArgumentError.value(topK, 'topK', 'must fit int32');
    }
    if (penaltyLastN < 0) {
      throw ArgumentError.value(
        penaltyLastN,
        'penaltyLastN',
        'must be non-negative',
      );
    }
    if (penaltyLastN > _maxInt32) {
      throw ArgumentError.value(penaltyLastN, 'penaltyLastN', 'must fit int32');
    }
    if (repeatPenalty < 0) {
      throw ArgumentError.value(
        repeatPenalty,
        'repeatPenalty',
        'must be non-negative',
      );
    }
    _validateFinite(repeatPenalty, 'repeatPenalty');
    _validateFinite(frequencyPenalty, 'frequencyPenalty');
    _validateFinite(presencePenalty, 'presencePenalty');
    _validatePositiveFinite(mirostatTau, 'mirostatTau');
    _validatePositiveFinite(mirostatEta, 'mirostatEta');
    final seed = this.seed;
    if (seed != null && (seed < 0 || seed > _maxUint32)) {
      throw ArgumentError.value(seed, 'seed', 'must fit uint32');
    }
    if (streamChunkTokens <= 0 || streamChunkTokens > _maxStreamChunkTokens) {
      throw ArgumentError.value(
        streamChunkTokens,
        'streamChunkTokens',
        'must be between 1 and $_maxStreamChunkTokens',
      );
    }
    _validateProbability(topP, 'topP');
    _validateProbability(minP, 'minP');
    _validateProbability(typicalP, 'typicalP');
    for (final marker in stop) {
      if (marker.isEmpty) {
        throw ArgumentError.value(
          stop,
          'stop',
          'must not contain empty strings',
        );
      }
      if (marker.contains('\u0000')) {
        throw ArgumentError.value(stop, 'stop', 'must not contain NUL');
      }
    }
    if (stopTokens.length > _maxStopTokens) {
      throw ArgumentError.value(
        stopTokens,
        'stopTokens',
        'must contain at most $_maxStopTokens token ids',
      );
    }
    final uniqueStopTokens = <int>{};
    for (final token in stopTokens) {
      if (token < 0 || token > _maxInt32) {
        throw ArgumentError.value(
          token,
          'stopTokens',
          'must contain non-negative int32 token ids',
        );
      }
      if (!uniqueStopTokens.add(token)) {
        throw ArgumentError.value(
          stopTokens,
          'stopTokens',
          'must contain unique token ids',
        );
      }
    }
    final loraScales = this.loraScales;
    if (loraScales != null) {
      for (final entry in loraScales.entries) {
        if (entry.key <= 0) {
          throw ArgumentError.value(
            entry.key,
            'loraScales',
            'adapter ids must be positive',
          );
        }
        _validateFinite(entry.value, 'loraScales');
        if (entry.value < 0) {
          throw ArgumentError.value(
            entry.value,
            'loraScales',
            'scales must be non-negative',
          );
        }
      }
    }
    final grammar = this.grammar;
    final jsonSchema = this.jsonSchema;
    if (grammar != null && jsonSchema != null) {
      throw ArgumentError.value(
        jsonSchema,
        'jsonSchema',
        'must not be provided with grammar',
      );
    }
    if ((enableThinking == true || this.reasoningBudgetTokens != null) &&
        (grammar != null || jsonSchema != null)) {
      throw ArgumentError.value(
        enableThinking,
        'enableThinking',
        'thinking output cannot be combined with request-owned structured '
            'output',
      );
    }
    final reasoningBudgetTokens = this.reasoningBudgetTokens;
    if (reasoningBudgetTokens != null) {
      if (reasoningBudgetTokens < 0) {
        throw ArgumentError.value(
          reasoningBudgetTokens,
          'reasoningBudgetTokens',
          'must be non-negative',
        );
      }
      if (reasoningBudgetTokens > _maxInt32) {
        throw ArgumentError.value(
          reasoningBudgetTokens,
          'reasoningBudgetTokens',
          'must fit int32',
        );
      }
      if (enableThinking == false) {
        throw ArgumentError.value(
          reasoningBudgetTokens,
          'reasoningBudgetTokens',
          'cannot be used when enableThinking is false',
        );
      }
    }
    if (jsonSchema != null) {
      _validateJsonValue(jsonSchema, 'jsonSchema');
    }
    if (grammar == null && grammarRoot != 'root') {
      throw ArgumentError.value(
        grammarRoot,
        'grammarRoot',
        'must be root when grammar is omitted',
      );
    }
    if (grammar != null && grammar.trim().isEmpty) {
      throw ArgumentError.value(grammar, 'grammar', 'must not be blank');
    }
    if (grammar != null && grammar.contains('\u0000')) {
      throw ArgumentError.value(grammar, 'grammar', 'must not contain NUL');
    }
    if (grammar != null && grammar.trim().isNotEmpty) {
      if (grammarRoot.trim().isEmpty) {
        throw ArgumentError.value(
          grammarRoot,
          'grammarRoot',
          'must not be blank when grammar is provided',
        );
      }
      if (grammarRoot.contains('\u0000')) {
        throw ArgumentError.value(
          grammarRoot,
          'grammarRoot',
          'must not contain NUL',
        );
      }
      if (grammarRoot.contains('\n') || grammarRoot.contains('\r')) {
        throw ArgumentError.value(
          grammarRoot,
          'grammarRoot',
          'must not contain line breaks',
        );
      }
      if (grammarRoot.contains(RegExp(r'\s'))) {
        throw ArgumentError.value(
          grammarRoot,
          'grammarRoot',
          'must not contain whitespace',
        );
      }
    }
    toolCalling.validate();
    if (toolCalling.tools.isNotEmpty &&
        toolCalling.toolChoice is! LlamaNoToolChoice &&
        (grammar != null || jsonSchema != null)) {
      throw ArgumentError.value(
        toolCalling,
        'toolCalling',
        'active tools cannot be combined with structured output',
      );
    }
  }
}

String llamaJsonSchemaGrammar(Map<String, Object?> schema) {
  return _JsonSchemaGrammarBuilder().build(schema);
}

void _validateProbability(double value, String name) {
  _validateFinite(value, name);
  if (value < 0 || value > 1) {
    throw ArgumentError.value(value, name, 'must be in [0, 1]');
  }
}

void _validateFinite(double value, String name) {
  if (!value.isFinite) {
    throw ArgumentError.value(value, name, 'must be finite');
  }
}

void _validatePositiveFinite(double value, String name) {
  _validateFinite(value, name);
  if (value <= 0) {
    throw ArgumentError.value(value, name, 'must be positive');
  }
}

final class _JsonSchemaGrammarBuilder {
  final Map<String, String> _rules = <String, String>{};
  final Set<Object> _activeSchemas = Set<Object>.identity();
  int _nextRule = 0;

  String build(Map<String, Object?> schema) {
    final rootRule = _schemaRule(schema);
    _rules['root'] = rootRule;
    _addCommonRules();
    return _rules.entries
        .map((entry) => '${entry.key} ::= ${entry.value}')
        .join('\n');
  }

  String _schemaRule(Object? schemaValue, [String schemaName = 'schema']) {
    if (schemaValue is Map<Object?, Object?> &&
        !_activeSchemas.add(schemaValue)) {
      throw ArgumentError.value(
        schemaValue,
        schemaName,
        'must not contain cycles',
      );
    }
    final ruleName = 'schema${_nextRule++}';
    try {
      final schema = _objectSchema(schemaValue, schemaName);
      _rules[ruleName] = _schemaExpression(schema);
      return ruleName;
    } finally {
      if (schemaValue is Map<Object?, Object?>) {
        _activeSchemas.remove(schemaValue);
      }
    }
  }

  String _schemaExpression(Map<String, Object?> schema) {
    if (schema.containsKey('const')) {
      return _constExpression(schema);
    }

    if (schema.containsKey('enum')) {
      return _enumExpression(schema);
    }

    if (schema.containsKey('anyOf')) {
      return _compositionExpression(schema, 'anyOf');
    }
    if (schema.containsKey('oneOf')) {
      return _compositionExpression(schema, 'oneOf');
    }

    final type = schema.containsKey('type')
        ? schema['type']
        : _inferredType(schema);
    if (type is List<Object?>) {
      final extras = schema.keys
          .where((key) => !_annotationKeys.contains(key) && key != 'type')
          .toList(growable: false);
      if (type.isEmpty || type.any((item) => item is! String)) {
        throw ArgumentError.value(type, 'schema.type', 'must contain strings');
      }
      final types = type.cast<String>().toList(growable: false);
      if (types.toSet().length != types.length) {
        throw ArgumentError.value(type, 'schema.type', 'must not repeat types');
      }
      if (extras.isEmpty) {
        return types
            .map((item) => _schemaRule(<String, Object?>{'type': item}))
            .join(' | ');
      }
      final nonNullTypes = types
          .where((item) => item != 'null')
          .toList(growable: false);
      if (!types.contains('null') || nonNullTypes.length != 1) {
        throw UnsupportedFeatureException(
          'JSON Schema constrained type arrays only support one non-null type plus null',
        );
      }
      return [
        for (final item in types)
          if (item == 'null')
            _schemaRule(<String, Object?>{'type': 'null'})
          else
            _schemaRule(<String, Object?>{...schema, 'type': item}),
      ].join(' | ');
    }
    if (type is! String) {
      throw ArgumentError.value(type, 'schema.type', 'must be a string');
    }

    return switch (type) {
      'object' => _objectExpression(schema),
      'array' => _arrayExpression(schema),
      'string' => _stringExpression(schema),
      'number' => _simpleExpression(schema, 'number'),
      'integer' => _integerExpression(schema),
      'boolean' => _simpleExpression(schema, '("true" | "false") ws'),
      'null' => _simpleExpression(schema, '"null" ws'),
      _ => throw UnsupportedFeatureException(
        'Unsupported JSON Schema type: $type',
      ),
    };
  }

  String? _inferredType(Map<String, Object?> schema) {
    final types = <String>{
      if (schema.keys.any(_objectKeyword)) 'object',
      if (schema.keys.any(_arrayKeyword)) 'array',
      if (schema.keys.any(_stringKeyword)) 'string',
    };
    if (types.length > 1) {
      throw UnsupportedFeatureException(
        'JSON Schema type could not be inferred',
      );
    }
    return types.isEmpty ? null : types.single;
  }

  String _compositionExpression(Map<String, Object?> schema, String key) {
    _rejectUnsupportedKeys(schema, <String>{..._annotationKeys, key});
    final schemas = schema[key];
    if (schemas is! List<Object?> || schemas.isEmpty) {
      throw ArgumentError.value(schemas, 'schema.$key', 'must be non-empty');
    }
    return schemas
        .map((value) => _schemaRule(value, 'schema.$key'))
        .join(' | ');
  }

  String _enumExpression(Map<String, Object?> schema) {
    _rejectUnsupportedKeys(schema, const <String>{..._annotationKeys, 'enum'});
    final values = schema['enum'];
    if (values is! List<Object?> || values.isEmpty) {
      throw ArgumentError.value(values, 'schema.enum', 'must be non-empty');
    }
    final seen = <String>{};
    return [
      for (var i = 0; i < values.length; i += 1) _enumValue(values[i], i, seen),
    ].join(' | ');
  }

  String _enumValue(Object? value, int index, Set<String> seen) {
    final literalValue = _jsonLiteralValue(
      value,
      'schema.enum[$index]',
      Set<Object>.identity(),
    );
    if (!seen.add(_jsonLiteralKey(literalValue))) {
      throw ArgumentError.value(value, 'schema.enum', 'must not repeat values');
    }
    return '${_gbnfLiteral(jsonEncode(literalValue))} ws';
  }

  String _constExpression(Map<String, Object?> schema) {
    _rejectUnsupportedKeys(schema, const <String>{..._annotationKeys, 'const'});
    return _jsonLiteralExpression(schema['const'], 'schema.const');
  }

  String _jsonLiteralExpression(Object? value, String name) {
    return '${_gbnfLiteral(jsonEncode(_jsonLiteralValue(value, name, Set<Object>.identity())))} ws';
  }

  Object? _jsonLiteralValue(
    Object? value,
    String name,
    Set<Object> activeContainers,
  ) {
    if (value == null || value is String || value is bool) {
      return value;
    }
    if (value is num) {
      if (!value.isFinite) {
        throw ArgumentError.value(value, name, 'must be finite');
      }
      return value;
    }
    if (value is List<Object?>) {
      if (!activeContainers.add(value)) {
        throw ArgumentError.value(value, name, 'must not contain cycles');
      }
      try {
        return [
          for (var i = 0; i < value.length; i += 1)
            _jsonLiteralValue(value[i], '$name[$i]', activeContainers),
        ];
      } finally {
        activeContainers.remove(value);
      }
    }
    if (value is Map<Object?, Object?>) {
      if (!activeContainers.add(value)) {
        throw ArgumentError.value(value, name, 'must not contain cycles');
      }
      final result = <String, Object?>{};
      try {
        for (final entry in value.entries) {
          final key = entry.key;
          if (key is! String) {
            throw ArgumentError.value(key, name, 'keys must be strings');
          }
          result[key] = _jsonLiteralValue(
            entry.value,
            '$name.$key',
            activeContainers,
          );
        }
        return result;
      } finally {
        activeContainers.remove(value);
      }
    }
    throw ArgumentError.value(value, name, 'unsupported JSON literal');
  }

  String _jsonLiteralKey(Object? value) {
    if (value == null) {
      return 'null';
    }
    if (value is bool) {
      return 'bool:$value';
    }
    if (value is String) {
      return 'string:${jsonEncode(value)}';
    }
    if (value is num) {
      return 'number:${_jsonNumberKey(value)}';
    }
    if (value is List<Object?>) {
      return 'list:${jsonEncode([for (final item in value) _jsonLiteralKey(item)])}';
    }
    if (value is Map<String, Object?>) {
      return 'map:${jsonEncode([
        for (final key in value.keys.toList()..sort()) <String>[key, _jsonLiteralKey(value[key])],
      ])}';
    }
    throw ArgumentError.value(value, 'schema.enum', 'unsupported JSON literal');
  }

  String _jsonNumberKey(num value) {
    if (value is int) {
      return value.toString();
    }
    final doubleValue = value.toDouble();
    if (doubleValue.truncateToDouble() == doubleValue) {
      return doubleValue.toInt().toString();
    }
    return doubleValue.toString();
  }

  String _objectExpression(Map<String, Object?> schema) {
    _rejectUnsupportedKeys(schema, const <String>{
      ..._annotationKeys,
      'type',
      'properties',
      'required',
      'additionalProperties',
      'minProperties',
      'maxProperties',
    });

    final required = schema.containsKey('required')
        ? schema['required']
        : const <Object?>[];
    if (required is! List<Object?> || required.any((item) => item is! String)) {
      throw ArgumentError.value(
        required,
        'schema.required',
        'must be a list of property names',
      );
    }
    final requiredSet = required.cast<String>().toSet();
    if (requiredSet.length != required.length) {
      throw ArgumentError.value(
        required,
        'schema.required',
        'must not contain duplicate properties',
      );
    }

    final properties = schema['properties'];
    if (!schema.containsKey('properties')) {
      if (requiredSet.isNotEmpty) {
        throw ArgumentError.value(
          required,
          'schema.required',
          'must not contain unknown properties',
        );
      }
      return _additionalPropertiesExpression(schema);
    }
    if (properties == null) {
      throw ArgumentError.value(
        properties,
        'schema.properties',
        'must be an object',
      );
    }
    final propertyMap = _stringMap(properties, 'schema.properties');
    if (propertyMap.isEmpty) {
      if (requiredSet.isNotEmpty) {
        throw ArgumentError.value(
          required,
          'schema.required',
          'must not contain unknown properties',
        );
      }
      return _additionalPropertiesExpression(schema);
    }
    if (schema.containsKey('minProperties') ||
        schema.containsKey('maxProperties')) {
      throw UnsupportedFeatureException(
        'JSON Schema minProperties/maxProperties only support map objects',
      );
    }
    final additionalProperties = _additionalPropertiesValue(schema);
    if (additionalProperties != false) {
      throw UnsupportedFeatureException(
        'Object schemas with properties require additionalProperties: false',
      );
    }

    if (!requiredSet.every(propertyMap.containsKey)) {
      throw ArgumentError.value(
        required,
        'schema.required',
        'must not contain unknown properties',
      );
    }

    final pairs = [
      for (final entry in propertyMap.entries)
        _JsonObjectPair(
          expression:
              '${_gbnfLiteral(jsonEncode(entry.key))} ws ":" ws '
              '${_schemaRule(entry.value, 'property ${entry.key}')}',
          required: requiredSet.contains(entry.key),
        ),
    ];
    return '"{" ws ${_objectPairsExpression(pairs)} "}" ws';
  }

  String _additionalPropertiesExpression(Map<String, Object?> schema) {
    final minProperties = _objectBound(schema, 'minProperties') ?? 0;
    final maxProperties = _objectBound(schema, 'maxProperties');
    if (maxProperties != null && minProperties > maxProperties) {
      throw ArgumentError.value(
        minProperties,
        'schema.minProperties',
        'must not exceed maxProperties',
      );
    }

    final additionalProperties = _additionalPropertiesValue(schema);
    if (additionalProperties == false) {
      if (minProperties > 0) {
        throw ArgumentError.value(
          minProperties,
          'schema.minProperties',
          'matches no object with additionalProperties: false',
        );
      }
      return '"{" ws "}" ws';
    }

    final valueRule =
        additionalProperties == null || additionalProperties == true
        ? 'value'
        : _schemaRule(additionalProperties, 'schema.additionalProperties');
    final entries = _objectEntriesExpression(
      'string ":" ws $valueRule',
      minProperties,
      maxProperties,
    );
    return '"{" ws $entries "}" ws';
  }

  Object? _additionalPropertiesValue(Map<String, Object?> schema) {
    final value = schema['additionalProperties'];
    if (value == null && schema.containsKey('additionalProperties')) {
      throw ArgumentError.value(
        value,
        'schema.additionalProperties',
        'must be a boolean or schema object',
      );
    }
    return value;
  }

  int? _objectBound(Map<String, Object?> schema, String key) {
    if (!schema.containsKey(key)) {
      return null;
    }
    final value = schema[key];
    if (value is! int || value < 0) {
      throw ArgumentError.value(value, 'schema.$key', 'must be non-negative');
    }
    if (value > _maxJsonSchemaExpandedItems) {
      throw UnsupportedFeatureException(
        'JSON Schema $key greater than $_maxJsonSchemaExpandedItems is not supported',
      );
    }
    return value;
  }

  String _objectEntriesExpression(
    String pairRule,
    int minProperties,
    int? maxProperties,
  ) {
    if (maxProperties == null) {
      if (minProperties == 0) {
        return '($pairRule ("," ws $pairRule)*)?';
      }
      return '${_objectEntrySequence(pairRule, minProperties)} ("," ws $pairRule)*';
    }

    final alternatives = <String>[
      for (var count = minProperties; count <= maxProperties; count += 1)
        _objectEntrySequence(pairRule, count),
    ];
    if (minProperties == 0) {
      if (maxProperties == 0) {
        return '';
      }
      return '(${alternatives.skip(1).join(' | ')})?';
    }
    if (alternatives.length == 1) {
      return alternatives.first;
    }
    return '(${alternatives.join(' | ')})';
  }

  String _objectEntrySequence(String pairRule, int count) {
    if (count == 0) {
      return '';
    }
    return List<String>.filled(count, pairRule).join(' "," ws ');
  }

  String _objectPairsExpression(List<_JsonObjectPair> pairs) {
    String tail(int index, bool hasPrevious) {
      if (index == pairs.length) {
        return '';
      }
      final pair = pairs[index];
      final present =
          '${hasPrevious ? ' "," ws ' : ''}${pair.expression}${tail(index + 1, true)}';
      if (pair.required) {
        return present;
      }
      final absent = tail(index + 1, hasPrevious);
      return '($present | $absent)';
    }

    return tail(0, false);
  }

  String _arrayExpression(Map<String, Object?> schema) {
    _rejectUnsupportedKeys(schema, const <String>{
      ..._annotationKeys,
      'type',
      'items',
      'prefixItems',
      'minItems',
      'maxItems',
    });
    final prefixItems = schema['prefixItems'];
    if (schema.containsKey('prefixItems')) {
      if (prefixItems == null) {
        throw ArgumentError.value(
          prefixItems,
          'schema.prefixItems',
          'must be a list',
        );
      }
      return _prefixItemsExpression(schema, prefixItems);
    }
    final items = schema['items'];
    if (items == null) {
      throw ArgumentError.value(items, 'schema.items', 'must be provided');
    }
    final itemRule = _schemaRule(items, 'schema.items');
    final minItems = _arrayBound(schema, 'minItems') ?? 0;
    final maxItems = _arrayBound(schema, 'maxItems');
    if (maxItems != null && minItems > maxItems) {
      throw ArgumentError.value(
        minItems,
        'schema.minItems',
        'must not exceed maxItems',
      );
    }
    final expression = _arrayItemsExpression(itemRule, minItems, maxItems);
    return '"[" ws $expression "]" ws';
  }

  String _prefixItemsExpression(
    Map<String, Object?> schema,
    Object? prefixItems,
  ) {
    if (schema.containsKey('items') ||
        schema.containsKey('minItems') ||
        schema.containsKey('maxItems')) {
      throw UnsupportedFeatureException(
        'JSON Schema prefixItems only supports fixed tuple arrays',
      );
    }
    if (prefixItems is! List<Object?>) {
      throw ArgumentError.value(
        prefixItems,
        'schema.prefixItems',
        'must be a list',
      );
    }
    if (prefixItems.length > _maxJsonSchemaExpandedItems) {
      throw UnsupportedFeatureException(
        'JSON Schema prefixItems longer than $_maxJsonSchemaExpandedItems is not supported',
      );
    }
    final itemRules = [
      for (var i = 0; i < prefixItems.length; i += 1)
        _schemaRule(prefixItems[i], 'schema.prefixItems[$i]'),
    ];
    return '"[" ws ${itemRules.join(' "," ws ')} "]" ws';
  }

  int? _arrayBound(Map<String, Object?> schema, String key) {
    if (!schema.containsKey(key)) {
      return null;
    }
    final value = schema[key];
    if (value is! int || value < 0) {
      throw ArgumentError.value(value, 'schema.$key', 'must be non-negative');
    }
    if (value > _maxJsonSchemaExpandedItems) {
      throw UnsupportedFeatureException(
        'JSON Schema $key greater than $_maxJsonSchemaExpandedItems is not supported',
      );
    }
    return value;
  }

  String _arrayItemsExpression(String itemRule, int minItems, int? maxItems) {
    if (maxItems == null) {
      if (minItems == 0) {
        return '($itemRule ("," ws $itemRule)*)?';
      }
      return '${_arraySequence(itemRule, minItems)} ("," ws $itemRule)*';
    }

    final alternatives = <String>[
      for (var count = minItems; count <= maxItems; count += 1)
        _arraySequence(itemRule, count),
    ];
    if (minItems == 0) {
      if (maxItems == 0) {
        return '';
      }
      return '(${alternatives.skip(1).join(' | ')})?';
    }
    if (alternatives.length == 1) {
      return alternatives.first;
    }
    return '(${alternatives.join(' | ')})';
  }

  String _arraySequence(String itemRule, int count) {
    if (count == 0) {
      return '';
    }
    return List<String>.filled(count, itemRule).join(' "," ws ');
  }

  String _stringExpression(Map<String, Object?> schema) {
    _rejectUnsupportedKeys(schema, const <String>{
      ..._annotationKeys,
      'type',
      'format',
      'minLength',
      'maxLength',
    });
    final format = schema['format'];
    if (schema.containsKey('format')) {
      if (format is! String) {
        throw ArgumentError.value(format, 'schema.format', 'must be a string');
      }
      if (schema.containsKey('minLength') || schema.containsKey('maxLength')) {
        throw UnsupportedFeatureException(
          'JSON Schema format cannot be combined with length bounds',
        );
      }
      return _formatExpression(format);
    }
    final minLength = _stringLengthBound(schema, 'minLength') ?? 0;
    final maxLength = _stringLengthBound(schema, 'maxLength');
    if (maxLength != null && minLength > maxLength) {
      throw ArgumentError.value(
        minLength,
        'schema.minLength',
        'must not exceed maxLength',
      );
    }
    if (minLength == 0 && maxLength == null) {
      return 'string';
    }
    return r'''"\"" char''' +
        _repeatRange(minLength, maxLength) +
        r''' "\"" ws''';
  }

  String _formatExpression(Object? format) {
    return switch (format) {
      'date' => r'''"\"" date "\"" ws''',
      'time' => r'''"\"" time "\"" ws''',
      'date-time' => r'''"\"" date-time "\"" ws''',
      'uuid' => r'''"\"" uuid "\"" ws''',
      _ => throw UnsupportedFeatureException(
        'Unsupported JSON Schema format: $format',
      ),
    };
  }

  String _integerExpression(Map<String, Object?> schema) {
    _rejectUnsupportedKeys(schema, const <String>{
      ..._annotationKeys,
      'type',
      'minimum',
      'maximum',
      'exclusiveMinimum',
      'exclusiveMaximum',
      'multipleOf',
    });
    final hasRange = schema.keys.any(_integerRangeKeyword);
    final multipleOf = _integerMultipleOf(schema);
    if (!hasRange && multipleOf == null) {
      return 'integer';
    }
    final minimum = _integerLowerBound(schema);
    final maximum = _integerUpperBound(schema);
    if (minimum == null || maximum == null) {
      throw UnsupportedFeatureException(
        'JSON Schema integer ranges require lower and upper bounds',
      );
    }
    if (minimum > maximum) {
      throw ArgumentError.value(
        minimum,
        'schema.integerRange',
        'lower bound must not exceed upper bound',
      );
    }
    final count = maximum - minimum + 1;
    if (count > _maxJsonSchemaExpandedItems) {
      throw UnsupportedFeatureException(
        'JSON Schema integer range larger than $_maxJsonSchemaExpandedItems is not supported',
      );
    }
    final values = <int>[
      for (var value = minimum; value <= maximum; value += 1)
        if (multipleOf == null || value % multipleOf == 0) value,
    ];
    if (values.isEmpty) {
      throw ArgumentError.value(
        multipleOf,
        'schema.multipleOf',
        'matches no integers in range',
      );
    }
    return [
      for (final value in values)
        _jsonLiteralExpression(value, 'schema.integerRange'),
    ].join(' | ');
  }

  int? _integerLowerBound(Map<String, Object?> schema) {
    final hasMinimum = schema.containsKey('minimum');
    final hasExclusiveMinimum = schema.containsKey('exclusiveMinimum');
    if (hasMinimum && hasExclusiveMinimum) {
      throw UnsupportedFeatureException(
        'JSON Schema integer ranges cannot mix minimum and exclusiveMinimum',
      );
    }
    if (hasMinimum) {
      return _integerBound(schema, 'minimum');
    }
    if (hasExclusiveMinimum) {
      return _integerBound(schema, 'exclusiveMinimum') + 1;
    }
    return null;
  }

  int? _integerUpperBound(Map<String, Object?> schema) {
    final hasMaximum = schema.containsKey('maximum');
    final hasExclusiveMaximum = schema.containsKey('exclusiveMaximum');
    if (hasMaximum && hasExclusiveMaximum) {
      throw UnsupportedFeatureException(
        'JSON Schema integer ranges cannot mix maximum and exclusiveMaximum',
      );
    }
    if (hasMaximum) {
      return _integerBound(schema, 'maximum');
    }
    if (hasExclusiveMaximum) {
      return _integerBound(schema, 'exclusiveMaximum') - 1;
    }
    return null;
  }

  int _integerBound(Map<String, Object?> schema, String key) {
    final value = schema[key];
    if (value is! int) {
      throw ArgumentError.value(value, 'schema.$key', 'must be an integer');
    }
    return value;
  }

  int? _integerMultipleOf(Map<String, Object?> schema) {
    if (!schema.containsKey('multipleOf')) {
      return null;
    }
    final value = schema['multipleOf'];
    if (value is! int || value <= 0) {
      throw ArgumentError.value(
        value,
        'schema.multipleOf',
        'must be a positive integer',
      );
    }
    return value;
  }

  int? _stringLengthBound(Map<String, Object?> schema, String key) {
    if (!schema.containsKey(key)) {
      return null;
    }
    final value = schema[key];
    if (value is! int || value < 0) {
      throw ArgumentError.value(value, 'schema.$key', 'must be non-negative');
    }
    if (value > _maxJsonSchemaStringLength) {
      throw UnsupportedFeatureException(
        'JSON Schema $key greater than $_maxJsonSchemaStringLength is not supported',
      );
    }
    return value;
  }

  String _repeatRange(int min, int? max) {
    if (max == null) {
      return '{$min,}';
    }
    if (min == max) {
      return '{$min}';
    }
    return '{$min,$max}';
  }

  String _simpleExpression(Map<String, Object?> schema, String expression) {
    _rejectUnsupportedKeys(schema, const <String>{..._annotationKeys, 'type'});
    return expression;
  }

  Map<String, Object?> _objectSchema(Object? value, String name) {
    if (value is Map<String, Object?>) {
      return value;
    }
    if (value is Map<Object?, Object?>) {
      return value.map((key, value) {
        if (key is! String) {
          throw ArgumentError.value(key, name, 'keys must be strings');
        }
        return MapEntry(key, value);
      });
    }
    throw ArgumentError.value(value, name, 'must be an object');
  }

  Map<String, Object?> _stringMap(Object? value, String name) {
    return _objectSchema(value, name);
  }

  void _rejectUnsupportedKeys(
    Map<String, Object?> schema,
    Set<String> supported,
  ) {
    final unsupported = schema.keys.where((key) => !supported.contains(key));
    if (unsupported.isNotEmpty) {
      throw UnsupportedFeatureException(
        'Unsupported JSON Schema key: ${unsupported.first}',
      );
    }
  }

  void _addCommonRules() {
    _rules.addAll(const <String, String>{
      'value':
          'object | array | string | number | ("true" | "false" | "null") ws',
      'object':
          r'''"{" ws (string ":" ws value ("," ws string ":" ws value)*)? "}" ws''',
      'array': r'''"[" ws (value ("," ws value)*)? "]" ws''',
      'string': r'''"\"" char* "\"" ws''',
      'char':
          r'''[^"\\\x7F\x00-\x1F] | "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4})''',
      'number':
          r'''"-"? ([0-9] | [1-9] [0-9]{0,15}) ("." [0-9]+)? ([eE] [-+]? [0-9] [0-9]{0,15})? ws''',
      'integer': r'''"-"? ([0-9] | [1-9] [0-9]{0,15}) ws''',
      'date':
          r'''[0-9]{4} "-" ("0" [1-9] | "1" [0-2]) "-" ("0" [1-9] | [1-2] [0-9] | "3" [0-1])''',
      'time':
          r'''([01] [0-9] | "2" [0-3]) ":" [0-5] [0-9] ":" [0-5] [0-9] ("." [0-9]{3})? ("Z" | ("+" | "-") ([01] [0-9] | "2" [0-3]) ":" [0-5] [0-9])''',
      'date-time': 'date "T" time',
      'uuid':
          r'''[0-9a-fA-F]{8} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{12}''',
      'ws': r'''| " " | "\n" [ \t]{0,20}''',
    });
  }

  String _gbnfLiteral(String value) {
    final buffer = StringBuffer('"');
    for (var i = 0; i < value.length; i += 1) {
      final code = value.codeUnitAt(i);
      switch (code) {
        case 0x08:
          buffer.write(r'\b');
        case 0x09:
          buffer.write(r'\t');
        case 0x0A:
          buffer.write(r'\n');
        case 0x0C:
          buffer.write(r'\f');
        case 0x0D:
          buffer.write(r'\r');
        case 0x22:
          buffer.write(r'\"');
        case 0x5C:
          buffer.write(r'\\');
        default:
          if (code < 0x20) {
            buffer.write(r'\u');
            buffer.write(code.toRadixString(16).padLeft(4, '0'));
          } else {
            buffer.writeCharCode(code);
          }
      }
    }
    buffer.write('"');
    return buffer.toString();
  }
}

final class _JsonObjectPair {
  const _JsonObjectPair({required this.expression, required this.required});

  final String expression;
  final bool required;
}

const Set<String> _annotationKeys = <String>{
  '\$schema',
  '\$id',
  '\$comment',
  'title',
  'description',
  'default',
  'deprecated',
  'examples',
  'readOnly',
  'writeOnly',
};

const Set<String> _objectKeywords = <String>{
  'properties',
  'required',
  'additionalProperties',
  'minProperties',
  'maxProperties',
};
const Set<String> _arrayKeywords = <String>{
  'items',
  'prefixItems',
  'minItems',
  'maxItems',
};
const Set<String> _stringKeywords = <String>{
  'format',
  'minLength',
  'maxLength',
};
const Set<String> _integerRangeKeywords = <String>{
  'minimum',
  'maximum',
  'exclusiveMinimum',
  'exclusiveMaximum',
};

bool _objectKeyword(String key) => _objectKeywords.contains(key);
bool _arrayKeyword(String key) => _arrayKeywords.contains(key);
bool _stringKeyword(String key) => _stringKeywords.contains(key);
bool _integerRangeKeyword(String key) => _integerRangeKeywords.contains(key);

const int _maxJsonSchemaExpandedItems = 64;
const int _maxJsonSchemaStringLength = 4096;
const int _maxStopTokens = 1024;
const int _maxStreamChunkTokens = 1024;
const int _maxSpeculativeNgramSize = 1024;
const int _maxSpeculativeDraftLength = 1024;

bool _isModelBackedSpeculation(SpeculativeDecodingConfig config) =>
    config is DraftModelSpeculation ||
    config is Eagle3Speculation ||
    config is DFlashSpeculation ||
    config is MtpSpeculation;

void _validateSpeculativeDecoding(SpeculativeDecodingConfig config) {
  switch (config) {
    case NoSpeculativeDecoding():
      return;
    case DraftModelSpeculation(:final draftModelPath, :final draftLength):
      _validatePathText(draftModelPath, 'draftModelPath');
      _validateSpeculativeDraftLength(draftLength);
    case Eagle3Speculation(:final draftModelPath, :final draftLength):
      _validatePathText(draftModelPath, 'draftModelPath');
      _validateSpeculativeDraftLength(draftLength);
    case DFlashSpeculation(:final draftModelPath, :final draftLength):
      _validatePathText(draftModelPath, 'draftModelPath');
      _validateSpeculativeDraftLength(draftLength);
    case MtpSpeculation(:final mtpModelPath, :final draftLength):
      final path = mtpModelPath;
      if (path != null) {
        _validatePathText(path, 'mtpModelPath');
      }
      _validateSpeculativeDraftLength(draftLength);
    case NGramSpeculation(
      :final strategy,
      :final ngramSize,
      :final draftLength,
    ):
      if (strategy.trim().isEmpty) {
        throw ArgumentError.value(strategy, 'strategy', 'must not be empty');
      }
      if (strategy.contains('\u0000')) {
        throw ArgumentError.value(strategy, 'strategy', 'must not contain NUL');
      }
      if (strategy.contains('\n') || strategy.contains('\r')) {
        throw ArgumentError.value(
          strategy,
          'strategy',
          'must not contain line breaks',
        );
      }
      if (strategy.contains(RegExp(r'\s'))) {
        throw ArgumentError.value(
          strategy,
          'strategy',
          'must not contain whitespace',
        );
      }
      if (ngramSize <= 0 || ngramSize > _maxSpeculativeNgramSize) {
        throw ArgumentError.value(
          ngramSize,
          'ngramSize',
          'must be between 1 and $_maxSpeculativeNgramSize',
        );
      }
      if (draftLength <= 0 || draftLength > _maxSpeculativeNgramSize) {
        throw ArgumentError.value(
          draftLength,
          'draftLength',
          'must be between 1 and $_maxSpeculativeNgramSize',
        );
      }
      if (draftLength < ngramSize) {
        throw ArgumentError.value(
          draftLength,
          'draftLength',
          'must be greater than or equal to ngramSize',
        );
      }
    case NGramModSpeculation(
      :final matchLength,
      :final minimumDraftLength,
      :final maximumDraftLength,
    ):
      if (matchLength <= 0 || matchLength > _maxSpeculativeNgramSize) {
        throw ArgumentError.value(
          matchLength,
          'matchLength',
          'must be between 1 and $_maxSpeculativeNgramSize',
        );
      }
      if (minimumDraftLength <= 0 ||
          minimumDraftLength > _maxSpeculativeNgramSize) {
        throw ArgumentError.value(
          minimumDraftLength,
          'minimumDraftLength',
          'must be between 1 and $_maxSpeculativeNgramSize',
        );
      }
      if (maximumDraftLength <= 0 ||
          maximumDraftLength > _maxSpeculativeNgramSize) {
        throw ArgumentError.value(
          maximumDraftLength,
          'maximumDraftLength',
          'must be between 1 and $_maxSpeculativeNgramSize',
        );
      }
      if (minimumDraftLength > maximumDraftLength) {
        throw ArgumentError.value(
          minimumDraftLength,
          'minimumDraftLength',
          'must not exceed maximumDraftLength',
        );
      }
    case NGramCacheSpeculation():
  }
}

void _validateSpeculativeDraftLength(int value) {
  if (value <= 0 || value > _maxSpeculativeDraftLength) {
    throw ArgumentError.value(
      value,
      'draftLength',
      'must be between 1 and $_maxSpeculativeDraftLength',
    );
  }
}

void _validatePathText(String value, String name) {
  if (value.trim().isEmpty) {
    throw ArgumentError.value(value, name, 'must not be empty');
  }
  if (value.contains('\u0000')) {
    throw ArgumentError.value(value, name, 'must not contain NUL');
  }
  if (value.contains('\n') || value.contains('\r')) {
    throw ArgumentError.value(value, name, 'must not contain line breaks');
  }
}

void _validateToolName(String value, String name) {
  if (value.trim().isEmpty) {
    throw ArgumentError.value(value, name, 'must not be empty');
  }
  if (value.contains('\n') || value.contains('\r')) {
    throw ArgumentError.value(value, name, 'must not contain line breaks');
  }
  if (value.contains(RegExp(r'\s'))) {
    throw ArgumentError.value(value, name, 'must not contain whitespace');
  }
  if (value.contains('\u0000')) {
    throw ArgumentError.value(value, name, 'must not contain NUL');
  }
}

List<LlamaToolCall> _toolCallList(Object? value) {
  if (value is! List<Object?>) {
    throw ArgumentError.value(value, 'tool_calls', 'must be a list');
  }
  if (value.isEmpty) {
    throw ArgumentError.value(value, 'tool_calls', 'must not be empty');
  }
  final calls = value.map(_toolCallFromJson).toList();
  _validateUniqueToolCallIds(calls);
  return calls;
}

void _validateUniqueToolCallIds(List<LlamaToolCall> calls) {
  final ids = <String>{};
  for (final call in calls) {
    final id = call.id;
    if (id != null && !ids.add(id)) {
      throw ArgumentError.value(id, 'toolCall.id', 'must be unique');
    }
  }
}

LlamaToolCall _toolCallFromJson(Object? value) {
  final json = _jsonMap(value, 'toolCall');
  final function = json['function'];
  _rejectUnexpectedKeys(
    json,
    function == null
        ? const <String>{'id', 'type', 'name', 'arguments'}
        : const <String>{'id', 'type', 'function'},
    'toolCall',
  );
  final type = json['type'];
  if (type != null && type != 'function') {
    throw ArgumentError.value(type, 'toolCall.type', 'must be function');
  }
  final functionJson = function == null
      ? json
      : _jsonMap(function, 'toolCall.function');
  if (function != null) {
    _rejectUnexpectedKeys(functionJson, const <String>{
      'name',
      'arguments',
    }, 'toolCall.function');
  }
  final id = json['id'];
  if (id != null && id is! String) {
    throw ArgumentError.value(id, 'toolCall.id', 'must be a string');
  }
  if (id is String) {
    _validateToolName(id, 'toolCall.id');
  }
  final name = functionJson['name'];
  if (name is! String) {
    throw ArgumentError.value(name, 'toolCall.name', 'must be a string');
  }
  _validateToolName(name, 'toolCall.name');
  final arguments = functionJson.containsKey('arguments')
      ? _toolArguments(functionJson['arguments'])
      : const <String, Object?>{};
  return LlamaToolCall(id: id as String?, name: name, arguments: arguments);
}

void _rejectUnexpectedKeys(
  Map<String, Object?> value,
  Set<String> allowed,
  String name,
) {
  for (final key in value.keys) {
    if (!allowed.contains(key)) {
      throw ArgumentError.value(key, name, 'contains unsupported key');
    }
  }
}

Map<String, Object?> _toolArguments(Object? value) {
  final arguments = value is String
      ? () {
          if (value.trim().isEmpty) {
            throw ArgumentError.value(
              value,
              'toolCall.arguments',
              'must not be empty',
            );
          }
          try {
            return _jsonMap(jsonDecode(value), 'toolCall.arguments');
          } on FormatException catch (error) {
            throw ArgumentError.value(
              value,
              'toolCall.arguments',
              error.message,
            );
          }
        }()
      : _jsonMap(value, 'toolCall.arguments');
  return _jsonObjectSnapshot(arguments, 'toolCall.arguments');
}

Map<String, Object?> _jsonMap(Object? value, String name) {
  if (value is Map<String, Object?>) {
    return Map<String, Object?>.unmodifiable(value);
  }
  if (value is Map<Object?, Object?>) {
    final result = <String, Object?>{};
    for (final entry in value.entries) {
      final key = entry.key;
      if (key is! String) {
        throw ArgumentError.value(key, name, 'keys must be strings');
      }
      result[key] = entry.value;
    }
    return Map<String, Object?>.unmodifiable(result);
  }
  throw ArgumentError.value(value, name, 'must be a JSON object');
}

void _validateJsonValue(
  Object? value,
  String name, [
  Set<Object>? activeContainers,
]) {
  if (value == null || value is bool) {
    return;
  }
  if (value is String) {
    if (value.contains('\u0000')) {
      throw ArgumentError.value(value, name, 'must not contain NUL');
    }
    return;
  }
  if (value is num) {
    if (!value.isFinite) {
      throw ArgumentError.value(value, name, 'must be finite');
    }
    return;
  }
  if (value is List<Object?>) {
    final active = activeContainers ?? Set<Object>.identity();
    if (!active.add(value)) {
      throw ArgumentError.value(value, name, 'must not contain cycles');
    }
    for (var i = 0; i < value.length; i += 1) {
      _validateJsonValue(value[i], '$name[$i]', active);
    }
    active.remove(value);
    return;
  }
  if (value is Map<Object?, Object?>) {
    final active = activeContainers ?? Set<Object>.identity();
    if (!active.add(value)) {
      throw ArgumentError.value(value, name, 'must not contain cycles');
    }
    for (final entry in value.entries) {
      final key = entry.key;
      if (key is! String) {
        throw ArgumentError.value(key, name, 'keys must be strings');
      }
      if (key.contains('\u0000')) {
        throw ArgumentError.value(key, name, 'keys must not contain NUL');
      }
      _validateJsonValue(entry.value, '$name.$key', active);
    }
    active.remove(value);
    return;
  }
  throw ArgumentError.value(value, name, 'must be a JSON value');
}

Map<String, Object?> _jsonObjectSnapshot(
  Map<String, Object?> value,
  String name,
) {
  _validateJsonValue(value, name);
  return _jsonSnapshotMap(value, name);
}

Map<String, Object?> _jsonSnapshotMap(
  Map<Object?, Object?> value,
  String name,
) {
  final result = <String, Object?>{};
  for (final entry in value.entries) {
    final key = entry.key;
    if (key is! String) {
      throw ArgumentError.value(key, name, 'keys must be strings');
    }
    result[key] = _jsonSnapshotValue(entry.value, '$name.$key');
  }
  return Map<String, Object?>.unmodifiable(result);
}

Object? _jsonSnapshotValue(Object? value, String name) {
  if (value == null || value is String || value is bool || value is num) {
    return value;
  }
  if (value is List<Object?>) {
    return List<Object?>.unmodifiable(
      value.indexed.map(
        (entry) => _jsonSnapshotValue(entry.$2, '$name[${entry.$1}]'),
      ),
    );
  }
  if (value is Map<Object?, Object?>) {
    return _jsonSnapshotMap(value, name);
  }
  throw ArgumentError.value(value, name, 'must be a JSON value');
}

String _textFromParts(List<ChatContentPart> parts) {
  return parts.whereType<TextPart>().map((part) => part.text).join();
}

enum EmbeddingPooling { model, mean, cls, last, rank }

final class EmbeddingConfig {
  const EmbeddingConfig({
    this.normalize = true,
    this.addSpecial = true,
    this.parseSpecial = false,
    this.pooling = EmbeddingPooling.model,
  });

  final bool normalize;
  final bool addSpecial;
  final bool parseSpecial;
  final EmbeddingPooling pooling;

  void validate() {
    if (pooling == EmbeddingPooling.rank) {
      throw const UnsupportedFeatureException(
        'Rank pooling is exposed through LlamaReranking, not LlamaEmbeddings.',
      );
    }
  }
}

final class RerankingConfig {
  const RerankingConfig({this.addSpecial = true, this.parseSpecial = false});

  final bool addSpecial;
  final bool parseSpecial;
}

final class LoraAdapterConfig {
  const LoraAdapterConfig({required this.path, this.scale = 1.0});

  final String path;
  final double scale;

  void validate() {
    _validatePathText(path, 'path');
    _validateFinite(scale, 'scale');
    if (scale < 0) {
      throw ArgumentError.value(scale, 'scale', 'must be non-negative');
    }
  }
}

final class LoraAdapterInfo {
  const LoraAdapterInfo({
    required this.id,
    required this.path,
    required this.scale,
  });

  final int id;
  final String path;
  final double scale;
}

enum ChatRole { system, user, assistant, tool }

final class ChatMessage {
  const ChatMessage({required this.role, required this.text})
    : parts = const <ChatContentPart>[],
      toolCalls = const <LlamaToolCall>[],
      toolName = null,
      toolCallId = null;

  ChatMessage.content({
    required this.role,
    required List<ChatContentPart> parts,
  }) : text = _textFromParts(parts),
       parts = List<ChatContentPart>.unmodifiable(parts),
       toolCalls = const <LlamaToolCall>[],
       toolName = null,
       toolCallId = null {
    if (parts.isEmpty) {
      throw ArgumentError.value(parts, 'parts', 'must not be empty');
    }
  }

  ChatMessage.assistantToolCalls({
    required List<LlamaToolCall> toolCalls,
    this.text = '',
  }) : role = ChatRole.assistant,
       parts = const <ChatContentPart>[],
       toolCalls = List<LlamaToolCall>.unmodifiable(
         toolCalls.map(_snapshotToolCallValue),
       ),
       toolName = null,
       toolCallId = null {
    if (toolCalls.isEmpty) {
      throw ArgumentError.value(toolCalls, 'toolCalls', 'must not be empty');
    }
  }

  const ChatMessage.toolResult({
    required this.toolCallId,
    required String name,
    required this.text,
  }) : role = ChatRole.tool,
       parts = const <ChatContentPart>[],
       toolCalls = const <LlamaToolCall>[],
       toolName = name;

  factory ChatMessage.system(String text) =>
      ChatMessage(role: ChatRole.system, text: text);

  factory ChatMessage.user(String text) =>
      ChatMessage(role: ChatRole.user, text: text);

  factory ChatMessage.assistant(String text) =>
      ChatMessage(role: ChatRole.assistant, text: text);

  factory ChatMessage.tool(String text) =>
      ChatMessage(role: ChatRole.tool, text: text);

  final ChatRole role;
  final String text;
  final List<ChatContentPart> parts;
  final List<LlamaToolCall> toolCalls;
  final String? toolName;
  final String? toolCallId;

  bool get hasNonTextParts => parts.any((part) => part is! TextPart);

  bool get hasToolData =>
      toolCalls.isNotEmpty || toolName != null || toolCallId != null;
}

sealed class ChatContentPart {
  const ChatContentPart();
}

final class TextPart extends ChatContentPart {
  const TextPart(this.text);

  final String text;
}

final class ImagePart extends ChatContentPart {
  const ImagePart.fromFile(this.path, {this.mimeType}) : bytes = null;

  ImagePart.fromBytes(Uint8List bytes, {this.mimeType})
    : path = null,
      bytes = _snapshotMediaBytes(bytes);

  final String? path;
  final Uint8List? bytes;
  final String? mimeType;
}

final class AudioPart extends ChatContentPart {
  const AudioPart.fromFile(this.path, {this.mimeType}) : bytes = null;

  AudioPart.fromBytes(Uint8List bytes, {this.mimeType})
    : path = null,
      bytes = _snapshotMediaBytes(bytes);

  final String? path;
  final Uint8List? bytes;
  final String? mimeType;
}

final class VideoPart extends ChatContentPart {
  const VideoPart.fromFile(this.path, {this.mimeType}) : bytes = null;

  VideoPart.fromBytes(Uint8List bytes, {this.mimeType})
    : path = null,
      bytes = _snapshotMediaBytes(bytes);

  final String? path;
  final Uint8List? bytes;
  final String? mimeType;
}

Uint8List _snapshotMediaBytes(Uint8List bytes) {
  if (bytes.isEmpty) {
    throw ArgumentError.value(bytes, 'bytes', 'must not be empty');
  }
  return Uint8List.fromList(bytes).asUnmodifiableView();
}

final class LlamaToolDefinition {
  const LlamaToolDefinition({
    required this.name,
    required this.description,
    required this.parametersSchema,
  });

  final String name;
  final String description;
  final Map<String, Object?> parametersSchema;

  void validate() {
    _validateToolName(name, 'name');
    if (description.trim().isEmpty) {
      throw ArgumentError.value(
        description,
        'description',
        'must not be empty',
      );
    }
    if (description.contains('\u0000')) {
      throw ArgumentError.value(
        description,
        'description',
        'must not contain NUL',
      );
    }
    _validateJsonValue(parametersSchema, 'parametersSchema');
  }

  /// Validates parsed tool [arguments] against [parametersSchema].
  ///
  /// This is applied automatically to model-generated calls before they are
  /// exposed by `LlamaEngine.chat`. Apps may also use it before replaying
  /// externally supplied tool-call history.
  void validateArguments(Map<String, Object?> arguments) {
    validate();
    final error = _ToolArgumentsValidator(parametersSchema).validate(arguments);
    if (error != null) {
      throw ArgumentError.value(arguments, 'arguments', error);
    }
  }

  Map<String, Object?> toJson() {
    validate();
    return Map<String, Object?>.unmodifiable(<String, Object?>{
      'type': 'function',
      'function': Map<String, Object?>.unmodifiable(<String, Object?>{
        'name': name,
        'description': description,
        'parameters': _jsonObjectSnapshot(parametersSchema, 'parametersSchema'),
      }),
    });
  }
}

final class _ToolArgumentsValidator {
  _ToolArgumentsValidator(this.rootSchema);

  final Map<String, Object?> rootSchema;

  String? validate(Object? value) => _validate(value, rootSchema, r'$');

  String? _validate(Object? value, Object? schemaValue, String path) {
    if (schemaValue is bool) {
      return schemaValue ? null : '$path is rejected by the schema';
    }
    final schema = _schemaMap(schemaValue);
    if (schema == null) {
      return '$path has an invalid schema';
    }

    if (schema['\$ref'] case final String reference) {
      final resolved = _resolveReference(reference);
      if (resolved == null) {
        return '$path uses unsupported or unresolved schema reference $reference';
      }
      final error = _validate(value, resolved, path);
      if (error != null) {
        return error;
      }
    }

    if (schema['allOf'] case final List<Object?> schemas) {
      for (final child in schemas) {
        final error = _validate(value, child, path);
        if (error != null) {
          return error;
        }
      }
    }
    if (schema['anyOf'] case final List<Object?> schemas) {
      if (!schemas.any((child) => _validate(value, child, path) == null)) {
        return '$path does not match any allowed schema';
      }
    }
    if (schema['oneOf'] case final List<Object?> schemas) {
      final matches = schemas
          .where((child) => _validate(value, child, path) == null)
          .length;
      if (matches != 1) {
        return '$path must match exactly one allowed schema';
      }
    }
    if (schema.containsKey('not') &&
        _validate(value, schema['not'], path) == null) {
      return '$path matches a disallowed schema';
    }

    if (schema.containsKey('const') &&
        !_jsonValuesEqual(value, schema['const'])) {
      return '$path must equal ${jsonEncode(schema['const'])}';
    }
    if (schema['enum'] case final List<Object?> values) {
      if (!values.any((candidate) => _jsonValuesEqual(value, candidate))) {
        return '$path is not one of the allowed values';
      }
    }

    final declaredType = schema['type'];
    if (declaredType is String) {
      final error = _validateType(value, declaredType, path);
      if (error != null) {
        return error;
      }
    } else if (declaredType is List<Object?>) {
      final types = declaredType.whereType<String>().toList(growable: false);
      if (types.length != declaredType.length ||
          !types.any((type) => _validateType(value, type, path) == null)) {
        return '$path does not match any declared type';
      }
    } else if (declaredType != null) {
      return '$path has an invalid schema type';
    }

    if (value is Map<Object?, Object?>) {
      final object = <String, Object?>{};
      for (final entry in value.entries) {
        if (entry.key is! String) {
          return '$path contains a non-string property name';
        }
        object[entry.key! as String] = entry.value;
      }
      final error = _validateObject(object, schema, path);
      if (error != null) {
        return error;
      }
    } else if (value is List<Object?>) {
      final error = _validateArray(value, schema, path);
      if (error != null) {
        return error;
      }
    } else if (value is String) {
      final error = _validateString(value, schema, path);
      if (error != null) {
        return error;
      }
    } else if (value is num) {
      final error = _validateNumber(value, schema, path);
      if (error != null) {
        return error;
      }
    }
    return null;
  }

  String? _validateType(Object? value, String type, String path) {
    final matches = switch (type) {
      'object' => value is Map<Object?, Object?>,
      'array' => value is List<Object?>,
      'string' => value is String,
      'number' => value is num && value.isFinite,
      'integer' =>
        value is int ||
            (value is double && value.isFinite && value == value.truncate()),
      'boolean' => value is bool,
      'null' => value == null,
      _ => false,
    };
    return matches ? null : '$path must be $type';
  }

  String? _validateObject(
    Map<String, Object?> value,
    Map<String, Object?> schema,
    String path,
  ) {
    if (schema['minProperties'] case final int minimum) {
      if (value.length < minimum) {
        return '$path must contain at least $minimum properties';
      }
    }
    if (schema['maxProperties'] case final int maximum) {
      if (value.length > maximum) {
        return '$path must contain at most $maximum properties';
      }
    }

    final properties = _schemaMap(schema['properties']) ?? const {};
    if (schema['required'] case final List<Object?> required) {
      for (final name in required.whereType<String>()) {
        if (!value.containsKey(name)) {
          return '$path.$name is required';
        }
      }
    }

    for (final entry in value.entries) {
      final propertyPath = _propertyPath(path, entry.key);
      if (properties.containsKey(entry.key)) {
        final error = _validate(
          entry.value,
          properties[entry.key],
          propertyPath,
        );
        if (error != null) {
          return error;
        }
        continue;
      }
      final additional = schema.containsKey('additionalProperties')
          ? schema['additionalProperties']
          : true;
      if (additional == false) {
        return '$propertyPath is not declared by the tool schema';
      }
      if (additional is Map<Object?, Object?> || additional is bool) {
        final error = _validate(entry.value, additional, propertyPath);
        if (error != null) {
          return error;
        }
      }
    }
    return null;
  }

  String? _validateArray(
    List<Object?> value,
    Map<String, Object?> schema,
    String path,
  ) {
    if (schema['minItems'] case final int minimum) {
      if (value.length < minimum) {
        return '$path must contain at least $minimum items';
      }
    }
    if (schema['maxItems'] case final int maximum) {
      if (value.length > maximum) {
        return '$path must contain at most $maximum items';
      }
    }
    if (schema['uniqueItems'] == true) {
      for (var i = 0; i < value.length; i += 1) {
        for (var j = i + 1; j < value.length; j += 1) {
          if (_jsonValuesEqual(value[i], value[j])) {
            return '$path must contain unique items';
          }
        }
      }
    }

    final prefixItems = schema['prefixItems'] is List<Object?>
        ? schema['prefixItems']! as List<Object?>
        : const <Object?>[];
    for (var i = 0; i < value.length; i += 1) {
      final Object? itemSchema;
      if (i < prefixItems.length) {
        itemSchema = prefixItems[i];
      } else if (schema.containsKey('items')) {
        itemSchema = schema['items'];
      } else {
        continue;
      }
      final error = _validate(value[i], itemSchema, '$path[$i]');
      if (error != null) {
        return error;
      }
    }
    return null;
  }

  String? _validateString(
    String value,
    Map<String, Object?> schema,
    String path,
  ) {
    final length = value.runes.length;
    if (schema['minLength'] case final int minimum) {
      if (length < minimum) {
        return '$path must contain at least $minimum characters';
      }
    }
    if (schema['maxLength'] case final int maximum) {
      if (length > maximum) {
        return '$path must contain at most $maximum characters';
      }
    }
    if (schema['pattern'] case final String pattern) {
      try {
        if (!RegExp(pattern, unicode: true).hasMatch(value)) {
          return '$path does not match the required pattern';
        }
      } on FormatException {
        return '$path uses an invalid schema pattern';
      }
    }
    if (schema['format'] case final String format) {
      final valid = switch (format) {
        'date' => RegExp(r'^\d{4}-\d{2}-\d{2}$').hasMatch(value),
        'time' => RegExp(
          r'^\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$',
        ).hasMatch(value),
        'date-time' => DateTime.tryParse(value) != null && value.contains('T'),
        'uuid' => RegExp(
          r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$',
        ).hasMatch(value),
        _ => true,
      };
      if (!valid) {
        return '$path is not a valid $format value';
      }
    }
    return null;
  }

  String? _validateNumber(num value, Map<String, Object?> schema, String path) {
    if (!value.isFinite) {
      return '$path must be finite';
    }
    if (schema['minimum'] case final num minimum) {
      if (value < minimum) {
        return '$path must be at least $minimum';
      }
    }
    if (schema['maximum'] case final num maximum) {
      if (value > maximum) {
        return '$path must be at most $maximum';
      }
    }
    if (schema['exclusiveMinimum'] case final num minimum) {
      if (value <= minimum) {
        return '$path must be greater than $minimum';
      }
    }
    if (schema['exclusiveMaximum'] case final num maximum) {
      if (value >= maximum) {
        return '$path must be less than $maximum';
      }
    }
    if (schema['multipleOf'] case final num factor) {
      if (factor <= 0 || !factor.isFinite) {
        return '$path uses an invalid multipleOf constraint';
      }
      final quotient = value / factor;
      if ((quotient - quotient.round()).abs() > 1e-9) {
        return '$path must be a multiple of $factor';
      }
    }
    return null;
  }

  Object? _resolveReference(String reference) {
    if (reference == '#') {
      return rootSchema;
    }
    if (!reference.startsWith('#/')) {
      return null;
    }
    Object? current = rootSchema;
    for (final encoded in reference.substring(2).split('/')) {
      final segment = encoded.replaceAll('~1', '/').replaceAll('~0', '~');
      final map = _schemaMap(current);
      if (map == null || !map.containsKey(segment)) {
        return null;
      }
      current = map[segment];
    }
    return current;
  }

  Map<String, Object?>? _schemaMap(Object? value) {
    if (value is Map<String, Object?>) {
      return value;
    }
    if (value is Map<Object?, Object?> &&
        value.keys.every((key) => key is String)) {
      return value.cast<String, Object?>();
    }
    return null;
  }

  String _propertyPath(String path, String property) {
    if (RegExp(r'^[A-Za-z_][A-Za-z0-9_]*$').hasMatch(property)) {
      return '$path.$property';
    }
    return '$path[${jsonEncode(property)}]';
  }
}

bool _jsonValuesEqual(Object? left, Object? right) {
  if (left is num && right is num) {
    return left == right;
  }
  if (left is List<Object?> && right is List<Object?>) {
    return left.length == right.length &&
        List<bool>.generate(
          left.length,
          (index) => _jsonValuesEqual(left[index], right[index]),
        ).every((matches) => matches);
  }
  if (left is Map<Object?, Object?> && right is Map<Object?, Object?>) {
    if (left.length != right.length || !left.keys.every(right.containsKey)) {
      return false;
    }
    return left.keys.every((key) => _jsonValuesEqual(left[key], right[key]));
  }
  return left == right;
}

final class LlamaToolCallingConfig {
  const LlamaToolCallingConfig({
    this.tools = const <LlamaToolDefinition>[],
    this.allowParallelToolCalls = true,
    this.toolChoice = const LlamaToolChoice.auto(),
  });

  final List<LlamaToolDefinition> tools;
  final bool allowParallelToolCalls;
  final LlamaToolChoice toolChoice;

  void validate() {
    final names = <String>{};
    for (final tool in tools) {
      tool.validate();
      if (!names.add(tool.name)) {
        throw ArgumentError.value(tools, 'tools', 'must have unique names');
      }
    }
    switch (toolChoice) {
      case LlamaAutoToolChoice() || LlamaNoToolChoice():
        return;
      case LlamaRequiredToolChoice():
        if (tools.isEmpty) {
          throw ArgumentError.value(
            tools,
            'tools',
            'must not be empty when toolChoice is required',
          );
        }
      case LlamaNamedToolChoice(:final name):
        _validateToolName(name, 'toolChoice.name');
        if (!names.contains(name)) {
          throw ArgumentError.value(
            name,
            'toolChoice.name',
            'must match a configured tool',
          );
        }
    }
  }

  List<Map<String, Object?>> toJson() {
    validate();
    return List<Map<String, Object?>>.unmodifiable(
      tools.map((tool) => tool.toJson()),
    );
  }

  Map<String, Object?> toOpenAiJson() {
    validate();
    final json = <String, Object?>{
      'tools': toJson(),
      'parallel_tool_calls': allowParallelToolCalls,
    };
    switch (toolChoice) {
      case LlamaAutoToolChoice():
        break;
      case LlamaNoToolChoice():
        json['tool_choice'] = 'none';
      case LlamaRequiredToolChoice():
        json['tool_choice'] = 'required';
      case LlamaNamedToolChoice(:final name):
        json['tool_choice'] = Map<String, Object?>.unmodifiable(
          <String, Object?>{
            'type': 'function',
            'function': Map<String, Object?>.unmodifiable(<String, Object?>{
              'name': name,
            }),
          },
        );
    }
    return Map<String, Object?>.unmodifiable(json);
  }
}

LlamaToolCallingConfig _snapshotToolCallingConfig(
  LlamaToolCallingConfig config,
) {
  return LlamaToolCallingConfig(
    tools: List<LlamaToolDefinition>.unmodifiable(
      config.tools.map(
        (tool) => LlamaToolDefinition(
          name: tool.name,
          description: tool.description,
          parametersSchema: _jsonObjectSnapshot(
            tool.parametersSchema,
            'parametersSchema',
          ),
        ),
      ),
    ),
    allowParallelToolCalls: config.allowParallelToolCalls,
    toolChoice: config.toolChoice,
  );
}

sealed class LlamaToolChoice {
  const LlamaToolChoice();

  const factory LlamaToolChoice.auto() = LlamaAutoToolChoice;

  const factory LlamaToolChoice.none() = LlamaNoToolChoice;

  const factory LlamaToolChoice.required() = LlamaRequiredToolChoice;

  const factory LlamaToolChoice.named(String name) = LlamaNamedToolChoice;
}

final class LlamaAutoToolChoice extends LlamaToolChoice {
  const LlamaAutoToolChoice();
}

final class LlamaNoToolChoice extends LlamaToolChoice {
  const LlamaNoToolChoice();
}

final class LlamaRequiredToolChoice extends LlamaToolChoice {
  const LlamaRequiredToolChoice();
}

final class LlamaNamedToolChoice extends LlamaToolChoice {
  const LlamaNamedToolChoice(this.name);

  final String name;
}

final class LlamaToolCall {
  const LlamaToolCall({this.id, required this.name, required this.arguments});

  final String? id;
  final String name;
  final Map<String, Object?> arguments;

  Map<String, Object?> toJson() {
    _validateToolName(name, 'name');
    final id = this.id;
    final json = <String, Object?>{
      'name': name,
      'arguments': _jsonObjectSnapshot(arguments, 'arguments'),
    };
    if (id != null) {
      _validateToolName(id, 'id');
      json['id'] = id;
    }
    return Map<String, Object?>.unmodifiable(json);
  }

  Map<String, Object?> toOpenAiJson() {
    _validateToolName(name, 'name');
    final arguments = _jsonObjectSnapshot(this.arguments, 'arguments');
    final id = this.id;
    final json = <String, Object?>{
      'type': 'function',
      'function': Map<String, Object?>.unmodifiable(<String, Object?>{
        'name': name,
        'arguments': jsonEncode(arguments),
      }),
    };
    if (id != null) {
      _validateToolName(id, 'id');
      json['id'] = id;
    }
    return Map<String, Object?>.unmodifiable(json);
  }
}

LlamaToolCall _snapshotToolCallValue(LlamaToolCall call) {
  return LlamaToolCall(
    id: call.id,
    name: call.name,
    arguments: _jsonObjectSnapshot(call.arguments, 'arguments'),
  );
}

abstract final class LlamaToolCalls {
  static List<LlamaToolCall> parse(
    String text, {
    bool allowParallelToolCalls = true,
  }) {
    if (text.trim().isEmpty) {
      throw ArgumentError.value(text, 'text', 'must not be empty');
    }
    try {
      return fromJson(
        jsonDecode(text),
        allowParallelToolCalls: allowParallelToolCalls,
      );
    } on FormatException catch (error) {
      throw ArgumentError.value(text, 'text', error.message);
    }
  }

  static List<LlamaToolCall> fromJson(
    Object? value, {
    bool allowParallelToolCalls = true,
  }) {
    final calls = value is List<Object?>
        ? _toolCallList(value)
        : _toolCallsFromMap(_jsonMap(value, 'value'));
    if (!allowParallelToolCalls && calls.length > 1) {
      throw UnsupportedFeatureException('Parallel tool calls are not allowed');
    }
    return List<LlamaToolCall>.unmodifiable(calls);
  }

  static Map<String, Object?> toOpenAiJson(List<LlamaToolCall> calls) {
    if (calls.isEmpty) {
      throw ArgumentError.value(calls, 'calls', 'must not be empty');
    }
    _validateUniqueToolCallIds(calls);
    return Map<String, Object?>.unmodifiable(<String, Object?>{
      'tool_calls': List<Map<String, Object?>>.unmodifiable(
        <Map<String, Object?>>[for (final call in calls) call.toOpenAiJson()],
      ),
    });
  }
}

List<LlamaToolCall> _toolCallsFromMap(Map<String, Object?> value) {
  if (!value.containsKey('tool_calls')) {
    return <LlamaToolCall>[_toolCallFromJson(value)];
  }
  _rejectUnexpectedKeys(value, const <String>{'tool_calls'}, 'value');
  return _toolCallList(value['tool_calls']);
}

sealed class SpeculativeDecodingConfig {
  const SpeculativeDecodingConfig();
}

final class NoSpeculativeDecoding extends SpeculativeDecodingConfig {
  const NoSpeculativeDecoding();
}

final class DraftModelSpeculation extends SpeculativeDecodingConfig {
  const DraftModelSpeculation({
    required this.draftModelPath,
    this.draftLength = 3,
  });

  final String draftModelPath;
  final int draftLength;
}

final class Eagle3Speculation extends SpeculativeDecodingConfig {
  const Eagle3Speculation({required this.draftModelPath, this.draftLength = 3});

  final String draftModelPath;
  final int draftLength;
}

final class DFlashSpeculation extends SpeculativeDecodingConfig {
  const DFlashSpeculation({
    required this.draftModelPath,
    this.draftLength = 15,
  });

  final String draftModelPath;
  final int draftLength;
}

final class MtpSpeculation extends SpeculativeDecodingConfig {
  const MtpSpeculation({this.mtpModelPath, this.draftLength = 3});

  final String? mtpModelPath;
  final int draftLength;
}

final class NGramSpeculation extends SpeculativeDecodingConfig {
  const NGramSpeculation({
    required this.strategy,
    this.ngramSize = 12,
    this.draftLength = 48,
  });

  final String strategy;
  final int ngramSize;
  final int draftLength;
}

final class NGramModSpeculation extends SpeculativeDecodingConfig {
  const NGramModSpeculation({
    this.matchLength = 24,
    this.minimumDraftLength = 48,
    this.maximumDraftLength = 64,
  });

  final int matchLength;
  final int minimumDraftLength;
  final int maximumDraftLength;
}

/// Uses the pinned upstream request-local n-gram cache without cache files.
final class NGramCacheSpeculation extends SpeculativeDecodingConfig {
  const NGramCacheSpeculation();
}

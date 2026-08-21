import 'native_bridge.dart';

/// Converts [schema] to GBNF with the pinned llama.cpp implementation.
///
/// This compatibility helper uses the bundled native bridge. New generation
/// code can pass the schema directly to [GenerationConfig.jsonSchema].
String llamaJsonSchemaGrammar(Map<String, Object?> schema) =>
    NativeLlamaBridge.jsonSchemaGrammar(schema);

/// Public API for fllamer, a Dart/Flutter wrapper around llama.cpp.
///
/// Native inference is provided by the bundled C ABI bridge and runs outside
/// the caller isolate. Dart-only RAG primitives do not require the bridge.
library;

export 'src/config.dart';
export 'src/engine.dart';
export 'src/errors.dart';
export 'src/json_schema_grammar.dart';
export 'src/model_info.dart';
export 'src/rag.dart';

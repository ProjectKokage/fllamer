sealed class LlamaException implements Exception {
  const LlamaException(this.message, {this.cause});

  final String message;
  final Object? cause;

  @override
  String toString() {
    final details = cause == null ? '' : ' ($cause)';
    return '$runtimeType: $message$details';
  }
}

final class ModelLoadException extends LlamaException {
  const ModelLoadException(super.message, {super.cause});
}

final class ModelFileException extends LlamaException {
  const ModelFileException(super.message, {super.cause});
}

final class StateFileException extends LlamaException {
  const StateFileException(super.message, {super.cause});
}

final class ContextCreateException extends LlamaException {
  const ContextCreateException(super.message, {super.cause});
}

final class NativeBridgeException extends LlamaException {
  const NativeBridgeException(super.message, {super.cause});
}

final class GenerationException extends LlamaException {
  const GenerationException(super.message, {super.cause});
}

final class EmbeddingException extends LlamaException {
  const EmbeddingException(super.message, {super.cause});
}

final class RerankingException extends LlamaException {
  const RerankingException(super.message, {super.cause});
}

final class RagIndexException extends LlamaException {
  const RagIndexException(super.message, {super.cause});
}

final class LoraException extends LlamaException {
  const LoraException(super.message, {super.cause});
}

final class UnsupportedFeatureException extends LlamaException {
  const UnsupportedFeatureException(super.message, {super.cause});
}

final class NativeOutOfMemoryException extends LlamaException {
  const NativeOutOfMemoryException(super.message, {super.cause});
}

final class ResourceDisposedException extends LlamaException {
  const ResourceDisposedException(super.message, {super.cause});
}

final class CancelledException extends LlamaException {
  const CancelledException(super.message, {super.cause});
}

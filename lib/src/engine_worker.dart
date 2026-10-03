import 'dart:isolate';

import 'errors.dart';

import 'engine_worker_protocol.dart';
import 'native_bridge.dart';
import 'native_engine_handles.dart';

void engineWorkerMain(EngineWorkerStart start) {
  NativeEngineHandles? handles;
  final commands = ReceivePort();
  try {
    final bridge = NativeLlamaBridge.tryOpen(start.config.nativeLibraryPath);
    if (bridge == null) {
      throw nativeBridgeUnavailable(start.config.nativeLibraryPath);
    }
    handles = bridge.openEngine(
      start.config,
      embeddings: start.embeddings,
      pooling: start.pooling,
    );
    start.reply.send(
      EngineWorkerReady(commands.sendPort, handles.context.address),
    );
  } catch (error) {
    start.reply.send(EngineWorkerFailure(NativeError.from(error)));
    return;
  }

  NativeStreamingGeneration? streamingGeneration;
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
      final chunk = generation.next(
        onProgress: (generatedTokens) =>
            reply.send(EngineWorkerStreamProgress(generatedTokens)),
      );
      if (chunk.isDone) {
        Object? closeError = closeStreamingGeneration();
        if (closeError != null) {
          closeError =
              closeStreamingGeneration(resetContext: true) ?? closeError;
          reply.send(EngineWorkerFailure(NativeError.from(closeError)));
        } else {
          reply.send(
            EngineWorkerStreamChunk(
              chunk.text,
              true,
              chunk.telemetry,
              chunk.assistantMessage,
            ),
          );
        }
      } else {
        reply.send(EngineWorkerStreamChunk(chunk.text, false, null, null));
      }
    } catch (error) {
      final closeError = closeStreamingGeneration(resetContext: true);
      reply.send(EngineWorkerFailure(NativeError.from(closeError ?? error)));
    }
  }

  handleMessage = (message) {
    if (message is EngineWorkerFinalize) {
      closeStreamingGeneration();
      try {
        handles?.close();
      } catch (_) {
        // No caller remains to receive a finalizer cleanup error.
      } finally {
        handles = null;
        commands.close();
      }
    } else if (message is EngineWorkerClose) {
      final activeReply = streamingReply;
      Object? cleanupError = closeStreamingGeneration();
      activeReply?.send(
        const EngineWorkerFailure(
          NativeError('cancelled', 'generation cancelled'),
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
            EngineWorkerFailure(NativeError.from(cleanupError)),
          );
        }
      }
    } else if (message is EngineWorkerRequest) {
      try {
        final active = handles;
        if (active == null) {
          throw const ResourceDisposedException('LlamaEngine is closed.');
        }
        message.reply.send(message.request.run(active));
      } catch (error) {
        message.reply.send(EngineWorkerFailure(NativeError.from(error)));
      }
    } else if (message is EngineWorkerStreamComplete) {
      try {
        final active = handles;
        if (active == null) {
          throw const ResourceDisposedException('LlamaEngine is closed.');
        }
        streamingGeneration = active.startChatStream(
          message.messages,
          message.config,
          reusePromptPrefix: message.reusePromptPrefix,
          maximumPromptBytes: message.maximumPromptBytes,
        );
        streamingId = message.id;
        streamingReply = message.reply;
        stepStreamingGeneration(message.id);
      } catch (error) {
        final closeError = closeStreamingGeneration(resetContext: true);
        message.reply.send(
          EngineWorkerFailure(NativeError.from(closeError ?? error)),
        );
      }
    } else if (message is EngineWorkerStreamPrompt) {
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
          EngineWorkerFailure(NativeError.from(closeError ?? error)),
        );
      }
    } else if (message is EngineWorkerStreamNext) {
      stepStreamingGeneration(message.id);
    } else if (message is EngineWorkerStreamDispose) {
      if (streamingGeneration != null && streamingId != message.id) {
        message.reply.send(
          const EngineWorkerFailure(
            NativeError(
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
            EngineWorkerFailure(NativeError.from(cleanupError)),
          );
        }
      }
    }
  };

  commands.listen((message) {
    if (streamingGeneration != null &&
        message is! EngineWorkerStreamNext &&
        message is! EngineWorkerStreamDispose &&
        message is! EngineWorkerClose &&
        message is! EngineWorkerFinalize) {
      engineWorkerReply(message)?.send(
        const EngineWorkerFailure(
          NativeError(
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

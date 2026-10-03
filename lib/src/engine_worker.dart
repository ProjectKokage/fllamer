part of 'native_bridge.dart';

void _engineWorkerMain(_EngineWorkerStart start) {
  _NativeEngineHandles? handles;
  final commands = ReceivePort();
  try {
    final bridge = NativeLlamaBridge.tryOpen(start.config.nativeLibraryPath);
    if (bridge == null) {
      throw _nativeBridgeUnavailable(start.config.nativeLibraryPath);
    }
    handles = bridge._openEngine(
      start.config,
      embeddings: start.embeddings,
      pooling: start.pooling,
    );
    start.reply.send(
      _EngineWorkerReady(commands.sendPort, handles.context.address),
    );
  } catch (error) {
    start.reply.send(_EngineWorkerFailure(_NativeError.from(error)));
    return;
  }

  _NativeStreamingGeneration? streamingGeneration;
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
            reply.send(_EngineWorkerStreamProgress(generatedTokens)),
      );
      if (chunk.isDone) {
        Object? closeError = closeStreamingGeneration();
        if (closeError != null) {
          closeError =
              closeStreamingGeneration(resetContext: true) ?? closeError;
          reply.send(_EngineWorkerFailure(_NativeError.from(closeError)));
        } else {
          reply.send(
            _EngineWorkerStreamChunk(
              chunk.text,
              true,
              chunk.telemetry,
              chunk.assistantMessage,
            ),
          );
        }
      } else {
        reply.send(_EngineWorkerStreamChunk(chunk.text, false, null, null));
      }
    } catch (error) {
      final closeError = closeStreamingGeneration(resetContext: true);
      reply.send(_EngineWorkerFailure(_NativeError.from(closeError ?? error)));
    }
  }

  handleMessage = (message) {
    if (message is _EngineWorkerFinalize) {
      closeStreamingGeneration();
      try {
        handles?.close();
      } catch (_) {
        // No caller remains to receive a finalizer cleanup error.
      } finally {
        handles = null;
        commands.close();
      }
    } else if (message is _EngineWorkerClose) {
      final activeReply = streamingReply;
      Object? cleanupError = closeStreamingGeneration();
      activeReply?.send(
        const _EngineWorkerFailure(
          _NativeError('cancelled', 'generation cancelled'),
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
            _EngineWorkerFailure(_NativeError.from(cleanupError)),
          );
        }
      }
    } else if (message is _EngineWorkerRequest) {
      try {
        final active = handles;
        if (active == null) {
          throw const ResourceDisposedException('LlamaEngine is closed.');
        }
        message.reply.send(message.request.run(active));
      } catch (error) {
        message.reply.send(_EngineWorkerFailure(_NativeError.from(error)));
      }
    } else if (message is _EngineWorkerStreamComplete) {
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
          _EngineWorkerFailure(_NativeError.from(closeError ?? error)),
        );
      }
    } else if (message is _EngineWorkerStreamPrompt) {
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
          _EngineWorkerFailure(_NativeError.from(closeError ?? error)),
        );
      }
    } else if (message is _EngineWorkerStreamNext) {
      stepStreamingGeneration(message.id);
    } else if (message is _EngineWorkerStreamDispose) {
      if (streamingGeneration != null && streamingId != message.id) {
        message.reply.send(
          const _EngineWorkerFailure(
            _NativeError(
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
            _EngineWorkerFailure(_NativeError.from(cleanupError)),
          );
        }
      }
    }
  };

  commands.listen((message) {
    if (streamingGeneration != null &&
        message is! _EngineWorkerStreamNext &&
        message is! _EngineWorkerStreamDispose &&
        message is! _EngineWorkerClose &&
        message is! _EngineWorkerFinalize) {
      _engineWorkerReply(message)?.send(
        const _EngineWorkerFailure(
          _NativeError(
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

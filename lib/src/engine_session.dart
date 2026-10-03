import 'dart:async';
import 'dart:isolate';
import 'dart:typed_data';

import 'config.dart';
import 'errors.dart';
import 'model_info.dart';

import 'engine_worker.dart';
import 'engine_worker_protocol.dart';
import 'native_bridge.dart';

Future<NativeLlamaEngineSession> spawnEngineSession(
  LlamaModelConfig config, {
  required bool embeddings,
  required EmbeddingPooling pooling,
}) async {
  final ready = ReceivePort();
  final lifecyclePort = ReceivePort();
  final lifecycle = _EngineWorkerLifecycle(lifecyclePort);
  final Isolate isolate;
  try {
    isolate = await Isolate.spawn(
      engineWorkerMain,
      EngineWorkerStart(
        config,
        ready.sendPort,
        embeddings: embeddings,
        pooling: pooling,
      ),
      errorsAreFatal: true,
      onError: lifecyclePort.sendPort,
      onExit: lifecyclePort.sendPort,
    );
  } catch (_) {
    ready.close();
    await lifecycle.dispose(expected: true);
    rethrow;
  }
  final message = await lifecycle.receive(ready);

  if (message is EngineWorkerReady) {
    return NativeLlamaEngineSession._(
      isolate,
      message.commands,
      config.nativeLibraryPath,
      message.contextAddress,
      lifecycle,
    );
  }
  isolate.kill(priority: Isolate.immediate);
  await lifecycle.dispose(expected: true);
  if (message is EngineWorkerFailure) {
    throw message.error.toException();
  }
  throw NativeBridgeException('Unexpected engine worker response: $message');
}

final class _EngineWorkerLifecycle {
  _EngineWorkerLifecycle(this._port) {
    _port.listen(_handleEvent);
  }

  final ReceivePort _port;
  final StreamController<EngineWorkerFailure> _failures =
      StreamController<EngineWorkerFailure>.broadcast(sync: true);
  EngineWorkerFailure? _failure;
  bool _expected = false;

  EngineWorkerFailure? get failure => _failure;
  Stream<EngineWorkerFailure> get failures => _failures.stream;

  Future<Object?> receive(ReceivePort reply) {
    final existingFailure = _failure;
    if (existingFailure != null) {
      reply.close();
      return Future<Object?>.value(existingFailure);
    }

    final completer = Completer<Object?>();
    late final StreamSubscription<Object?> replySubscription;
    late final StreamSubscription<EngineWorkerFailure> failureSubscription;

    void complete(Object? value) {
      if (!completer.isCompleted) {
        completer.complete(value);
      }
    }

    replySubscription = reply.listen(complete);
    failureSubscription = failures.listen(complete);
    final racedFailure = _failure;
    if (racedFailure != null) {
      complete(racedFailure);
    }

    return completer.future.whenComplete(() async {
      await replySubscription.cancel();
      await failureSubscription.cancel();
      reply.close();
    });
  }

  Future<void> dispose({required bool expected}) async {
    _expected = _expected || expected;
    _port.close();
    if (!_failures.isClosed) {
      await _failures.close();
    }
  }

  void _handleEvent(Object? event) {
    if (!_expected && _failure == null) {
      final details = event is List<Object?> && event.isNotEmpty
          ? ': ${event.first}'
          : '';
      final failure = EngineWorkerFailure(
        NativeError(
          'nativeBridge',
          'Inference worker exited unexpectedly$details',
        ),
      );
      _failure = failure;
      _failures.add(failure);
    }
    _port.close();
    if (!_failures.isClosed) {
      unawaited(_failures.close());
    }
  }
}

final class NativeLlamaEngineSession {
  NativeLlamaEngineSession._(
    this._isolate,
    this._commands,
    this._nativeLibraryPath,
    this._contextAddress,
    this._lifecycle,
  ) {
    _finalizer.attach(
      this,
      _EngineFinalizerToken(_commands, _nativeLibraryPath, _contextAddress),
      detach: _finalizerDetach,
    );
  }

  static final Finalizer<_EngineFinalizerToken> _finalizer = Finalizer(
    (token) => token.release(),
  );

  final Isolate _isolate;
  final SendPort _commands;
  final String? _nativeLibraryPath;
  final int _contextAddress;
  final _EngineWorkerLifecycle _lifecycle;
  final Object _finalizerDetach = Object();
  int _nextStreamId = 1;
  int? _activeStreamId;
  bool _closed = false;

  bool get hasActiveGeneration => _activeStreamId != null;

  Stream<
    ({
      String text,
      bool isDone,
      GenerationTelemetry? telemetry,
      ChatMessage? assistantMessage,
      int? generatedTokens,
    })
  >
  completeChatStream(
    List<ChatMessage> messages,
    GenerationConfig config, {
    required bool reusePromptPrefix,
    int? maximumPromptBytes,
  }) {
    if (_closed) {
      throw const ResourceDisposedException('LlamaEngine is closed.');
    }
    return _streamGeneration(
      (id, reply) => EngineWorkerStreamComplete(
        id,
        messages,
        config,
        reusePromptPrefix,
        maximumPromptBytes,
        reply,
      ),
    );
  }

  Stream<
    ({
      String text,
      bool isDone,
      GenerationTelemetry? telemetry,
      ChatMessage? assistantMessage,
      int? generatedTokens,
    })
  >
  completeStream(String prompt, GenerationConfig config) {
    if (_closed) {
      throw const ResourceDisposedException('LlamaEngine is closed.');
    }
    return _streamGeneration(
      (id, reply) => EngineWorkerStreamPrompt(id, prompt, config, reply),
    );
  }

  Stream<
    ({
      String text,
      bool isDone,
      GenerationTelemetry? telemetry,
      ChatMessage? assistantMessage,
      int? generatedTokens,
    })
  >
  _streamGeneration(Object Function(int id, SendPort reply) createMessage) {
    late final StreamController<
      ({
        String text,
        bool isDone,
        GenerationTelemetry? telemetry,
        ChatMessage? assistantMessage,
        int? generatedTokens,
      })
    >
    controller;
    ReceivePort? reply;
    // stopLifecycleListening cancels it on every terminal path.
    // ignore: cancel_subscriptions
    StreamSubscription<EngineWorkerFailure>? lifecycleSubscription;
    var streamId = 0;
    var paused = false;
    var waitingForWorker = false;
    var terminalResponseReceived = false;
    var slotClaimed = false;

    void releaseSlot() {
      if (slotClaimed && _activeStreamId == streamId) {
        _activeStreamId = null;
      }
      slotClaimed = false;
    }

    void stopLifecycleListening() {
      final subscription = lifecycleSubscription;
      lifecycleSubscription = null;
      if (subscription != null) {
        unawaited(subscription.cancel());
      }
    }

    void handleWorkerFailure(EngineWorkerFailure failure) {
      if (terminalResponseReceived) {
        return;
      }
      terminalResponseReceived = true;
      waitingForWorker = false;
      reply?.close();
      stopLifecycleListening();
      releaseSlot();
      if (!controller.isClosed) {
        controller.addError(failure.error.toException());
        unawaited(controller.close());
      }
    }

    void requestNext() {
      if (streamId == 0 ||
          paused ||
          waitingForWorker ||
          terminalResponseReceived ||
          controller.isClosed) {
        return;
      }
      waitingForWorker = true;
      _commands.send(EngineWorkerStreamNext(streamId));
    }

    controller =
        StreamController<
          ({
            String text,
            bool isDone,
            GenerationTelemetry? telemetry,
            ChatMessage? assistantMessage,
            int? generatedTokens,
          })
        >(
          onListen: () {
            if (_closed) {
              terminalResponseReceived = true;
              controller.addError(
                const ResourceDisposedException('LlamaEngine is closed.'),
              );
              unawaited(controller.close());
              return;
            }
            final existingFailure = _lifecycle.failure;
            if (existingFailure != null) {
              handleWorkerFailure(existingFailure);
              return;
            }
            if (_activeStreamId != null) {
              terminalResponseReceived = true;
              controller.addError(
                const GenerationException(
                  'Another generation is already active on this engine.',
                ),
              );
              unawaited(controller.close());
              return;
            }
            streamId = _nextStreamId;
            _nextStreamId += 1;
            _activeStreamId = streamId;
            slotClaimed = true;
            reply = ReceivePort();
            lifecycleSubscription = _lifecycle.failures.listen(
              handleWorkerFailure,
            );
            reply!.listen((message) {
              if (terminalResponseReceived || controller.isClosed) return;
              if (message is EngineWorkerStreamProgress) {
                // Progress belongs to the outstanding batch. It neither
                // completes that request nor permits another native batch.
                controller.add((
                  text: '',
                  isDone: false,
                  telemetry: null,
                  assistantMessage: null,
                  generatedTokens: message.generatedTokens,
                ));
                return;
              }
              waitingForWorker = false;
              if (message is EngineWorkerStreamChunk) {
                if (message.isDone) {
                  terminalResponseReceived = true;
                  releaseSlot();
                }
                if (!controller.isClosed &&
                    (message.text.isNotEmpty || message.isDone)) {
                  controller.add((
                    text: message.text,
                    isDone: message.isDone,
                    telemetry: message.telemetry,
                    assistantMessage: message.assistantMessage,
                    generatedTokens: null,
                  ));
                }
                if (message.isDone) {
                  reply?.close();
                  stopLifecycleListening();
                  if (!controller.isClosed) {
                    unawaited(controller.close());
                  }
                } else {
                  scheduleMicrotask(requestNext);
                }
              } else if (message is EngineWorkerFailure) {
                terminalResponseReceived = true;
                reply?.close();
                stopLifecycleListening();
                releaseSlot();
                if (!controller.isClosed) {
                  controller.addError(message.error.toException());
                  unawaited(controller.close());
                }
              } else {
                terminalResponseReceived = true;
                reply?.close();
                stopLifecycleListening();
                releaseSlot();
                if (!controller.isClosed) {
                  controller.addError(
                    NativeBridgeException(
                      'Unexpected engine worker response: $message',
                    ),
                  );
                  unawaited(controller.close());
                }
              }
            });
            waitingForWorker = true;
            _commands.send(createMessage(streamId, reply!.sendPort));
          },
          onPause: () {
            paused = true;
          },
          onResume: () {
            paused = false;
            requestNext();
          },
          onCancel: () async {
            if (terminalResponseReceived || !slotClaimed || streamId == 0) {
              reply?.close();
              stopLifecycleListening();
              return;
            }

            terminalResponseReceived = true;
            waitingForWorker = false;
            reply?.close();
            stopLifecycleListening();

            (Object, StackTrace)? cancellationFailure;
            try {
              requestCancel();
            } catch (error, stackTrace) {
              cancellationFailure = (error, stackTrace);
            }

            final disposeReply = ReceivePort();
            try {
              _commands.send(
                EngineWorkerStreamDispose(streamId, disposeReply.sendPort),
              );
              final message = await _lifecycle.receive(disposeReply);
              if (message is EngineWorkerFailure) {
                throw message.error.toException();
              }
              if (message != null) {
                throw NativeBridgeException(
                  'Unexpected engine worker stream-dispose response: $message',
                );
              }
            } finally {
              releaseSlot();
            }

            final failure = cancellationFailure;
            if (failure != null) {
              Error.throwWithStackTrace(failure.$1, failure.$2);
            }
          },
        );
    return controller.stream;
  }

  /// Sends one request to the worker and returns its answer.
  Future<T> _request<T>(EngineRequest<T> request) async {
    if (_closed) {
      throw const ResourceDisposedException('LlamaEngine is closed.');
    }
    final reply = ReceivePort();
    _commands.send(EngineWorkerRequest(request, reply.sendPort));
    final message = await _lifecycle.receive(reply);
    if (message is EngineWorkerFailure) {
      throw message.error.toException();
    }
    if (message is T) {
      return message;
    }
    throw NativeBridgeException('Unexpected engine worker response: $message');
  }

  Future<void> reset() => _request(const ResetRequest());

  Future<void> warmUp() => _request(const WarmUpRequest());

  Future<LlamaModelInfo> modelInfo() => _request(const ModelInfoRequest());

  Future<Map<String, String>> modelMetadata() async {
    final message = await _request(const ModelMetadataRequest());
    return Map<String, String>.unmodifiable(message);
  }

  Future<String> chatTemplate() => _request(const ChatTemplateRequest());

  Future<List<int>> tokenize(
    String text, {
    required bool addSpecial,
    required bool parseSpecial,
  }) => _request(TokenizeRequest(text, addSpecial, parseSpecial));

  Future<int> countTokens(
    String text, {
    required bool addSpecial,
    required bool parseSpecial,
  }) => _request(CountTokensRequest(text, addSpecial, parseSpecial));

  Future<String> detokenize(
    List<int> tokens, {
    required bool removeSpecial,
    required bool unparseSpecial,
  }) => _request(DetokenizeRequest(tokens, removeSpecial, unparseSpecial));

  Future<EmbeddingBatch> embedTexts(
    List<String> texts,
    EmbeddingConfig config,
  ) => _request(EmbedTextsRequest(texts, config));

  Future<LlamaChatTemplateCapabilities> chatTemplateCapabilities() =>
      _request(const ChatTemplateCapabilitiesRequest());

  Future<String> formatChat(
    List<ChatMessage> messages, {
    required bool addAssistantPrompt,
    required LlamaToolCallingConfig toolCalling,
    bool? enableThinking,
    int? reasoningBudgetTokens,
    int? maximumPromptBytes,
  }) => _request(
    FormatChatRequest(
      messages,
      addAssistantPrompt,
      toolCalling,
      enableThinking,
      reasoningBudgetTokens,
      maximumPromptBytes,
    ),
  );

  Future<int> countChatTokens(
    List<ChatMessage> messages, {
    required bool addAssistantPrompt,
    required LlamaToolCallingConfig toolCalling,
    bool? enableThinking,
    int? reasoningBudgetTokens,
    int? maximumPromptBytes,
  }) => _request(
    CountChatTokensRequest(
      messages,
      addAssistantPrompt,
      toolCalling,
      enableThinking,
      reasoningBudgetTokens,
      maximumPromptBytes,
    ),
  );

  Future<PrefillTelemetry> prefill(
    String prompt, {
    required bool? addSpecial,
    required bool parseSpecial,
  }) => _request(PrefillRequest(prompt, addSpecial, parseSpecial));

  Future<int> shiftContext({
    required int keepTokens,
    required int? discardTokens,
  }) => _request(ShiftContextRequest(keepTokens, discardTokens));

  Future<LlamaContextInfo> contextInfo() =>
      _request(const ContextInfoRequest());

  Future<Uint8List> saveState() => _request(const SaveStateRequest());

  Future<void> restoreState(Uint8List state) =>
      _request(RestoreStateRequest(state));

  Future<LoraAdapterInfo> loadLora(LoraAdapterConfig config) =>
      _request(LoadLoraRequest(config));

  Future<List<LoraAdapterInfo>> loraAdapters() async {
    final message = await _request(const ListLorasRequest());
    return List<LoraAdapterInfo>.unmodifiable(message);
  }

  Future<void> setLoraScale(int adapterId, double scale) =>
      _request(SetLoraScaleRequest(adapterId, scale));

  Future<void> unloadLora(int adapterId) =>
      _request(UnloadLoraRequest(adapterId));

  Future<void> close() async {
    if (_closed) {
      return;
    }
    _finalizer.detach(_finalizerDetach);
    (Object, StackTrace)? cancellationFailure;
    if (_lifecycle.failure == null) {
      try {
        _requestCancel();
      } catch (error, stackTrace) {
        cancellationFailure = (error, stackTrace);
      }
    }
    _closed = true;
    final reply = ReceivePort();
    _commands.send(EngineWorkerClose(reply.sendPort));
    final Object? message;
    try {
      message = await _lifecycle.receive(reply);
    } finally {
      _isolate.kill(priority: Isolate.immediate);
      await _lifecycle.dispose(expected: true);
    }
    if (message is EngineWorkerFailure) {
      throw message.error.toException();
    }
    if (message != null) {
      throw NativeBridgeException(
        'Unexpected engine worker close response: $message',
      );
    }
    final failure = cancellationFailure;
    if (failure != null) {
      Error.throwWithStackTrace(failure.$1, failure.$2);
    }
  }

  void requestCancel() {
    if (_closed || _lifecycle.failure != null) {
      return;
    }
    _requestCancel();
  }

  void _requestCancel() {
    final bridge = NativeLlamaBridge.tryOpen(_nativeLibraryPath);
    bridge?.cancelContextAddress(_contextAddress);
  }
}

final class _EngineFinalizerToken {
  const _EngineFinalizerToken(
    this.commands,
    this.nativeLibraryPath,
    this.contextAddress,
  );

  final SendPort commands;
  final String? nativeLibraryPath;
  final int contextAddress;

  void release() {
    try {
      NativeLlamaBridge.tryOpen(
        nativeLibraryPath,
      )?.cancelContextAddress(contextAddress);
    } catch (_) {
      // Finalizers cannot report cleanup failures to an owning caller.
    }
    try {
      commands.send(const EngineWorkerFinalize());
    } catch (_) {
      // The worker may already have terminated during process shutdown.
    }
  }
}

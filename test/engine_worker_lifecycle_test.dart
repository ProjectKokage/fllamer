import 'dart:isolate';

import 'package:fllamer/fllamer.dart';
import 'package:fllamer/src/engine_session.dart';
import 'package:fllamer/src/engine_worker_protocol.dart';
import 'package:test/test.dart';

void main() {
  test('a pending worker reply ends when the worker is closed', () async {
    final lifecycle = EngineWorkerLifecycle(ReceivePort());
    final pending = lifecycle.receive(ReceivePort());

    await lifecycle.dispose(expected: true);

    final message = await pending.timeout(const Duration(seconds: 2));
    expect(
      message,
      isA<EngineWorkerFailure>().having(
        (failure) => failure.error.toException(),
        'exception',
        isA<ResourceDisposedException>(),
      ),
    );
  });

  test('a reply requested after the worker closed ends at once', () async {
    final lifecycle = EngineWorkerLifecycle(ReceivePort());
    await lifecycle.dispose(expected: true);

    final message = await lifecycle
        .receive(ReceivePort())
        .timeout(const Duration(seconds: 2));
    expect(message, isA<EngineWorkerFailure>());
  });
}

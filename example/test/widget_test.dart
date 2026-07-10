import 'package:fllamer_example/main.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  testWidgets('shows the local model workflow', (tester) async {
    await tester.pumpWidget(const FllamerExampleApp());

    expect(find.textContaining('Native bridge'), findsOneWidget);
    expect(
      find.byKey(const ValueKey<String>('select-model-button')),
      findsOneWidget,
    );
    expect(
      find.byKey(const ValueKey<String>('select-projector-button')),
      findsOneWidget,
    );
    expect(
      find.byKey(const ValueKey<String>('load-model-button')),
      findsOneWidget,
    );
    expect(find.byKey(const ValueKey<String>('prompt-field')), findsOneWidget);
    expect(find.byKey(const ValueKey<String>('send-button')), findsOneWidget);
    expect(tester.takeException(), isNull);
  });

  testWidgets('fits a compact phone viewport', (tester) async {
    tester.view.physicalSize = const Size(320, 568);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);

    await tester.pumpWidget(const FllamerExampleApp());

    expect(find.text('fllamer'), findsOneWidget);
    expect(find.text('Local model'), findsOneWidget);
    expect(find.byKey(const ValueKey<String>('prompt-field')), findsOneWidget);
    expect(tester.takeException(), isNull);
  });
}

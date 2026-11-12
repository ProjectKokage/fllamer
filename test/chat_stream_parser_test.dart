import 'dart:io';

import 'package:fllamer/fllamer.dart';
import 'package:fllamer/src/native_bridge.dart';
import 'package:fllamer/src/native_chat.dart';
import 'package:fllamer/src/native_model_ops.dart';
import 'package:test/test.dart';

import 'src/native_test_support.dart';

const _models = 'third_party/llama.cpp/models';

/// What each of [chunks] adds to the reasoning and to the response text of a
/// chat read by the built bridge's own parser. The last chunk ends the
/// output. Null when the bridge or a fixture is missing.
List<(String, String)>? _streamed(
  List<String> chunks, {
  required String vocabulary,
  String? template,
  required bool enableThinking,
}) {
  final bridgePath = nativeBridgePath;
  final modelPath = '$_models/ggml-vocab-$vocabulary.gguf';
  final templatePath = template == null ? null : '$_models/templates/$template';
  for (final path in <String?>[bridgePath, modelPath, templatePath]) {
    if (path != null && !File(path).existsSync()) {
      markTestSkipped('$path is missing');
      return null;
    }
  }
  final bridge = NativeLlamaBridge.tryOpen(bridgePath)!;
  return bridge.withLoadedModel(
    LlamaModelConfig(
      modelPath: modelPath,
      nativeLibraryPath: bridgePath,
      chatTemplate: templatePath == null
          ? null
          : File(templatePath).readAsStringSync(),
    ),
    (model) {
      final plan = bridge
          .renderPreparedChat(
            model,
            <ChatMessage>[ChatMessage.user('hello')],
            addAssistantPrompt: true,
            toolCalling: const LlamaToolCallingConfig(),
            enableThinking: enableThinking,
            withOutputParser: true,
          )
          .plan!;
      final parser = plan.outputParser!;
      try {
        final output = StringBuffer();
        return <(String, String)>[
          for (final (index, chunk) in chunks.indexed)
            () {
              output.write(chunk);
              final added = parser.advance(
                output.toString(),
                isDone: index == chunks.length - 1,
              );
              return (added.reasoningText, added.text);
            }(),
        ];
      } finally {
        plan.close();
      }
    },
  );
}

void main() {
  group('chat stream read by the built bridge', () {
    test('a chunk that ends where the reasoning could end is not streamed as '
        'response text', () {
      // Ling 3 reads reasoning up to "</think>" or "<tool_call>", and
      // response text after it. "<t" could begin the second marker.
      final streamed = _streamed(
        <String>['weigh <t', 'able> more</think>', 'Hi'],
        vocabulary: 'qwen35',
        template: 'inclusionai-ling-3.0-flash.jinja',
        enableThinking: true,
      );
      if (streamed == null) {
        return;
      }
      expect(streamed, <(String, String)>[
        ('weigh ', ''),
        ('<table> more', ''),
        ('', 'Hi'),
      ]);
    });

    test('text withheld as a possible marker is delivered when the response '
        'ends', () {
      final streamed = _streamed(
        <String>["I'm fine ", '>_<'],
        vocabulary: 'gemma-4',
        enableThinking: false,
      );
      if (streamed == null) {
        return;
      }
      expect(streamed, <(String, String)>[('', "I'm fine "), ('', '>_<')]);

      // The same piece in a reasoning block that never closed is reasoning.
      expect(
        _streamed(
          <String>['<|channel>thought\nweigh<chan'],
          vocabulary: 'gemma-4',
          enableThinking: true,
        ),
        <(String, String)>[('weigh<chan', '')],
      );
    });
  });
}

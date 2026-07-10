import 'dart:io';

import 'package:fllamer/fllamer.dart';

Future<void> main(List<String> args) async {
  if (args.length < 3 || args.length > 4) {
    stderr.writeln(
      'Usage: dart run example/dart_cli/audio.dart '
      'MODEL.gguf MMPROJ.gguf AUDIO [PROMPT]',
    );
    exitCode = 64;
    return;
  }

  final engine = await LlamaEngine.load(
    LlamaModelConfig(modelPath: args[0], mmprojPath: args[1]),
  );
  try {
    final capabilities = await engine.contextInfo();
    if (!capabilities.supportsAudio) {
      throw const UnsupportedFeatureException(
        'The loaded mmproj does not support audio input.',
      );
    }
    await for (final chunk in engine.chat(
      messages: <ChatMessage>[
        ChatMessage.content(
          role: ChatRole.user,
          parts: <ChatContentPart>[
            AudioPart.fromFile(args[2]),
            TextPart(args.length == 4 ? args[3] : 'Transcribe this audio.'),
          ],
        ),
      ],
    )) {
      stdout.write(chunk.text);
    }
    stdout.writeln();
  } finally {
    await engine.close();
  }
}

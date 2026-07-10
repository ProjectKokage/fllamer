import 'dart:io';

import 'package:fllamer/fllamer.dart';

Future<void> main(List<String> args) async {
  if (args.length < 3 || args.length > 4) {
    stderr.writeln(
      'Usage: dart run example/dart_cli/multimodal.dart '
      'MODEL.gguf MMPROJ.gguf IMAGE [PROMPT]',
    );
    exitCode = 64;
    return;
  }

  final engine = await LlamaEngine.load(
    LlamaModelConfig(modelPath: args[0], mmprojPath: args[1]),
  );
  try {
    final capabilities = await engine.contextInfo();
    if (!capabilities.supportsVision) {
      throw const UnsupportedFeatureException(
        'The loaded mmproj does not support image input.',
      );
    }
    await for (final chunk in engine.chat(
      messages: <ChatMessage>[
        ChatMessage.content(
          role: ChatRole.user,
          parts: <ChatContentPart>[
            ImagePart.fromFile(args[2]),
            TextPart(args.length == 4 ? args[3] : 'Describe this image.'),
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

import 'dart:io';

import 'package:fllamer/fllamer.dart';

const _maxInt32 = 0x7FFFFFFF;
const _maxUint32 = 0xFFFFFFFF;

Future<void> main(List<String> args) async {
  final status = await runChatCli(args);
  if (status != 0) {
    exitCode = status;
  }
}

Future<int> runChatCli(
  List<String> args, {
  StringSink? output,
  StringSink? errorOutput,
}) async {
  final out = output ?? stdout;
  final errors = errorOutput ?? stderr;
  final _ChatOptions options;
  try {
    options = _ChatOptions.parse(args);
  } on FormatException catch (error) {
    errors.writeln(error.message);
    errors.writeln();
    errors.writeln(_usage);
    return 64;
  }
  if (options.help) {
    out.writeln(_usage);
    return 0;
  }
  final modelPath = options.modelPath;
  if (modelPath == null) {
    errors.writeln('Missing required --model path.\n');
    errors.writeln(_usage);
    return 64;
  }

  final engine = await LlamaEngine.load(
    LlamaModelConfig(
      modelPath: modelPath,
      nativeLibraryPath: options.nativeLibraryPath,
      contextSize: options.contextSize,
    ),
  );
  try {
    await for (final chunk in engine.chat(
      messages: <ChatMessage>[ChatMessage.user(options.prompt)],
      config: GenerationConfig(maxTokens: options.maxTokens),
    )) {
      out.write(chunk.text);
    }
    out.writeln();
  } finally {
    await engine.close();
  }
  return 0;
}

const _usage = '''
Usage: dart run example/dart_cli/chat.dart --model MODEL.gguf [options]

Options:
  --native-library PATH   Native bridge library path.
  --prompt TEXT           User prompt. Default: Write one sentence about local inference.
  --max-tokens N          Generated-token limit. Default: 128.
  --context-size N        Context size. Default: 4096.
  --help                  Show this help.
''';

final class _ChatOptions {
  const _ChatOptions({
    this.help = false,
    this.modelPath,
    this.nativeLibraryPath,
    this.prompt = 'Write one sentence about local inference.',
    this.maxTokens = 128,
    this.contextSize = 4096,
  });

  final bool help;
  final String? modelPath;
  final String? nativeLibraryPath;
  final String prompt;
  final int maxTokens;
  final int contextSize;

  static _ChatOptions parse(List<String> args) {
    var options = const _ChatOptions();
    for (var i = 0; i < args.length; i += 1) {
      final arg = args[i];
      if (arg == '--help' || arg == '-h') {
        return const _ChatOptions(help: true);
      }

      String value() {
        final equals = arg.indexOf('=');
        if (equals != -1) {
          return arg.substring(equals + 1);
        }
        if (i + 1 >= args.length) {
          throw FormatException('Missing value for $arg.');
        }
        i += 1;
        if (args[i].startsWith('--')) {
          throw FormatException('Missing value for $arg.');
        }
        return args[i];
      }

      final flag = arg.split('=').first;
      switch (flag) {
        case '--model':
          options = options._copy(modelPath: _path(value(), flag));
        case '--native-library':
          options = options._copy(nativeLibraryPath: _path(value(), flag));
        case '--prompt':
          options = options._copy(prompt: _text(value(), flag));
        case '--max-tokens':
          options = options._copy(
            maxTokens: _positiveInt(value(), flag, _maxInt32),
          );
        case '--context-size':
          options = options._copy(
            contextSize: _positiveInt(value(), flag, _maxUint32),
          );
        default:
          throw FormatException('Unknown option: $arg.');
      }
    }
    return options;
  }

  _ChatOptions _copy({
    String? modelPath,
    String? nativeLibraryPath,
    String? prompt,
    int? maxTokens,
    int? contextSize,
  }) {
    return _ChatOptions(
      modelPath: modelPath ?? this.modelPath,
      nativeLibraryPath: nativeLibraryPath ?? this.nativeLibraryPath,
      prompt: prompt ?? this.prompt,
      maxTokens: maxTokens ?? this.maxTokens,
      contextSize: contextSize ?? this.contextSize,
    );
  }
}

String _text(String value, String name) {
  if (value.trim().isEmpty) {
    throw FormatException('$name must not be empty.');
  }
  if (value.contains('\u0000')) {
    throw FormatException('$name must not contain NUL.');
  }
  return value;
}

String _path(String value, String name) {
  _text(value, name);
  if (value.contains('\n') || value.contains('\r')) {
    throw FormatException('$name must not contain line breaks.');
  }
  return value;
}

int _positiveInt(String value, String name, int max) {
  final parsed = int.tryParse(value);
  if (parsed == null || parsed <= 0 || parsed > max) {
    throw FormatException('$name must be between 1 and $max.');
  }
  return parsed;
}

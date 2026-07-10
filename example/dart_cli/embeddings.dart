import 'dart:io';

import 'package:fllamer/fllamer.dart';

Future<void> main(List<String> args) async {
  if (args.length == 1 && (args.single == '--help' || args.single == '-h')) {
    stdout.writeln(_usage);
    return;
  }
  if (args.isEmpty) {
    stderr.writeln('Missing required embedding model path.\n');
    stderr.writeln(_usage);
    exitCode = 64;
    return;
  }

  final modelPath = args.first;
  if (modelPath.trim().isEmpty ||
      modelPath.contains('\u0000') ||
      modelPath.contains('\n') ||
      modelPath.contains('\r')) {
    stderr.writeln(
      'Model path must be non-empty and contain no control bytes.',
    );
    exitCode = 64;
    return;
  }
  final texts = args.length == 1
      ? _defaultTexts
      : List<String>.unmodifiable(args.skip(1));

  final batch = await LlamaEmbeddings.embedTexts(
    LlamaModelConfig(modelPath: modelPath),
    texts,
  );
  stdout.writeln(
    '${batch.count} embeddings, ${batch.dimensions} dimensions, '
    '${batch.values.length} flat values, normalized=${batch.normalized}, '
    'pooling=${batch.pooling.name}',
  );
  for (var i = 0; i < batch.length; i += 1) {
    final preview = batch[i]
        .take(8)
        .map((value) => value.toStringAsFixed(6))
        .join(', ');
    stdout.writeln('$i: [$preview${batch.dimensions > 8 ? ', ...' : ''}]');
  }
}

const _usage = '''
Usage: dart run example/dart_cli/embeddings.dart MODEL.gguf [TEXT ...]

The default normalized embedding configuration is used. Set
FLLAMER_NATIVE_LIBRARY when the native bridge is not bundled or discoverable.
''';

const _defaultTexts = <String>[
  'Local inference keeps model data on the device.',
  'On-device models can keep private data local.',
  'The train arrives at the station before noon.',
];

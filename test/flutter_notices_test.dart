import 'dart:io';

import 'package:test/test.dart';

import '../tool/flutter_notices.dart';

void main() {
  test('pubspec lists exactly the written notices', () {
    // The list items under `flutter:` → `licenses:`.
    final lines = File('pubspec.yaml').readAsLinesSync();
    final start = lines.indexOf('  licenses:');
    expect(start, greaterThan(lines.indexOf('flutter:')));
    final listed = [
      for (final line
          in lines
              .skip(start + 1)
              .takeWhile((line) => line.startsWith('    - ')))
        line.substring('    - '.length),
    ];
    expect(listed, [
      for (final (file, _, _) in flutterNotices)
        '$flutterNoticeDirectory/$file',
    ]);
  });

  for (final (file, names, source) in flutterNotices) {
    test('$file names what it covers and matches $source', () {
      final written = File('$flutterNoticeDirectory/$file').readAsStringSync();
      expect(
        written,
        flutterNoticeText(names, File(source).readAsStringSync()),
        reason: 'Run dart --disable-dart-dev tool/flutter_notices.dart.',
      );
      // Flutter's license page reads the lines before the first blank line
      // as names and splits notices at a line of 80 hyphens.
      expect(written.substring(0, written.indexOf('\n\n')).split('\n'), names);
      expect(written, isNot(contains('\n${'-' * 80}\n')));
    });
  }
}

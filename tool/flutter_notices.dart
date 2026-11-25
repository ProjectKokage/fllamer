import 'dart:io';

/// The notices `pubspec.yaml` lists under `flutter.licenses`, written in the
/// form Flutter's license page reads: the names the notice covers, one per
/// line, a blank line, then the notice. Flutter appends these files to the
/// app's license bundle as they are, so a notice without the names would show
/// its own first lines as package names.
///
/// Each one is a copy of a pinned upstream notice. After an upstream sync,
/// run from the package root, without `dart run`, which builds the native
/// assets first:
///
///     dart --disable-dart-dev tool/flutter_notices.dart
///
/// `test/flutter_notices_test.dart` fails while a copy differs from its source.
const flutterNoticeDirectory = 'third_party/flutter_notices';

/// Each written notice: its file name, the names it covers and its source.
const flutterNotices = <(String, List<String>, String)>[
  ('llama.cpp.txt', ['llama.cpp', 'ggml'], 'third_party/llama.cpp/LICENSE'),
  ('llama.cpp-authors.txt', ['llama.cpp'], 'third_party/llama.cpp/AUTHORS'),
  (
    'nlohmann-json.txt',
    ['nlohmann/json'],
    'third_party/llama.cpp/licenses/LICENSE-jsonhpp',
  ),
  (
    'cpp-httplib.txt',
    ['cpp-httplib'],
    'third_party/llama.cpp/vendor/cpp-httplib/LICENSE',
  ),
  (
    'rotate-bits.txt',
    ['rotate-bits'],
    'third_party/llama.cpp/vendor/hash/rotate-bits/LICENSE.md',
  ),
  (
    'sha256.txt',
    ['sha256'],
    'third_party/llama.cpp/vendor/hash/sha256/LICENSE',
  ),
  (
    'xxhash.txt',
    ['xxHash'],
    'third_party/llama.cpp/vendor/hash/xxhash/LICENSE',
  ),
  ('miniaudio.txt', ['miniaudio'], 'third_party/licenses/LICENSE-miniaudio'),
  ('stb_image.txt', ['stb_image'], 'third_party/licenses/LICENSE-stb'),
  (
    'vulkan-headers.txt',
    ['Vulkan-Headers'],
    'third_party/vulkan_headers/LICENSE.md',
  ),
];

/// The text of a written notice for [names] and the [source] notice's text.
String flutterNoticeText(List<String> names, String source) =>
    '${names.join('\n')}\n\n${source.trim()}\n';

void main() {
  for (final (file, names, source) in flutterNotices) {
    File('$flutterNoticeDirectory/$file')
      ..createSync(recursive: true)
      ..writeAsStringSync(
        flutterNoticeText(names, File(source).readAsStringSync()),
      );
  }
  stdout.writeln(
    'Wrote ${flutterNotices.length} notices to $flutterNoticeDirectory.',
  );
}

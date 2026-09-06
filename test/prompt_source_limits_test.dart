import 'dart:convert';

import 'package:fllamer/src/errors.dart';
import 'package:fllamer/src/prompt_source_limits.dart';
import 'package:test/test.dart';

void main() {
  test('counts Unicode bytes and rejects malformed text before a copy', () {
    expect(promptUtf8Bytes('a日😀', 8), 8);
    expect(
      () => promptUtf8Bytes('a日😀', 7),
      throwsA(isA<PromptBufferException>()),
    );
    for (final text in ['a\u0000', '\ud800', '\udc00', '\ud800x']) {
      expect(() => promptUtf8Bytes(text, 100), throwsArgumentError);
    }
  });

  test(
    'wire JSON charges escaping and metadata before retaining encoder chunks',
    () {
      final body = {
        'messages': [
          {'role': 'user', 'content': '日😀"\n\\'},
        ],
      };
      final expected = utf8.encode(jsonEncode(body));
      expect(encodePromptJson(body, expected.length), expected);
      expect(
        () => encodePromptJson(body, expected.length - 1),
        throwsA(isA<PromptBufferException>()),
      );
      expect(encodePromptJson(body, null), expected);
    },
  );

  test('source/wire guard does not impose an old 16 or 64 KiB text cap', () {
    final body = {
      'messages': [
        {'role': 'user', 'content': 'x' * (70 * 1024)},
      ],
    };
    expect(encodePromptJson(body, 4096 * 256).length, greaterThan(64 * 1024));
    expect(
      () => encodePromptJson(body, 4096 * 16),
      throwsA(isA<PromptBufferException>()),
    );
  });

  test(
    'returned native buffer size is checked independently of prompt tokens',
    () {
      expect(
        () => checkPromptBufferSize(4097, 4096),
        throwsA(isA<PromptBufferException>()),
      );
      checkPromptBufferSize(4096, 4096);
      for (final invalid in [0, -1, 0x80000000]) {
        expect(() => validateMaximumPromptBytes(invalid), throwsArgumentError);
      }
    },
  );
}

import 'dart:convert';
import 'dart:typed_data';

import 'errors.dart';

void validateMaximumPromptBytes(int? maximumBytes) {
  if (maximumBytes != null &&
      (maximumBytes <= 0 || maximumBytes > 0x7fffffff)) {
    throw ArgumentError.value(
      maximumBytes,
      'maximumPromptBytes',
      'must fit positive int32',
    );
  }
}

void checkPromptBufferSize(int bytes, int? maximumBytes) {
  if (maximumBytes != null && bytes > maximumBytes) {
    throw const PromptBufferException(
      'The prompt exceeds its configured source buffer.',
    );
  }
}

/// Validates/counts source before allocating an encoded copy. This optional
/// policy is independent of token count and includes malformed UTF-16 refusal.
int promptUtf8Bytes(String source, int? maximumBytes) {
  var bytes = 0;
  for (var i = 0; i < source.length; i++) {
    final unit = source.codeUnitAt(i);
    late final int width;
    if (unit == 0) {
      throw ArgumentError('Prompt text contains NUL.');
    }
    if (unit < 0x80) {
      width = 1;
    } else if (unit < 0x800) {
      width = 2;
    } else if (unit >= 0xd800 && unit <= 0xdbff) {
      if (++i >= source.length) {
        throw ArgumentError('Prompt text has invalid UTF-16.');
      }
      final low = source.codeUnitAt(i);
      if (low < 0xdc00 || low > 0xdfff) {
        throw ArgumentError('Prompt text has invalid UTF-16.');
      }
      width = 4;
    } else if (unit >= 0xdc00 && unit <= 0xdfff) {
      throw ArgumentError('Prompt text has invalid UTF-16.');
    } else {
      width = 3;
    }
    bytes += width;
    checkPromptBufferSize(bytes, maximumBytes);
  }
  return bytes;
}

/// Uses the SDK's bounded encoder chunks, avoiding an unbounded intermediate
/// JSON string. Each chunk is charged before retention in the outgoing buffer.
List<int> encodePromptJson(Object? source, int? maximumBytes) {
  if (maximumBytes == null) return utf8.encode(jsonEncode(source));
  final sink = _PromptBytes(maximumBytes);
  final encoder = JsonUtf8Encoder().startChunkedConversion(sink);
  encoder.add(source);
  encoder.close();
  return sink.bytes.takeBytes();
}

final class _PromptBytes implements Sink<List<int>> {
  _PromptBytes(this.maximumBytes);
  final int maximumBytes;
  final BytesBuilder bytes = BytesBuilder(copy: false);
  @override
  void add(List<int> data) {
    checkPromptBufferSize(bytes.length + data.length, maximumBytes);
    bytes.add(data);
  }

  @override
  void close() {}
}

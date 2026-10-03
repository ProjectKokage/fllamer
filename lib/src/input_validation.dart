/// Checks that several parts of the package apply to caller input before it
/// reaches native code or a file. Each throws [ArgumentError] naming the
/// offending argument.
library;

/// Rejects blank text, NUL and line breaks. Paths and identifiers use it.
void validateSingleLineText(String value, String name) {
  if (value.trim().isEmpty) {
    throw ArgumentError.value(value, name, 'must not be empty');
  }
  validateNulFreeText(value, name);
  if (value.contains('\n') || value.contains('\r')) {
    throw ArgumentError.value(value, name, 'must not contain line breaks');
  }
}

/// Rejects NUL, which would end the text early on the native side.
void validateNulFreeText(String value, String name) {
  if (value.contains('\u0000')) {
    throw ArgumentError.value(value, name, 'must not contain NUL');
  }
}

/// Rejects token ids outside the non-negative int32 range.
void validateTokenIds(List<int> tokens) {
  for (final token in tokens) {
    if (token < 0 || token > 0x7FFFFFFF) {
      throw ArgumentError.value(
        tokens,
        'tokens',
        'must contain int32 token ids',
      );
    }
  }
}

/// Whether [value] holds NUL, literally or percent-encoded.
bool containsNulOctet(String value) {
  return value.contains('\u0000') || value.toLowerCase().contains('%00');
}

/// Whether [value] holds a line break, literally or percent-encoded.
bool containsLineBreakOctet(String value) {
  final lower = value.toLowerCase();
  return value.contains('\n') ||
      value.contains('\r') ||
      lower.contains('%0a') ||
      lower.contains('%0d');
}

/// Accepts null; otherwise rejects a blank URI and one that holds NUL or a
/// line break, literally or percent-encoded.
void validateSourceUri(Uri? value, String name) {
  final text = value?.toString();
  if (text == null) {
    return;
  }
  if (text.trim().isEmpty) {
    throw ArgumentError.value(value, name, 'must not be empty');
  }
  if (containsNulOctet(text)) {
    throw ArgumentError.value(value, name, 'must not contain NUL');
  }
  if (containsLineBreakOctet(text)) {
    throw ArgumentError.value(value, name, 'must not contain line breaks');
  }
}

/// Returns a deeply unmodifiable copy of a JSON object.
///
/// Rejects non-string keys, NUL in keys or strings, non-finite numbers,
/// values that are not JSON, and cycles. [active] holds the containers on the
/// current path.
Map<String, Object?> snapshotJsonObject(
  Map<Object?, Object?> value,
  String name,
  Set<Object> active,
) {
  if (!active.add(value)) {
    throw ArgumentError.value(value, name, 'must not contain cycles');
  }
  final result = <String, Object?>{};
  try {
    for (final entry in value.entries) {
      final key = entry.key;
      if (key is! String) {
        throw ArgumentError.value(key, name, 'keys must be strings');
      }
      if (key.contains('\u0000')) {
        throw ArgumentError.value(key, name, 'keys must not contain NUL');
      }
      result[key] = snapshotJsonValue(entry.value, '$name.$key', active);
    }
    return Map<String, Object?>.unmodifiable(result);
  } finally {
    active.remove(value);
  }
}

/// Returns a deeply unmodifiable copy of one JSON value; see
/// [snapshotJsonObject].
Object? snapshotJsonValue(Object? value, String name, Set<Object> active) {
  if (value == null || value is bool) {
    return value;
  }
  if (value is String) {
    validateNulFreeText(value, name);
    return value;
  }
  if (value is num) {
    if (!value.isFinite) {
      throw ArgumentError.value(value, name, 'must be finite');
    }
    return value;
  }
  if (value is List<Object?>) {
    if (!active.add(value)) {
      throw ArgumentError.value(value, name, 'must not contain cycles');
    }
    try {
      return List<Object?>.unmodifiable(<Object?>[
        for (var i = 0; i < value.length; i += 1)
          snapshotJsonValue(value[i], '$name[$i]', active),
      ]);
    } finally {
      active.remove(value);
    }
  }
  if (value is Map<Object?, Object?>) {
    return snapshotJsonObject(value, name, active);
  }
  throw ArgumentError.value(value, name, 'must be a JSON value');
}

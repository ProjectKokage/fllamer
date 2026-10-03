import 'dart:io';

import 'package:fllamer/fllamer.dart';
import 'package:test/test.dart';

final throwsUnsupportedFeature = throwsA(isA<UnsupportedFeatureException>());

String get nativeBridgePath {
  if (Platform.isMacOS || Platform.isIOS) {
    return 'build/native/libllama_dart_bridge.dylib';
  }
  if (Platform.isWindows) {
    return r'build\native\llama_dart_bridge.dll';
  }
  return 'build/native/libllama_dart_bridge.so';
}

String get nativeBridgeExtension {
  if (Platform.isMacOS || Platform.isIOS) {
    return '.dylib';
  }
  if (Platform.isWindows) {
    return '.dll';
  }
  return '.so';
}

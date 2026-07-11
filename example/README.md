# fllamer example

Mobile text and image chat using app-owned GGUF files. Select a model, add its
matching mmproj when needed, load it, and stream local responses. The app
supports image selection for vision projectors, generation cancellation,
conversation reset, explicit model unload, capability reporting, and final
generation telemetry.

The example requires Flutter 3.44 or newer (Dart 3.12 or newer). Its Android
project uses Flutter's AGP 9 migration configuration, and its iOS plugins are
integrated with Swift Package Manager.

The example does not bundle models or download model data. Android file
selection uses the platform document picker, so no broad storage or internet
permission is required. The Android picker returns a sanitized temporary cache
copy, which can be reclaimed. Production apps should copy selected GGUF,
mmproj, and LoRA files into durable app-private storage, verify their checksums,
and pass those filesystem paths to `fllamer`; raw `content://` URIs are not
native model paths.

```sh
flutter pub get
flutter test
flutter build apk --debug --target-platform android-arm64,android-x64
flutter build ios --no-codesign --config-only
```

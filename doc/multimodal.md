# Multimodal

`fllamer` provides experimental local image and audio chat through the pinned
upstream `mtmd` library. Media preprocessing and inference run in the engine
worker isolate.

## Usage

Load the text model with its matching app-owned mmproj file:

```dart
final engine = await LlamaEngine.load(
  LlamaModelConfig(
    modelPath: modelPath,
    mmprojPath: mmprojPath,
  ),
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
          ImagePart.fromFile(imagePath),
          const TextPart('Describe this image.'),
        ],
      ),
    ],
  )) {
    // Render chunk.text in the UI layer.
  }
} finally {
  await engine.close();
}
```

The same request can use `ImagePart.fromBytes`. Audio-capable mmproj files use
`AudioPart.fromFile` or `AudioPart.fromBytes`; check
`LlamaContextInfo.supportsAudio` first. The runnable
`example/dart_cli/multimodal.dart` and `example/dart_cli/audio.dart` examples
cover image and audio chat. The Flutter `example/` uses the platform document
picker to select an app-owned image, checks `supportsVision`, and streams the
request without adding storage or internet permissions.

## Input handling

- Encoded image and audio files or byte buffers are decoded by upstream
  `mtmd`.
- Each media input is limited to 64 MiB and each request to 64 media inputs.
- Byte constructors snapshot their input and reject empty buffers.
- Paths, MIME types, and media kinds are validated before native processing.
- Prompt markers come from the pinned `mtmd` runtime and are inserted in
  content-part order before the model chat template is applied.
- A modality unsupported by the loaded mmproj fails with
  `UnsupportedFeatureException`.

`LlamaRuntime.currentCapabilities().multimodal` reports whether the native
bridge was compiled with mtmd. `LlamaContextInfo.supportsVision` and
`supportsAudio` report the loaded mmproj's model-specific capabilities.

## Limits

Video input is not available in mobile builds. Pinned upstream video helpers
launch `ffmpeg`/`ffprobe`, which conflicts with the in-process mobile runtime
contract. `VideoPart` remains typed so a future in-process upstream decoder can
be added without changing chat message structure.

Image/audio execution remains experimental. Native and Dart tests cover ABI
validation, mmproj ownership, media ordering, worker isolation, capability
propagation, and streaming. Opt-in checksum-pinned TinyGemma3 CIFAR and official
Qwen3 ASR fixtures verify real model/mmproj loading, modality capability
reporting, identical file/byte preprocessing, deterministic image
classification and audio transcription, telemetry, and cleanup. Exact fixture
metadata and test commands are recorded in [native_builds.md](native_builds.md).

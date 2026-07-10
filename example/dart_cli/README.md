# Dart CLI Examples

Run a local chat completion with your own GGUF model:

```sh
dart run example/dart_cli/chat.dart --model /path/to/model.gguf
```

Run a model-free RAG smoke example:

```sh
dart run example/dart_cli/local_rag.dart
```

Run a real batch embedding with an app-owned embedding GGUF:

```sh
dart run example/dart_cli/embeddings.dart /path/to/embedding-model.gguf
```

The output uses one row-major `EmbeddingBatch`; pass it directly to an
`InMemoryVectorIndex` as shown in [the RAG guide](../../doc/rag.md).

Run local image or audio chat with a matching app-owned mmproj:

```sh
dart run example/dart_cli/multimodal.dart MODEL.gguf MMPROJ.gguf IMAGE
dart run example/dart_cli/audio.dart MODEL.gguf MMPROJ.gguf AUDIO
```

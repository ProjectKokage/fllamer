import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:code_assets/code_assets.dart';
import 'package:fllamer/fllamer.dart';
import 'package:fllamer/src/native_bridge.dart' as native_bridge;
import 'package:hooks/hooks.dart' as hooks;
import 'package:test/test.dart';

import '../benchmark/desktop_smoke/benchmark.dart' as desktop_benchmark;
import '../example/dart_cli/chat.dart' as chat_cli;
import '../example/dart_cli/local_rag.dart' as local_rag_cli;
import '../hook/build.dart' as build_hook;

final _throwsUnsupportedFeature = throwsA(isA<UnsupportedFeatureException>());

void main() {
  group('runtime capabilities', () {
    test('reports native features as unavailable for a missing override', () {
      final capabilities = LlamaRuntime.currentCapabilities(
        nativeLibraryPath: '/missing/fllamer/native/bridge',
      );

      expect(capabilities.nativeBridgeAvailable, isFalse);
      expect(capabilities.bridgeAbiVersion, LlamaRuntime.bridgeAbiVersion);
      expect(capabilities.modelLoading, isFalse);
      expect(capabilities.textGeneration, isFalse);
      expect(capabilities.structuredOutput, isFalse);
      expect(capabilities.embeddings, isFalse);
      expect(capabilities.reranking, isFalse);
      expect(capabilities.multimodal, isFalse);
      expect(capabilities.lora, isFalse);
      expect(capabilities.speculativeDecoding, isFalse);
      expect(capabilities.mtp, isFalse);
      expect(capabilities.metal, isFalse);
      expect(capabilities.vulkan, isFalse);
      expect(capabilities.toolCalling, isFalse);
      expect(capabilities.nativeLogging, isFalse);
      expect(capabilities.prefill, isFalse);
      expect(capabilities.rag, isTrue);
      expect(capabilities.upstreamCommit, isNull);
      expect(capabilities.nativeBuildFlags, isNull);
      expect(
        () => LlamaRuntime.currentCapabilities(nativeLibraryPath: ' '),
        throwsArgumentError,
      );
      expect(
        () =>
            LlamaRuntime.currentCapabilities(nativeLibraryPath: 'bridge\u0000'),
        throwsArgumentError,
      );
    });

    test('resolves the bundled native asset without a path override', () {
      if (Platform.isWindows) {
        markTestSkipped(
          'The native-assets hook is not configured for Windows.',
        );
      }

      final capabilities = LlamaRuntime.currentCapabilities();

      expect(capabilities.nativeBridgeAvailable, isTrue);
      expect(capabilities.bridgeAbiVersion, LlamaRuntime.bridgeAbiVersion);
      expect(capabilities.modelLoading, isTrue);
      expect(capabilities.textGeneration, isTrue);
    });

    test('can read native capabilities from a built bridge', () {
      final path = _nativeBridgePath;
      if (!File(path).existsSync()) {
        markTestSkipped('native bridge has not been built at $path');
      }

      final capabilities = LlamaRuntime.currentCapabilities(
        nativeLibraryPath: path,
      );

      expect(capabilities.nativeBridgeAvailable, isTrue);
      expect(capabilities.bridgeAbiVersion, LlamaRuntime.bridgeAbiVersion);
      expect(capabilities.modelLoading, isTrue);
      expect(capabilities.tokenization, isTrue);
      expect(capabilities.textGeneration, isTrue);
      expect(capabilities.structuredOutput, isTrue);
      expect(capabilities.embeddings, isTrue);
      expect(capabilities.reranking, isTrue);
      expect(capabilities.multimodal, isTrue);
      expect(capabilities.lora, isTrue);
      expect(capabilities.speculativeDecoding, isTrue);
      expect(capabilities.mtp, isTrue);
      expect(capabilities.metal, Platform.isMacOS || Platform.isIOS);
      expect(capabilities.toolCalling, isTrue);
      expect(capabilities.nativeLogging, isTrue);
      expect(capabilities.prefill, isTrue);
      expect(capabilities.upstreamCommit, isNotEmpty);
      expect(capabilities.nativeBuildFlags, contains('LLAMA_BUILD_COMMON=ON'));
      expect(capabilities.nativeBuildFlags, contains('LLAMA_BUILD_MTMD=ON'));
      expect(capabilities.nativeBuildFlags, contains('MTMD_VIDEO=OFF'));
      expect(capabilities.nativeBuildFlags, contains('CMAKE_BUILD_TYPE='));
      expect(
        capabilities.nativeBuildFlags,
        contains('LLAMA_DART_NO_NETWORK=ON'),
      );
      expect(capabilities.nativeBuildFlags, contains('LLAMA_LLGUIDANCE=OFF'));
      expect(
        capabilities.nativeBuildFlags,
        contains(
          Platform.isMacOS || Platform.isIOS
              ? 'GGML_METAL=ON'
              : 'GGML_METAL=OFF',
        ),
      );
      expect(
        capabilities.nativeBuildFlags,
        contains(capabilities.vulkan ? 'GGML_VULKAN=ON' : 'GGML_VULKAN=OFF'),
      );
      if (Platform.isMacOS || Platform.isIOS) {
        expect(
          capabilities.nativeBuildFlags,
          contains('GGML_METAL_EMBED_LIBRARY=ON'),
        );
      }
    });

    test('formats ordinary Gemma 4 chat through the Jinja fallback', () async {
      final bridgePath = _nativeBridgePath;
      const modelPath = 'third_party/llama.cpp/models/ggml-vocab-gemma-4.gguf';
      if (!File(bridgePath).existsSync()) {
        markTestSkipped('native bridge has not been built at $bridgePath');
      }
      if (!File(modelPath).existsSync()) {
        markTestSkipped('Gemma 4 vocab fixture is missing at $modelPath');
      }

      final config = LlamaModelConfig(
        modelPath: modelPath,
        mmprojPath: 'format-only-mmproj.gguf',
        nativeLibraryPath: bridgePath,
        gpu: const GpuConfig.cpu(),
      );
      final textMessages = <ChatMessage>[ChatMessage.user('hello-gemma4')];
      final withAssistant = await LlamaChatTemplate.format(
        config,
        textMessages,
      );
      final withoutAssistant = await LlamaChatTemplate.format(
        config,
        textMessages,
        addAssistantPrompt: false,
      );

      expect(withAssistant, contains('hello-gemma4'));
      expect(withoutAssistant, contains('hello-gemma4'));
      expect(withAssistant, isNot(withoutAssistant));
      expect(
        await LlamaChatTemplate.countTokens(config, textMessages),
        await LlamaTokenizer.countTokens(
          config,
          withAssistant,
          addSpecial: true,
          parseSpecial: true,
        ),
      );

      final mediaPrompt = await LlamaChatTemplate.format(config, <ChatMessage>[
        ChatMessage.content(
          role: ChatRole.user,
          parts: const <ChatContentPart>[
            TextPart('before-media'),
            ImagePart.fromFile('image.png'),
            TextPart('between-media'),
            AudioPart.fromFile('audio.wav'),
            TextPart('after-media'),
          ],
        ),
      ]);
      final before = mediaPrompt.indexOf('before-media');
      final firstMarker = mediaPrompt.indexOf('<__media__>');
      final between = mediaPrompt.indexOf('between-media');
      final secondMarker = mediaPrompt.indexOf('<__media__>', firstMarker + 1);
      final after = mediaPrompt.indexOf('after-media');
      expect(before, greaterThanOrEqualTo(0));
      expect(firstMarker, greaterThan(before));
      expect(between, greaterThan(firstMarker));
      expect(secondMarker, greaterThan(between));
      expect(after, greaterThan(secondMarker));
      expect(mediaPrompt, isNot(contains('forbidden_tool_name_9')));
    });

    test('captures native logs only when configured', () async {
      final bridgePath = _nativeBridgePath;
      if (!File(bridgePath).existsSync()) {
        markTestSkipped('native bridge has not been built at $bridgePath');
      }
      const modelPath = 'third_party/llama.cpp/models/ggml-vocab-gpt-2.gguf';
      if (!File(modelPath).existsSync()) {
        markTestSkipped('vocab fixture is missing at $modelPath');
      }

      expect(
        () => LlamaRuntime.configureNativeLogging(
          LlamaLogLevel.debug,
          nativeLibraryPath: ' ',
        ),
        throwsArgumentError,
      );
      expect(
        () => LlamaRuntime.configureNativeLogging(
          LlamaLogLevel.debug,
          nativeLibraryPath: 'missing-bridge.dylib',
        ),
        throwsA(isA<UnsupportedFeatureException>()),
      );
      LlamaRuntime.configureNativeLogging(
        LlamaLogLevel.debug,
        nativeLibraryPath: bridgePath,
      );
      try {
        expect(
          LlamaRuntime.drainNativeLogs(nativeLibraryPath: bridgePath),
          isEmpty,
        );
        await LlamaModel.inspect(
          LlamaModelConfig(modelPath: modelPath, nativeLibraryPath: bridgePath),
        );
        final records = LlamaRuntime.drainNativeLogs(
          nativeLibraryPath: bridgePath,
        );
        expect(records, isNotEmpty);
        expect(records, everyElement(isA<LlamaLogRecord>()));
        expect(
          records.map((record) => record.level),
          isNot(contains(LlamaLogLevel.disabled)),
        );
        expect(
          records.map((record) => record.message),
          everyElement(isNotEmpty),
        );
        expect(
          () => records.add(
            const LlamaLogRecord(level: LlamaLogLevel.info, message: 'blocked'),
          ),
          throwsUnsupportedError,
        );
        expect(
          LlamaRuntime.drainNativeLogs(nativeLibraryPath: bridgePath),
          isEmpty,
        );
        LlamaRuntime.configureNativeLogging(
          LlamaLogLevel.error,
          nativeLibraryPath: bridgePath,
        );
        await LlamaModel.inspect(
          LlamaModelConfig(modelPath: modelPath, nativeLibraryPath: bridgePath),
        );
        expect(
          LlamaRuntime.drainNativeLogs(
            nativeLibraryPath: bridgePath,
          ).map((record) => record.level),
          everyElement(LlamaLogLevel.error),
        );
      } finally {
        LlamaRuntime.configureNativeLogging(
          LlamaLogLevel.disabled,
          nativeLibraryPath: bridgePath,
        );
      }
      await LlamaModel.inspect(
        LlamaModelConfig(modelPath: modelPath, nativeLibraryPath: bridgePath),
      );
      expect(
        LlamaRuntime.drainNativeLogs(nativeLibraryPath: bridgePath),
        isEmpty,
      );
    });

    test('opens exact explicit native library paths', () async {
      final path = _nativeBridgePath;
      if (!File(path).existsSync()) {
        markTestSkipped('native bridge has not been built at $path');
      }

      final source = File(path);
      final copy = File(
        '${source.parent.path}${Platform.pathSeparator}'
        'libllama_dart_bridge_path_test_'
        '${DateTime.now().microsecondsSinceEpoch}'
        '$_nativeBridgeExtension ',
      );
      try {
        await source.copy(copy.path);

        final capabilities = LlamaRuntime.currentCapabilities(
          nativeLibraryPath: copy.path,
        );

        expect(capabilities.nativeBridgeAvailable, isTrue);
        expect(capabilities.bridgeAbiVersion, LlamaRuntime.bridgeAbiVersion);
      } finally {
        if (await copy.exists()) {
          await copy.delete();
        }
      }
    });

    test('reports non-bridge native libraries as unavailable', () {
      final capabilities = LlamaRuntime.currentCapabilities(
        nativeLibraryPath: Platform.resolvedExecutable,
      );

      expect(capabilities.nativeBridgeAvailable, isFalse);
      expect(capabilities.bridgeAbiVersion, LlamaRuntime.bridgeAbiVersion);
      expect(capabilities.modelLoading, isFalse);
      expect(capabilities.rag, isTrue);
    });

    test('reports ABI-mismatched native libraries as unavailable', () async {
      final path = await _buildAbiMismatchBridge();
      if (path == null) {
        markTestSkipped('C compiler is not available for fake bridge build');
        return;
      }

      final capabilities = LlamaRuntime.currentCapabilities(
        nativeLibraryPath: path,
      );

      expect(capabilities.nativeBridgeAvailable, isFalse);
      expect(capabilities.bridgeAbiVersion, LlamaRuntime.bridgeAbiVersion);
      expect(capabilities.modelLoading, isFalse);
      expect(capabilities.rag, isTrue);
    });

    test('reports incomplete native bridge libraries as unavailable', () async {
      final path = await _buildAbiOnlyBridge();
      if (path == null) {
        markTestSkipped('C compiler is not available for fake bridge build');
        return;
      }

      final capabilities = LlamaRuntime.currentCapabilities(
        nativeLibraryPath: path,
      );

      expect(capabilities.nativeBridgeAvailable, isFalse);
      expect(capabilities.bridgeAbiVersion, LlamaRuntime.bridgeAbiVersion);
      expect(capabilities.modelLoading, isFalse);
      expect(capabilities.rag, isTrue);
    });

    test('rejects malformed environment native library paths', () {
      expect(
        () => native_bridge.nativeLibraryPathFromEnvironment(
          const <String, String>{'FLLAMER_NATIVE_LIBRARY': 'bad\nbridge'},
        ),
        throwsArgumentError,
      );
      expect(
        native_bridge.nativeLibraryPathFromEnvironment(const <String, String>{
          'FLLAMER_NATIVE_LIBRARY': '  ',
        }),
        isNull,
      );
      expect(
        native_bridge.nativeLibraryPathFromEnvironment(const <String, String>{
          'FLLAMER_NATIVE_LIBRARY': 'bridge.dylib',
        }),
        'bridge.dylib',
      );
    });

    test('example Android manifests do not request internet permission', () {
      final manifests = Directory('example/android/app/src')
          .listSync(recursive: true)
          .whereType<File>()
          .where((file) => file.path.endsWith('AndroidManifest.xml'))
          .toList();

      expect(manifests, isNotEmpty);
      for (final manifest in manifests) {
        expect(
          manifest.readAsStringSync(),
          isNot(contains('android.permission.INTERNET')),
          reason: manifest.path,
        );
      }
    });

    test('Flutter notices include every bundled native license', () {
      final pubspec = File('pubspec.yaml').readAsStringSync();
      for (final path in const <String>[
        'third_party/llama.cpp/LICENSE',
        'third_party/llama.cpp/AUTHORS',
        'third_party/llama.cpp/licenses/LICENSE-jsonhpp',
        'third_party/llama.cpp/vendor/cpp-httplib/LICENSE',
        'third_party/licenses/LICENSE-miniaudio',
        'third_party/licenses/LICENSE-stb',
      ]) {
        expect(File(path).existsSync(), isTrue, reason: path);
        expect(pubspec, contains('- $path'));
      }
    });

    test('native-asset lookup covers every generated bridge symbol', () {
      final generated = File(
        'lib/src/ffi/generated_native_asset_bindings.dart',
      ).readAsStringSync();
      final lookup = File(
        'lib/src/ffi/native_asset_lookup.dart',
      ).readAsStringSync();
      final symbols = RegExp(
        r'get\s+(llama_dart_[a-z0-9_]+)\s*=>',
      ).allMatches(generated).map((match) => match.group(1)!).toSet();

      expect(symbols, isNotEmpty);
      for (final symbol in symbols) {
        expect(lookup, contains("'$symbol'"), reason: symbol);
      }
    });

    test('build hook emits a bundled native code asset', () async {
      if (Platform.isWindows) {
        markTestSkipped(
          'CMake native-assets hook is not configured for Windows.',
        );
      }

      await testCodeBuildHook(
        mainMethod: build_hook.main,
        targetOS: OS.current,
        check: (_, output) {
          final assets = output.assets.encodedAssets
              .map((asset) => asset.asCodeAsset)
              .toList();

          expect(assets, hasLength(1));
          expect(assets.single.id, 'package:fllamer/llama_dart_bridge');
          expect(assets.single.linkMode, isA<DynamicLoadingBundled>());
          expect(File.fromUri(assets.single.file!).existsSync(), isTrue);

          final dependencies = output.dependencies
              .map((uri) => uri.toFilePath())
              .toList();
          expect(
            dependencies,
            contains(endsWith('native/llama_dart_bridge/src/state_snapshot.h')),
          );
          expect(
            dependencies,
            contains(endsWith('third_party/llama.cpp/src/llama.cpp')),
          );
          expect(
            dependencies,
            contains(endsWith('third_party/llama.cpp/common/speculative.cpp')),
          );
          expect(
            dependencies,
            contains(
              endsWith(
                'third_party/llama.cpp/ggml/src/ggml-metal/ggml-metal.cpp',
              ),
            ),
          );
          expect(
            dependencies,
            contains(
              endsWith('third_party/llama.cpp/tools/mtmd/mtmd-helper.cpp'),
            ),
          );
          expect(
            dependencies,
            contains(
              endsWith('third_party/llama.cpp/vendor/miniaudio/miniaudio.h'),
            ),
          );
          expect(
            dependencies,
            contains(
              endsWith('third_party/llama.cpp/vendor/cpp-httplib/httplib.h'),
            ),
          );
          expect(
            dependencies,
            contains(
              endsWith('third_party/llama.cpp/vendor/nlohmann/json.hpp'),
            ),
          );
          expect(
            dependencies,
            contains(endsWith('third_party/llama.cpp/vendor/stb/stb_image.h')),
          );
          expect(
            dependencies,
            isNot(
              contains(
                endsWith('third_party/llama.cpp/models/ggml-vocab-gpt-2.gguf'),
              ),
            ),
          );
        },
      );
    }, timeout: const Timeout(Duration(minutes: 5)));

    test('build hook maps only supported Android ABIs', () {
      expect(
        build_hook.androidAbiForNativeAssetsBuild(Architecture.arm64),
        'arm64-v8a',
      );
      expect(
        build_hook.androidAbiForNativeAssetsBuild(Architecture.x64),
        'x86_64',
      );
      expect(
        () => build_hook.androidAbiForNativeAssetsBuild(Architecture.arm),
        throwsA(
          isA<hooks.BuildError>().having(
            (error) => error.message,
            'message',
            contains('--target-platform android-arm64,android-x64'),
          ),
        ),
      );
    });

    test('build hook applies the Metal-compatible iOS minimum', () {
      expect(build_hook.minimumIosVersion, 15);
      expect(build_hook.iosDeploymentTargetForNativeAssetsBuild(15), '15.0');
      expect(build_hook.iosDeploymentTargetForNativeAssetsBuild(18), '18.0');
      expect(build_hook.iosDeploymentTargetForNativeAssetsBuild(13), '15.0');
    });

    test('build hook bounds default CMake parallelism', () {
      expect(
        build_hook.cmakeBuildParallelism(
          environment: const <String, String>{},
          processorCount: 32,
        ),
        build_hook.maximumDefaultBuildJobs,
      );
      expect(
        build_hook.cmakeBuildParallelism(
          environment: const <String, String>{'FLLAMER_BUILD_JOBS': '2'},
          processorCount: 32,
        ),
        2,
      );
      expect(
        build_hook.cmakeBuildParallelism(
          environment: const <String, String>{'FLLAMER_BUILD_JOBS': 'bad'},
          processorCount: 2,
        ),
        2,
      );
    });

    test('build hook infers Android NDK from native-assets compiler', () async {
      final temp = await Directory.systemTemp.createTemp('fllamer_ndk_');
      try {
        final ndk = Directory('${temp.path}/android-sdk/ndk/30.0.14904198');
        await File(
          '${ndk.path}/build/cmake/android.toolchain.cmake',
        ).create(recursive: true);
        final compiler = File(
          '${ndk.path}/toolchains/llvm/prebuilt/darwin-x86_64/bin/clang',
        );
        await compiler.create(recursive: true);

        expect(
          build_hook.androidNdkForNativeAssetsCompiler(compiler.uri),
          ndk.path,
        );
        expect(
          build_hook.androidNdkForNativeAssetsCompiler(
            Uri.https('example.com', 'clang'),
          ),
          isNull,
        );
        final malformedNdk = Directory('${temp.path}/bad\nndk');
        await File(
          '${malformedNdk.path}/build/cmake/android.toolchain.cmake',
        ).create(recursive: true);
        final malformedCompiler = File(
          '${malformedNdk.path}/toolchains/llvm/prebuilt/darwin-x86_64/bin/clang',
        );
        await malformedCompiler.create(recursive: true);

        expect(
          build_hook.androidNdkForNativeAssetsCompiler(malformedCompiler.uri),
          isNull,
        );
        expect(build_hook.androidNdkForNativeAssetsCompiler(null), isNull);
      } finally {
        await temp.delete(recursive: true);
      }
    });

    test('build hook picks the newest Android NDK numerically', () async {
      final temp = await Directory.systemTemp.createTemp('fllamer_ndk_home_');
      try {
        final androidHome = Directory('${temp.path}/android-sdk');
        final oldNdk = Directory('${androidHome.path}/ndk/9.0.0');
        final newNdk = Directory('${androidHome.path}/ndk/30.0.14904198');
        final malformedNdk = Directory('${androidHome.path}/ndk/bad\n31.0');
        for (final ndk in <Directory>[oldNdk, newNdk, malformedNdk]) {
          await File(
            '${ndk.path}/build/cmake/android.toolchain.cmake',
          ).create(recursive: true);
        }

        expect(build_hook.newestAndroidNdkInSdk(androidHome.path), newNdk.path);
        expect(build_hook.newestAndroidNdkInSdk(''), isNull);
        expect(
          build_hook.newestAndroidNdkInSdk('${temp.path}/missing'),
          isNull,
        );
      } finally {
        await temp.delete(recursive: true);
      }
    });

    test('build hook finds generator-specific CMake output dirs', () async {
      final temp = await Directory.systemTemp.createTemp('fllamer_cmake_out_');
      try {
        await File(
          '${temp.path}/Debug-iphoneos/libfixture.dylib',
        ).create(recursive: true);
        await File(
          '${temp.path}/Release-iphoneos/libfixture.dylib',
        ).create(recursive: true);
        final output = File(
          '${temp.path}/RelWithDebInfo-iphoneos/libfixture.dylib',
        );
        await output.create(recursive: true);

        final found = await build_hook.builtLibraryForCmakeOutput(
          temp.uri,
          'libfixture.dylib',
        );

        expect(found.path, output.path);
      } finally {
        await temp.delete(recursive: true);
      }
    });

    test('loads fail with a typed unsupported-feature exception', () {
      const missingBridge = '/missing/libllama_dart_bridge.dylib';
      expect(
        LlamaEngine.load(
          const LlamaModelConfig(
            modelPath: 'model.gguf',
            nativeLibraryPath: missingBridge,
          ),
        ),
        throwsA(isA<UnsupportedFeatureException>()),
      );
      expect(
        LlamaEngine.load(
          const LlamaModelConfig(
            modelPath: 'model.gguf',
            mmprojPath: 'vision-mmproj.gguf',
            speculativeDecoding: DraftModelSpeculation(
              draftModelPath: 'draft.gguf',
            ),
          ),
        ),
        throwsA(isA<UnsupportedFeatureException>()),
      );
      expect(
        LlamaEngine.load(
          LlamaModelConfig(
            modelPath: 'model.gguf',
            nativeLibraryPath: Platform.resolvedExecutable,
          ),
        ),
        throwsA(
          isA<UnsupportedFeatureException>().having(
            (error) => error.message,
            'message',
            contains('not found or is incompatible'),
          ),
        ),
      );
      expect(
        LlamaEmbeddings.embedText(
          const LlamaModelConfig(
            modelPath: 'model.gguf',
            nativeLibraryPath: missingBridge,
          ),
          'hello',
        ),
        throwsA(isA<UnsupportedFeatureException>()),
      );
      expect(
        LlamaEmbeddings.embedTexts(
          const LlamaModelConfig(
            modelPath: 'model.gguf',
            nativeLibraryPath: missingBridge,
          ),
          <String>['hello'],
        ),
        throwsA(isA<UnsupportedFeatureException>()),
      );
      expect(
        LlamaEngine.load(
          const LlamaModelConfig(
            modelPath: 'model.gguf',
            nativeLibraryPath: missingBridge,
            speculativeDecoding: DraftModelSpeculation(
              draftModelPath: 'draft.gguf',
            ),
          ),
        ),
        throwsA(isA<UnsupportedFeatureException>()),
      );
    });

    test('engine load maps native model load failures', () async {
      final path = _nativeBridgePath;
      if (!File(path).existsSync()) {
        markTestSkipped('native bridge has not been built at $path');
      }

      await expectLater(
        LlamaEngine.load(
          LlamaModelConfig(
            modelPath: 'missing-model.gguf',
            nativeLibraryPath: path,
          ),
        ),
        throwsA(isA<ModelLoadException>()),
      );
    });

    test('streams mmproj image and audio inputs through the worker', () async {
      final path = await _buildMultimodalCaptureBridge();
      if (path == null) {
        markTestSkipped('C compiler is not available for fake bridge build');
        return;
      }

      final modelConfig = LlamaModelConfig(
        modelPath: 'model.gguf',
        mmprojPath: 'mmproj.gguf',
        nativeLibraryPath: path,
        kvCache: const KvCacheConfig(
          keyType: KvCacheType.q4Zero,
          valueType: KvCacheType.q8Zero,
          offload: false,
          flashAttention: FlashAttentionMode.enabled,
          swaFull: false,
          unified: true,
        ),
      );
      final engine = await LlamaEngine.load(modelConfig);
      try {
        await expectLater(engine.tokenize('bad\u0000'), throwsArgumentError);
        await expectLater(
          engine.detokenize(const <int>[-1]),
          throwsArgumentError,
        );
        await expectLater(
          engine.countChatTokens(<ChatMessage>[
            ChatMessage.content(
              role: ChatRole.user,
              parts: const <ChatContentPart>[ImagePart.fromFile('image.png')],
            ),
          ]),
          _throwsUnsupportedFeature,
        );
        final tokens = await engine.tokenize(
          'Hi',
          addSpecial: true,
          parseSpecial: true,
        );
        expect(tokens, <int>[256, 257, 72, 105]);
        expect(
          await engine.countTokens('Hi', addSpecial: true, parseSpecial: true),
          4,
        );
        expect(
          await engine.detokenize(
            tokens,
            removeSpecial: true,
            unparseSpecial: true,
          ),
          'Hi',
        );
        final templateCapabilities = await engine.chatTemplateCapabilities();
        expect(templateCapabilities.supportsTools, isTrue);
        expect(templateCapabilities.supportsToolCalls, isTrue);
        expect(templateCapabilities.supportsParallelToolCalls, isFalse);
        final textMessages = <ChatMessage>[ChatMessage.user('hello')];
        expect(await engine.formatChat(textMessages), 'hello');
        expect(await engine.countChatTokens(textMessages), 7);
        expect(
          await LlamaChatTemplate.countTokens(modelConfig, textMessages),
          7,
        );

        await engine.warmUp();
        await expectLater(engine.prefill(prompt: ' \t'), throwsArgumentError);
        await expectLater(
          engine.prefill(prompt: 'bad\u0000prompt'),
          throwsArgumentError,
        );
        final firstPrefill = await engine.prefill(prompt: 'prefix');
        expect(firstPrefill.promptTokens, 7);
        expect(firstPrefill.promptEvalMs, 2.5);
        expect(firstPrefill.totalMs, 3.0);
        expect(firstPrefill.promptEvalTokensPerSecond, 2800);
        final continuedPrefill = await engine.prefill(
          prompt: 'tail',
          addSpecial: false,
          parseSpecial: true,
        );
        expect(continuedPrefill.promptTokens, 4);
        final info = await engine.contextInfo();
        expect(info.supportsVision, isTrue);
        expect(info.supportsAudio, isTrue);
        expect(info.gpuBackend, GpuBackend.auto);
        expect(info.usedTokens, 11);
        expect(info.kvCacheKeyType, KvCacheType.q4Zero);
        expect(info.kvCacheValueType, KvCacheType.q8Zero);
        expect(info.flashAttention, FlashAttentionMode.enabled);
        expect(info.kvCacheOffload, isFalse);
        expect(info.swaFull, isFalse);
        expect(info.kvUnified, isTrue);

        final adapter = await engine.loadLora(
          const LoraAdapterConfig(path: 'adapter.gguf', scale: 0.25),
        );
        final messages = <ChatMessage>[
          ChatMessage.content(
            role: ChatRole.user,
            parts: <ChatContentPart>[
              const TextPart('Inspect '),
              ImagePart.fromBytes(Uint8List.fromList(<int>[1, 2, 3])),
              const TextPart(' and transcribe '),
              const AudioPart.fromFile('clip.wav'),
            ],
          ),
        ];
        final requestScales = <int, double>{adapter.id: 0.75};
        final request = engine.chat(
          messages: messages,
          config: GenerationConfig(loraScales: requestScales),
        );
        requestScales[adapter.id] = 0.5;
        final chunks = await request.toList();

        expect(chunks, hasLength(1));
        expect(chunks.single.text, 'override-ok');
        expect(chunks.single.isDone, isTrue);
        expect(chunks.single.telemetry?.promptTokens, 42);
        await Future<void>.delayed(Duration.zero);
        expect((await engine.contextInfo()).contextSize, 128);
        expect((await engine.loraAdapters()).single.scale, 0.25);

        final global = await engine.chat(messages: messages).toList();
        expect(global.single.text, 'global-ok');
        final disabled = await engine
            .chat(
              messages: messages,
              config: const GenerationConfig(loraScales: <int, double>{}),
            )
            .toList();
        expect(disabled.single.text, 'disabled-ok');
        await expectLater(
          engine
              .chat(
                messages: messages,
                config: GenerationConfig(
                  loraScales: <int, double>{adapter.id: 0.5},
                ),
              )
              .toList(),
          throwsA(isA<CancelledException>()),
        );
        final restoredAfterCancellation = await engine
            .chat(messages: messages)
            .toList();
        expect(restoredAfterCancellation.single.text, 'global-ok');
        await expectLater(
          engine
              .chat(
                messages: messages,
                config: const GenerationConfig(
                  loraScales: <int, double>{999: 1},
                ),
              )
              .toList(),
          throwsA(isA<LoraException>()),
        );
        final restored = await engine.chat(messages: messages).toList();
        expect(restored.single.text, 'global-ok');
      } finally {
        await engine.close();
      }
    });

    test(
      'forwards integrated MTP tensor-load intent only for target heads',
      () async {
        final fixture = await _buildStreamingCaptureBridge();
        if (fixture == null) {
          markTestSkipped('C compiler is not available for fake bridge build');
          return;
        }

        for (final speculation in const <SpeculativeDecodingConfig>[
          NoSpeculativeDecoding(),
          MtpSpeculation(),
          MtpSpeculation(mtpModelPath: 'sidecar.gguf'),
        ]) {
          final engine = await LlamaEngine.load(
            LlamaModelConfig(
              modelPath: fixture.markerPath,
              nativeLibraryPath: fixture.libraryPath,
              speculativeDecoding: speculation,
            ),
          );
          await engine.close();
        }
      },
    );

    test('chat forwards exact prompt-prefix reuse opt in', () async {
      final fixture = await _buildStreamingCaptureBridge();
      if (fixture == null) {
        markTestSkipped('C compiler is not available for fake bridge build');
        return;
      }

      final engine = await LlamaEngine.load(
        LlamaModelConfig(
          modelPath: fixture.markerPath,
          nativeLibraryPath: fixture.libraryPath,
        ),
      );
      try {
        final reused = await engine
            .chat(
              messages: <ChatMessage>[ChatMessage.user('First')],
              config: const GenerationConfig(maxTokens: 1, temperature: 0),
              reusePromptPrefix: true,
            )
            .toList();
        final reset = await engine
            .chat(
              messages: <ChatMessage>[ChatMessage.user('Second')],
              config: const GenerationConfig(maxTokens: 1, temperature: 0),
            )
            .toList();

        expect(reused.single.text, 'A');
        expect(reset.single.text, 'a');
        expect(await File(fixture.markerPath).readAsString(), 'Aa');
      } finally {
        await engine.close();
      }
    });

    test(
      'streaming coalesces tokens and rejects context work while paused',
      () async {
        final fixture = await _buildStreamingCaptureBridge();
        if (fixture == null) {
          markTestSkipped('C compiler is not available for fake bridge build');
          return;
        }

        final engine = await LlamaEngine.load(
          LlamaModelConfig(
            modelPath: fixture.markerPath,
            nativeLibraryPath: fixture.libraryPath,
            chatTemplate: 'custom-template',
          ),
        );
        try {
          expect(await engine.chatTemplate(), 'custom-template');
          final chunks = <GenerationChunk>[];
          final firstChunk = Completer<void>();
          final done = Completer<void>();
          late final StreamSubscription<GenerationChunk> subscription;
          subscription = engine
              .complete(
                prompt: 'go',
                config: const GenerationConfig(
                  maxTokens: 6,
                  temperature: 0,
                  streamChunkTokens: 2,
                ),
              )
              .listen(
                (chunk) {
                  chunks.add(chunk);
                  if (chunks.length == 1) {
                    subscription.pause();
                    firstChunk.complete();
                  }
                },
                onError: (Object error, StackTrace stackTrace) {
                  if (!done.isCompleted) {
                    done.completeError(error, stackTrace);
                  }
                },
                onDone: () {
                  if (!done.isCompleted) {
                    done.complete();
                  }
                },
              );

          await firstChunk.future;
          await Future<void>.delayed(const Duration(milliseconds: 50));
          expect(await File(fixture.markerPath).length(), 2);
          expect(chunks.single.text, 'ab');

          await expectLater(
            engine.contextInfo(),
            throwsA(isA<GenerationException>()),
          );

          subscription.resume();
          await done.future;
          expect(chunks.map((chunk) => chunk.text), <String>['ab', 'cd', 'ef']);
          expect(chunks.map((chunk) => chunk.isDone), <bool>[
            false,
            false,
            true,
          ]);
          expect(chunks.last.telemetry?.promptTokens, 2);
          expect(chunks.last.telemetry?.generatedTokens, 6);
          expect(chunks.last.stopReason, GenerationStopReason.maxTokens);
          expect(await File(fixture.markerPath).length(), 6);
          expect((await engine.contextInfo()).usedTokens, 6);
          await subscription.cancel();
          expect((await engine.contextInfo()).usedTokens, 6);
        } finally {
          await engine.close();
        }
      },
    );

    test('stream cancellation awaits reset and permits recovery', () async {
      final fixture = await _buildStreamingCaptureBridge();
      if (fixture == null) {
        markTestSkipped('C compiler is not available for fake bridge build');
        return;
      }

      final engine = await LlamaEngine.load(
        LlamaModelConfig(
          modelPath: fixture.markerPath,
          nativeLibraryPath: fixture.libraryPath,
        ),
      );
      try {
        final firstChunk = Completer<void>();
        late final StreamSubscription<GenerationChunk> subscription;
        subscription = engine
            .complete(
              prompt: 'go',
              config: const GenerationConfig(
                maxTokens: 6,
                temperature: 0,
                streamChunkTokens: 2,
              ),
            )
            .listen((chunk) {
              if (!firstChunk.isCompleted) {
                subscription.pause();
                firstChunk.complete();
              }
            });

        await firstChunk.future;
        var cancellationCompleted = false;
        final cancellation = subscription.cancel().then((_) {
          cancellationCompleted = true;
        });
        await Future<void>.delayed(const Duration(milliseconds: 10));
        expect(cancellationCompleted, isFalse);
        await cancellation;
        expect(cancellationCompleted, isTrue);
        final info = await engine.contextInfo();
        expect(info.usedTokens, 0);
        expect(await File(fixture.markerPath).length(), 2);

        final recovered = await engine
            .complete(
              prompt: 'again',
              config: const GenerationConfig(
                maxTokens: 2,
                temperature: 0,
                streamChunkTokens: 1,
              ),
            )
            .toList();
        expect(recovered.map((chunk) => chunk.text).join(), 'ab');
        expect(recovered.last.isDone, isTrue);
        expect((await engine.contextInfo()).usedTokens, 2);
        expect(await File(fixture.markerPath).readAsString(), 'abab');
      } finally {
        await engine.close();
      }
    });

    test('closing a paused stream rejects concurrent context work', () async {
      final fixture = await _buildStreamingCaptureBridge();
      if (fixture == null) {
        markTestSkipped('C compiler is not available for fake bridge build');
        return;
      }

      final engine = await LlamaEngine.load(
        LlamaModelConfig(
          modelPath: fixture.markerPath,
          nativeLibraryPath: fixture.libraryPath,
        ),
      );
      try {
        final firstChunk = Completer<void>();
        late final StreamSubscription<GenerationChunk> subscription;
        subscription = engine
            .complete(
              prompt: 'go',
              config: const GenerationConfig(
                maxTokens: 6,
                temperature: 0,
                streamChunkTokens: 2,
              ),
            )
            .listen((chunk) {
              if (!firstChunk.isCompleted) {
                subscription.pause();
                firstChunk.complete();
              }
            }, onError: (Object _) {});

        await firstChunk.future;
        await expectLater(
          engine.contextInfo(),
          throwsA(isA<GenerationException>()),
        );
        await engine.close();
        await subscription.cancel();
        expect(await File(fixture.markerPath).length(), 2);
      } finally {
        await engine.close();
      }
    });

    test(
      'a rejected stream cannot cancel or delay the active generation',
      () async {
        final fixture = await _buildStreamingCaptureBridge();
        if (fixture == null) {
          markTestSkipped('C compiler is not available for fake bridge build');
          return;
        }

        final engine = await LlamaEngine.load(
          LlamaModelConfig(
            modelPath: fixture.markerPath,
            nativeLibraryPath: fixture.libraryPath,
          ),
        );
        try {
          final pendingSecond = engine.complete(
            prompt: 'two',
            config: const GenerationConfig(
              maxTokens: 2,
              temperature: 0,
              streamChunkTokens: 1,
            ),
          );
          final pendingCancelled = engine.complete(
            prompt: 'cancel-rejected',
            config: const GenerationConfig(
              maxTokens: 2,
              temperature: 0,
              streamChunkTokens: 1,
            ),
          );
          final firstChunk = Completer<void>();
          final firstDone = Completer<void>();
          late final StreamSubscription<GenerationChunk> firstSubscription;
          firstSubscription = engine
              .complete(
                prompt: 'one',
                config: const GenerationConfig(
                  maxTokens: 6,
                  temperature: 0,
                  streamChunkTokens: 2,
                ),
              )
              .listen(
                (chunk) {
                  if (!firstChunk.isCompleted) {
                    firstSubscription.pause();
                    firstChunk.complete();
                  }
                },
                onError: (Object error, StackTrace stackTrace) {
                  if (!firstDone.isCompleted) {
                    firstDone.completeError(error, stackTrace);
                  }
                },
                onDone: () {
                  if (!firstDone.isCompleted) {
                    firstDone.complete();
                  }
                },
              );

          await firstChunk.future;
          await expectLater(
            pendingSecond.toList(),
            throwsA(isA<GenerationException>()),
          );

          final rejected = pendingCancelled.listen(
            null,
            onError: (Object _) {},
          );
          await rejected.cancel();
          expect(await File(fixture.markerPath).length(), 2);

          firstSubscription.resume();
          await firstDone.future;
          expect(await File(fixture.markerPath).readAsString(), 'abcdef');

          final nextChunks = await engine
              .complete(
                prompt: 'after',
                config: const GenerationConfig(
                  maxTokens: 2,
                  temperature: 0,
                  streamChunkTokens: 1,
                ),
              )
              .toList();
          expect(nextChunks.map((chunk) => chunk.text).join(), 'ab');
          expect(nextChunks.last.isDone, isTrue);
          expect(await File(fixture.markerPath).readAsString(), 'abcdefab');
          await firstSubscription.cancel();
        } finally {
          await engine.close();
        }
      },
    );

    test('concurrent close callers await the same native cleanup', () async {
      final fixture = await _buildStreamingCaptureBridge();
      if (fixture == null) {
        markTestSkipped('C compiler is not available for fake bridge build');
        return;
      }

      final engine = await LlamaEngine.load(
        LlamaModelConfig(
          modelPath: fixture.markerPath,
          nativeLibraryPath: fixture.libraryPath,
        ),
      );
      var completed = false;
      final first = engine.close();
      final second = engine.close();
      expect(identical(first, second), isTrue);
      unawaited(first.then((_) => completed = true));

      await Future<void>.delayed(const Duration(milliseconds: 10));
      expect(completed, isFalse);
      await Future.wait(<Future<void>>[first, second]);
      expect(completed, isTrue);
    });

    test('engine close reports native context free failures', () async {
      final path = await _buildContextFreeFailureBridge();
      if (path == null) {
        markTestSkipped('C compiler is not available for fake bridge build');
        return;
      }

      final engine = await LlamaEngine.load(
        LlamaModelConfig(modelPath: 'model.gguf', nativeLibraryPath: path),
      );

      await expectLater(
        engine.shiftContext(keepTokens: -1),
        throwsArgumentError,
      );
      await expectLater(
        engine.shiftContext(discardTokens: 0),
        throwsArgumentError,
      );
      await expectLater(
        engine.shiftContext(keepTokens: 0x100000000),
        throwsArgumentError,
      );
      expect(await engine.shiftContext(keepTokens: 1, discardTokens: 2), 2);
      await expectLater(
        engine.close(),
        throwsA(
          isA<NativeBridgeException>().having(
            (error) => error.message,
            'message',
            contains('context free blocked'),
          ),
        ),
      );
    });

    test('model inspection reports native model free failures', () async {
      final path = await _buildModelFreeFailureBridge();
      if (path == null) {
        markTestSkipped('C compiler is not available for fake bridge build');
        return;
      }

      await expectLater(
        LlamaModel.inspect(
          LlamaModelConfig(modelPath: 'model.gguf', nativeLibraryPath: path),
        ),
        throwsA(
          isA<NativeBridgeException>().having(
            (error) => error.message,
            'message',
            contains('model free blocked'),
          ),
        ),
      );
    });

    test(
      'model metadata rejects oversized entry counts before copying',
      () async {
        final path = await _buildOversizedMetadataBridge();
        if (path == null) {
          markTestSkipped(
            'Oversized-metadata fake bridge could not be compiled',
          );
          return;
        }

        await expectLater(
          LlamaModel.metadata(
            LlamaModelConfig(modelPath: 'model.gguf', nativeLibraryPath: path),
          ),
          throwsA(
            isA<ModelLoadException>().having(
              (error) => error.message,
              'message',
              contains('65536-entry safety limit'),
            ),
          ),
        );
      },
    );

    test('model inspection maps native load failures', () async {
      final path = _nativeBridgePath;
      if (!File(path).existsSync()) {
        markTestSkipped('native bridge has not been built at $path');
      }

      await expectLater(
        LlamaModel.inspect(
          LlamaModelConfig(
            modelPath: 'missing-model.gguf',
            nativeLibraryPath: path,
          ),
        ),
        throwsA(isA<ModelLoadException>()),
      );
    });

    test('explicit GPU backends are runtime-gated', () async {
      final path = _nativeBridgePath;
      if (!File(path).existsSync()) {
        markTestSkipped('native bridge has not been built at $path');
      }

      final capabilities = LlamaRuntime.currentCapabilities(
        nativeLibraryPath: path,
      );
      // Compiled capability does not guarantee a usable runtime device. A
      // present device reaches the missing-model check; an absent device fails
      // earlier with a typed unsupported error.
      await expectLater(
        LlamaModel.inspect(
          LlamaModelConfig(
            modelPath: 'missing-model.gguf',
            nativeLibraryPath: path,
            gpu: const GpuConfig.metal(),
          ),
        ),
        throwsA(
          capabilities.metal
              ? anyOf(
                  isA<ModelLoadException>(),
                  isA<UnsupportedFeatureException>(),
                )
              : isA<UnsupportedFeatureException>(),
        ),
      );
      await expectLater(
        LlamaModel.inspect(
          LlamaModelConfig(
            modelPath: 'missing-model.gguf',
            nativeLibraryPath: path,
            gpu: const GpuConfig.vulkan(),
          ),
        ),
        throwsA(
          capabilities.vulkan
              ? anyOf(
                  isA<ModelLoadException>(),
                  isA<UnsupportedFeatureException>(),
                )
              : isA<UnsupportedFeatureException>(),
        ),
      );
    });

    test('validates local model files before native loading', () async {
      final temp = await Directory.systemTemp.createTemp('fllamer_model_');
      try {
        final model = File('${temp.path}/model.gguf');
        await model.writeAsBytes(<int>[0x47, 0x47, 0x55, 0x46, 0, 0, 0, 0]);
        final notGguf = File('${temp.path}/not-model.bin');
        await notGguf.writeAsBytes(<int>[1, 2, 3]);
        final digest = await LlamaModel.sha256(model.path);

        final info = await LlamaModel.validateFile(
          model.path,
          expectedSizeBytes: 8,
          expectedSha256: digest.toUpperCase(),
        );

        expect(info.path, model.path);
        expect(info.sizeBytes, 8);
        expect(info.sha256, digest);
        await expectLater(
          LlamaModel.validateFile('${temp.path}/missing.gguf'),
          throwsA(isA<ModelFileException>()),
        );
        await IOOverrides.runZoned(
          () => expectLater(
            LlamaModel.validateFile(model.path),
            throwsA(
              isA<ModelFileException>().having(
                (error) => error.cause,
                'cause',
                isA<FileSystemException>(),
              ),
            ),
          ),
          fseGetType: (_, _) async {
            throw const FileSystemException('blocked');
          },
        );
        await expectLater(
          LlamaModel.validateFile(temp.path),
          throwsA(isA<ModelFileException>()),
        );
        await expectLater(
          LlamaModel.validateFile(model.path, expectedSizeBytes: 4),
          throwsA(isA<ModelFileException>()),
        );
        await expectLater(
          LlamaModel.validateFile(notGguf.path),
          throwsA(isA<ModelFileException>()),
        );
        final rawInfo = await LlamaModel.validateFile(
          notGguf.path,
          requireGgufMagic: false,
        );
        expect(rawInfo.sizeBytes, 3);
        await expectLater(
          LlamaModel.validateFile(
            model.path,
            expectedSha256: ''.padLeft(64, '0'),
          ),
          throwsA(isA<ModelFileException>()),
        );
        await expectLater(
          LlamaModel.validateFile(model.path, expectedSha256: 'bad'),
          throwsArgumentError,
        );
        await expectLater(
          LlamaModel.validateFile('', expectedSizeBytes: 3),
          throwsArgumentError,
        );
        await expectLater(
          LlamaModel.validateFile('bad\npath.gguf'),
          throwsArgumentError,
        );
        await expectLater(
          LlamaModel.validateFile(model.path, expectedSizeBytes: 0),
          throwsArgumentError,
        );
        await expectLater(
          LlamaModel.deleteFile(
            model.path,
            expectedSha256: ''.padLeft(64, '0'),
          ),
          throwsA(isA<ModelFileException>()),
        );
        expect(await model.exists(), isTrue);
        if (!Platform.isWindows) {
          final link = Link('${temp.path}/linked.gguf');
          await link.create(model.path);
          await expectLater(
            LlamaModel.deleteFile(link.path, expectedSha256: digest),
            throwsA(isA<ModelFileException>()),
          );
          expect(await link.exists(), isTrue);
          await link.delete();
          expect(await model.exists(), isTrue);
        }
        final deleteFailureDelegate = File(model.path);
        await IOOverrides.runZoned(
          () => expectLater(
            LlamaModel.deleteFile(model.path),
            throwsA(
              isA<ModelFileException>().having(
                (error) => error.cause,
                'cause',
                isA<FileSystemException>(),
              ),
            ),
          ),
          createFile: (_) => _DeleteFailsFile(deleteFailureDelegate),
          fseGetType: (_, _) async => FileSystemEntityType.file,
        );
        expect(await model.exists(), isTrue);

        final deleted = await LlamaModel.deleteFile(
          model.path,
          expectedSha256: digest,
        );

        expect(deleted.sizeBytes, 8);
        expect(deleted.sha256, digest);
        expect(await model.exists(), isFalse);
      } finally {
        await temp.delete(recursive: true);
      }
    });

    test('reads tokenizer metadata with a tiny vocab fixture', () async {
      final bridgePath = _nativeBridgePath;
      if (!File(bridgePath).existsSync()) {
        markTestSkipped('native bridge has not been built at $bridgePath');
      }
      const modelPath = 'third_party/llama.cpp/models/ggml-vocab-gpt-2.gguf';
      if (!File(modelPath).existsSync()) {
        markTestSkipped('vocab fixture is missing at $modelPath');
      }

      final info = await LlamaModel.inspect(
        LlamaModelConfig(modelPath: modelPath, nativeLibraryPath: bridgePath),
      );

      expect(info.description, 'gpt-2');
      expect(info.vocabSize, greaterThan(0));
      expect(info.trainingContextSize, greaterThan(0));
      expect(info.embeddingSize, greaterThan(0));
      expect(info.inputEmbeddingSize, greaterThan(0));
      expect(info.outputEmbeddingSize, greaterThan(0));
      expect(info.layerCount, greaterThan(0));
      expect(info.attentionHeadCount, greaterThan(0));
      expect(info.keyValueHeadCount, greaterThan(0));
      expect(info.bosToken, greaterThanOrEqualTo(-1));
      expect(info.eosToken, greaterThanOrEqualTo(-1));
      expect(info.newlineToken, greaterThanOrEqualTo(-1));
      expect(info.nextnLayerCount, greaterThanOrEqualTo(0));
      expect(info.hasMtpLayers, info.nextnLayerCount > 0);
      expect(info.fileTypeName, isNotEmpty);
      expect(info.chatTemplate, isNull);
    });

    test('reads GGUF metadata with a tiny vocab fixture', () async {
      final bridgePath = _nativeBridgePath;
      if (!File(bridgePath).existsSync()) {
        markTestSkipped('native bridge has not been built at $bridgePath');
      }
      const modelPath = 'third_party/llama.cpp/models/ggml-vocab-gpt-2.gguf';
      if (!File(modelPath).existsSync()) {
        markTestSkipped('vocab fixture is missing at $modelPath');
      }

      final config = LlamaModelConfig(
        modelPath: modelPath,
        nativeLibraryPath: bridgePath,
      );
      final metadata = await LlamaModel.metadata(config);
      final architecture = await LlamaModel.architecture(config);

      expect(metadata, isNotEmpty);
      expect(metadata.keys, everyElement(isNotEmpty));
      expect(architecture, metadata['general.architecture']);
    });

    test('embedding maps native model load failures', () async {
      final path = _nativeBridgePath;
      if (!File(path).existsSync()) {
        markTestSkipped('native bridge has not been built at $path');
      }

      await expectLater(
        LlamaEmbeddings.embedText(
          LlamaModelConfig(
            modelPath: 'missing-model.gguf',
            nativeLibraryPath: path,
          ),
          'hello',
        ),
        throwsA(isA<ModelLoadException>()),
      );
    });

    test('batch embedding maps native model load failures once', () async {
      final path = _nativeBridgePath;
      if (!File(path).existsSync()) {
        markTestSkipped('native bridge has not been built at $path');
      }

      await expectLater(
        LlamaEmbeddings.embedTexts(
          LlamaModelConfig(
            modelPath: 'missing-model.gguf',
            nativeLibraryPath: path,
          ),
          <String>['hello', 'world'],
        ),
        throwsA(isA<ModelLoadException>()),
      );
    });

    test(
      'engine load owns native context lifecycle when fixture supports it',
      () async {
        final bridgePath = _nativeBridgePath;
        if (!File(bridgePath).existsSync()) {
          markTestSkipped('native bridge has not been built at $bridgePath');
        }
        const modelPath = 'third_party/llama.cpp/models/ggml-vocab-gpt-2.gguf';
        if (!File(modelPath).existsSync()) {
          markTestSkipped('vocab fixture is missing at $modelPath');
        }

        try {
          final engine = await LlamaEngine.load(
            LlamaModelConfig(
              modelPath: modelPath,
              nativeLibraryPath: bridgePath,
              chatTemplate: 'chatml',
              contextSize: 128,
              batchSize: 16,
              ubatchSize: 8,
              threads: 1,
              batchThreads: 1,
              gpu: const GpuConfig.cpu(),
            ),
          );
          final contextInfo = await engine.contextInfo();
          final loadedModelInfo = await engine.modelInfo();
          final loadedMetadata = await engine.modelMetadata();
          expect(loadedModelInfo.chatTemplate, 'chatml');
          expect(loadedMetadata['general.architecture'], isNotEmpty);
          expect(await engine.chatTemplate(), 'chatml');
          expect(contextInfo.contextSize, greaterThan(0));
          expect(contextInfo.batchSize, greaterThan(0));
          expect(contextInfo.ubatchSize, greaterThan(0));
          expect(contextInfo.gpuBackend, GpuBackend.cpu);
          expect(contextInfo.kvCacheOffload, isFalse);
          final state = await engine.saveState();
          expect(state, isNotEmpty);
          await engine.restoreState(state);
          final temp = await Directory.systemTemp.createTemp('fllamer_state_');
          try {
            final statePath = '${temp.path}/session.bin';
            await engine.saveStateToFile(statePath);
            expect(await File(statePath).length(), state.length);
            await engine.restoreStateFromFile(statePath);
            await expectLater(
              engine.restoreStateFromFile('${temp.path}/missing.bin'),
              throwsA(isA<StateFileException>()),
            );
            final emptyStatePath = '${temp.path}/empty.bin';
            await File(emptyStatePath).writeAsBytes(const <int>[]);
            await expectLater(
              engine.restoreStateFromFile(emptyStatePath),
              throwsA(isA<StateFileException>()),
            );
            if (!Platform.isWindows) {
              final target = File('${temp.path}/target.bin');
              await target.writeAsString('original');
              final link = Link('${temp.path}/linked.bin');
              await link.create(target.path);
              await expectLater(
                engine.saveStateToFile(link.path),
                throwsA(isA<StateFileException>()),
              );
              await expectLater(
                engine.restoreStateFromFile(link.path),
                throwsA(isA<StateFileException>()),
              );
              expect(await link.exists(), isTrue);
              expect(await target.readAsString(), 'original');
            }
          } finally {
            await temp.delete(recursive: true);
          }
          await expectLater(
            engine.restoreState(Uint8List(0)),
            throwsArgumentError,
          );
          await expectLater(engine.saveStateToFile(''), throwsArgumentError);
          await expectLater(
            engine.restoreStateFromFile(''),
            throwsArgumentError,
          );
          await expectLater(
            engine.saveStateToFile('bad\nstate.bin'),
            throwsArgumentError,
          );
          await engine.reset();
          final adapters = await engine.loraAdapters();
          expect(adapters, isEmpty);
          expect(
            () => adapters.add(
              const LoraAdapterInfo(id: 1, path: 'adapter.gguf', scale: 1),
            ),
            throwsUnsupportedError,
          );
          final unopenedStream = engine.complete(prompt: 'hello');
          await engine.close();
          await expectLater(
            unopenedStream.toList(),
            throwsA(isA<ResourceDisposedException>()),
          );
          await engine.close();
          await expectLater(
            engine.reset(),
            throwsA(isA<ResourceDisposedException>()),
          );
          await expectLater(
            engine.warmUp(),
            throwsA(isA<ResourceDisposedException>()),
          );
          await expectLater(
            engine.tokenize('hello'),
            throwsA(isA<ResourceDisposedException>()),
          );
          await expectLater(
            engine.modelInfo(),
            throwsA(isA<ResourceDisposedException>()),
          );
          await expectLater(
            engine.modelMetadata(),
            throwsA(isA<ResourceDisposedException>()),
          );
          await expectLater(
            engine.chatTemplate(),
            throwsA(isA<ResourceDisposedException>()),
          );
          await expectLater(
            engine.countTokens('hello'),
            throwsA(isA<ResourceDisposedException>()),
          );
          await expectLater(
            engine.detokenize(const <int>[1]),
            throwsA(isA<ResourceDisposedException>()),
          );
          await expectLater(
            engine.chatTemplateCapabilities(),
            throwsA(isA<ResourceDisposedException>()),
          );
          await expectLater(
            engine.formatChat(<ChatMessage>[ChatMessage.user('hello')]),
            throwsA(isA<ResourceDisposedException>()),
          );
          await expectLater(
            engine.countChatTokens(<ChatMessage>[ChatMessage.user('hello')]),
            throwsA(isA<ResourceDisposedException>()),
          );
          await expectLater(
            engine.prefill(prompt: 'hello'),
            throwsA(isA<ResourceDisposedException>()),
          );
          await expectLater(
            engine.shiftContext(),
            throwsA(isA<ResourceDisposedException>()),
          );
          await expectLater(
            engine.contextInfo(),
            throwsA(isA<ResourceDisposedException>()),
          );
          await expectLater(
            engine.saveState(),
            throwsA(isA<ResourceDisposedException>()),
          );
          await expectLater(
            engine.restoreState(state),
            throwsA(isA<ResourceDisposedException>()),
          );
          await expectLater(
            engine.saveStateToFile('state.bin'),
            throwsA(isA<ResourceDisposedException>()),
          );
          await expectLater(
            engine.restoreStateFromFile('state.bin'),
            throwsA(isA<ResourceDisposedException>()),
          );
          await expectLater(
            engine.loadLora(const LoraAdapterConfig(path: 'adapter.gguf')),
            throwsA(isA<ResourceDisposedException>()),
          );
          await expectLater(
            engine.loraAdapters(),
            throwsA(isA<ResourceDisposedException>()),
          );
          await expectLater(
            engine.setLoraScale(1, 1),
            throwsA(isA<ResourceDisposedException>()),
          );
          await expectLater(
            engine.unloadLora(1),
            throwsA(isA<ResourceDisposedException>()),
          );
          await expectLater(
            engine.complete(prompt: 'hello').toList(),
            throwsA(isA<ResourceDisposedException>()),
          );
          await expectLater(
            engine.continueCompletion().toList(),
            throwsA(isA<ResourceDisposedException>()),
          );
          await expectLater(
            engine.chat(messages: [ChatMessage.user('hello')]).toList(),
            throwsA(isA<ResourceDisposedException>()),
          );
        } on ModelLoadException {
          // Vocab-only GGUF fixtures may not contain weights needed for full load.
        } on ContextCreateException {
          // Some vocab fixtures load but cannot create a usable context.
        }
      },
    );

    test('engine load accepts pinned ngram speculative strategies', () async {
      final bridgePath = _nativeBridgePath;
      if (!File(bridgePath).existsSync()) {
        markTestSkipped('native bridge has not been built at $bridgePath');
      }
      const modelPath = 'third_party/llama.cpp/models/ggml-vocab-gpt-2.gguf';
      if (!File(modelPath).existsSync()) {
        markTestSkipped('vocab fixture is missing at $modelPath');
      }

      for (final strategy in const <String>[
        'ngram-simple',
        'ngram-map-k',
        'ngram-map-k4v',
      ]) {
        try {
          final engine = await LlamaEngine.load(
            LlamaModelConfig(
              modelPath: modelPath,
              nativeLibraryPath: bridgePath,
              contextSize: 128,
              batchSize: 16,
              ubatchSize: 8,
              threads: 1,
              batchThreads: 1,
              speculativeDecoding: NGramSpeculation(
                strategy: strategy,
                ngramSize: 8,
                draftLength: 16,
              ),
            ),
          );
          await engine.close();
        } on ModelLoadException {
          // Vocab-only GGUF fixtures may not contain weights needed for load.
        } on ContextCreateException {
          // Some vocab fixtures cannot create a usable context.
        }
      }
    });

    test('weighted fixture warms prefills shifts and stops cleanly', () async {
      final modelPath = Platform.environment['LLAMA_DART_TEST_MODEL'];
      if (modelPath == null || modelPath.isEmpty) {
        markTestSkipped('LLAMA_DART_TEST_MODEL is not set');
        return;
      }
      final bridgePath = _nativeBridgePath;
      if (!File(bridgePath).existsSync()) {
        markTestSkipped('native bridge has not been built at $bridgePath');
      }
      final modelConfig = LlamaModelConfig(
        modelPath: modelPath,
        nativeLibraryPath: bridgePath,
        contextSize: 128,
        batchSize: 32,
        threads: 1,
        batchThreads: 1,
        gpu: const GpuConfig.cpu(),
        kvCache: const KvCacheConfig(
          keyType: KvCacheType.f32,
          valueType: KvCacheType.f16,
          offload: false,
          flashAttention: FlashAttentionMode.disabled,
          swaFull: false,
          unified: true,
        ),
      );
      final engine = await LlamaEngine.load(modelConfig);
      try {
        final configuredContext = await engine.contextInfo();
        expect(configuredContext.kvCacheKeyType, KvCacheType.f32);
        expect(configuredContext.kvCacheValueType, KvCacheType.f16);
        expect(configuredContext.flashAttention, FlashAttentionMode.disabled);
        expect(configuredContext.kvCacheOffload, isFalse);
        expect(configuredContext.swaFull, isFalse);
        expect(configuredContext.kvUnified, isTrue);
        await engine.warmUp();
        const prompt = 'Once upon a time';
        final loadedPromptTokens = await engine.tokenize(
          prompt,
          addSpecial: true,
        );
        expect(
          await engine.countTokens(prompt, addSpecial: true),
          loadedPromptTokens.length,
        );
        expect(
          await engine.detokenize(loadedPromptTokens, removeSpecial: true),
          await LlamaTokenizer.detokenize(
            modelConfig,
            loadedPromptTokens,
            removeSpecial: true,
          ),
        );
        const generation = GenerationConfig(
          maxTokens: 1,
          temperature: 0,
          seed: 42,
        );
        await expectLater(
          engine.continueCompletion(config: generation).toList(),
          throwsA(isA<GenerationException>()),
        );
        final baseline = await engine
            .complete(prompt: prompt, config: generation)
            .toList();
        final baselineText = baseline.map((chunk) => chunk.text).join();
        expect(baselineText, isNotEmpty);
        final promptTokens = await LlamaTokenizer.tokenize(
          modelConfig,
          prompt,
          addSpecial: true,
        );
        expect(loadedPromptTokens, promptTokens);
        final history = _snapshotTokenHistory(await engine.saveState());
        expect(history, hasLength(promptTokens.length + 1));
        final sampled = history.last;
        final beforeShift = await engine.contextInfo();
        expect(beforeShift.usedTokens, history.length);
        expect(history.length, greaterThan(2));
        if (beforeShift.supportsContextShift) {
          expect(await engine.shiftContext(keepTokens: 1, discardTokens: 1), 1);
          expect(_snapshotTokenHistory(await engine.saveState()), <int>[
            history.first,
            ...history.skip(2),
          ]);
          final continued = await engine
              .complete(
                prompt: ' Continue',
                config: const GenerationConfig(
                  maxTokens: 1,
                  temperature: 0,
                  seed: 42,
                ),
              )
              .toList();
          expect(continued.last.isDone, isTrue);
          final beforeAutoShift = _snapshotTokenHistory(
            await engine.saveState(),
          );
          final expectedAutoDiscard = (beforeAutoShift.length - 1) ~/ 2;
          expect(await engine.shiftContext(keepTokens: 1), expectedAutoDiscard);
          final afterAutoShift = _snapshotTokenHistory(
            await engine.saveState(),
          );
          expect(afterAutoShift, <int>[
            beforeAutoShift.first,
            ...beforeAutoShift.skip(1 + expectedAutoDiscard),
          ]);
          expect(
            (await engine.contextInfo()).usedTokens,
            beforeAutoShift.length - expectedAutoDiscard,
          );
        } else {
          await expectLater(
            engine.shiftContext(keepTokens: 1, discardTokens: 1),
            throwsA(isA<UnsupportedFeatureException>()),
          );
          expect(_snapshotTokenHistory(await engine.saveState()), history);
          expect((await engine.contextInfo()).usedTokens, history.length);
        }
        await expectLater(
          engine.warmUp(),
          throwsA(isA<NativeBridgeException>()),
        );

        await engine.reset();
        await engine.warmUp();
        final stopped = await engine
            .complete(
              prompt: prompt,
              config: GenerationConfig(
                maxTokens: 1,
                temperature: 0,
                seed: 42,
                stopTokens: <int>[sampled],
              ),
            )
            .toList();

        expect(stopped.map((chunk) => chunk.text).join(), isEmpty);
        expect(stopped.last.telemetry?.generatedTokens, 0);

        await engine.reset();
        const firstPrefix = 'Once upon';
        final firstPrefill = await engine.prefill(prompt: firstPrefix);
        final firstPrefixTokens = await LlamaTokenizer.tokenize(
          modelConfig,
          firstPrefix,
          addSpecial: true,
        );
        expect(firstPrefill.promptTokens, firstPrefixTokens.length);
        expect(firstPrefill.promptEvalMs, greaterThanOrEqualTo(0));
        expect(firstPrefill.totalMs, greaterThanOrEqualTo(0));
        expect(
          _snapshotTokenHistory(await engine.saveState()),
          firstPrefixTokens,
        );

        const secondPrefix = ' a time';
        final secondPrefill = await engine.prefill(prompt: secondPrefix);
        final secondPrefixTokens = await LlamaTokenizer.tokenize(
          modelConfig,
          secondPrefix,
          addSpecial: false,
        );
        expect(secondPrefill.promptTokens, secondPrefixTokens.length);
        final prefetchedHistory = <int>[
          ...firstPrefixTokens,
          ...secondPrefixTokens,
        ];
        expect(
          _snapshotTokenHistory(await engine.saveState()),
          prefetchedHistory,
        );
        expect(
          (await engine.contextInfo()).usedTokens,
          prefetchedHistory.length,
        );

        final afterPrefill = await engine
            .continueCompletion(config: generation)
            .toList();
        expect(afterPrefill.last.isDone, isTrue);
        expect(afterPrefill.last.telemetry?.promptTokens, 0);
        expect(afterPrefill.last.telemetry?.generatedTokens, 1);
        final continuedHistory = _snapshotTokenHistory(
          await engine.saveState(),
        );
        expect(
          continuedHistory.take(prefetchedHistory.length),
          prefetchedHistory,
        );
        expect(continuedHistory, hasLength(prefetchedHistory.length + 1));

        final withSuffix = await engine
            .complete(prompt: ' Continue', config: generation)
            .toList();
        expect(withSuffix.last.isDone, isTrue);
        final continuationTokens = await LlamaTokenizer.tokenize(
          modelConfig,
          ' Continue',
          addSpecial: false,
        );
        final completedHistory = _snapshotTokenHistory(
          await engine.saveState(),
        );
        expect(
          completedHistory.take(continuedHistory.length),
          continuedHistory,
        );
        expect(
          completedHistory
              .skip(continuedHistory.length)
              .take(continuationTokens.length),
          continuationTokens,
        );

        await engine.reset();
        final stepped = await engine
            .complete(
              prompt: prompt,
              config: const GenerationConfig(
                maxTokens: 2,
                temperature: 0,
                seed: 42,
                streamChunkTokens: 1,
              ),
            )
            .toList();
        expect(stepped, hasLength(2));
        expect(stepped.first.isDone, isFalse);
        expect(stepped.first.text, isNotEmpty);
        expect(stepped.last.isDone, isTrue);
        expect(stepped.last.telemetry?.generatedTokens, inInclusiveRange(1, 2));
      } finally {
        await engine.close();
      }
    });

    test(
      'weighted fixture preserves output across draftless ngram strategies',
      () async {
        final modelPath = Platform.environment['LLAMA_DART_TEST_MODEL'];
        if (modelPath == null || modelPath.isEmpty) {
          markTestSkipped('LLAMA_DART_TEST_MODEL is not set');
          return;
        }
        final bridgePath = _nativeBridgePath;
        if (!File(bridgePath).existsSync()) {
          markTestSkipped('native bridge has not been built at $bridgePath');
        }

        const prompt = 'Once upon a time. Once upon a time. Once upon a time.';
        const generation = GenerationConfig(
          maxTokens: 8,
          temperature: 0,
          seed: 42,
        );
        LlamaModelConfig modelConfig(SpeculativeDecodingConfig speculation) {
          return LlamaModelConfig(
            modelPath: modelPath,
            nativeLibraryPath: bridgePath,
            contextSize: 256,
            batchSize: 64,
            ubatchSize: 64,
            threads: 1,
            batchThreads: 1,
            gpu: const GpuConfig.cpu(),
            speculativeDecoding: speculation,
          );
        }

        final baselineEngine = await LlamaEngine.load(
          modelConfig(const NoSpeculativeDecoding()),
        );
        late final String baseline;
        late final bool rejectsNgramSpeculation;
        try {
          final modelInfo = await baselineEngine.modelInfo();
          rejectsNgramSpeculation = modelInfo.isRecurrent || modelInfo.isHybrid;
          baseline =
              (await baselineEngine
                      .complete(prompt: prompt, config: generation)
                      .toList())
                  .map((chunk) => chunk.text)
                  .join();
        } finally {
          await baselineEngine.close();
        }

        for (final strategy
            in const <({String name, SpeculativeDecodingConfig config})>[
              (
                name: 'ngram-simple',
                config: NGramSpeculation(
                  strategy: 'ngram-simple',
                  ngramSize: 3,
                  draftLength: 8,
                ),
              ),
              (
                name: 'ngram-map-k',
                config: NGramSpeculation(
                  strategy: 'ngram-map-k',
                  ngramSize: 3,
                  draftLength: 8,
                ),
              ),
              (
                name: 'ngram-map-k4v',
                config: NGramSpeculation(
                  strategy: 'ngram-map-k4v',
                  ngramSize: 3,
                  draftLength: 8,
                ),
              ),
              (
                name: 'ngram-mod',
                config: NGramModSpeculation(
                  matchLength: 3,
                  minimumDraftLength: 1,
                  maximumDraftLength: 8,
                ),
              ),
              (name: 'ngram-cache', config: NGramCacheSpeculation()),
            ]) {
          if (rejectsNgramSpeculation) {
            await expectLater(
              LlamaEngine.load(modelConfig(strategy.config)),
              throwsA(isA<UnsupportedFeatureException>()),
              reason: strategy.name,
            );
            continue;
          }
          final engine = await LlamaEngine.load(modelConfig(strategy.config));
          try {
            final chunks = await engine
                .complete(prompt: prompt, config: generation)
                .toList();
            expect(
              chunks.map((chunk) => chunk.text).join(),
              baseline,
              reason: strategy.name,
            );
            final telemetry = chunks.last.telemetry!;
            expect(
              telemetry.speculativeAcceptedTokens,
              lessThanOrEqualTo(telemetry.speculativeDraftTokens),
              reason: strategy.name,
            );
            final state = await engine.saveState();
            final usedTokens = (await engine.contextInfo()).usedTokens;
            await engine.reset();
            await engine.restoreState(state);
            expect(
              (await engine.contextInfo()).usedTokens,
              usedTokens,
              reason: strategy.name,
            );
            expect(
              await engine.shiftContext(keepTokens: 1, discardTokens: 1),
              1,
              reason: strategy.name,
            );
            final continued = await engine
                .continueCompletion(
                  config: const GenerationConfig(
                    maxTokens: 1,
                    temperature: 0,
                    seed: 42,
                  ),
                )
                .toList();
            expect(continued.last.isDone, isTrue, reason: strategy.name);
          } finally {
            await engine.close();
          }
        }
      },
    );

    test('real embedding fixture returns stable semantic vectors', () async {
      final modelPath = Platform.environment['LLAMA_DART_TEST_EMBEDDING_MODEL'];
      if (modelPath == null || modelPath.isEmpty) {
        markTestSkipped('LLAMA_DART_TEST_EMBEDDING_MODEL is not set');
        return;
      }
      final bridgePath = _nativeBridgePath;
      if (!File(bridgePath).existsSync()) {
        markTestSkipped('native bridge has not been built at $bridgePath');
      }

      const expectedSize = 20999104;
      const expectedSha256 =
          '2ec4cee28a27a9c973d5f5230930d6ef6e52694bd2bc71be26a9bef5b1d755e6';
      final fixture = await LlamaModel.validateFile(
        modelPath,
        expectedSizeBytes: expectedSize,
        expectedSha256: expectedSha256,
      );
      expect(fixture.sizeBytes, expectedSize);
      expect(fixture.sha256, expectedSha256);

      final modelConfig = LlamaModelConfig(
        modelPath: modelPath,
        nativeLibraryPath: bridgePath,
        contextSize: 384,
        batchSize: 384,
        ubatchSize: 384,
        threads: 1,
        batchThreads: 1,
        gpu: const GpuConfig.cpu(),
      );
      final info = await LlamaModel.inspect(modelConfig);
      expect(info.description, isNotEmpty);
      expect(info.vocabSize, greaterThan(0));
      final metadata = await LlamaModel.metadata(modelConfig);
      expect(metadata['general.architecture'], 'bert');

      const query = 'A child is playing with a dog in the park.';
      const related = 'A kid plays outdoors with a puppy.';
      const unrelated = 'Quarterly tax filings are due next month.';
      final single = await LlamaEmbeddings.embedText(modelConfig, query);
      final batch = await LlamaEmbeddings.embedTexts(
        modelConfig,
        const <String>[query, related, unrelated],
      );
      final persistentEngine = await LlamaEmbeddingEngine.load(modelConfig);
      late final EmbeddingBatch persistentBatch;
      try {
        expect(
          (await persistentEngine.modelMetadata())['general.architecture'],
          'bert',
        );
        expect((await persistentEngine.modelInfo()).outputEmbeddingSize, 384);
        final tokens = await persistentEngine.tokenize(query, addSpecial: true);
        expect(tokens, isNotEmpty);
        expect(
          await persistentEngine.detokenize(tokens, removeSpecial: true),
          isNotEmpty,
        );
        persistentBatch = await persistentEngine.embedTexts(const <String>[
          query,
          related,
          unrelated,
        ]);
      } finally {
        await persistentEngine.close();
        await persistentEngine.close();
      }
      final raw = await LlamaEmbeddings.embedText(
        modelConfig,
        query,
        config: const EmbeddingConfig(normalize: false),
      );

      expect(single, hasLength(384));
      expect(batch, hasLength(3));
      expect(batch.count, 3);
      expect(batch.dimensions, 384);
      expect(batch.values, hasLength(3 * 384));
      expect(batch.normalized, isTrue);
      expect(batch.pooling, EmbeddingPooling.model);
      expect(persistentBatch.count, batch.count);
      expect(persistentBatch.dimensions, batch.dimensions);
      expect(persistentBatch.normalized, isTrue);
      expect(persistentBatch.pooling, EmbeddingPooling.model);
      expect(
        _maxAbsoluteDifference(persistentBatch.first, batch.first),
        lessThan(1e-6),
      );
      expect(raw, hasLength(single.length));
      for (final vector in <Float32List>[single, ...batch]) {
        expect(vector, hasLength(single.length));
        expect(vector.every((value) => value.isFinite), isTrue);
        expect(_squaredMagnitude(vector), closeTo(1, 1e-5));
      }
      expect(raw.every((value) => value.isFinite), isTrue);
      expect(_maxAbsoluteDifference(single, batch.first), lessThan(1e-6));

      final rawMagnitude = math.sqrt(_squaredMagnitude(raw));
      expect(rawMagnitude, greaterThan(0));
      final normalizedRaw = Float32List.fromList(<double>[
        for (final value in raw) value / rawMagnitude,
      ]);
      expect(_maxAbsoluteDifference(single, normalizedRaw), lessThan(1e-6));

      final relatedScore = _dotProduct(batch[0], batch[1]);
      final unrelatedScore = _dotProduct(batch[0], batch[2]);
      expect(
        relatedScore,
        greaterThan(unrelatedScore + 0.05),
        reason:
            'expected related score $relatedScore to exceed unrelated score '
            '$unrelatedScore',
      );
    });

    test(
      'real target and draft fixtures preserve deterministic output',
      () async {
        final targetPath =
            Platform.environment['LLAMA_DART_TEST_SPECULATIVE_TARGET'];
        final draftPath =
            Platform.environment['LLAMA_DART_TEST_SPECULATIVE_DRAFT'];
        if (targetPath == null ||
            targetPath.isEmpty ||
            draftPath == null ||
            draftPath.isEmpty) {
          markTestSkipped(
            'LLAMA_DART_TEST_SPECULATIVE_TARGET and '
            'LLAMA_DART_TEST_SPECULATIVE_DRAFT are not both set',
          );
          return;
        }
        final bridgePath = _nativeBridgePath;
        if (!File(bridgePath).existsSync()) {
          markTestSkipped('native bridge has not been built at $bridgePath');
        }

        await LlamaModel.validateFile(
          targetPath,
          expectedSizeBytes: 26671328,
          expectedSha256:
              '2eda49203f2f044f3dddf29a7dd7cc861ef5a0340f518a19613d73ba6d9c06b6',
        );
        await LlamaModel.validateFile(
          draftPath,
          expectedSizeBytes: 19077344,
          expectedSha256:
              '6151b1929d7f5aa3385d9ddef3393e55587c0a55de661562322bc51dfda93a04',
        );

        const prompt = 'I believe the meaning of life is';
        const generationConfig = GenerationConfig(
          maxTokens: 16,
          temperature: 0,
          topK: 1,
          seed: 42,
          streamChunkTokens: 4,
        );
        LlamaModelConfig modelConfig({
          SpeculativeDecodingConfig speculation = const NoSpeculativeDecoding(),
        }) {
          return LlamaModelConfig(
            modelPath: targetPath,
            nativeLibraryPath: bridgePath,
            contextSize: 128,
            batchSize: 64,
            ubatchSize: 64,
            threads: 1,
            batchThreads: 1,
            gpu: const GpuConfig.cpu(),
            speculativeDecoding: speculation,
          );
        }

        final baselineEngine = await LlamaEngine.load(modelConfig());
        late final List<GenerationChunk> baseline;
        try {
          baseline = await baselineEngine
              .complete(prompt: prompt, config: generationConfig)
              .toList();
        } finally {
          await baselineEngine.close();
        }

        final speculativeEngine = await LlamaEngine.load(
          modelConfig(
            speculation: DraftModelSpeculation(
              draftModelPath: draftPath,
              draftLength: 4,
            ),
          ),
        );
        late final List<GenerationChunk> speculative;
        try {
          speculative = await speculativeEngine
              .complete(prompt: prompt, config: generationConfig)
              .toList();
        } finally {
          await speculativeEngine.close();
        }

        final baselineText = baseline.map((chunk) => chunk.text).join();
        final speculativeText = speculative.map((chunk) => chunk.text).join();
        expect(baselineText, isNotEmpty);
        expect(speculativeText, baselineText);
        final baselineTelemetry = baseline.last.telemetry;
        final speculativeTelemetry = speculative.last.telemetry;
        expect(baselineTelemetry, isNotNull);
        expect(speculativeTelemetry, isNotNull);
        expect(
          speculativeTelemetry?.generatedTokens,
          baselineTelemetry?.generatedTokens,
        );
        final draftTokens = speculativeTelemetry!.speculativeDraftTokens;
        final acceptedTokens = speculativeTelemetry.speculativeAcceptedTokens;
        expect(draftTokens, greaterThan(0));
        expect(acceptedTokens, inInclusiveRange(1, draftTokens));
        expect(
          speculativeTelemetry.speculativeDraftMs,
          greaterThanOrEqualTo(0),
        );
        expect(
          speculativeTelemetry.speculativeVerifyMs,
          greaterThanOrEqualTo(0),
        );
      },
    );

    test('real EAGLE-3 fixtures preserve deterministic output', () async {
      final targetPath = Platform.environment['LLAMA_DART_TEST_EAGLE_TARGET'];
      final draftPath = Platform.environment['LLAMA_DART_TEST_EAGLE_DRAFT'];
      if (targetPath == null ||
          targetPath.isEmpty ||
          draftPath == null ||
          draftPath.isEmpty) {
        markTestSkipped(
          'LLAMA_DART_TEST_EAGLE_TARGET and '
          'LLAMA_DART_TEST_EAGLE_DRAFT are not both set',
        );
        return;
      }
      final bridgePath = _nativeBridgePath;
      if (!File(bridgePath).existsSync()) {
        markTestSkipped('native bridge has not been built at $bridgePath');
      }

      const expectedTargetSize = 777796160;
      const expectedTargetSha256 =
          '62b3fb705434cb57fabc59d59aa7b4c6fb558fff7c7c4b2ce67456373bc30fd3';
      const expectedDraftSize = 279901088;
      const expectedDraftSha256 =
          '3785856376f55cc634e637ff4bdbf8b319e62b78680de117eaf3ed728e26b526';
      final targetFixture = await LlamaModel.validateFile(
        targetPath,
        expectedSizeBytes: expectedTargetSize,
        expectedSha256: expectedTargetSha256,
      );
      final draftFixture = await LlamaModel.validateFile(
        draftPath,
        expectedSizeBytes: expectedDraftSize,
        expectedSha256: expectedDraftSha256,
      );
      expect(targetFixture.sizeBytes, expectedTargetSize);
      expect(targetFixture.sha256, expectedTargetSha256);
      expect(draftFixture.sizeBytes, expectedDraftSize);
      expect(draftFixture.sha256, expectedDraftSha256);

      const prompt = 'The capital of France is';
      const generationConfig = GenerationConfig(
        maxTokens: 16,
        temperature: 0,
        topK: 1,
        seed: 42,
        streamChunkTokens: 4,
      );
      LlamaModelConfig modelConfig({
        SpeculativeDecodingConfig speculation = const NoSpeculativeDecoding(),
      }) {
        return LlamaModelConfig(
          modelPath: targetPath,
          nativeLibraryPath: bridgePath,
          contextSize: 256,
          batchSize: 64,
          ubatchSize: 64,
          threads: 2,
          batchThreads: 2,
          gpu: const GpuConfig.cpu(),
          speculativeDecoding: speculation,
        );
      }

      final baselineEngine = await LlamaEngine.load(modelConfig());
      late final List<GenerationChunk> baseline;
      try {
        baseline = await baselineEngine
            .complete(prompt: prompt, config: generationConfig)
            .toList();
      } finally {
        await baselineEngine.close();
      }

      final eagleEngine = await LlamaEngine.load(
        modelConfig(
          speculation: Eagle3Speculation(
            draftModelPath: draftPath,
            draftLength: 3,
          ),
        ),
      );
      late final List<GenerationChunk> eagle;
      try {
        eagle = await eagleEngine
            .complete(prompt: prompt, config: generationConfig)
            .toList();
      } finally {
        await eagleEngine.close();
      }

      final baselineText = baseline.map((chunk) => chunk.text).join();
      final eagleText = eagle.map((chunk) => chunk.text).join();
      expect(baselineText, isNotEmpty);
      expect(eagleText, baselineText);
      final baselineTelemetry = baseline.last.telemetry;
      final eagleTelemetry = eagle.last.telemetry;
      expect(baselineTelemetry, isNotNull);
      expect(eagleTelemetry, isNotNull);
      expect(
        eagleTelemetry?.generatedTokens,
        baselineTelemetry?.generatedTokens,
      );
      final draftTokens = eagleTelemetry!.speculativeDraftTokens;
      final acceptedTokens = eagleTelemetry.speculativeAcceptedTokens;
      expect(draftTokens, greaterThan(0));
      expect(acceptedTokens, inInclusiveRange(1, draftTokens));
      expect(eagleTelemetry.speculativeDraftMs, greaterThanOrEqualTo(0));
      expect(eagleTelemetry.speculativeVerifyMs, greaterThanOrEqualTo(0));
    });

    test(
      'real integrated MTP fixture preserves deterministic output',
      () async {
        final modelPath = Platform.environment['LLAMA_DART_TEST_MTP_MODEL'];
        if (modelPath == null || modelPath.isEmpty) {
          markTestSkipped('LLAMA_DART_TEST_MTP_MODEL is not set');
          return;
        }
        final bridgePath = _nativeBridgePath;
        if (!File(bridgePath).existsSync()) {
          markTestSkipped('native bridge has not been built at $bridgePath');
        }

        const expectedSize = 549698976;
        const expectedSha256 =
            'ac7c9d7a1b3e3695bb3bd50f8ceaa97f9c93e99ccc3d3d1a620301b6dd6d3d86';
        final fixture = await LlamaModel.validateFile(
          modelPath,
          expectedSizeBytes: expectedSize,
          expectedSha256: expectedSha256,
        );
        expect(fixture.sizeBytes, expectedSize);
        expect(fixture.sha256, expectedSha256);

        final inspectionConfig = LlamaModelConfig(
          modelPath: modelPath,
          nativeLibraryPath: bridgePath,
          gpu: const GpuConfig.cpu(),
        );
        final inspection = await LlamaModel.inspect(inspectionConfig);
        final metadata = await LlamaModel.metadata(inspectionConfig);
        expect(metadata['qwen35.nextn_predict_layers'], '1');
        expect(inspection.nextnLayerCount, 1);
        expect(inspection.hasMtpLayers, isTrue);

        const prompt = 'The capital of France is';
        const generationConfig = GenerationConfig(
          maxTokens: 16,
          temperature: 0,
          topK: 1,
          seed: 42,
          streamChunkTokens: 4,
        );
        LlamaModelConfig modelConfig({
          SpeculativeDecodingConfig speculation = const NoSpeculativeDecoding(),
        }) {
          return LlamaModelConfig(
            modelPath: modelPath,
            nativeLibraryPath: bridgePath,
            contextSize: 256,
            batchSize: 64,
            ubatchSize: 64,
            threads: 2,
            batchThreads: 2,
            gpu: const GpuConfig.cpu(),
            speculativeDecoding: speculation,
          );
        }

        final baselineEngine = await LlamaEngine.load(modelConfig());
        late final List<GenerationChunk> baseline;
        try {
          baseline = await baselineEngine
              .complete(prompt: prompt, config: generationConfig)
              .toList();
        } finally {
          await baselineEngine.close();
        }

        final mtpEngine = await LlamaEngine.load(
          modelConfig(speculation: const MtpSpeculation(draftLength: 3)),
        );
        late final List<GenerationChunk> mtp;
        try {
          mtp = await mtpEngine
              .complete(prompt: prompt, config: generationConfig)
              .toList();
        } finally {
          await mtpEngine.close();
        }

        final baselineText = baseline.map((chunk) => chunk.text).join();
        final mtpText = mtp.map((chunk) => chunk.text).join();
        expect(baselineText, isNotEmpty);
        expect(mtpText, baselineText);
        final baselineTelemetry = baseline.last.telemetry;
        final mtpTelemetry = mtp.last.telemetry;
        expect(baselineTelemetry, isNotNull);
        expect(mtpTelemetry, isNotNull);
        expect(
          mtpTelemetry?.generatedTokens,
          baselineTelemetry?.generatedTokens,
        );
        final draftTokens = mtpTelemetry!.speculativeDraftTokens;
        final acceptedTokens = mtpTelemetry.speculativeAcceptedTokens;
        expect(draftTokens, greaterThan(0));
        expect(acceptedTokens, inInclusiveRange(1, draftTokens));
        expect(mtpTelemetry.speculativeDraftMs, greaterThanOrEqualTo(0));
        expect(mtpTelemetry.speculativeVerifyMs, greaterThanOrEqualTo(0));
      },
    );

    test('real reranker fixture scores and reorders documents', () async {
      final modelPath = Platform.environment['LLAMA_DART_TEST_RERANKER_MODEL'];
      if (modelPath == null || modelPath.isEmpty) {
        markTestSkipped('LLAMA_DART_TEST_RERANKER_MODEL is not set');
        return;
      }
      final bridgePath = _nativeBridgePath;
      if (!File(bridgePath).existsSync()) {
        markTestSkipped('native bridge has not been built at $bridgePath');
      }

      const expectedSize = 67504480;
      const expectedSha256 =
          'ad9f450c1053a431e2e3746d1f9f7768fb9183cf0796e046983a3449dca093c2';
      final fixture = await LlamaModel.validateFile(
        modelPath,
        expectedSizeBytes: expectedSize,
        expectedSha256: expectedSha256,
      );
      expect(fixture.sizeBytes, expectedSize);
      expect(fixture.sha256, expectedSha256);

      final modelConfig = LlamaModelConfig(
        modelPath: modelPath,
        nativeLibraryPath: bridgePath,
        contextSize: 256,
        batchSize: 256,
        ubatchSize: 256,
        threads: 1,
        batchThreads: 1,
        gpu: const GpuConfig.cpu(),
      );
      const query = 'What is the capital of France?';
      const relevant = 'Paris is the capital and largest city of France.';
      const unrelated = 'The Pacific Ocean is the largest ocean on Earth.';

      final pairScore = await LlamaReranking.scorePair(
        modelConfig,
        query: query,
        document: relevant,
      );
      final batchScores = await LlamaReranking.scoreDocuments(
        modelConfig,
        query: query,
        documents: const <String>[relevant, unrelated],
      );
      expect(pairScore.isFinite, isTrue);
      expect(batchScores, hasLength(2));
      expect(batchScores.every((score) => score.isFinite), isTrue);
      expect(pairScore, closeTo(batchScores[0], 1e-6));
      expect(batchScores[0], greaterThan(batchScores[1]));

      const relevantChunk = TextChunk(
        documentId: 'france',
        id: 'france:0',
        text: relevant,
        tokenCount: 10,
        metadata: <String, Object?>{'kind': 'answer'},
      );
      const unrelatedChunk = TextChunk(
        documentId: 'ocean',
        id: 'ocean:0',
        text: unrelated,
        tokenCount: 10,
      );
      final reranked = await LlamaReranker(modelConfig)
          .rerank(query, const <VectorSearchResult>[
            VectorSearchResult(chunk: unrelatedChunk, score: 0.9),
            VectorSearchResult(chunk: relevantChunk, score: 0.1),
          ]);
      expect(reranked.map((result) => result.chunk.documentId), <String>[
        'france',
        'ocean',
      ]);
      expect(reranked[0].score, closeTo(batchScores[0], 1e-6));
      expect(reranked[1].score, closeTo(batchScores[1], 1e-6));
      expect(reranked[0].chunk.metadata['kind'], 'answer');
    });

    test('real LoRA fixture changes and restores generation', () async {
      final modelPath = Platform.environment['LLAMA_DART_TEST_LORA_MODEL'];
      final adapterPath = Platform.environment['LLAMA_DART_TEST_LORA_ADAPTER'];
      if (modelPath == null ||
          modelPath.isEmpty ||
          adapterPath == null ||
          adapterPath.isEmpty) {
        markTestSkipped(
          'LLAMA_DART_TEST_LORA_MODEL and LLAMA_DART_TEST_LORA_ADAPTER '
          'are not both set',
        );
        return;
      }
      final bridgePath = _nativeBridgePath;
      if (!File(bridgePath).existsSync()) {
        markTestSkipped('native bridge has not been built at $bridgePath');
      }

      await LlamaModel.validateFile(
        modelPath,
        expectedSizeBytes: 39390272,
        expectedSha256:
            'c7aa6863f9a4b3cdf19716e2c95622dcbd3bd06989324bf1ac8e60486ef8e881',
      );
      await LlamaModel.validateFile(
        adapterPath,
        expectedSizeBytes: 16364896,
        expectedSha256:
            'd1e0617d7e10de960639d18a4620ec8c6bb56343f45692830d3634a1a3e1fe1a',
      );

      final engine = await LlamaEngine.load(
        LlamaModelConfig(
          modelPath: modelPath,
          nativeLibraryPath: bridgePath,
          contextSize: 256,
          batchSize: 128,
          ubatchSize: 128,
          threads: 1,
          batchThreads: 1,
          gpu: const GpuConfig.cpu(),
        ),
      );
      try {
        Future<String> generate({Map<int, double>? loraScales}) async {
          await engine.reset();
          final chunks = await engine
              .complete(
                prompt: 'Look in thy glass',
                config: GenerationConfig(
                  maxTokens: 16,
                  temperature: 0,
                  topK: 1,
                  seed: 42,
                  streamChunkTokens: 4,
                  loraScales: loraScales,
                ),
              )
              .toList();
          return chunks.map((chunk) => chunk.text).join();
        }

        final baseline = await generate();
        expect(baseline, isNotEmpty);

        final adapter = await engine.loadLora(
          LoraAdapterConfig(path: adapterPath),
        );
        expect(adapter.id, greaterThan(0));
        expect(adapter.path, adapterPath);
        expect(adapter.scale, 1);
        final adapters = await engine.loraAdapters();
        expect(adapters, hasLength(1));
        expect(adapters.single.id, adapter.id);
        expect(adapters.single.path, adapter.path);
        expect(adapters.single.scale, adapter.scale);

        final enabled = await generate();
        expect(enabled, isNot(baseline));
        final requestDisabled = await generate(
          loraScales: const <int, double>{},
        );
        expect(requestDisabled, baseline);
        expect(await generate(), enabled);

        await engine.setLoraScale(adapter.id, 0);
        expect((await engine.loraAdapters()).single.scale, 0);
        expect(await generate(), baseline);

        await engine.setLoraScale(adapter.id, 1);
        expect(await generate(), enabled);

        await engine.unloadLora(adapter.id);
        expect(await engine.loraAdapters(), isEmpty);
        expect(await generate(), baseline);
      } finally {
        await engine.close();
      }
    });

    test('real multimodal fixture classifies file and byte images', () async {
      final modelPath =
          Platform.environment['LLAMA_DART_TEST_MULTIMODAL_MODEL'];
      final mmprojPath =
          Platform.environment['LLAMA_DART_TEST_MULTIMODAL_MMPROJ'];
      final imagePath =
          Platform.environment['LLAMA_DART_TEST_MULTIMODAL_IMAGE'];
      if (modelPath == null ||
          modelPath.isEmpty ||
          mmprojPath == null ||
          mmprojPath.isEmpty ||
          imagePath == null ||
          imagePath.isEmpty) {
        markTestSkipped(
          'LLAMA_DART_TEST_MULTIMODAL_MODEL, '
          'LLAMA_DART_TEST_MULTIMODAL_MMPROJ, and '
          'LLAMA_DART_TEST_MULTIMODAL_IMAGE are not all set',
        );
        return;
      }
      final bridgePath = _nativeBridgePath;
      if (!File(bridgePath).existsSync()) {
        markTestSkipped('native bridge has not been built at $bridgePath');
      }

      await LlamaModel.validateFile(
        modelPath,
        expectedSizeBytes: 47227552,
        expectedSha256:
            '7566ae7219c93ea2ecc692a931ee122d30c55261d0e2c3347acb8b939d2e9abd',
      );
      await LlamaModel.validateFile(
        mmprojPath,
        expectedSizeBytes: 1039072,
        expectedSha256:
            '93c2ba8c34574dd8f2dfda64931fc20943de2f941bfe03e6e9eca68951b80604',
      );
      await LlamaModel.validateFile(
        imagePath,
        expectedSizeBytes: 3134,
        expectedSha256:
            '2935c6e3f3d4d78284e8ab4fb89f271c057b77956f6fc3c43daf3d6374296108',
        requireGgufMagic: false,
      );

      final engine = await LlamaEngine.load(
        LlamaModelConfig(
          modelPath: modelPath,
          mmprojPath: mmprojPath,
          nativeLibraryPath: bridgePath,
          contextSize: 1024,
          batchSize: 32,
          ubatchSize: 32,
          threads: 1,
          batchThreads: 1,
          gpu: const GpuConfig.cpu(),
        ),
      );
      try {
        final info = await engine.contextInfo();
        expect(info.supportsVision, isTrue);
        expect(info.supportsAudio, isFalse);

        Future<List<GenerationChunk>> classify(ImagePart image) async {
          await engine.reset();
          return engine
              .chat(
                messages: <ChatMessage>[
                  ChatMessage.content(
                    role: ChatRole.user,
                    parts: <ChatContentPart>[
                      const TextPart('What is this:\n'),
                      image,
                    ],
                  ),
                ],
                config: const GenerationConfig(
                  maxTokens: 4,
                  temperature: 0,
                  topK: 1,
                  seed: 42,
                  streamChunkTokens: 1,
                ),
              )
              .toList();
        }

        final fromFile = await classify(
          ImagePart.fromFile(imagePath, mimeType: 'image/png'),
        );
        final fileText = fromFile.map((chunk) => chunk.text).join();
        expect(fileText.toLowerCase(), contains('cat'));
        expect(fromFile.last.isDone, isTrue);
        expect(fromFile.last.telemetry?.promptTokens, greaterThan(10));
        expect(
          fromFile.last.telemetry?.generatedTokens,
          inInclusiveRange(1, 4),
        );

        final fromBytes = await classify(
          ImagePart.fromBytes(
            await File(imagePath).readAsBytes(),
            mimeType: 'image/png',
          ),
        );
        expect(fromBytes.map((chunk) => chunk.text).join(), fileText);
        expect(fromBytes.last.telemetry?.promptTokens, greaterThan(10));
      } finally {
        await engine.close();
      }
    });

    test(
      'real Gemma 4 rejects an unsafe non-causal microbatch recoverably',
      () async {
        final modelPath = Platform.environment['LLAMA_DART_TEST_GEMMA4_MODEL'];
        final mmprojPath =
            Platform.environment['LLAMA_DART_TEST_GEMMA4_MMPROJ'];
        final imagePath = Platform.environment['LLAMA_DART_TEST_GEMMA4_IMAGE'];
        if (modelPath == null ||
            modelPath.isEmpty ||
            mmprojPath == null ||
            mmprojPath.isEmpty ||
            imagePath == null ||
            imagePath.isEmpty) {
          markTestSkipped(
            'LLAMA_DART_TEST_GEMMA4_MODEL, '
            'LLAMA_DART_TEST_GEMMA4_MMPROJ, and '
            'LLAMA_DART_TEST_GEMMA4_IMAGE are not all set',
          );
          return;
        }
        final bridgePath = _nativeBridgePath;
        if (!File(bridgePath).existsSync()) {
          markTestSkipped('native bridge has not been built at $bridgePath');
        }

        await LlamaModel.validateFile(
          modelPath,
          expectedSizeBytes: 2186184768,
          expectedSha256:
              '8279c8b153490e400831e89fc8162348911dfbe3c70d22055c70abaa9b05a0b4',
        );
        await LlamaModel.validateFile(
          mmprojPath,
          expectedSizeBytes: 986833728,
          expectedSha256:
              '38b33846f56426cd650e0e574d78de125abdfcedf35c0d7f6929f6ffe26efe02',
        );
        await LlamaModel.validateFile(
          imagePath,
          expectedSizeBytes: 124071,
          expectedSha256:
              '2dff664c0c8aaea18aff8cbe7e868845b775e90cdd7a0bac98df709b131deaa3',
          requireGgufMagic: false,
        );

        final engine = await LlamaEngine.load(
          LlamaModelConfig(
            modelPath: modelPath,
            mmprojPath: mmprojPath,
            nativeLibraryPath: bridgePath,
            contextSize: 4096,
            batchSize: 512,
            ubatchSize: 128,
            gpu: const GpuConfig.cpu(),
          ),
        );
        try {
          await expectLater(
            engine
                .chat(
                  messages: <ChatMessage>[
                    ChatMessage.content(
                      role: ChatRole.user,
                      parts: <ChatContentPart>[
                        ImagePart.fromFile(imagePath, mimeType: 'image/jpeg'),
                        const TextPart('Describe this image.'),
                      ],
                    ),
                  ],
                  config: const GenerationConfig(
                    maxTokens: 1,
                    temperature: 0,
                    topK: 1,
                    seed: 42,
                  ),
                )
                .toList(),
            throwsA(
              isA<GenerationException>().having(
                (error) => error.message,
                'message',
                contains('non-causal media chunk exceeds ubatch_size'),
              ),
            ),
          );
        } finally {
          await engine.close();
        }
      },
    );

    test('real audio fixture transcribes file and byte inputs', () async {
      final modelPath = Platform.environment['LLAMA_DART_TEST_AUDIO_MODEL'];
      final mmprojPath = Platform.environment['LLAMA_DART_TEST_AUDIO_MMPROJ'];
      final audioPath = Platform.environment['LLAMA_DART_TEST_AUDIO_FILE'];
      if (modelPath == null ||
          modelPath.isEmpty ||
          mmprojPath == null ||
          mmprojPath.isEmpty ||
          audioPath == null ||
          audioPath.isEmpty) {
        markTestSkipped(
          'LLAMA_DART_TEST_AUDIO_MODEL, LLAMA_DART_TEST_AUDIO_MMPROJ, '
          'and LLAMA_DART_TEST_AUDIO_FILE are not all set',
        );
        return;
      }
      final bridgePath = _nativeBridgePath;
      if (!File(bridgePath).existsSync()) {
        markTestSkipped('native bridge has not been built at $bridgePath');
      }

      const expectedModelSize = 804749248;
      const expectedModelSha256 =
          'bca259818b50ca7c4c05e9bdb35a5dc04fa039653a6d6f3f0f331f96f6aa1971';
      const expectedMmprojSize = 214392480;
      const expectedMmprojSha256 =
          '41a342b5e4c514e968cb756de6cd1b7be39eff43c44c57a2ef5fc6522e36603d';
      const expectedAudioSize = 140060;
      const expectedAudioSha256 =
          'cdeac0ded280e18b99afbb7fac86130e4dda4b7d0b252cc22aa3d20679270a1e';
      final modelFixture = await LlamaModel.validateFile(
        modelPath,
        expectedSizeBytes: expectedModelSize,
        expectedSha256: expectedModelSha256,
      );
      final mmprojFixture = await LlamaModel.validateFile(
        mmprojPath,
        expectedSizeBytes: expectedMmprojSize,
        expectedSha256: expectedMmprojSha256,
      );
      expect(modelFixture.sizeBytes, expectedModelSize);
      expect(mmprojFixture.sizeBytes, expectedMmprojSize);
      expect(await File(audioPath).length(), expectedAudioSize);
      expect(await LlamaModel.sha256(audioPath), expectedAudioSha256);

      final engine = await LlamaEngine.load(
        LlamaModelConfig(
          modelPath: modelPath,
          mmprojPath: mmprojPath,
          nativeLibraryPath: bridgePath,
          contextSize: 4096,
          batchSize: 512,
          ubatchSize: 128,
          threads: 2,
          batchThreads: 2,
          gpu: const GpuConfig.cpu(),
        ),
      );
      try {
        final context = await engine.contextInfo();
        expect(context.supportsAudio, isTrue);
        expect(context.supportsVision, isFalse);

        Future<List<GenerationChunk>> transcribe(AudioPart audio) {
          return engine
              .chat(
                messages: <ChatMessage>[
                  ChatMessage.content(
                    role: ChatRole.user,
                    parts: <ChatContentPart>[
                      audio,
                      const TextPart('Transcribe this audio.'),
                    ],
                  ),
                ],
                config: const GenerationConfig(
                  maxTokens: 128,
                  temperature: 0,
                  topK: 1,
                  seed: 42,
                  streamChunkTokens: 8,
                ),
              )
              .toList();
        }

        final fromFile = await transcribe(
          AudioPart.fromFile(audioPath, mimeType: 'audio/mpeg'),
        );
        final fileText = fromFile.map((chunk) => chunk.text).join();
        final normalized = fileText.toLowerCase();
        expect(
          normalized,
          anyOf(contains('new york'), allOf(contains('men'), contains('walk'))),
        );
        expect(fromFile.last.isDone, isTrue);
        expect(fromFile.last.telemetry?.promptTokens, greaterThan(10));
        expect(
          fromFile.last.telemetry?.generatedTokens,
          inInclusiveRange(1, 128),
        );

        final fromBytes = await transcribe(
          AudioPart.fromBytes(
            await File(audioPath).readAsBytes(),
            mimeType: 'audio/mpeg',
          ),
        );
        expect(fromBytes.map((chunk) => chunk.text).join(), fileText);
        expect(fromBytes.last.telemetry?.promptTokens, greaterThan(10));
      } finally {
        await engine.close();
      }
    });

    test('real tool fixture generates parses and consumes a call', () async {
      final modelPath = Platform.environment['LLAMA_DART_TEST_TOOL_MODEL'];
      if (modelPath == null || modelPath.isEmpty) {
        markTestSkipped('LLAMA_DART_TEST_TOOL_MODEL is not set');
        return;
      }
      final bridgePath = _nativeBridgePath;
      if (!File(bridgePath).existsSync()) {
        markTestSkipped('native bridge has not been built at $bridgePath');
      }

      const expectedSize = 491400032;
      const expectedSha256 =
          '74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db';
      final fixture = await LlamaModel.validateFile(
        modelPath,
        expectedSizeBytes: expectedSize,
        expectedSha256: expectedSha256,
      );
      expect(fixture.sizeBytes, expectedSize);
      expect(fixture.sha256, expectedSha256);

      final modelConfig = LlamaModelConfig(
        modelPath: modelPath,
        nativeLibraryPath: bridgePath,
        contextSize: 2048,
        batchSize: 512,
        ubatchSize: 128,
        threads: 2,
        batchThreads: 2,
        gpu: const GpuConfig.cpu(),
      );
      const weatherTool = LlamaToolDefinition(
        name: 'get_current_weather',
        description: 'Get the current weather in a city.',
        parametersSchema: <String, Object?>{
          'type': 'object',
          'properties': <String, Object?>{
            'location': <String, Object?>{'type': 'string'},
          },
          'required': <Object?>['location'],
          'additionalProperties': false,
        },
      );
      const toolCalling = LlamaToolCallingConfig(
        tools: <LlamaToolDefinition>[weatherTool],
        allowParallelToolCalls: false,
        toolChoice: LlamaToolChoice.named('get_current_weather'),
      );

      final capabilities = await LlamaChatTemplate.capabilities(modelConfig);
      expect(capabilities.supportsTools, isTrue);
      expect(capabilities.supportsToolCalls, isTrue);

      final engine = await LlamaEngine.load(modelConfig);
      try {
        final user = ChatMessage.user(
          'What is the weather in Paris, France? Use the weather tool.',
        );
        final chunks = await engine
            .chat(
              messages: <ChatMessage>[user],
              config: const GenerationConfig(
                maxTokens: 128,
                temperature: 0,
                topK: 1,
                seed: 42,
                streamChunkTokens: 8,
                toolCalling: toolCalling,
              ),
            )
            .toList();
        expect(chunks.last.isDone, isTrue);
        expect(chunks.last.telemetry?.generatedTokens, greaterThan(0));
        final assistant = chunks.last.assistantMessage;
        expect(assistant, isNotNull);
        expect(assistant?.role, ChatRole.assistant);
        expect(assistant?.toolCalls, hasLength(1));
        final call = assistant!.toolCalls.single;
        expect(call.name, weatherTool.name);
        expect(call.id, isNotNull);
        expect(call.id, isNotEmpty);
        final location = call.arguments['location'];
        expect(location, isA<String>());
        expect((location! as String).toLowerCase(), contains('paris'));

        await engine.reset();
        final followUp = await engine
            .chat(
              messages: <ChatMessage>[
                user,
                assistant,
                ChatMessage.toolResult(
                  toolCallId: call.id!,
                  name: call.name,
                  text: 'Paris is sunny and 18 degrees Celsius.',
                ),
              ],
              config: const GenerationConfig(
                maxTokens: 64,
                temperature: 0,
                topK: 1,
                seed: 42,
                streamChunkTokens: 8,
                toolCalling: LlamaToolCallingConfig(
                  tools: <LlamaToolDefinition>[weatherTool],
                  allowParallelToolCalls: false,
                  toolChoice: LlamaToolChoice.none(),
                ),
              ),
            )
            .toList();
        final finalAssistant = followUp.last.assistantMessage;
        expect(finalAssistant, isNotNull);
        expect(finalAssistant?.toolCalls, isEmpty);
        expect(finalAssistant?.text.trim(), isNotEmpty);
      } finally {
        await engine.close();
      }
    });

    test('tokenizes and detokenizes with a tiny vocab fixture', () async {
      final bridgePath = _nativeBridgePath;
      if (!File(bridgePath).existsSync()) {
        markTestSkipped('native bridge has not been built at $bridgePath');
      }
      const modelPath = 'third_party/llama.cpp/models/ggml-vocab-gpt-2.gguf';
      if (!File(modelPath).existsSync()) {
        markTestSkipped('vocab fixture is missing at $modelPath');
      }
      final config = LlamaModelConfig(
        modelPath: modelPath,
        nativeLibraryPath: bridgePath,
      );

      final tokens = await LlamaTokenizer.tokenize(config, 'hello');
      final count = await LlamaTokenizer.countTokens(config, 'hello');
      final text = await LlamaTokenizer.detokenize(config, tokens);

      expect(tokens, isNotEmpty);
      expect(count, tokens.length);
      expect(text, 'hello');
    });

    test('tokenization matches upstream GPT-2 golden fixture', () async {
      final bridgePath = _nativeBridgePath;
      if (!File(bridgePath).existsSync()) {
        markTestSkipped('native bridge has not been built at $bridgePath');
      }
      const modelPath = 'third_party/llama.cpp/models/ggml-vocab-gpt-2.gguf';
      final inputFile = File('$modelPath.inp');
      final outputFile = File('$modelPath.out');
      if (!File(modelPath).existsSync() ||
          !inputFile.existsSync() ||
          !outputFile.existsSync()) {
        markTestSkipped('vocab fixture is missing at $modelPath');
      }
      final config = LlamaModelConfig(
        modelPath: modelPath,
        nativeLibraryPath: bridgePath,
      );
      final inputs = _vocabGoldenInputs(await inputFile.readAsString());
      final expected = _vocabGoldenTokens(await outputFile.readAsLines());

      expect(inputs, hasLength(expected.length));
      for (var i = 0; i < inputs.length; i += 1) {
        expect(
          await LlamaTokenizer.tokenize(config, inputs[i]),
          expected[i],
          reason: 'upstream tokenizer case $i',
        );
      }
    });

    test('requires or explicitly selects a chat template', () async {
      final bridgePath = _nativeBridgePath;
      if (!File(bridgePath).existsSync()) {
        markTestSkipped('native bridge has not been built at $bridgePath');
      }
      const modelPath = 'third_party/llama.cpp/models/ggml-vocab-gpt-2.gguf';
      if (!File(modelPath).existsSync()) {
        markTestSkipped('vocab fixture is missing at $modelPath');
      }

      final withoutTemplate = LlamaModelConfig(
        modelPath: modelPath,
        nativeLibraryPath: bridgePath,
      );
      await expectLater(
        LlamaModel.chatTemplate(withoutTemplate),
        throwsA(isA<UnsupportedFeatureException>()),
      );
      await expectLater(
        LlamaChatTemplate.format(withoutTemplate, <ChatMessage>[
          ChatMessage.user('hello'),
        ]),
        throwsA(isA<UnsupportedFeatureException>()),
      );

      final config = LlamaModelConfig(
        modelPath: modelPath,
        nativeLibraryPath: bridgePath,
        chatTemplate: 'chatml',
      );
      final prompt = await LlamaChatTemplate.format(config, <ChatMessage>[
        ChatMessage.user('hello'),
      ]);

      expect(await LlamaModel.chatTemplate(config), 'chatml');
      expect((await LlamaModel.inspect(config)).chatTemplate, 'chatml');
      expect(prompt, contains('hello'));
      expect(prompt, contains('assistant'));
    });

    test('formats tool-aware chat with the upstream Jinja helper', () async {
      final bridgePath = _nativeBridgePath;
      if (!File(bridgePath).existsSync()) {
        markTestSkipped('native bridge has not been built at $bridgePath');
      }
      const modelPath = 'third_party/llama.cpp/models/ggml-vocab-gemma-4.gguf';
      if (!File(modelPath).existsSync()) {
        markTestSkipped('vocab fixture is missing at $modelPath');
      }
      final config = LlamaModelConfig(
        modelPath: modelPath,
        nativeLibraryPath: bridgePath,
      );
      const tool = LlamaToolDefinition(
        name: 'lookup',
        description: 'Look up local data.',
        parametersSchema: <String, Object?>{
          'type': 'object',
          'properties': <String, Object?>{
            'query': <String, Object?>{'type': 'string'},
          },
          'required': <Object?>['query'],
          'additionalProperties': false,
        },
      );

      final capabilities = await LlamaChatTemplate.capabilities(config);
      final prompt = await LlamaChatTemplate.format(
        config,
        <ChatMessage>[ChatMessage.user('hello')],
        toolCalling: const LlamaToolCallingConfig(
          tools: <LlamaToolDefinition>[tool],
        ),
      );

      expect(capabilities.supportsTools, isTrue);
      expect(capabilities.supportsToolCalls, isTrue);
      expect(capabilities.supportsParallelToolCalls, isTrue);
      expect(prompt, contains('hello'));
      expect(prompt, contains('lookup'));
    });

    test('rejects tools when the model template lacks capability', () async {
      final bridgePath = _nativeBridgePath;
      if (!File(bridgePath).existsSync()) {
        markTestSkipped('native bridge has not been built at $bridgePath');
      }
      const modelPath = 'third_party/llama.cpp/models/ggml-vocab-gpt-2.gguf';
      if (!File(modelPath).existsSync()) {
        markTestSkipped('vocab fixture is missing at $modelPath');
      }
      final config = LlamaModelConfig(
        modelPath: modelPath,
        nativeLibraryPath: bridgePath,
        chatTemplate: 'chatml',
      );

      final capabilities = await LlamaChatTemplate.capabilities(config);
      expect(capabilities.supportsTools, isFalse);
      expect(capabilities.supportsToolCalls, isFalse);
      await expectLater(
        LlamaChatTemplate.format(
          config,
          <ChatMessage>[ChatMessage.user('hello')],
          toolCalling: const LlamaToolCallingConfig(
            tools: <LlamaToolDefinition>[
              LlamaToolDefinition(
                name: 'lookup',
                description: 'Look up local data.',
                parametersSchema: <String, Object?>{'type': 'object'},
              ),
            ],
          ),
        ),
        throwsA(isA<UnsupportedFeatureException>()),
      );
    });

    test('formats typed tool call and result history', () async {
      final bridgePath = _nativeBridgePath;
      if (!File(bridgePath).existsSync()) {
        markTestSkipped('native bridge has not been built at $bridgePath');
      }
      const modelPath = 'third_party/llama.cpp/models/ggml-vocab-gemma-4.gguf';
      if (!File(modelPath).existsSync()) {
        markTestSkipped('vocab fixture is missing at $modelPath');
      }

      final prompt = await LlamaChatTemplate.format(
        LlamaModelConfig(modelPath: modelPath, nativeLibraryPath: bridgePath),
        <ChatMessage>[
          ChatMessage.user('Find alpha.'),
          ChatMessage.assistantToolCalls(
            toolCalls: const <LlamaToolCall>[
              LlamaToolCall(
                id: 'call_1',
                name: 'lookup',
                arguments: <String, Object?>{'query': 'alpha'},
              ),
            ],
          ),
          const ChatMessage.toolResult(
            toolCallId: 'call_1',
            name: 'lookup',
            text: 'local result',
          ),
        ],
        toolCalling: const LlamaToolCallingConfig(
          tools: <LlamaToolDefinition>[
            LlamaToolDefinition(
              name: 'lookup',
              description: 'Look up local data.',
              parametersSchema: <String, Object?>{
                'type': 'object',
                'properties': <String, Object?>{
                  'query': <String, Object?>{'type': 'string'},
                },
              },
            ),
          ],
        ),
      );

      expect(prompt, contains('lookup'));
      expect(prompt, contains('local result'));
    });

    test('snapshots chat template messages before worker formatting', () async {
      final bridgePath = _nativeBridgePath;
      if (!File(bridgePath).existsSync()) {
        markTestSkipped('native bridge has not been built at $bridgePath');
      }
      const modelPath = 'third_party/llama.cpp/models/ggml-vocab-gpt-2.gguf';
      if (!File(modelPath).existsSync()) {
        markTestSkipped('vocab fixture is missing at $modelPath');
      }

      final messages = <ChatMessage>[ChatMessage.user('hello')];
      final prompt = LlamaChatTemplate.format(
        LlamaModelConfig(
          modelPath: modelPath,
          nativeLibraryPath: bridgePath,
          chatTemplate: 'chatml',
        ),
        messages,
      );
      messages[0] = ChatMessage.user('bad\u0000');

      expect(await prompt, contains('hello'));
    });

    test('counts chat-template tokens with the upstream tokenizer', () async {
      final bridgePath = _nativeBridgePath;
      if (!File(bridgePath).existsSync()) {
        markTestSkipped('native bridge has not been built at $bridgePath');
      }
      const modelPath = 'third_party/llama.cpp/models/ggml-vocab-gpt-2.gguf';
      if (!File(modelPath).existsSync()) {
        markTestSkipped('vocab fixture is missing at $modelPath');
      }

      final count = await LlamaChatTemplate.countTokens(
        LlamaModelConfig(
          modelPath: modelPath,
          nativeLibraryPath: bridgePath,
          chatTemplate: 'chatml',
        ),
        <ChatMessage>[ChatMessage.user('hello')],
      );

      expect(count, greaterThan(0));
    });
  });

  group('config validation', () {
    test('rejects empty model paths', () {
      expect(
        () => const LlamaModelConfig(modelPath: '').validate(),
        throwsArgumentError,
      );
      expect(
        () => const LlamaModelConfig(
          modelPath: 'model.gguf',
          nativeLibraryPath: ' ',
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => const LlamaModelConfig(
          modelPath: 'model.gguf',
          speculativeDecoding: DraftModelSpeculation(
            draftModelPath: 'draft.gguf',
            draftLength: 0,
          ),
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => const LlamaModelConfig(
          modelPath: 'model.gguf',
          speculativeDecoding: MtpSpeculation(draftLength: 1025),
        ).validate(),
        throwsArgumentError,
      );
      const draft = DraftModelSpeculation(draftModelPath: 'draft.gguf');
      const eagle = Eagle3Speculation(draftModelPath: 'eagle.gguf');
      const dflash = DFlashSpeculation(draftModelPath: 'dflash.gguf');
      const mtp = MtpSpeculation();
      const ngramMod = NGramModSpeculation();
      expect(draft.draftLength, 3);
      expect(eagle.draftLength, 3);
      expect(dflash.draftLength, 15);
      expect(mtp.draftLength, 3);
      expect(ngramMod.matchLength, 24);
      expect(ngramMod.minimumDraftLength, 48);
      expect(ngramMod.maximumDraftLength, 64);
      expect(
        () => const LlamaModelConfig(
          modelPath: 'model.gguf',
          speculativeDecoding: NGramCacheSpeculation(),
        ).validate(),
        returnsNormally,
      );
    });

    test('accepts model loading flags', () {
      const defaults = LlamaModelConfig(modelPath: 'model.gguf');
      expect(defaults.useMmap, isTrue);
      expect(defaults.useMlock, isFalse);
      expect(defaults.checkTensors, isTrue);

      expect(
        () => const LlamaModelConfig(
          modelPath: 'model.gguf',
          useMmap: false,
          useMlock: true,
          checkTensors: false,
        ).validate(),
        returnsNormally,
      );
    });

    test('validates explicit chat template overrides', () {
      expect(
        () => const LlamaModelConfig(
          modelPath: 'model.gguf',
          chatTemplate: 'chatml',
        ).validate(),
        returnsNormally,
      );
      expect(
        () => const LlamaModelConfig(
          modelPath: 'model.gguf',
          chatTemplate: ' \n\t',
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => const LlamaModelConfig(
          modelPath: 'model.gguf',
          chatTemplate: 'bad\u0000template',
        ).validate(),
        throwsArgumentError,
      );
    });

    test('validates typed KV cache tuning', () {
      const defaults = LlamaModelConfig(modelPath: 'model.gguf');
      expect(defaults.kvCache.keyType, KvCacheType.f16);
      expect(defaults.kvCache.valueType, KvCacheType.f16);
      expect(defaults.kvCache.offload, isTrue);
      expect(defaults.kvCache.flashAttention, FlashAttentionMode.auto);
      expect(defaults.kvCache.swaFull, isTrue);
      expect(defaults.kvCache.unified, isFalse);

      for (final type in KvCacheType.values) {
        expect(
          () => LlamaModelConfig(
            modelPath: 'model.gguf',
            kvCache: KvCacheConfig(
              keyType: type,
              valueType: type,
              flashAttention: FlashAttentionMode.enabled,
            ),
          ).validate(),
          returnsNormally,
        );
      }
      expect(
        () => const LlamaModelConfig(
          modelPath: 'model.gguf',
          kvCache: KvCacheConfig(
            keyType: KvCacheType.q8Zero,
            flashAttention: FlashAttentionMode.disabled,
          ),
        ).validate(),
        returnsNormally,
      );
      expect(
        () => const LlamaModelConfig(
          modelPath: 'model.gguf',
          kvCache: KvCacheConfig(
            valueType: KvCacheType.q8Zero,
            flashAttention: FlashAttentionMode.disabled,
          ),
        ).validate(),
        throwsArgumentError,
      );
    });

    test('rejects native integer overflow config', () {
      expect(
        () => const GpuConfig.auto(layers: -1).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GpuConfig.auto(layers: 0x80000000).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GpuConfig.metal(layers: 0).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GpuConfig.vulkan(layers: 0).validate(),
        throwsArgumentError,
      );
      expect(
        () => const LlamaModelConfig(
          modelPath: 'model.gguf',
          contextSize: 0x100000000,
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => const LlamaModelConfig(
          modelPath: 'model.gguf',
          batchSize: 0x100000000,
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => const LlamaModelConfig(
          modelPath: 'model.gguf',
          ubatchSize: 0x100000000,
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => const LlamaModelConfig(
          modelPath: 'model.gguf',
          threads: 0x80000000,
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => const LlamaModelConfig(
          modelPath: 'model.gguf',
          batchThreads: 0x80000000,
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => const LlamaModelConfig(
          modelPath: 'model.gguf',
          ubatchSize: 0,
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => const LlamaModelConfig(
          modelPath: 'model.gguf',
          batchThreads: 0,
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => const LlamaModelConfig(
          modelPath: 'model.gguf',
          batchSize: 16,
          ubatchSize: 32,
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => const LlamaModelConfig(
          modelPath: 'model.gguf',
          contextSize: 0x80000000,
          speculativeDecoding: MtpSpeculation(),
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => const LlamaModelConfig(
          modelPath: 'model.gguf',
          batchSize: 0x80000000,
          speculativeDecoding: DraftModelSpeculation(
            draftModelPath: 'draft.gguf',
          ),
        ).validate(),
        throwsArgumentError,
      );
    });

    test('rejects unsafe model paths', () {
      expect(
        () => const LlamaModelConfig(modelPath: 'bad\u0000path').validate(),
        throwsArgumentError,
      );
      expect(
        () => const LlamaModelConfig(modelPath: 'bad\npath').validate(),
        throwsArgumentError,
      );
      expect(
        () => const LlamaModelConfig(
          modelPath: 'model.gguf',
          nativeLibraryPath: 'bridge\u0000',
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => const LlamaModelConfig(
          modelPath: 'model.gguf',
          nativeLibraryPath: 'bad\nbridge',
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => const LlamaModelConfig(
          modelPath: 'model.gguf',
          mmprojPath: 'bad\u0000mmproj.gguf',
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => const LlamaModelConfig(
          modelPath: 'model.gguf',
          mmprojPath: 'bad\nmmproj.gguf',
        ).validate(),
        throwsArgumentError,
      );
    });

    test('rejects invalid speculative decoding config', () {
      expect(
        () => const LlamaModelConfig(
          modelPath: 'model.gguf',
          speculativeDecoding: DraftModelSpeculation(draftModelPath: ''),
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => const LlamaModelConfig(
          modelPath: 'model.gguf',
          speculativeDecoding: Eagle3Speculation(
            draftModelPath: 'draft.gguf\u0000',
          ),
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => const LlamaModelConfig(
          modelPath: 'model.gguf',
          speculativeDecoding: DFlashSpeculation(draftModelPath: ''),
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => const LlamaModelConfig(
          modelPath: 'model.gguf',
          speculativeDecoding: NGramModSpeculation(
            minimumDraftLength: 9,
            maximumDraftLength: 8,
          ),
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => const LlamaModelConfig(
          modelPath: 'model.gguf',
          speculativeDecoding: MtpSpeculation(mtpModelPath: ''),
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => const LlamaModelConfig(
          modelPath: 'model.gguf',
          speculativeDecoding: NGramSpeculation(strategy: ''),
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => const LlamaModelConfig(
          modelPath: 'model.gguf',
          speculativeDecoding: NGramSpeculation(strategy: 'ngram-simple\n'),
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => const LlamaModelConfig(
          modelPath: 'model.gguf',
          speculativeDecoding: NGramSpeculation(strategy: ' ngram-simple '),
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => const LlamaModelConfig(
          modelPath: 'model.gguf',
          speculativeDecoding: NGramSpeculation(
            strategy: 'ngram-simple',
            ngramSize: 0,
          ),
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => const LlamaModelConfig(
          modelPath: 'model.gguf',
          speculativeDecoding: NGramSpeculation(
            strategy: 'ngram-simple',
            ngramSize: 16,
            draftLength: 8,
          ),
        ).validate(),
        throwsArgumentError,
      );
      expect(
        LlamaEngine.load(
          const LlamaModelConfig(
            modelPath: 'model.gguf',
            speculativeDecoding: NGramSpeculation(strategy: 'unknown-ngram'),
          ),
        ),
        throwsA(isA<UnsupportedFeatureException>()),
      );
      expect(
        LlamaEngine.load(
          const LlamaModelConfig(
            modelPath: 'model.gguf',
            mmprojPath: 'vision-mmproj.gguf',
          ),
        ),
        throwsA(isA<ModelLoadException>()),
      );
    });

    test('rejects empty raw completion prompts', () async {
      final bridgePath = _nativeBridgePath;
      if (!File(bridgePath).existsSync()) {
        markTestSkipped('native bridge has not been built at $bridgePath');
      }
      const modelPath = 'third_party/llama.cpp/models/ggml-vocab-gpt-2.gguf';
      if (!File(modelPath).existsSync()) {
        markTestSkipped('vocab fixture is missing at $modelPath');
      }

      try {
        final engine = await LlamaEngine.load(
          LlamaModelConfig(
            modelPath: modelPath,
            nativeLibraryPath: bridgePath,
            contextSize: 128,
            batchSize: 16,
            threads: 1,
          ),
        );
        try {
          await expectLater(
            engine.complete(prompt: '').toList(),
            throwsArgumentError,
          );
          await expectLater(
            engine.complete(prompt: '  \n\t').toList(),
            throwsArgumentError,
          );
        } finally {
          await engine.close();
        }
      } on ModelLoadException {
        // Vocab-only GGUF fixtures may not contain weights needed for full load.
      } on ContextCreateException {
        // Some vocab fixtures load but cannot create a usable context.
      }
    });

    test('rejects invalid sampling config', () {
      expect(
        () => const GenerationConfig(maxTokens: 0x100000000).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(maxTokens: 0x80000000).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(streamChunkTokens: 0).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(streamChunkTokens: 1025).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(temperature: double.nan).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(topK: -1).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(topK: 0x80000000).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(topP: 1.1).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(topP: double.nan).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(minP: -0.1).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(typicalP: 1.1).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(penaltyLastN: -1).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(penaltyLastN: 0x80000000).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(repeatPenalty: -0.1).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(
          frequencyPenalty: double.infinity,
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(mirostatTau: 0).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(mirostatEta: double.nan).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(seed: -1).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(seed: 0x100000000).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(stop: <String>['']).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(stop: <String>['bad\u0000']).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(stopTokens: <int>[-1]).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(stopTokens: <int>[0x80000000]).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(stopTokens: <int>[1, 1]).validate(),
        throwsArgumentError,
      );
      expect(
        () => GenerationConfig(
          stopTokens: List<int>.generate(1025, (index) => index),
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () =>
            const GenerationConfig(loraScales: <int, double>{0: 1}).validate(),
        throwsArgumentError,
      );
      expect(
        () =>
            const GenerationConfig(loraScales: <int, double>{1: -1}).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(
          loraScales: <int, double>{1: double.nan},
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(grammar: '').validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(grammar: '  \n\t').validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(grammar: 'root ::= "x"\u0000').validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(grammarRoot: 'custom').validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(
          grammar: 'root ::= "x"',
          enableThinking: true,
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(
          maxTokens: 16,
          enableThinking: true,
          reasoningBudgetTokens: -1,
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(
          maxTokens: 16,
          reasoningBudgetTokens: 8,
        ).validate(),
        returnsNormally,
      );
      expect(
        () => const GenerationConfig(
          maxTokens: 16,
          enableThinking: false,
          reasoningBudgetTokens: 8,
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(
          maxTokens: 16,
          enableThinking: true,
          reasoningBudgetTokens: 16,
        ).validate(),
        returnsNormally,
      );
      expect(
        () => const GenerationConfig(
          maxTokens: 16,
          enableThinking: true,
          reasoningBudgetTokens: 8,
        ).validate(),
        returnsNormally,
      );
      expect(
        () => GenerationConfig.jsonSchema(
          schema: const <String, Object?>{'type': 'string'},
          enableThinking: true,
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(
          grammar: 'root ::= "x"',
          grammarRoot: '',
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(
          grammar: 'root ::= "x"',
          grammarRoot: '  \t',
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(
          grammar: 'root ::= "x"',
          grammarRoot: 'root\u0000',
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(
          grammar: 'root ::= "x"',
          grammarRoot: 'root\nother',
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(
          grammar: 'root ::= "x"',
          grammarRoot: 'root other',
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(
          mirostat: MirostatMode.v1,
          mirostatTau: 0,
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(
          mirostat: MirostatMode.v2,
          mirostatEta: double.nan,
        ).validate(),
        throwsArgumentError,
      );
    });

    test('json schema generation config snapshots mutable inputs', () {
      final stops = <String>['</json>'];
      final stopTokens = <int>[1, 2];
      final loraScales = <int, double>{1: 0.5};
      final properties = <String, Object?>{
        'query': <String, Object?>{'type': 'string'},
      };
      final toolSchema = <String, Object?>{
        'type': 'object',
        'additionalProperties': false,
        'required': <Object?>['query'],
        'properties': properties,
      };
      final tool = LlamaToolDefinition(
        name: 'search_local',
        description: 'Search local notes.',
        parametersSchema: toolSchema,
      );
      final tools = <LlamaToolDefinition>[tool];
      final outputSchema = <String, Object?>{
        'type': 'object',
        'properties': <String, Object?>{
          'answer': <String, Object?>{'type': 'string'},
        },
      };
      final config = GenerationConfig.jsonSchema(
        schema: outputSchema,
        streamChunkTokens: 7,
        stop: stops,
        stopTokens: stopTokens,
        loraScales: loraScales,
        toolCalling: LlamaToolCallingConfig(tools: tools),
      );

      stops[0] = 'changed';
      stopTokens[0] = 99;
      loraScales[1] = 2;
      tools.clear();
      properties['query'] = <String, Object?>{'type': 'number'};
      (outputSchema['properties']! as Map<String, Object?>)['answer'] =
          <String, Object?>{'type': 'number'};

      expect(config.stop, const <String>['</json>']);
      expect(config.streamChunkTokens, 7);
      expect(() => config.stop[0] = 'changed', throwsUnsupportedError);
      expect(config.stopTokens, const <int>[1, 2]);
      expect(() => config.stopTokens[0] = 99, throwsUnsupportedError);
      expect(config.loraScales, const <int, double>{1: 0.5});
      expect(() => config.loraScales![1] = 2, throwsUnsupportedError);
      expect(config.toolCalling.tools.single.name, 'search_local');
      expect(() => config.toolCalling.tools.clear(), throwsUnsupportedError);
      final exportedTool = config.toolCalling.toJson().single;
      final exportedFunction =
          exportedTool['function']! as Map<String, Object?>;
      final exportedSchema =
          exportedFunction['parameters']! as Map<String, Object?>;
      final exportedProperties =
          exportedSchema['properties']! as Map<String, Object?>;
      expect(exportedProperties['query'], <String, Object?>{'type': 'string'});
      final savedOutputProperties =
          config.jsonSchema!['properties']! as Map<String, Object?>;
      expect(savedOutputProperties['answer'], <String, Object?>{
        'type': 'string',
      });
      expect(
        () => savedOutputProperties['answer'] = <String, Object?>{},
        throwsUnsupportedError,
      );
    });

    test('rejects invalid LoRA adapter config', () {
      expect(
        () => const LoraAdapterConfig(path: '').validate(),
        throwsArgumentError,
      );
      expect(
        () => const LoraAdapterConfig(path: 'adapter.gguf\u0000').validate(),
        throwsArgumentError,
      );
      expect(
        () => const LoraAdapterConfig(path: 'adapter\n.gguf').validate(),
        throwsArgumentError,
      );
      expect(
        () => const LoraAdapterConfig(
          path: 'adapter.gguf',
          scale: double.nan,
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => const LoraAdapterConfig(
          path: 'adapter.gguf',
          scale: -0.1,
        ).validate(),
        throwsArgumentError,
      );
    });

    test('rejects rank pooling through embedding config validation', () {
      expect(
        () => const EmbeddingConfig(pooling: EmbeddingPooling.rank).validate(),
        _throwsUnsupportedFeature,
      );
      expect(
        () => const EmbeddingConfig(pooling: EmbeddingPooling.mean).validate(),
        returnsNormally,
      );
    });

    test('json mode uses the built-in JSON grammar', () {
      const config = GenerationConfig.jsonMode(maxTokens: 32);

      expect(config.maxTokens, 32);
      expect(config.grammar, llamaJsonGrammar);
      expect(config.grammar, contains('[0-9] [0-9]{0,15})? ws'));
      expect(config.grammarRoot, 'root');
      expect(() => config.validate(), returnsNormally);
    });

    test('json schema mode preserves schema for native conversion', () {
      final config = GenerationConfig.jsonSchema(
        schema: <String, Object?>{
          'type': 'object',
          'additionalProperties': false,
          'required': <Object?>['answer', 'score', 'tags'],
          'properties': <String, Object?>{
            'answer': <String, Object?>{'type': 'string'},
            'score': <String, Object?>{'type': 'integer'},
            'tags': <String, Object?>{
              'type': 'array',
              'items': <String, Object?>{
                'enum': <Object?>['a', 'b'],
              },
            },
          },
        },
      );

      expect(config.grammarRoot, 'root');
      expect(config.grammar, isNull);
      expect(config.jsonSchema?['type'], 'object');
      expect(
        config.jsonSchema?['properties'],
        containsPair('answer', <String, Object?>{'type': 'string'}),
      );
      expect(() => config.validate(), returnsNormally);
    });

    test('json schema mode delegates broader schemas to upstream', () {
      final config = GenerationConfig.jsonSchema(
        schema: <String, Object?>{
          r'$defs': <String, Object?>{
            'code': <String, Object?>{
              'type': 'string',
              'pattern': r'^[A-Z]{2}$',
            },
          },
          r'$ref': r'#/$defs/code',
        },
      );

      expect(config.grammar, isNull);
      expect(config.jsonSchema?[r'$ref'], r'#/$defs/code');
      expect(() => config.validate(), returnsNormally);
      expect(
        () => GenerationConfig(
          grammar: llamaJsonGrammar,
          jsonSchema: const <String, Object?>{'type': 'string'},
        ).validate(),
        throwsArgumentError,
      );
    });

    test('json schema helper delegates to the pinned native converter', () {
      final grammar = llamaJsonSchemaGrammar(<String, Object?>{
        'type': 'object',
        'additionalProperties': false,
        'required': <Object?>['answer'],
        'properties': <String, Object?>{
          'answer': <String, Object?>{'type': 'string'},
        },
      });

      expect(grammar, contains('root ::='));
      expect(grammar, contains('answer'));
    });

    test('generation chunks can carry telemetry', () {
      const telemetry = GenerationTelemetry(
        promptTokens: 2,
        generatedTokens: 3,
        promptEvalMs: 1.5,
        decodeMs: 2.5,
        totalMs: 4.0,
        timeToFirstTokenMs: 2.0,
        speculativeDraftTokens: 4,
        speculativeAcceptedTokens: 3,
        speculativeDraftMs: 1.25,
        speculativeVerifyMs: 2.5,
      );
      const chunk = GenerationChunk(
        text: 'ok',
        isDone: true,
        telemetry: telemetry,
      );

      expect(chunk.telemetry?.generatedTokens, 3);
      expect(chunk.telemetry?.totalMs, 4.0);
      expect(chunk.telemetry?.timeToFirstTokenMs, 2.0);
      expect(
        chunk.telemetry?.promptEvalTokensPerSecond,
        closeTo(1333.333, 0.001),
      );
      expect(chunk.telemetry?.decodeTokensPerSecond, 1200.0);
      expect(chunk.telemetry?.totalTokensPerSecond, 1250.0);
      expect(chunk.telemetry?.speculativeAcceptanceRate, 0.75);
      expect(chunk.telemetry?.speculativeDraftMs, 1.25);
      expect(chunk.telemetry?.speculativeVerifyMs, 2.5);
      expect(
        const GenerationTelemetry(
          promptTokens: 0,
          generatedTokens: 0,
          promptEvalMs: 0,
          decodeMs: double.nan,
          totalMs: 0,
        ).decodeTokensPerSecond,
        0,
      );
      expect(
        const GenerationTelemetry(
          promptTokens: 0,
          generatedTokens: 0,
          promptEvalMs: 0,
          decodeMs: 0,
          totalMs: 0,
          speculativeDraftTokens: 2,
          speculativeAcceptedTokens: 3,
        ).speculativeAcceptanceRate,
        1,
      );
      expect(
        const GenerationTelemetry(
          promptTokens: 0,
          generatedTokens: 0,
          promptEvalMs: 0,
          decodeMs: 0,
          totalMs: 0,
          speculativeDraftTokens: 2,
          speculativeAcceptedTokens: -1,
        ).speculativeAcceptanceRate,
        0,
      );
    });

    test('rejects empty embedding text', () {
      expect(
        LlamaEmbeddings.embedText(
          const LlamaModelConfig(
            modelPath: 'model.gguf',
            nativeLibraryPath: 'bridge',
          ),
          '',
        ),
        throwsArgumentError,
      );
      expect(
        LlamaEmbeddings.embedText(
          const LlamaModelConfig(modelPath: 'model.gguf'),
          ' ',
        ),
        throwsArgumentError,
      );
    });

    test('rejects rank pooling through embedding APIs before native work', () {
      expect(
        LlamaEmbeddings.embedText(
          const LlamaModelConfig(modelPath: 'model.gguf'),
          'query',
          config: const EmbeddingConfig(pooling: EmbeddingPooling.rank),
        ),
        throwsA(isA<UnsupportedFeatureException>()),
      );
      expect(
        LlamaEmbeddings.embedTexts(
          const LlamaModelConfig(modelPath: 'model.gguf'),
          const <String>['query'],
          config: const EmbeddingConfig(pooling: EmbeddingPooling.rank),
        ),
        throwsA(isA<UnsupportedFeatureException>()),
      );
      expect(
        LlamaEmbeddingEngine.load(
          const LlamaModelConfig(modelPath: 'model.gguf'),
          config: const EmbeddingConfig(pooling: EmbeddingPooling.rank),
        ),
        throwsA(isA<UnsupportedFeatureException>()),
      );
    });

    test('embedding contexts reject generation-only model settings', () {
      const speculative = LlamaModelConfig(
        modelPath: 'model.gguf',
        speculativeDecoding: NGramSpeculation(strategy: 'ngram-simple'),
      );
      final throwsSpeculative = throwsA(
        isA<UnsupportedFeatureException>().having(
          (error) => error.message,
          'message',
          contains('Speculative decoding'),
        ),
      );

      expect(
        LlamaEmbeddings.embedText(speculative, 'query'),
        throwsSpeculative,
      );
      expect(
        LlamaEmbeddings.embedTexts(speculative, const <String>['query']),
        throwsSpeculative,
      );
      expect(LlamaEmbeddingEngine.load(speculative), throwsSpeculative);
      expect(
        LlamaReranking.scorePair(
          speculative,
          query: 'query',
          document: 'document',
        ),
        throwsSpeculative,
      );
      expect(
        const LlamaReranker(
          speculative,
        ).rerank('query', const <VectorSearchResult>[]),
        throwsSpeculative,
      );
      expect(
        LlamaEmbeddings.embedText(
          const LlamaModelConfig(
            modelPath: 'model.gguf',
            mmprojPath: 'vision-mmproj.gguf',
          ),
          'query',
        ),
        throwsA(
          isA<UnsupportedFeatureException>().having(
            (error) => error.message,
            'message',
            contains('mmprojPath'),
          ),
        ),
      );
    });

    test('rejects NUL bytes in native text inputs before native work', () {
      expect(
        LlamaTokenizer.tokenize(
          const LlamaModelConfig(modelPath: 'model.gguf'),
          'bad\u0000',
        ),
        throwsArgumentError,
      );
      expect(
        LlamaEmbeddings.embedText(
          const LlamaModelConfig(modelPath: 'model.gguf'),
          'bad\u0000',
        ),
        throwsArgumentError,
      );
      expect(
        LlamaEmbeddings.embedTexts(
          const LlamaModelConfig(modelPath: 'model.gguf'),
          const <String>['ok', 'bad\u0000'],
        ),
        throwsArgumentError,
      );
    });

    test('rejects invalid detokenize token ids before native work', () {
      expect(
        LlamaTokenizer.detokenize(
          const LlamaModelConfig(modelPath: 'model.gguf'),
          const <int>[-1],
        ),
        throwsArgumentError,
      );
      expect(
        LlamaTokenizer.detokenize(
          const LlamaModelConfig(modelPath: 'model.gguf'),
          const <int>[0x80000000],
        ),
        throwsArgumentError,
      );
    });

    test('detokenize allows empty token lists without native work', () async {
      final text = await LlamaTokenizer.detokenize(
        const LlamaModelConfig(modelPath: 'model.gguf'),
        const <int>[],
      );

      expect(text, isEmpty);
    });

    test('batch embeddings allow empty batches without native work', () async {
      final embeddings = await LlamaEmbeddings.embedTexts(
        const LlamaModelConfig(modelPath: 'model.gguf'),
        const <String>[],
        config: const EmbeddingConfig(
          normalize: false,
          pooling: EmbeddingPooling.mean,
        ),
      );

      expect(embeddings, isEmpty);
      expect(embeddings.count, 0);
      expect(embeddings.dimensions, 0);
      expect(embeddings.values, isEmpty);
      expect(embeddings.normalized, isFalse);
      expect(embeddings.pooling, EmbeddingPooling.mean);
    });

    test('embedding batches expose flat storage and vector views', () {
      final batch = EmbeddingBatch(
        count: 2,
        dimensions: 3,
        values: Float32List.fromList(<double>[1, 2, 3, 4, 5, 6]),
        normalized: false,
        pooling: EmbeddingPooling.mean,
      );

      expect(batch.values, <double>[1, 2, 3, 4, 5, 6]);
      expect(batch[0], <double>[1, 2, 3]);
      expect(batch.vectorAt(1), <double>[4, 5, 6]);
      expect(batch.normalized, isFalse);
      expect(batch.pooling, EmbeddingPooling.mean);
      expect(batch.toList(), <List<double>>[
        <double>[1, 2, 3],
        <double>[4, 5, 6],
      ]);
      expect(() => batch[2], throwsRangeError);
      expect(() => batch.values[0] = 0, throwsUnsupportedError);
      expect(
        () => EmbeddingBatch(count: 2, dimensions: 3, values: Float32List(5)),
        throwsArgumentError,
      );
    });

    test(
      'reranking allows empty document batches without native work',
      () async {
        final scores = await LlamaReranking.scoreDocuments(
          const LlamaModelConfig(modelPath: 'model.gguf'),
          query: 'query',
          documents: const <String>[],
          config: const RerankingConfig(addSpecial: false, parseSpecial: true),
        );

        expect(scores, isEmpty);
        const reranker = LlamaReranker(
          LlamaModelConfig(modelPath: 'model.gguf'),
          config: RerankingConfig(addSpecial: false, parseSpecial: true),
        );
        expect(reranker.config.addSpecial, isFalse);
        expect(reranker.config.parseSpecial, isTrue);
      },
    );

    test('reranker rejects invalid query before empty-candidate return', () {
      const reranker = LlamaReranker(LlamaModelConfig(modelPath: 'model.gguf'));

      expect(
        reranker.rerank('', const <VectorSearchResult>[]),
        throwsArgumentError,
      );
      expect(
        reranker.rerank('bad\u0000query', const <VectorSearchResult>[]),
        throwsArgumentError,
      );
    });

    test('reranker rejects malformed candidates before native work', () {
      const reranker = LlamaReranker(LlamaModelConfig(modelPath: 'model.gguf'));

      expect(
        reranker.rerank('query', const <VectorSearchResult>[
          VectorSearchResult(
            chunk: TextChunk(
              documentId: 'doc',
              id: '0',
              text: 'candidate',
              tokenCount: 1,
              metadata: <String, Object?>{'bad': Object()},
            ),
            score: 1,
          ),
        ]),
        throwsArgumentError,
      );
    });

    test('reranking rejects empty or NUL text before native work', () {
      expect(
        LlamaReranking.scorePair(
          const LlamaModelConfig(modelPath: 'model.gguf'),
          query: '',
          document: 'document',
        ),
        throwsArgumentError,
      );
      expect(
        LlamaReranking.scorePair(
          const LlamaModelConfig(modelPath: 'model.gguf'),
          query: ' ',
          document: 'document',
        ),
        throwsArgumentError,
      );
      expect(
        LlamaReranking.scorePair(
          const LlamaModelConfig(modelPath: 'model.gguf'),
          query: 'query',
          document: 'bad\u0000document',
        ),
        throwsArgumentError,
      );
      expect(
        LlamaReranking.scorePair(
          const LlamaModelConfig(modelPath: 'model.gguf'),
          query: 'query',
          document: ' ',
        ),
        throwsArgumentError,
      );
    });

    test('rejects empty batch embedding text', () {
      expect(
        LlamaEmbeddings.embedTexts(
          const LlamaModelConfig(modelPath: 'model.gguf'),
          const <String>['ok', ''],
        ),
        throwsArgumentError,
      );
      expect(
        LlamaEmbeddings.embedTexts(
          const LlamaModelConfig(modelPath: 'model.gguf'),
          const <String>['ok', ' '],
        ),
        throwsArgumentError,
      );
    });

    test('rejects unsupported or invalid chat template inputs', () {
      expect(
        LlamaChatTemplate.format(
          const LlamaModelConfig(
            modelPath: 'model.gguf',
            nativeLibraryPath: 'bridge',
          ),
          const <ChatMessage>[],
        ),
        throwsArgumentError,
      );
      expect(
        LlamaChatTemplate.format(
          const LlamaModelConfig(modelPath: 'model.gguf'),
          <ChatMessage>[ChatMessage.user(' ')],
        ),
        throwsArgumentError,
      );
      expect(
        LlamaChatTemplate.format(
          const LlamaModelConfig(modelPath: 'model.gguf'),
          <ChatMessage>[
            ChatMessage.content(
              role: ChatRole.user,
              parts: const <ChatContentPart>[TextPart('')],
            ),
          ],
        ),
        throwsArgumentError,
      );
      expect(
        LlamaChatTemplate.countTokens(
          const LlamaModelConfig(
            modelPath: 'model.gguf',
            mmprojPath: 'mmproj.gguf',
          ),
          <ChatMessage>[
            ChatMessage.content(
              role: ChatRole.user,
              parts: const <ChatContentPart>[ImagePart.fromFile('image.png')],
            ),
          ],
        ),
        _throwsUnsupportedFeature,
      );
      expect(
        LlamaChatTemplate.format(
          const LlamaModelConfig(
            modelPath: 'model.gguf',
            mmprojPath: 'mmproj.gguf',
          ),
          <ChatMessage>[
            ChatMessage.content(
              role: ChatRole.user,
              parts: const <ChatContentPart>[VideoPart.fromFile('video.mp4')],
            ),
          ],
        ),
        _throwsUnsupportedFeature,
      );
      expect(
        LlamaChatTemplate.format(
          const LlamaModelConfig(
            modelPath: 'model.gguf',
            mmprojPath: 'mmproj.gguf',
          ),
          <ChatMessage>[
            ChatMessage.content(
              role: ChatRole.user,
              parts: <ChatContentPart>[
                for (var i = 0; i < 65; i += 1)
                  ImagePart.fromFile('image-$i.png'),
              ],
            ),
          ],
        ),
        throwsArgumentError,
      );
    });

    test('rejects NUL bytes in chat messages', () {
      expect(
        LlamaChatTemplate.format(
          const LlamaModelConfig(
            modelPath: 'model.gguf',
            nativeLibraryPath: 'bridge',
          ),
          <ChatMessage>[ChatMessage.user('bad\u0000')],
        ),
        throwsArgumentError,
      );
    });

    test('chat content parts aggregate text and gate media', () {
      final textOnly = ChatMessage.content(
        role: ChatRole.user,
        parts: const <ChatContentPart>[TextPart('hello'), TextPart(' world')],
      );

      expect(textOnly.text, 'hello world');
      expect(textOnly.hasNonTextParts, isFalse);
      expect(
        () => ChatMessage.content(
          role: ChatRole.user,
          parts: const <ChatContentPart>[],
        ),
        throwsArgumentError,
      );
      final imageBytes = Uint8List.fromList(<int>[1]);
      final imagePart = ImagePart.fromBytes(imageBytes);
      imageBytes[0] = 2;
      expect(imagePart.bytes!.single, 1);
      expect(() => imagePart.bytes![0] = 3, throwsUnsupportedError);
      expect(() => ImagePart.fromBytes(Uint8List(0)), throwsArgumentError);
      expect(() => AudioPart.fromBytes(Uint8List(0)), throwsArgumentError);
      expect(() => VideoPart.fromBytes(Uint8List(0)), throwsArgumentError);
      expect(
        LlamaChatTemplate.format(
          const LlamaModelConfig(
            modelPath: 'model.gguf',
            nativeLibraryPath: 'bridge',
          ),
          <ChatMessage>[
            ChatMessage.content(
              role: ChatRole.user,
              parts: const <ChatContentPart>[
                ImagePart.fromFile('image.png', mimeType: 'image/ '),
              ],
            ),
          ],
        ),
        throwsArgumentError,
      );
      expect(
        LlamaChatTemplate.format(
          const LlamaModelConfig(
            modelPath: 'model.gguf',
            nativeLibraryPath: 'bridge',
          ),
          <ChatMessage>[
            ChatMessage.content(
              role: ChatRole.user,
              parts: const <ChatContentPart>[
                ImagePart.fromFile('image.png', mimeType: 'image/png '),
              ],
            ),
          ],
        ),
        throwsArgumentError,
      );
      expect(
        LlamaChatTemplate.format(
          const LlamaModelConfig(
            modelPath: 'model.gguf',
            nativeLibraryPath: 'bridge',
          ),
          <ChatMessage>[
            ChatMessage.content(
              role: ChatRole.user,
              parts: const <ChatContentPart>[
                TextPart('look'),
                ImagePart.fromFile('image.png'),
              ],
            ),
          ],
        ),
        throwsA(
          isA<UnsupportedFeatureException>().having(
            (error) => error.message,
            'message',
            allOf(contains('image'), contains('mmproj')),
          ),
        ),
      );
      expect(
        LlamaChatTemplate.format(
          const LlamaModelConfig(
            modelPath: 'model.gguf',
            nativeLibraryPath: 'bridge',
          ),
          <ChatMessage>[
            ChatMessage.content(
              role: ChatRole.user,
              parts: const <ChatContentPart>[
                ImagePart.fromFile('image.png', mimeType: 'text/plain'),
              ],
            ),
          ],
        ),
        throwsArgumentError,
      );
      expect(
        LlamaChatTemplate.format(
          const LlamaModelConfig(
            modelPath: 'model.gguf',
            nativeLibraryPath: 'bridge',
          ),
          <ChatMessage>[
            ChatMessage.content(
              role: ChatRole.user,
              parts: const <ChatContentPart>[
                ImagePart.fromFile('image.png', mimeType: 'image/'),
              ],
            ),
          ],
        ),
        throwsArgumentError,
      );
    });

    test('tool definitions validate and serialize', () {
      const tool = LlamaToolDefinition(
        name: 'search_local',
        description: 'Search local notes.',
        parametersSchema: <String, Object?>{
          'type': 'object',
          'additionalProperties': false,
          'required': <Object?>['query'],
          'properties': <String, Object?>{
            'query': <String, Object?>{'type': 'string'},
          },
        },
      );

      final exportedTool = tool.toJson();
      expect(exportedTool['type'], 'function');
      expect(() => exportedTool['type'] = 'bad', throwsUnsupportedError);
      final exportedToolFunction =
          exportedTool['function']! as Map<String, Object?>;
      expect(
        () => exportedToolFunction['name'] = 'changed',
        throwsUnsupportedError,
      );
      final exportedSchema =
          exportedToolFunction['parameters']! as Map<String, Object?>;
      final exportedProperties =
          exportedSchema['properties']! as Map<String, Object?>;
      expect(
        () => exportedProperties['query'] = <String, Object?>{'type': 'number'},
        throwsUnsupportedError,
      );
      expect(
        const LlamaToolCallingConfig(
          tools: <LlamaToolDefinition>[tool],
        ).toJson().single['type'],
        'function',
      );
      expect(
        const LlamaToolCallingConfig(
          tools: <LlamaToolDefinition>[tool],
          allowParallelToolCalls: false,
        ).toOpenAiJson()['parallel_tool_calls'],
        isFalse,
      );
      expect(
        const LlamaToolCallingConfig(
          tools: <LlamaToolDefinition>[tool],
          toolChoice: LlamaToolChoice.required(),
        ).toOpenAiJson()['tool_choice'],
        'required',
      );
      final namedToolChoice = const LlamaToolCallingConfig(
        tools: <LlamaToolDefinition>[tool],
        toolChoice: LlamaToolChoice.named('search_local'),
      ).toOpenAiJson();
      expect(namedToolChoice['tool_choice'], <String, Object?>{
        'type': 'function',
        'function': <String, Object?>{'name': 'search_local'},
      });
      expect(
        () => namedToolChoice['tool_choice'] = 'none',
        throwsUnsupportedError,
      );
      final namedToolChoiceValue =
          namedToolChoice['tool_choice']! as Map<String, Object?>;
      expect(
        () => namedToolChoiceValue['type'] = 'changed',
        throwsUnsupportedError,
      );
      final namedToolChoiceFunction =
          namedToolChoiceValue['function']! as Map<String, Object?>;
      expect(
        () => namedToolChoiceFunction['name'] = 'changed',
        throwsUnsupportedError,
      );
      expect(
        const LlamaToolCallingConfig(
          tools: <LlamaToolDefinition>[tool],
        ).validate,
        returnsNormally,
      );
      expect(
        const LlamaToolCallingConfig(
          tools: <LlamaToolDefinition>[tool, tool],
        ).validate,
        throwsArgumentError,
      );
      expect(
        const GenerationConfig(
          toolCalling: LlamaToolCallingConfig(
            tools: <LlamaToolDefinition>[tool],
            allowParallelToolCalls: false,
          ),
        ).validate,
        returnsNormally,
      );
      expect(
        const LlamaToolDefinition(
          name: '',
          description: 'bad',
          parametersSchema: <String, Object?>{},
        ).validate,
        throwsArgumentError,
      );
      expect(
        const LlamaToolDefinition(
          name: 'bad\nname',
          description: 'bad',
          parametersSchema: <String, Object?>{},
        ).validate,
        throwsArgumentError,
      );
      expect(
        const LlamaToolDefinition(
          name: 'bad name',
          description: 'bad',
          parametersSchema: <String, Object?>{},
        ).validate,
        throwsArgumentError,
      );
      expect(
        const LlamaToolDefinition(
          name: 'bad_description',
          description: ' ',
          parametersSchema: <String, Object?>{},
        ).validate,
        throwsArgumentError,
      );
      expect(
        const LlamaToolDefinition(
          name: 'bad_schema',
          description: 'bad',
          parametersSchema: <String, Object?>{
            'type': 'object',
            'properties': <String, Object?>{
              'bad': <String, Object?>{
                'enum': <Object?>[double.nan],
              },
            },
          },
        ).validate,
        throwsArgumentError,
      );
      expect(
        const LlamaToolCallingConfig(
          tools: <LlamaToolDefinition>[tool],
          toolChoice: LlamaToolChoice.named('missing'),
        ).validate,
        throwsArgumentError,
      );
      expect(
        const LlamaToolCallingConfig(
          tools: <LlamaToolDefinition>[tool],
          toolChoice: LlamaToolChoice.named('bad\tname'),
        ).validate,
        throwsArgumentError,
      );
      expect(
        const LlamaToolCallingConfig(
          toolChoice: LlamaToolChoice.required(),
        ).validate,
        throwsArgumentError,
      );
      expect(
        () => const GenerationConfig(
          grammar: 'root ::= "x"',
          toolCalling: LlamaToolCallingConfig(
            tools: <LlamaToolDefinition>[tool],
          ),
        ).validate(),
        throwsArgumentError,
      );
      expect(
        () => GenerationConfig.jsonSchema(
          schema: const <String, Object?>{'type': 'object'},
          toolCalling: const LlamaToolCallingConfig(
            tools: <LlamaToolDefinition>[tool],
          ),
        ).validate(),
        throwsArgumentError,
      );
    });

    test('tool definitions validate parsed arguments against their schema', () {
      const tool = LlamaToolDefinition(
        name: 'record_probe',
        description: 'Record a local probe.',
        parametersSchema: <String, Object?>{
          'type': 'object',
          'properties': <String, Object?>{
            'sentinel': <String, Object?>{
              'type': 'string',
              'enum': <Object?>['EXPECTED_SENTINEL'],
            },
            'samples': <String, Object?>{
              'type': 'array',
              'items': <String, Object?>{'\$ref': '#/\$defs/sample'},
              'minItems': 1,
            },
          },
          'required': <Object?>['sentinel'],
          'additionalProperties': false,
          '\$defs': <String, Object?>{
            'sample': <String, Object?>{
              'type': 'integer',
              'minimum': 0,
              'maximum': 10,
            },
          },
        },
      );

      expect(
        () => tool.validateArguments(<String, Object?>{
          'sentinel': 'EXPECTED_SENTINEL',
          'samples': <Object?>[0, 10],
        }),
        returnsNormally,
      );
      expect(
        () => tool.validateArguments(<String, Object?>{
          'sentinel': 'EXPECTED_SENTINEL',
          'markdown': '**reasoning**',
        }),
        throwsArgumentError,
      );
      expect(
        () => tool.validateArguments(<String, Object?>{}),
        throwsArgumentError,
      );
      expect(
        () => tool.validateArguments(<String, Object?>{'sentinel': 'wrong'}),
        throwsArgumentError,
      );
      expect(
        () => tool.validateArguments(<String, Object?>{
          'sentinel': 'EXPECTED_SENTINEL',
          'samples': <Object?>[11],
        }),
        throwsArgumentError,
      );
    });

    test('typed tool messages snapshot calls and expose result metadata', () {
      final arguments = <String, Object?>{'query': 'alpha'};
      final calls = <LlamaToolCall>[
        LlamaToolCall(id: 'call_1', name: 'lookup', arguments: arguments),
      ];
      final assistant = ChatMessage.assistantToolCalls(toolCalls: calls);
      arguments['query'] = 'changed';
      calls.clear();

      expect(assistant.role, ChatRole.assistant);
      expect(assistant.text, isEmpty);
      expect(assistant.toolCalls.single.arguments['query'], 'alpha');
      expect(() => assistant.toolCalls.clear(), throwsUnsupportedError);

      const result = ChatMessage.toolResult(
        toolCallId: 'call_1',
        name: 'lookup',
        text: 'local result',
      );
      expect(result.role, ChatRole.tool);
      expect(result.toolCallId, 'call_1');
      expect(result.toolName, 'lookup');
      expect(
        () => ChatMessage.assistantToolCalls(toolCalls: const []),
        throwsArgumentError,
      );

      final finalChunk = GenerationChunk(
        text: '',
        isDone: true,
        assistantMessage: assistant,
      );
      expect(finalChunk.assistantMessage, same(assistant));
    });

    test('rejects malformed typed tool history before native work', () {
      const config = LlamaModelConfig(
        modelPath: 'model.gguf',
        nativeLibraryPath: 'bridge',
      );
      expect(
        LlamaChatTemplate.format(config, <ChatMessage>[
          ChatMessage.assistantToolCalls(
            toolCalls: const <LlamaToolCall>[
              LlamaToolCall(
                id: 'bad id',
                name: 'lookup',
                arguments: <String, Object?>{},
              ),
            ],
          ),
        ]),
        throwsArgumentError,
      );
      expect(
        LlamaChatTemplate.format(config, const <ChatMessage>[
          ChatMessage.toolResult(
            toolCallId: 'bad id',
            name: 'lookup',
            text: 'result',
          ),
        ]),
        throwsArgumentError,
      );
    });

    test('tool call parser handles direct and OpenAI-shaped JSON', () {
      final direct = LlamaToolCalls.parse(
        '{"name":"search_local","arguments":{"query":"llama"}}',
      );

      expect(direct.single.name, 'search_local');
      expect(direct.single.arguments['query'], 'llama');
      final typedAsObject = LlamaToolCalls.fromJson(<Object?, Object?>{
        'name': 'search_local',
        'arguments': <Object?, Object?>{'query': 'typed'},
      });

      expect(typedAsObject.single.arguments['query'], 'typed');

      final openAi = LlamaToolCalls.parse(
        '{"tool_calls":[{"id":"call_1","function":{"name":"search_local",'
        '"arguments":"{\\"query\\":\\"rag\\"}"}}]}',
      );

      expect(openAi.single.id, 'call_1');
      expect(openAi.single.arguments['query'], 'rag');
      final nestedArguments = <Object?>['original'];
      final arguments = <String, Object?>{'query': nestedArguments};
      final exported = LlamaToolCall(
        name: 'search_local',
        arguments: arguments,
      ).toJson();
      final exportedOpenAi = LlamaToolCall(
        name: 'search_local',
        arguments: arguments,
      ).toOpenAiJson();
      nestedArguments[0] = 'changed';

      expect(openAi.single.toJson()['name'], 'search_local');
      expect(() => exported['name'] = 'changed', throwsUnsupportedError);
      final exportedArguments = exported['arguments']! as Map<String, Object?>;
      final exportedQuery = exportedArguments['query']! as List<Object?>;
      expect(exportedQuery.single, 'original');
      expect(() => exportedQuery[0] = 'changed', throwsUnsupportedError);
      final exportedOpenAiFunction =
          exportedOpenAi['function']! as Map<String, Object?>;
      expect(() => exportedOpenAi['type'] = 'bad', throwsUnsupportedError);
      expect(
        () => exportedOpenAiFunction['name'] = 'changed',
        throwsUnsupportedError,
      );
      expect(
        jsonDecode(exportedOpenAiFunction['arguments']! as String),
        <String, Object?>{
          'query': <Object?>['original'],
        },
      );
      final exportedOpenAiCalls = LlamaToolCalls.toOpenAiJson(openAi);
      final exportedOpenAiCallsList =
          exportedOpenAiCalls['tool_calls']! as List<Map<String, Object?>>;
      expect(
        () => exportedOpenAiCalls['tool_calls'] = <Object?>[],
        throwsUnsupportedError,
      );
      expect(
        () => exportedOpenAiCallsList.add(exportedOpenAi),
        throwsUnsupportedError,
      );
      expect(
        () => LlamaToolCalls.toOpenAiJson(const <LlamaToolCall>[]),
        throwsArgumentError,
      );
      expect(
        () => LlamaToolCalls.toOpenAiJson(const <LlamaToolCall>[
          LlamaToolCall(
            id: 'call_1',
            name: 'search_local',
            arguments: <String, Object?>{},
          ),
          LlamaToolCall(
            id: 'call_1',
            name: 'lookup',
            arguments: <String, Object?>{},
          ),
        ]),
        throwsArgumentError,
      );
      final roundTrip = LlamaToolCalls.fromJson(exportedOpenAiCalls);
      expect(roundTrip.single.id, 'call_1');
      expect(roundTrip.single.arguments['query'], 'rag');
      expect(
        () => LlamaToolCalls.parse(
          '[{"name":"a","arguments":{}},{"name":"b","arguments":{}}]',
          allowParallelToolCalls: false,
        ),
        _throwsUnsupportedFeature,
      );
      expect(() => LlamaToolCalls.parse('{'), throwsArgumentError);
      expect(() => LlamaToolCalls.parse('[]'), throwsArgumentError);
      expect(
        () => LlamaToolCalls.parse('{"name":"","arguments":{}}'),
        throwsArgumentError,
      );
      expect(
        () => LlamaToolCalls.parse('{"tool_calls":null}'),
        throwsArgumentError,
      );
      expect(
        () => LlamaToolCalls.parse('{"tool_calls":[]}'),
        throwsArgumentError,
      );
      expect(
        () => LlamaToolCalls.parse('{"name":"search_local","arguments":null}'),
        throwsArgumentError,
      );
      expect(
        () => LlamaToolCalls.parse('{"name":"search_local","arguments":""}'),
        throwsArgumentError,
      );
      expect(
        () => LlamaToolCalls.parse('{"tool_calls":[],"extra":true}'),
        throwsArgumentError,
      );
      expect(
        () => LlamaToolCalls.parse(
          '{"type":"custom","name":"search_local","arguments":{}}',
        ),
        throwsArgumentError,
      );
      expect(
        () => LlamaToolCalls.parse(
          '{"name":"search_local","arguments":{},"extra":true}',
        ),
        throwsArgumentError,
      );
      expect(
        () => LlamaToolCalls.parse(
          '{"tool_calls":[{"id":"","function":{"name":"search_local",'
          '"arguments":"{}"}}]}',
        ),
        throwsArgumentError,
      );
      expect(
        () => LlamaToolCalls.parse('{"name":"bad\\nname","arguments":{}}'),
        throwsArgumentError,
      );
      expect(
        () => LlamaToolCalls.parse('{"name":"bad name","arguments":{}}'),
        throwsArgumentError,
      );
      expect(
        () => LlamaToolCalls.parse(
          '{"tool_calls":[{"id":"bad\\nid","function":{"name":"search_local",'
          '"arguments":"{}"}}]}',
        ),
        throwsArgumentError,
      );
      expect(
        () => LlamaToolCalls.parse(
          '{"tool_calls":[{"type":"custom","function":{"name":"search_local",'
          '"arguments":"{}"}}]}',
        ),
        throwsArgumentError,
      );
      expect(
        () => LlamaToolCalls.parse(
          '{"tool_calls":[{"id":"call_1","extra":true,'
          '"function":{"name":"search_local","arguments":"{}"}}]}',
        ),
        throwsArgumentError,
      );
      expect(
        () => LlamaToolCalls.parse(
          '{"tool_calls":[{"id":"call_1","function":{"name":"search_local",'
          '"arguments":"{}","extra":true}}]}',
        ),
        throwsArgumentError,
      );
      expect(
        () => LlamaToolCalls.parse(
          '{"tool_calls":[{"id":"call_1","function":{"name":"search_local",'
          '"arguments":null}}]}',
        ),
        throwsArgumentError,
      );
      expect(
        () => LlamaToolCalls.parse(
          '{"tool_calls":[{"id":"call_1","function":{"name":"search_local",'
          '"arguments":""}}]}',
        ),
        throwsArgumentError,
      );
      expect(
        () => LlamaToolCalls.parse(
          '{"tool_calls":[{"id":"call_1","function":{"name":"search_local",'
          '"arguments":"{"}}]}',
        ),
        throwsArgumentError,
      );
      expect(
        () => LlamaToolCalls.parse(
          '{"tool_calls":[{"id":"call_1","function":{"name":"search_local",'
          '"arguments":"{}"}},{"id":"call_1","function":{"name":"lookup",'
          '"arguments":"{}"}}]}',
        ),
        throwsArgumentError,
      );
      expect(
        () => const LlamaToolCall(
          name: 'search local',
          arguments: <String, Object?>{},
        ).toJson(),
        throwsArgumentError,
      );
      expect(
        () => LlamaToolCalls.fromJson(<String, Object?>{
          'name': 'search_local',
          'arguments': <String, Object?>{'bad': Object()},
        }),
        throwsArgumentError,
      );
      expect(
        () => const LlamaToolCall(
          name: 'search_local',
          arguments: <String, Object?>{'bad': Object()},
        ).toOpenAiJson(),
        throwsArgumentError,
      );
      expect(
        () => const LlamaToolCall(
          name: 'search_local',
          arguments: <String, Object?>{'bad': 'value\u0000'},
        ).toOpenAiJson(),
        throwsArgumentError,
      );
    });
  });

  group('desktop smoke benchmark', () {
    test('reports Dart build mode metadata', () {
      expect(desktop_benchmark.benchmarkBuildMode(), 'jit');
      expect(Platform.version, isNotEmpty);
    });

    test('reports process memory metadata', () {
      final memory = desktop_benchmark.benchmarkMemorySnapshot();

      expect(memory['currentRssBytes'], greaterThan(0));
      expect(memory['maxRssBytes'], greaterThan(0));
    });

    test('reports stable benchmark model file names', () {
      expect(
        desktop_benchmark.benchmarkModelFileName('/models/tiny-q4.gguf'),
        'tiny-q4.gguf',
      );
      expect(
        desktop_benchmark.benchmarkModelFileName(r'C:\models\tiny-q4.gguf'),
        'tiny-q4.gguf',
      );
      expect(
        desktop_benchmark.benchmarkKvCacheTypeName(KvCacheType.iq4Nl),
        'iq4_nl',
      );
      expect(
        desktop_benchmark.benchmarkFlashAttentionName(
          FlashAttentionMode.disabled,
        ),
        'disabled',
      );
    });

    test('parses benchmark CLI options', () {
      final options = desktop_benchmark.parseBenchmarkArgs(<String>[
        '--model',
        'model.gguf',
        '--native-library=build/native/libllama_dart_bridge.dylib',
        '--device-model',
        'MacBookPro18,3',
        '--prompt',
        'hello',
        '--max-tokens',
        '8',
        '--iterations',
        '5',
        '--no-warm-up',
        '--context-size',
        '1024',
        '--batch-size',
        '128',
        '--ubatch-size',
        '64',
        '--threads',
        '2',
        '--batch-threads',
        '3',
        '--gpu-layers',
        '0',
        '--kv-cache-key',
        'q4_0',
        '--kv-cache-value',
        'q8_0',
        '--no-kv-offload',
        '--flash-attention',
        'enabled',
        '--no-swa-full',
        '--kv-unified',
        '--seed',
        '42',
        '--spec-ngram-simple',
        '--spec-ngram-size',
        '8',
        '--spec-min-draft-length',
        '12',
        '--spec-draft-length',
        '16',
        '--json-out',
        'benchmark.json',
      ]);

      expect(options.modelPath, 'model.gguf');
      expect(
        options.nativeLibraryPath,
        'build/native/libllama_dart_bridge.dylib',
      );
      expect(options.deviceModel, 'MacBookPro18,3');
      expect(options.prompt, 'hello');
      expect(options.maxTokens, 8);
      expect(options.iterations, 5);
      expect(options.warmUp, isFalse);
      expect(options.contextSize, 1024);
      expect(options.batchSize, 128);
      expect(options.ubatchSize, 64);
      expect(options.threads, 2);
      expect(options.batchThreads, 3);
      expect(options.gpuLayers, 0);
      expect(options.kvCacheKeyType, KvCacheType.q4Zero);
      expect(options.kvCacheValueType, KvCacheType.q8Zero);
      expect(options.kvCacheOffload, isFalse);
      expect(options.flashAttention, FlashAttentionMode.enabled);
      expect(options.swaFull, isFalse);
      expect(options.kvUnified, isTrue);
      expect(options.seed, 42);
      expect(options.speculativeNgramSimple, isTrue);
      expect(options.speculativeNgramStrategy, 'ngram-simple');
      expect(options.speculativeNgramSize, 8);
      expect(options.speculativeMinimumDraftLength, 12);
      expect(options.speculativeDraftLength, 16);
      expect(options.jsonOut, 'benchmark.json');
      expect(
        desktop_benchmark.parseBenchmarkArgs(<String>[
          '--help',
          '--max-tokens',
        ]).help,
        isTrue,
      );
      expect(
        desktop_benchmark.parseBenchmarkArgs(<String>[
          '--spec-ngram',
          'ngram-map-k4v',
        ]).speculativeNgramStrategy,
        'ngram-map-k4v',
      );
      final modOptions = desktop_benchmark.parseBenchmarkArgs(<String>[
        '--spec-ngram',
        'ngram-mod',
        '--spec-ngram-size',
        '24',
        '--spec-min-draft-length',
        '48',
        '--spec-draft-length',
        '64',
      ]);
      expect(
        desktop_benchmark.benchmarkSpeculativeDecodingConfig(modOptions),
        isA<NGramModSpeculation>()
            .having((config) => config.matchLength, 'matchLength', 24)
            .having(
              (config) => config.minimumDraftLength,
              'minimumDraftLength',
              48,
            )
            .having(
              (config) => config.maximumDraftLength,
              'maximumDraftLength',
              64,
            ),
      );
      final cacheConfig = desktop_benchmark.benchmarkSpeculativeDecodingConfig(
        desktop_benchmark.parseBenchmarkArgs(<String>[
          '--spec-ngram',
          'ngram-cache',
        ]),
      );
      expect(cacheConfig, isA<NGramCacheSpeculation>());
      expect(
        desktop_benchmark.benchmarkSpeculativeDecodingRecord(cacheConfig),
        <String, Object?>{'strategy': 'ngram-cache'},
      );
    });

    test('rejects invalid benchmark CLI options', () {
      expect(
        () => desktop_benchmark.parseBenchmarkArgs(<String>['--max-tokens']),
        throwsFormatException,
      );
      expect(
        () => desktop_benchmark.parseBenchmarkArgs(<String>[
          '--model',
          '--max-tokens',
          '8',
        ]),
        throwsA(
          isA<FormatException>().having(
            (error) => error.message,
            'message',
            'Missing value for --model.',
          ),
        ),
      );
      expect(
        () => desktop_benchmark.parseBenchmarkArgs(<String>[
          '--gpu-layers',
          '-1',
        ]),
        throwsFormatException,
      );
      expect(
        () => desktop_benchmark.parseBenchmarkArgs(<String>['--model=']),
        throwsFormatException,
      );
      expect(
        () => desktop_benchmark.parseBenchmarkArgs(<String>[
          '--model',
          'bad\nmodel.gguf',
        ]),
        throwsFormatException,
      );
      expect(
        () => desktop_benchmark.parseBenchmarkArgs(<String>[
          '--native-library',
          'bad\rbridge',
        ]),
        throwsFormatException,
      );
      expect(
        () => desktop_benchmark.parseBenchmarkArgs(<String>[
          '--prompt',
          'bad\u0000',
        ]),
        throwsFormatException,
      );
      expect(
        () => desktop_benchmark.parseBenchmarkArgs(<String>['--json-out=']),
        throwsFormatException,
      );
      expect(
        () => desktop_benchmark.parseBenchmarkArgs(<String>[
          '--json-out',
          'bad\nbenchmark.json',
        ]),
        throwsFormatException,
      );
      expect(
        () => desktop_benchmark.parseBenchmarkArgs(<String>['--unknown']),
        throwsFormatException,
      );
      expect(
        () => desktop_benchmark.parseBenchmarkArgs(<String>[
          '--spec-ngram-simple=true',
        ]),
        throwsFormatException,
      );
      expect(
        () => desktop_benchmark.parseBenchmarkArgs(<String>[
          '--spec-ngram',
          'ngram-unknown',
        ]),
        throwsFormatException,
      );
      expect(
        () =>
            desktop_benchmark.parseBenchmarkArgs(<String>['--iterations', '0']),
        throwsFormatException,
      );
      expect(
        () => desktop_benchmark.parseBenchmarkArgs(<String>[
          '--iterations',
          '1001',
        ]),
        throwsFormatException,
      );
      expect(
        () =>
            desktop_benchmark.parseBenchmarkArgs(<String>['--no-warm-up=true']),
        throwsFormatException,
      );
      expect(
        () => desktop_benchmark.parseBenchmarkArgs(<String>[
          '--kv-cache-key',
          'q2_bad',
        ]),
        throwsFormatException,
      );
      expect(
        () => desktop_benchmark.parseBenchmarkArgs(<String>[
          '--flash-attention',
          'sometimes',
        ]),
        throwsFormatException,
      );
      expect(
        () => desktop_benchmark.parseBenchmarkArgs(<String>[
          '--no-kv-offload=true',
        ]),
        throwsFormatException,
      );
      expect(
        () => desktop_benchmark.parseBenchmarkArgs(<String>[
          '--no-swa-full=true',
        ]),
        throwsFormatException,
      );
      expect(
        () =>
            desktop_benchmark.parseBenchmarkArgs(<String>['--kv-unified=true']),
        throwsFormatException,
      );
      expect(
        () => desktop_benchmark.parseBenchmarkArgs(<String>[
          '--kv-cache-value',
          'q8_0',
          '--flash-attention',
          'disabled',
        ]),
        throwsFormatException,
      );
      expect(
        () => desktop_benchmark.parseBenchmarkArgs(<String>[
          '--batch-size',
          '8',
          '--ubatch-size',
          '16',
        ]),
        throwsFormatException,
      );
      expect(
        () => desktop_benchmark.parseBenchmarkArgs(<String>[
          '--spec-ngram-size',
          '16',
          '--spec-draft-length',
          '8',
        ]),
        throwsFormatException,
      );
      expect(
        () => desktop_benchmark.parseBenchmarkArgs(<String>[
          '--spec-ngram',
          'ngram-mod',
          '--spec-min-draft-length',
          '65',
          '--spec-draft-length',
          '64',
        ]),
        throwsFormatException,
      );
    });

    test('benchmark CLI reports usage for invalid options', () async {
      final output = StringBuffer();
      final errors = StringBuffer();
      final status = await desktop_benchmark.runBenchmarkCli(
        const <String>['--max-tokens'],
        output: output,
        errorOutput: errors,
      );

      expect(status, 64);
      expect(output.toString(), isEmpty);
      expect(errors.toString(), contains('Missing value for --max-tokens.'));
      expect(errors.toString(), contains('Usage: dart run'));
      expect(errors.toString(), isNot(contains('Unhandled exception')));
    });

    test('benchmark CLI reports help in process', () async {
      final output = StringBuffer();
      final errors = StringBuffer();
      final status = await desktop_benchmark.runBenchmarkCli(
        const <String>['--help'],
        output: output,
        errorOutput: errors,
      );

      expect(status, 0);
      expect(output.toString(), contains('Usage: dart run'));
      expect(errors.toString(), isEmpty);
    });

    test('chat CLI reports usage for invalid options', () async {
      Future<(int, String, String)> invoke(List<String> args) async {
        final output = StringBuffer();
        final errors = StringBuffer();
        final status = await chat_cli.runChatCli(
          args,
          output: output,
          errorOutput: errors,
        );
        return (status, output.toString(), errors.toString());
      }

      final result = await invoke(const <String>['--model', 'bad\nmodel.gguf']);
      expect(result.$1, 64);
      expect(result.$2, isEmpty);
      expect(result.$3, contains('--model must not contain line breaks.'));
      expect(result.$3, contains('Usage: dart run'));
      expect(result.$3, isNot(contains('Unhandled exception')));

      final tooLarge = await invoke(const <String>[
        '--model',
        'model.gguf',
        '--max-tokens',
        '2147483648',
      ]);
      expect(tooLarge.$1, 64);
      expect(tooLarge.$2, isEmpty);
      expect(tooLarge.$3, contains('--max-tokens must be between 1'));
      expect(tooLarge.$3, contains('Usage: dart run'));
      expect(tooLarge.$3, isNot(contains('Unhandled exception')));

      final missingBeforeFlag = await invoke(const <String>[
        '--model',
        '--max-tokens',
        '8',
      ]);
      expect(missingBeforeFlag.$1, 64);
      expect(missingBeforeFlag.$2, isEmpty);
      expect(missingBeforeFlag.$3, contains('Missing value for --model.'));
      expect(missingBeforeFlag.$3, contains('Usage: dart run'));
      expect(missingBeforeFlag.$3, isNot(contains('Unhandled exception')));
    });

    test('local RAG CLI reports usage for invalid query text', () async {
      final output = StringBuffer();
      final errors = StringBuffer();
      final status = await local_rag_cli.runLocalRagCli(
        const <String>[''],
        output: output,
        errorOutput: errors,
      );

      expect(status, 64);
      expect(output.toString(), isEmpty);
      expect(errors.toString(), contains('Query must not be empty.'));
      expect(errors.toString(), contains('Usage: dart run'));
      expect(errors.toString(), isNot(contains('Unhandled exception')));
    });

    test('local RAG CLI runs in process', () async {
      final output = StringBuffer();
      final errors = StringBuffer();
      final status = await local_rag_cli.runLocalRagCli(
        const <String>[],
        output: output,
        errorOutput: errors,
      );

      expect(status, 0);
      expect(output.toString(), contains('Citations:'));
      expect(output.toString(), contains('privacy#privacy:0'));
      expect(errors.toString(), isEmpty);
    });

    test('rejects invalid direct benchmark options before native work', () {
      expect(
        desktop_benchmark.runDesktopSmokeBenchmark(
          const desktop_benchmark.DesktopSmokeBenchmarkOptions(
            modelPath: 'model.gguf',
            prompt: '',
          ),
        ),
        throwsFormatException,
      );
      expect(
        desktop_benchmark.runDesktopSmokeBenchmark(
          const desktop_benchmark.DesktopSmokeBenchmarkOptions(
            modelPath: 'model.gguf',
            kvCacheValueType: KvCacheType.q8Zero,
            flashAttention: FlashAttentionMode.disabled,
          ),
        ),
        throwsFormatException,
      );
      expect(
        desktop_benchmark.runDesktopSmokeBenchmark(
          const desktop_benchmark.DesktopSmokeBenchmarkOptions(
            modelPath: 'model.gguf',
            batchSize: 8,
            ubatchSize: 16,
          ),
        ),
        throwsFormatException,
      );
      expect(
        desktop_benchmark.runDesktopSmokeBenchmark(
          const desktop_benchmark.DesktopSmokeBenchmarkOptions(
            modelPath: 'model.gguf',
            maxTokens: 0,
          ),
        ),
        throwsFormatException,
      );
      expect(
        desktop_benchmark.runDesktopSmokeBenchmark(
          const desktop_benchmark.DesktopSmokeBenchmarkOptions(
            modelPath: 'model.gguf',
            gpuLayers: -1,
          ),
        ),
        throwsFormatException,
      );
      expect(
        desktop_benchmark.runDesktopSmokeBenchmark(
          const desktop_benchmark.DesktopSmokeBenchmarkOptions(
            modelPath: 'model.gguf',
            seed: 0x100000000,
          ),
        ),
        throwsFormatException,
      );
      expect(
        desktop_benchmark.runDesktopSmokeBenchmark(
          const desktop_benchmark.DesktopSmokeBenchmarkOptions(
            modelPath: 'model.gguf',
            jsonOut: 'bad\nbenchmark.json',
          ),
        ),
        throwsFormatException,
      );
      expect(
        desktop_benchmark.runDesktopSmokeBenchmark(
          const desktop_benchmark.DesktopSmokeBenchmarkOptions(
            modelPath: 'model.gguf',
            iterations: 0,
          ),
        ),
        throwsFormatException,
      );
      expect(
        desktop_benchmark.runDesktopSmokeBenchmark(
          const desktop_benchmark.DesktopSmokeBenchmarkOptions(
            modelPath: 'model.gguf',
            speculativeNgramSize: 1025,
          ),
        ),
        throwsFormatException,
      );
    });

    test('aggregates sustained benchmark telemetry', () {
      const first = GenerationTelemetry(
        promptTokens: 10,
        generatedTokens: 20,
        promptEvalMs: 100,
        decodeMs: 200,
        totalMs: 300,
        timeToFirstTokenMs: 40,
        speculativeDraftTokens: 8,
        speculativeAcceptedTokens: 4,
        speculativeDraftMs: 12,
        speculativeVerifyMs: 15,
      );
      const second = GenerationTelemetry(
        promptTokens: 10,
        generatedTokens: 20,
        promptEvalMs: 100,
        decodeMs: 200,
        totalMs: 300,
        timeToFirstTokenMs: 60,
        speculativeDraftTokens: 12,
        speculativeAcceptedTokens: 6,
        speculativeDraftMs: 18,
        speculativeVerifyMs: 25,
      );

      final aggregate = desktop_benchmark.benchmarkAggregateTelemetry(
        const <GenerationTelemetry>[first, second],
      );

      expect(aggregate['iterations'], 2);
      expect(aggregate['promptTokens'], 20);
      expect(aggregate['generatedTokens'], 40);
      expect(aggregate['timeToFirstTokenMsMean'], 50);
      expect(aggregate['timeToFirstTokenMsMin'], 40);
      expect(aggregate['timeToFirstTokenMsMax'], 60);
      expect(aggregate['promptEvalTokensPerSecond'], 100);
      expect(aggregate['decodeTokensPerSecond'], 100);
      expect(aggregate['speculativeDraftTokens'], 20);
      expect(aggregate['speculativeAcceptedTokens'], 10);
      expect(aggregate['speculativeAcceptanceRate'], 0.5);
      expect(aggregate['speculativeDraftMs'], 30);
      expect(aggregate['speculativeVerifyMs'], 40);
      expect(
        () => desktop_benchmark.benchmarkAggregateTelemetry(
          const <GenerationTelemetry>[],
        ),
        throwsArgumentError,
      );
    });
  });

  group('character splitter', () {
    test('splits deterministically and snapshots metadata', () {
      final splitter = CharacterTextSplitter(maxLength: 8, overlap: 2);
      final tags = <Object?>['fixture'];
      final chunks = splitter.split(
        Document(
          id: 'doc',
          text: 'alpha beta gamma',
          metadata: <String, Object?>{'tags': tags},
        ),
      );
      tags[0] = 'changed';

      expect(chunks.map((chunk) => chunk.text), <String>[
        'alpha be',
        'beta gam',
        'amma',
      ]);
      expect(chunks.map((chunk) => chunk.id), <String>[
        'doc:0',
        'doc:1',
        'doc:2',
      ]);
      final storedTags = chunks.first.metadata['tags']! as List<Object?>;
      expect(storedTags.single, 'fixture');
      expect(() => storedTags[0] = 'changed', throwsUnsupportedError);
      expect(chunks.first.tokenCount, 2);
    });

    test('records trimmed character source spans', () {
      final splitter = CharacterTextSplitter(maxLength: 9);
      final chunks = splitter.split(
        const Document(id: 'doc', text: '  alpha  beta'),
      );

      expect(chunks.map((chunk) => chunk.text), <String>['alpha', 'beta']);
      expect(chunks.first.metadata['start'], 2);
      expect(chunks.first.metadata['end'], 7);
      expect(chunks.last.metadata['start'], 9);
      expect(chunks.last.metadata['end'], 13);
    });

    test('does not split surrogate pairs', () {
      final splitter = CharacterTextSplitter(maxLength: 2, overlap: 1);
      final chunks = splitter.split(const Document(id: 'doc', text: 'a😀b'));

      expect(chunks.map((chunk) => chunk.text), <String>['a😀', '😀b']);
      expect(chunks.first.metadata['end'], 3);
      expect(chunks.last.metadata['start'], 1);
    });

    test('rejects empty document ids', () {
      final splitter = CharacterTextSplitter(maxLength: 8);

      expect(
        () => splitter.split(const Document(id: '', text: 'alpha')),
        throwsArgumentError,
      );
      expect(
        () => splitter.split(const Document(id: 'doc', text: 'bad\u0000')),
        throwsArgumentError,
      );
    });

    test('rejects invalid document source URIs', () {
      final splitter = CharacterTextSplitter(maxLength: 8);

      expect(
        () => splitter.split(
          Document(id: 'doc', text: 'alpha', sourceUri: Uri.parse('')),
        ),
        throwsArgumentError,
      );
      expect(
        () => splitter.split(
          Document(
            id: 'doc',
            text: 'alpha',
            sourceUri: Uri.parse('file:///bad%00path'),
          ),
        ),
        throwsArgumentError,
      );
    });
  });

  group('token splitter', () {
    test('splits by token count with overlap and snapshots metadata', () async {
      final splitter = TokenTextSplitter(
        maxTokens: 2,
        overlap: 1,
        tokenize: (_) async => <int>[1, 2, 3, 4],
        detokenize: (tokens) async => tokens.join(' '),
      );
      final tags = <Object?>['fixture'];

      final chunks = await splitter.split(
        Document(
          id: 'doc',
          text: 'ignored by fake tokenizer',
          metadata: <String, Object?>{'tags': tags},
        ),
      );
      tags[0] = 'changed';

      expect(chunks.map((chunk) => chunk.text), <String>['1 2', '2 3', '3 4']);
      expect(chunks.map((chunk) => chunk.id), <String>[
        'doc:0',
        'doc:1',
        'doc:2',
      ]);
      expect(chunks.map((chunk) => chunk.tokenCount), <int>[2, 2, 2]);
      expect(chunks.first.metadata['tokenStart'], 0);
      expect(chunks.first.metadata['tokenEnd'], 2);
      final storedTags = chunks.first.metadata['tags']! as List<Object?>;
      expect(storedTags.single, 'fixture');
      expect(() => storedTags[0] = 'changed', throwsUnsupportedError);
    });

    test('rejects empty document ids', () async {
      final splitter = TokenTextSplitter(
        maxTokens: 2,
        tokenize: (_) async => <int>[1],
        detokenize: (_) async => 'x',
      );

      await expectLater(
        splitter.split(const Document(id: '', text: 'alpha')),
        throwsArgumentError,
      );
      await expectLater(
        splitter.split(const Document(id: 'doc', text: 'bad\u0000')),
        throwsArgumentError,
      );
    });

    test('rejects invalid document source URIs', () async {
      final splitter = TokenTextSplitter(
        maxTokens: 2,
        tokenize: (_) async => <int>[1],
        detokenize: (_) async => 'x',
      );

      await expectLater(
        splitter.split(
          Document(id: 'doc', text: 'alpha', sourceUri: Uri.parse('')),
        ),
        throwsArgumentError,
      );
      await expectLater(
        splitter.split(
          Document(
            id: 'doc',
            text: 'alpha',
            sourceUri: Uri.parse('file:///bad%0Apath'),
          ),
        ),
        throwsArgumentError,
      );
    });

    test('rejects detokenized NUL bytes', () async {
      final splitter = TokenTextSplitter(
        maxTokens: 2,
        tokenize: (_) async => <int>[1],
        detokenize: (_) async => 'bad\u0000',
      );

      await expectLater(
        splitter.split(const Document(id: 'doc', text: 'alpha')),
        throwsArgumentError,
      );
    });

    test('rejects invalid tokenizer token ids before detokenizing', () async {
      final splitter = TokenTextSplitter(
        maxTokens: 2,
        tokenize: (_) async => <int>[1, -1],
        detokenize: (_) async => throw StateError('detokenizer should not run'),
      );

      await expectLater(
        splitter.split(const Document(id: 'doc', text: 'alpha')),
        throwsArgumentError,
      );
    });

    test('snapshots tokenizer output before detokenizing', () async {
      final tokenBuffer = <int>[1, 2, 3];
      final splitter = TokenTextSplitter(
        maxTokens: 1,
        tokenize: (_) async {
          Timer.run(() {
            tokenBuffer[1] = 99;
          });
          return tokenBuffer;
        },
        detokenize: (tokens) async {
          await Future<void>.delayed(Duration.zero);
          return tokens.single.toString();
        },
      );

      final chunks = await splitter.split(const Document(id: 'doc', text: 'x'));

      expect(chunks.map((chunk) => chunk.text), <String>['1', '2', '3']);
    });
  });

  group('in-memory vector index', () {
    test('returns nearest chunks by cosine score', () {
      final chunks = <TextChunk>[
        const TextChunk(documentId: 'a', id: '0', text: 'alpha', tokenCount: 1),
        const TextChunk(documentId: 'b', id: '0', text: 'beta', tokenCount: 1),
      ];
      final index = InMemoryVectorIndex(dimensions: 2)
        ..add(chunks, <Float32List>[
          Float32List.fromList(<double>[1, 0]),
          Float32List.fromList(<double>[0, 1]),
        ]);

      final results = index.search(Float32List.fromList(<double>[0.8, 0.2]));

      expect(results.first.chunk.documentId, 'a');
      expect(results.first.score, greaterThan(results.last.score));
    });

    test('adds flat embedding batches without nested vector storage', () {
      final chunks = <TextChunk>[
        const TextChunk(documentId: 'a', id: '0', text: 'alpha', tokenCount: 1),
        const TextChunk(documentId: 'b', id: '0', text: 'beta', tokenCount: 1),
      ];
      final batch = EmbeddingBatch(
        count: 2,
        dimensions: 2,
        values: Float32List.fromList(<double>[1, 0, 0, 1]),
      );
      final index = InMemoryVectorIndex(dimensions: 2)..add(chunks, batch);

      expect(
        index.search(Float32List.fromList(<double>[1, 0])).first.chunk.text,
        'alpha',
      );
      expect(
        () => InMemoryVectorIndex(dimensions: 2).add(chunks, batch.take(1)),
        throwsArgumentError,
      );
    });

    test('orders equal vector scores deterministically', () {
      final index = InMemoryVectorIndex(dimensions: 2)
        ..add(
          const <TextChunk>[
            TextChunk(documentId: 'b', id: '1', text: 'beta', tokenCount: 1),
            TextChunk(documentId: 'a', id: '1', text: 'alpha', tokenCount: 1),
            TextChunk(documentId: 'a', id: '0', text: 'alpha', tokenCount: 1),
          ],
          <Float32List>[
            Float32List.fromList(<double>[1, 0]),
            Float32List.fromList(<double>[1, 0]),
            Float32List.fromList(<double>[1, 0]),
          ],
        );

      final results = index.search(Float32List.fromList(<double>[1, 0]));

      expect(
        results.map(
          (result) => '${result.chunk.documentId}:${result.chunk.id}',
        ),
        <String>['a:0', 'a:1', 'b:1'],
      );
    });

    test('does not partially add invalid batches', () {
      final index = InMemoryVectorIndex(dimensions: 2);

      expect(
        () => index.add(
          const <TextChunk>[
            TextChunk(documentId: 'a', id: '0', text: 'alpha', tokenCount: 1),
            TextChunk(documentId: 'b', id: '0', text: 'beta', tokenCount: 1),
          ],
          <Float32List>[
            Float32List.fromList(<double>[1, 0]),
            Float32List.fromList(<double>[double.nan, 0]),
          ],
        ),
        throwsArgumentError,
      );

      expect(index.isEmpty, isTrue);
    });

    test('rejects duplicate chunks in add batches', () {
      final index = InMemoryVectorIndex(dimensions: 2);

      expect(
        () => index.add(
          const <TextChunk>[
            TextChunk(documentId: 'a', id: '0', text: 'alpha', tokenCount: 1),
            TextChunk(documentId: 'a', id: '0', text: 'alpha', tokenCount: 1),
          ],
          <Float32List>[
            Float32List.fromList(<double>[1, 0]),
            Float32List.fromList(<double>[0, 1]),
          ],
        ),
        throwsArgumentError,
      );

      expect(index.isEmpty, isTrue);
    });

    test('retriever embeds query and searches the vector index', () async {
      final chunks = <TextChunk>[
        const TextChunk(documentId: 'a', id: '0', text: 'alpha', tokenCount: 1),
        const TextChunk(documentId: 'b', id: '0', text: 'beta', tokenCount: 1),
      ];
      final index = InMemoryVectorIndex(dimensions: 2)
        ..add(chunks, <Float32List>[
          Float32List.fromList(<double>[1, 0]),
          Float32List.fromList(<double>[0, 1]),
        ]);
      final retriever = VectorIndexRetriever(
        embeddingModel: _FakeEmbeddingModel(
          Float32List.fromList(<double>[0.9, 0.1]),
        ),
        index: index,
      );

      final results = await retriever.retrieve('query');

      expect(results.first.chunk.documentId, 'a');
    });

    test('retriever snapshots vector index dimensions once', () async {
      final index = _ChangingDimensionsVectorIndex();
      final retriever = VectorIndexRetriever(
        embeddingModel: _FakeEmbeddingModel(
          Float32List.fromList(<double>[1, 0]),
        ),
        index: index,
      );

      final results = await retriever.retrieve('query');

      expect(results.single.chunk.id, '0');
      expect(index.searchQueryLength, 2);
    });

    test('retriever rejects invalid search args before embedding', () async {
      final retriever = VectorIndexRetriever(
        embeddingModel: const _ThrowingEmbeddingModel(),
        index: InMemoryVectorIndex(dimensions: 2),
      );

      await expectLater(
        retriever.retrieve('query', topK: 0),
        throwsArgumentError,
      );
      await expectLater(
        retriever.retrieve('query', minScore: double.nan),
        throwsArgumentError,
      );
      await expectLater(
        retriever.retrieve('bad\u0000query'),
        throwsArgumentError,
      );
    });

    test('retriever rejects malformed vector index dimensions', () async {
      const retriever = VectorIndexRetriever(
        embeddingModel: _ThrowingEmbeddingModel(),
        index: _MalformedVectorIndex(dimensions: 0, isEmpty: true),
      );

      await expectLater(retriever.retrieve('query'), throwsArgumentError);
    });

    test('retriever rejects malformed vector index results', () async {
      final retriever = VectorIndexRetriever(
        embeddingModel: _FakeEmbeddingModel(
          Float32List.fromList(<double>[1, 0]),
        ),
        index: const _MalformedVectorIndex(),
      );

      await expectLater(retriever.retrieve('query'), throwsArgumentError);
      await expectLater(
        VectorIndexRetriever(
          embeddingModel: _FakeEmbeddingModel(
            Float32List.fromList(<double>[1, 0]),
          ),
          index: const _MalformedVectorIndex(
            resultCount: 2,
            score: 1,
            duplicateChunks: true,
          ),
        ).retrieve('query'),
        throwsArgumentError,
      );
    });

    test('retriever rejects vector indexes that ignore topK', () async {
      final retriever = VectorIndexRetriever(
        embeddingModel: _FakeEmbeddingModel(
          Float32List.fromList(<double>[1, 0]),
        ),
        index: const _MalformedVectorIndex(resultCount: 2, score: 1),
      );

      await expectLater(
        retriever.retrieve('query', topK: 1),
        throwsArgumentError,
      );
    });

    test('retriever rejects vector indexes that ignore minScore', () async {
      final retriever = VectorIndexRetriever(
        embeddingModel: _FakeEmbeddingModel(
          Float32List.fromList(<double>[1, 0]),
        ),
        index: const _MalformedVectorIndex(score: -1),
      );

      await expectLater(
        retriever.retrieve('query', minScore: 0),
        throwsArgumentError,
      );
    });

    test(
      'retriever rejects malformed embedding vectors before search',
      () async {
        await expectLater(
          VectorIndexRetriever(
            embeddingModel: _FakeEmbeddingModel(Float32List(1)),
            index: const _MalformedVectorIndex(),
          ).retrieve('query'),
          throwsArgumentError,
        );
        await expectLater(
          VectorIndexRetriever(
            embeddingModel: _FakeEmbeddingModel(
              Float32List.fromList(<double>[double.nan, 0]),
            ),
            index: const _MalformedVectorIndex(),
          ).retrieve('query'),
          throwsArgumentError,
        );
      },
    );

    test('retriever snapshots custom vector index results', () async {
      final tags = <Object?>['original'];
      final metadata = <String, Object?>{'title': 'fixture', 'tags': tags};
      final retriever = VectorIndexRetriever(
        embeddingModel: _FakeEmbeddingModel(
          Float32List.fromList(<double>[1, 0]),
        ),
        index: _MutableMetadataVectorIndex(metadata),
      );

      final results = await retriever.retrieve('query');
      metadata['title'] = 'changed';
      tags[0] = 'changed';

      final resultMetadata = results.single.chunk.metadata;
      expect(resultMetadata['title'], 'fixture');
      expect(resultMetadata['tags'], <Object?>['original']);
      expect(() => resultMetadata['title'] = 'changed', throwsUnsupportedError);
      final resultTags = resultMetadata['tags']! as List<Object?>;
      expect(() => resultTags[0] = 'changed', throwsUnsupportedError);
    });

    test('retriever skips embedding when vector index is empty', () async {
      final retriever = VectorIndexRetriever(
        embeddingModel: const _ThrowingEmbeddingModel(),
        index: InMemoryVectorIndex(dimensions: 2),
      );

      expect(await retriever.retrieve('query'), isEmpty);
    });

    test('retriever can rerank vector candidates', () async {
      final chunks = <TextChunk>[
        const TextChunk(documentId: 'a', id: '0', text: 'alpha', tokenCount: 1),
        const TextChunk(documentId: 'b', id: '0', text: 'beta', tokenCount: 1),
      ];
      final index = InMemoryVectorIndex(dimensions: 2)
        ..add(chunks, <Float32List>[
          Float32List.fromList(<double>[1, 0]),
          Float32List.fromList(<double>[0, 1]),
        ]);
      final retriever = VectorIndexRetriever(
        embeddingModel: _FakeEmbeddingModel(
          Float32List.fromList(<double>[0.9, 0.1]),
        ),
        index: index,
        reranker: const _ReverseReranker(),
      );

      final results = await retriever.retrieve('query');

      expect(results.first.chunk.documentId, 'b');
      expect(
        () => results.add(
          const VectorSearchResult(
            chunk: TextChunk(
              documentId: 'c',
              id: '0',
              text: 'gamma',
              tokenCount: 1,
            ),
            score: 0,
          ),
        ),
        throwsUnsupportedError,
      );
    });

    test('retriever rejects non-finite reranker scores', () async {
      final index = InMemoryVectorIndex(dimensions: 2)
        ..add(
          const <TextChunk>[
            TextChunk(documentId: 'a', id: '0', text: 'alpha', tokenCount: 1),
          ],
          <Float32List>[
            Float32List.fromList(<double>[1, 0]),
          ],
        );
      final retriever = VectorIndexRetriever(
        embeddingModel: _FakeEmbeddingModel(
          Float32List.fromList(<double>[1, 0]),
        ),
        index: index,
        reranker: const _NonFiniteReranker(),
      );

      await expectLater(retriever.retrieve('query'), throwsArgumentError);
    });

    test(
      'retriever rejects reranker chunks outside vector candidates',
      () async {
        final index = InMemoryVectorIndex(dimensions: 2)
          ..add(
            const <TextChunk>[
              TextChunk(documentId: 'a', id: '0', text: 'alpha', tokenCount: 1),
              TextChunk(documentId: 'b', id: '0', text: 'beta', tokenCount: 1),
            ],
            <Float32List>[
              Float32List.fromList(<double>[1, 0]),
              Float32List.fromList(<double>[0, 1]),
            ],
          );
        final embedding = _FakeEmbeddingModel(
          Float32List.fromList(<double>[1, 0]),
        );

        await expectLater(
          VectorIndexRetriever(
            embeddingModel: embedding,
            index: index,
            reranker: const _UnknownChunkReranker(),
          ).retrieve('query'),
          throwsArgumentError,
        );
        await expectLater(
          VectorIndexRetriever(
            embeddingModel: embedding,
            index: index,
            reranker: const _DuplicateReranker(),
          ).retrieve('query'),
          throwsArgumentError,
        );
        await expectLater(
          VectorIndexRetriever(
            embeddingModel: embedding,
            index: index,
            reranker: const _DroppingReranker(),
          ).retrieve('query'),
          throwsArgumentError,
        );
      },
    );

    test(
      'retriever skips reranking when vector search has no candidates',
      () async {
        final retriever = VectorIndexRetriever(
          embeddingModel: _FakeEmbeddingModel(
            Float32List.fromList(<double>[1, 0]),
          ),
          index: InMemoryVectorIndex(dimensions: 2),
          reranker: const _ThrowingReranker(),
        );

        final results = await retriever.retrieve('query');

        expect(results, isEmpty);
      },
    );

    test('removes generated chunk ids without touching other documents', () {
      final chunks = <TextChunk>[
        const TextChunk(
          documentId: 'a',
          id: 'a:0',
          text: 'alpha',
          tokenCount: 1,
        ),
        const TextChunk(
          documentId: 'b',
          id: 'b:0',
          text: 'beta',
          tokenCount: 1,
        ),
      ];
      final index = InMemoryVectorIndex(dimensions: 2)
        ..add(chunks, <Float32List>[
          Float32List.fromList(<double>[1, 0]),
          Float32List.fromList(<double>[0, 1]),
        ]);

      expect(index.remove('a:0'), isTrue);
      final results = index.search(Float32List.fromList(<double>[1, 0]));

      expect(results.single.chunk.documentId, 'b');
    });

    test('removes a chunk exactly when document id is provided', () {
      final index = InMemoryVectorIndex(dimensions: 2)
        ..add(
          const <TextChunk>[
            TextChunk(documentId: 'a', id: '0', text: 'alpha', tokenCount: 1),
            TextChunk(documentId: 'b', id: '0', text: 'beta', tokenCount: 1),
          ],
          <Float32List>[
            Float32List.fromList(<double>[1, 0]),
            Float32List.fromList(<double>[0, 1]),
          ],
        );

      expect(index.remove('0', documentId: 'a'), isTrue);
      final results = index.search(Float32List.fromList(<double>[0, 1]));

      expect(results.single.chunk.documentId, 'b');
      expect(index.remove('0', documentId: 'a'), isFalse);
    });

    test('rejects NUL bytes in vector index ids', () {
      final index = InMemoryVectorIndex(dimensions: 2);

      expect(
        () => index.add(
          const <TextChunk>[
            TextChunk(
              documentId: 'doc\u0000bad',
              id: '0',
              text: 'alpha',
              tokenCount: 1,
            ),
          ],
          <Float32List>[
            Float32List.fromList(<double>[1, 0]),
          ],
        ),
        throwsArgumentError,
      );
      expect(
        () => index.add(
          const <TextChunk>[
            TextChunk(
              documentId: 'doc',
              id: '0',
              text: 'bad\u0000',
              tokenCount: 1,
            ),
          ],
          <Float32List>[
            Float32List.fromList(<double>[1, 0]),
          ],
        ),
        throwsArgumentError,
      );
      expect(
        () => index.add(
          const <TextChunk>[
            TextChunk(documentId: 'doc', id: 'blank', text: ' ', tokenCount: 1),
          ],
          <Float32List>[
            Float32List.fromList(<double>[1, 0]),
          ],
        ),
        throwsArgumentError,
      );
      expect(
        () => index.add(
          const <TextChunk>[
            TextChunk(
              documentId: 'doc\nbad',
              id: '0',
              text: 'alpha',
              tokenCount: 1,
            ),
          ],
          <Float32List>[
            Float32List.fromList(<double>[1, 0]),
          ],
        ),
        throwsArgumentError,
      );
      expect(() => index.remove(''), throwsArgumentError);
      expect(() => index.remove('0', documentId: ''), throwsArgumentError);
      expect(() => index.removeDocument(''), throwsArgumentError);
    });

    test('rejects non-positive vector index token counts', () {
      final index = InMemoryVectorIndex(dimensions: 2);

      expect(
        () => index.add(
          const <TextChunk>[
            TextChunk(
              documentId: 'doc',
              id: '0',
              text: 'alpha',
              tokenCount: -1,
            ),
          ],
          <Float32List>[
            Float32List.fromList(<double>[1, 0]),
          ],
        ),
        throwsArgumentError,
      );
      expect(
        () => index.add(
          const <TextChunk>[
            TextChunk(
              documentId: 'doc',
              id: 'zero',
              text: 'alpha',
              tokenCount: 0,
            ),
          ],
          <Float32List>[
            Float32List.fromList(<double>[1, 0]),
          ],
        ),
        throwsArgumentError,
      );
      expect(
        () => InMemoryVectorIndex.fromJson(const <String, Object?>{
          'version': 1,
          'dimensions': 2,
          'records': <Object?>[
            <String, Object?>{
              'chunk': <String, Object?>{
                'documentId': 'doc',
                'id': 'json',
                'text': 'bad',
                'tokenCount': -1,
              },
              'vector': <Object?>[1, 0],
            },
          ],
        }),
        throwsFormatException,
      );
    });

    test('rejects non-finite vector search inputs', () {
      final index = InMemoryVectorIndex(dimensions: 2)
        ..add(
          const <TextChunk>[
            TextChunk(documentId: 'doc', id: '0', text: 'alpha', tokenCount: 1),
          ],
          <Float32List>[
            Float32List.fromList(<double>[1, 0]),
          ],
        );

      expect(
        () => index.add(
          const <TextChunk>[
            TextChunk(documentId: 'doc', id: '1', text: 'beta', tokenCount: 1),
          ],
          <Float32List>[
            Float32List.fromList(<double>[double.nan, 0]),
          ],
        ),
        throwsArgumentError,
      );
      expect(
        () => index.search(
          Float32List.fromList(<double>[1, 0]),
          minScore: double.nan,
        ),
        throwsArgumentError,
      );
      expect(
        () => index.search(Float32List.fromList(<double>[double.infinity, 0])),
        throwsArgumentError,
      );
      expect(
        () => InMemoryVectorIndex.fromJson(<String, Object?>{
          'version': 1,
          'dimensions': 2,
          'records': <Object?>[
            <String, Object?>{
              'chunk': <String, Object?>{
                'documentId': 'doc',
                'id': 'json',
                'text': 'bad',
                'tokenCount': 1,
              },
              'vector': <Object?>[double.nan, 0],
            },
          ],
        }),
        throwsFormatException,
      );
    });

    test('persists and loads vector records', () async {
      final temp = await Directory.systemTemp.createTemp('fllamer_index_');
      try {
        final path = '${temp.path}/index.json';
        final index = InMemoryVectorIndex(dimensions: 2)
          ..add(
            <TextChunk>[
              TextChunk(
                documentId: 'doc',
                id: '0',
                text: 'alpha',
                tokenCount: 1,
                metadata: const <String, Object?>{'title': 'fixture'},
                sourceUri: Uri.parse('file:///fixture.txt'),
              ),
            ],
            <Float32List>[
              Float32List.fromList(<double>[1, 0]),
            ],
          );

        await index.persist(path);
        final loaded = await InMemoryVectorIndex.load(path);
        final results = loaded.search(Float32List.fromList(<double>[1, 0]));

        expect(results.single.chunk.text, 'alpha');
        expect(results.single.chunk.metadata['title'], 'fixture');
        expect(
          results.single.chunk.sourceUri,
          Uri.parse('file:///fixture.txt'),
        );
      } finally {
        await temp.delete(recursive: true);
      }
    });

    test('rejects empty vector source URIs before indexing', () {
      expect(
        () => InMemoryVectorIndex(dimensions: 2).add(
          <TextChunk>[
            TextChunk(
              documentId: 'doc',
              id: '0',
              text: 'alpha',
              tokenCount: 1,
              sourceUri: Uri.parse(''),
            ),
          ],
          <Float32List>[
            Float32List.fromList(<double>[1, 0]),
          ],
        ),
        throwsArgumentError,
      );
      expect(
        () => InMemoryVectorIndex(dimensions: 2).add(
          <TextChunk>[
            TextChunk(
              documentId: 'doc',
              id: '0',
              text: 'alpha',
              tokenCount: 1,
              sourceUri: Uri.parse('file:///bad%00path'),
            ),
          ],
          <Float32List>[
            Float32List.fromList(<double>[1, 0]),
          ],
        ),
        throwsArgumentError,
      );
      expect(
        () => InMemoryVectorIndex(dimensions: 2).add(
          <TextChunk>[
            TextChunk(
              documentId: 'doc',
              id: '0',
              text: 'alpha',
              tokenCount: 1,
              sourceUri: Uri.parse('file:///bad%0Apath'),
            ),
          ],
          <Float32List>[
            Float32List.fromList(<double>[1, 0]),
          ],
        ),
        throwsArgumentError,
      );
    });

    test('rejects non-JSON vector metadata before indexing', () {
      final index = InMemoryVectorIndex(dimensions: 2);

      expect(
        () => index.add(
          <TextChunk>[
            TextChunk(
              documentId: 'doc',
              id: '0',
              text: 'alpha',
              tokenCount: 1,
              metadata: <String, Object?>{'bad': Object()},
            ),
          ],
          <Float32List>[
            Float32List.fromList(<double>[1, 0]),
          ],
        ),
        throwsArgumentError,
      );
    });

    test('rejects NUL bytes in vector metadata before indexing', () {
      final index = InMemoryVectorIndex(dimensions: 2);

      void addWithMetadata(Map<String, Object?> metadata) {
        index.add(
          <TextChunk>[
            TextChunk(
              documentId: 'doc',
              id: '0',
              text: 'alpha',
              tokenCount: 1,
              metadata: metadata,
            ),
          ],
          <Float32List>[
            Float32List.fromList(<double>[1, 0]),
          ],
        );
      }

      expect(
        () => addWithMetadata(<String, Object?>{'title': 'bad\u0000value'}),
        throwsArgumentError,
      );
      expect(
        () => addWithMetadata(<String, Object?>{'bad\u0000key': 'value'}),
        throwsArgumentError,
      );
    });

    test('rejects cyclic vector metadata before indexing', () {
      final metadata = <String, Object?>{};
      metadata['self'] = metadata;
      final index = InMemoryVectorIndex(dimensions: 2);

      expect(
        () => index.add(
          <TextChunk>[
            TextChunk(
              documentId: 'doc',
              id: '0',
              text: 'alpha',
              tokenCount: 1,
              metadata: metadata,
            ),
          ],
          <Float32List>[
            Float32List.fromList(<double>[1, 0]),
          ],
        ),
        throwsArgumentError,
      );
    });

    test('snapshots vector metadata at add and JSON export', () {
      final nested = <Object?>['fixture'];
      final metadata = <String, Object?>{'tags': nested};
      final index = InMemoryVectorIndex(dimensions: 2)
        ..add(
          <TextChunk>[
            TextChunk(
              documentId: 'doc',
              id: '0',
              text: 'alpha',
              tokenCount: 1,
              metadata: metadata,
            ),
          ],
          <Float32List>[
            Float32List.fromList(<double>[1, 0]),
          ],
        );

      nested[0] = 'changed';
      final results = index.search(Float32List.fromList(<double>[1, 0]));
      final storedTags =
          results.single.chunk.metadata['tags']! as List<Object?>;
      expect(storedTags.single, 'fixture');
      expect(() => storedTags[0] = 'changed', throwsUnsupportedError);

      final json = index.toJson();
      expect(() => json['records'] = <Object?>[], throwsUnsupportedError);
      final records = json['records']! as List<Object?>;
      expect(() => records.add(<String, Object?>{}), throwsUnsupportedError);
      final record = records.single! as Map<String, Object?>;
      expect(() => record['vector'] = <double>[0, 1], throwsUnsupportedError);
      final chunk = record['chunk']! as Map<String, Object?>;
      expect(() => chunk['id'] = 'changed', throwsUnsupportedError);
      final exportedVector = record['vector']! as List<Object?>;
      expect(() => exportedVector[0] = 0.5, throwsUnsupportedError);
      final exportedMetadata = chunk['metadata']! as Map<String, Object?>;
      final exportedTags = exportedMetadata['tags']! as List<Object?>;

      expect(exportedTags.single, 'fixture');
      expect(() => exportedTags[0] = 'changed', throwsUnsupportedError);
    });

    test('exports vector records deterministically', () {
      final index = InMemoryVectorIndex(dimensions: 2)
        ..add(
          const <TextChunk>[
            TextChunk(documentId: 'b', id: '0', text: 'beta', tokenCount: 1),
            TextChunk(documentId: 'a', id: '1', text: 'alpha', tokenCount: 1),
            TextChunk(documentId: 'a', id: '0', text: 'alpha', tokenCount: 1),
          ],
          <Float32List>[
            Float32List.fromList(<double>[0, 1]),
            Float32List.fromList(<double>[1, 0]),
            Float32List.fromList(<double>[1, 0]),
          ],
        );

      final records = index.toJson()['records']! as List<Object?>;

      expect(
        records.map((record) {
          final chunk =
              (record! as Map<String, Object?>)['chunk']!
                  as Map<String, Object?>;
          return '${chunk['documentId']}:${chunk['id']}';
        }),
        <String>['a:0', 'a:1', 'b:0'],
      );
    });

    test('rejects invalid persisted vector index paths', () async {
      final index = InMemoryVectorIndex(dimensions: 2);

      expect(() => index.persist(''), throwsArgumentError);
      expect(() => index.persist('bad\u0000path'), throwsArgumentError);
      expect(() => index.persist('bad\npath'), throwsArgumentError);
      await expectLater(InMemoryVectorIndex.load(''), throwsArgumentError);
      await expectLater(
        InMemoryVectorIndex.load('bad\u0000path'),
        throwsArgumentError,
      );
      await expectLater(
        InMemoryVectorIndex.load('bad\npath'),
        throwsArgumentError,
      );
    });

    test('maps persisted vector index file failures', () async {
      final temp = await Directory.systemTemp.createTemp('fllamer_index_');
      try {
        final index = InMemoryVectorIndex(dimensions: 2);

        await expectLater(
          InMemoryVectorIndex.load('${temp.path}/missing.json'),
          throwsA(isA<RagIndexException>()),
        );
        await expectLater(
          index.persist(temp.path),
          throwsA(isA<RagIndexException>()),
        );
        if (!Platform.isWindows) {
          final target = File('${temp.path}/target.json');
          await target.writeAsString('original');
          final link = Link('${temp.path}/linked.json');
          await link.create(target.path);
          await expectLater(
            index.persist(link.path),
            throwsA(isA<RagIndexException>()),
          );
          await expectLater(
            InMemoryVectorIndex.load(link.path),
            throwsA(isA<RagIndexException>()),
          );
          expect(await link.exists(), isTrue);
          expect(await target.readAsString(), 'original');
        }
      } finally {
        await temp.delete(recursive: true);
      }
    });

    test('labels malformed persisted vector index JSON', () async {
      final temp = await Directory.systemTemp.createTemp('fllamer_index_');
      try {
        final path = '${temp.path}/index.json';
        await File(path).writeAsString('{');

        await expectLater(
          InMemoryVectorIndex.load(path),
          throwsA(
            isA<FormatException>().having(
              (error) => error.message,
              'message',
              contains('vector index JSON is malformed'),
            ),
          ),
        );
      } finally {
        await temp.delete(recursive: true);
      }
    });

    test('rejects unsupported persisted vector index versions', () {
      expect(
        () => InMemoryVectorIndex.fromJson(const <String, Object?>{
          'version': 2,
          'dimensions': 2,
          'records': <Object?>[],
        }),
        throwsFormatException,
      );
    });

    test('rejects malformed persisted vector index headers', () {
      expect(
        () => InMemoryVectorIndex.fromJson(const <String, Object?>{
          'version': 1,
          'dimensions': '2',
        }),
        throwsFormatException,
      );
      expect(
        () => InMemoryVectorIndex.fromJson(const <String, Object?>{
          'version': 1,
          'dimensions': 0,
        }),
        throwsFormatException,
      );
      expect(
        () => InMemoryVectorIndex.fromJson(const <String, Object?>{
          'version': 1,
          'dimensions': 2,
          'normalize': null,
        }),
        throwsFormatException,
      );
      expect(
        () => InMemoryVectorIndex.fromJson(const <String, Object?>{
          'version': 1,
          'dimensions': 2,
          'normalize': 'true',
        }),
        throwsFormatException,
      );
      expect(
        () => InMemoryVectorIndex.fromJson(const <String, Object?>{
          'version': 1,
          'dimensions': 2,
          'extra': true,
        }),
        throwsFormatException,
      );
      expect(
        () => InMemoryVectorIndex.fromJson(const <String, Object?>{
          'version': 1,
          'dimensions': 2,
          'records': null,
        }),
        throwsFormatException,
      );
      expect(
        () => InMemoryVectorIndex.fromJson(const <String, Object?>{
          'version': 1,
          'dimensions': 2,
          'records': <String, Object?>{},
        }),
        throwsFormatException,
      );
    });

    test('rejects malformed persisted vector records', () {
      Map<String, Object?> indexWith(Object? record) {
        return <String, Object?>{
          'version': 1,
          'dimensions': 2,
          'records': <Object?>[record],
        };
      }

      expect(
        () => InMemoryVectorIndex.fromJson(indexWith('bad')),
        throwsFormatException,
      );
      expect(
        () => InMemoryVectorIndex.fromJson(
          indexWith(const <Object?, Object?>{1: 'bad'}),
        ),
        throwsFormatException,
      );
      expect(
        () => InMemoryVectorIndex.fromJson(
          indexWith(const <String, Object?>{'chunk': 'bad', 'vector': <int>[]}),
        ),
        throwsFormatException,
      );
      expect(
        () => InMemoryVectorIndex.fromJson(
          indexWith(const <String, Object?>{
            'chunk': <Object?, Object?>{1: 'bad'},
            'vector': <int>[],
          }),
        ),
        throwsFormatException,
      );
      expect(
        () => InMemoryVectorIndex.fromJson(
          indexWith(const <String, Object?>{
            'extra': true,
            'chunk': <String, Object?>{
              'documentId': 'doc',
              'id': '0',
              'text': 'alpha',
              'tokenCount': 1,
            },
            'vector': <int>[1, 0],
          }),
        ),
        throwsFormatException,
      );
      expect(
        () => InMemoryVectorIndex.fromJson(
          indexWith(const <String, Object?>{
            'chunk': <String, Object?>{
              'documentId': 'doc',
              'id': '0',
              'extra': true,
              'text': 'alpha',
              'tokenCount': 1,
            },
            'vector': <int>[1, 0],
          }),
        ),
        throwsFormatException,
      );
      expect(
        () => InMemoryVectorIndex.fromJson(const <String, Object?>{
          'version': 1,
          'dimensions': 2,
          'records': <Object?>[
            <String, Object?>{
              'chunk': <String, Object?>{
                'documentId': 'doc',
                'id': '0',
                'text': 'alpha',
                'tokenCount': 1,
              },
              'vector': <int>[1, 0],
            },
            <String, Object?>{
              'chunk': <String, Object?>{
                'documentId': 'doc',
                'id': '0',
                'text': 'alpha again',
                'tokenCount': 2,
              },
              'vector': <int>[0, 1],
            },
          ],
        }),
        throwsFormatException,
      );
      expect(
        () => InMemoryVectorIndex.fromJson(
          indexWith(const <String, Object?>{
            'chunk': <String, Object?>{
              'documentId': 'doc',
              'id': '0',
              'text': 'alpha',
              'tokenCount': 1,
              'sourceUri': 'file:///bad%0Dpath',
            },
            'vector': <int>[1, 0],
          }),
        ),
        throwsFormatException,
      );
      expect(
        () => InMemoryVectorIndex.fromJson(
          indexWith(const <String, Object?>{
            'chunk': <String, Object?>{
              'documentId': 'doc',
              'id': '0',
              'text': 'alpha',
              'tokenCount': 1,
              'metadata': null,
            },
            'vector': <int>[1, 0],
          }),
        ),
        throwsFormatException,
      );
      expect(
        () => InMemoryVectorIndex.fromJson(
          indexWith(const <String, Object?>{
            'chunk': <String, Object?>{
              'documentId': 'doc',
              'id': '0',
              'text': 'alpha',
              'tokenCount': 1,
              'metadata': <String, Object?>{'title': 'bad\u0000value'},
            },
            'vector': <int>[1, 0],
          }),
        ),
        throwsFormatException,
      );
      expect(
        () => InMemoryVectorIndex.fromJson(
          indexWith(const <String, Object?>{
            'chunk': <String, Object?>{
              'documentId': 'doc',
              'id': '0',
              'text': 'alpha',
              'tokenCount': 1,
              'metadata': <String, Object?>{'bad\u0000key': 'value'},
            },
            'vector': <int>[1, 0],
          }),
        ),
        throwsFormatException,
      );
      expect(
        () => InMemoryVectorIndex.fromJson(
          indexWith(const <String, Object?>{
            'chunk': <String, Object?>{
              'documentId': 'doc',
              'id': '0',
              'text': 'alpha',
              'tokenCount': 1,
              'sourceUri': '',
            },
            'vector': <int>[1, 0],
          }),
        ),
        throwsFormatException,
      );
      expect(
        () => InMemoryVectorIndex.fromJson(
          indexWith(const <String, Object?>{
            'chunk': <String, Object?>{
              'documentId': 'doc',
              'id': '0',
              'text': 'alpha',
              'tokenCount': '1',
            },
            'vector': <int>[1, 0],
          }),
        ),
        throwsFormatException,
      );
      expect(
        () => InMemoryVectorIndex.fromJson(
          indexWith(const <String, Object?>{
            'chunk': <String, Object?>{
              'documentId': 'doc',
              'id': '0',
              'text': ' ',
              'tokenCount': 1,
            },
            'vector': <int>[1, 0],
          }),
        ),
        throwsFormatException,
      );
      expect(
        () => InMemoryVectorIndex.fromJson(
          indexWith(const <String, Object?>{
            'chunk': <String, Object?>{
              'documentId': 'doc',
              'id': '0',
              'text': 'alpha',
              'tokenCount': 1,
              'sourceUri': 'file:///bad\u0000path',
            },
            'vector': <int>[1, 0],
          }),
        ),
        throwsFormatException,
      );
      expect(
        () => InMemoryVectorIndex.fromJson(
          indexWith(const <String, Object?>{
            'chunk': <String, Object?>{
              'documentId': 'doc',
              'id': '0',
              'text': 'alpha',
              'tokenCount': 1,
              'sourceUri': 'http://[::1',
            },
            'vector': <int>[1, 0],
          }),
        ),
        throwsFormatException,
      );
      expect(
        () => InMemoryVectorIndex.fromJson(
          indexWith(const <String, Object?>{
            'chunk': <String, Object?>{
              'documentId': 'doc',
              'id': '0',
              'text': 'alpha',
              'tokenCount': 1,
            },
            'vector': <Object?>[1],
          }),
        ),
        throwsFormatException,
      );
      expect(
        () => InMemoryVectorIndex.fromJson(
          indexWith(const <String, Object?>{
            'chunk': <String, Object?>{
              'documentId': 'doc',
              'id': '0',
              'text': 'alpha',
              'tokenCount': 1,
              'metadata': <String, Object?>{
                'nested': <Object?, Object?>{1: 'bad'},
              },
            },
            'vector': <int>[1, 0],
          }),
        ),
        throwsFormatException,
      );
      expect(
        () => InMemoryVectorIndex.fromJson(
          indexWith(<String, Object?>{
            'chunk': <String, Object?>{
              'documentId': 'doc',
              'id': '0',
              'text': 'alpha',
              'tokenCount': 1,
              'metadata': <String, Object?>{'bad': Object()},
            },
            'vector': <int>[1, 0],
          }),
        ),
        throwsFormatException,
      );
      expect(
        () => InMemoryVectorIndex.fromJson(
          indexWith(const <String, Object?>{
            'chunk': <String, Object?>{
              'documentId': 'doc',
              'id': '0',
              'text': 'alpha',
              'tokenCount': 1,
              'metadata': <Object?, Object?>{1: 'bad'},
            },
            'vector': <int>[1, 0],
          }),
        ),
        throwsFormatException,
      );
      expect(
        () => InMemoryVectorIndex.fromJson(
          indexWith(const <String, Object?>{
            'chunk': <String, Object?>{
              'documentId': 'doc',
              'id': '0',
              'text': 'alpha',
              'tokenCount': 1,
            },
            'vector': 'bad',
          }),
        ),
        throwsFormatException,
      );
      expect(
        () => InMemoryVectorIndex.fromJson(
          indexWith(const <String, Object?>{
            'chunk': <String, Object?>{
              'documentId': 'doc',
              'id': '0',
              'text': 'alpha',
              'tokenCount': 1,
            },
            'vector': <Object?>[1, 'bad'],
          }),
        ),
        throwsFormatException,
      );
    });

    test('rejects non-object persisted vector index roots', () async {
      final temp = await Directory.systemTemp.createTemp('fllamer_index_');
      try {
        final path = '${temp.path}/index.json';
        await File(path).writeAsString('[]');

        await expectLater(
          InMemoryVectorIndex.load(path),
          throwsFormatException,
        );
      } finally {
        await temp.delete(recursive: true);
      }
    });

    test('normalizes loaded vector records', () {
      final index = InMemoryVectorIndex.fromJson(const <String, Object?>{
        'version': 1,
        'dimensions': 2,
        'records': <Object?>[
          <String, Object?>{
            'chunk': <String, Object?>{
              'documentId': 'a',
              'id': '0',
              'text': 'scaled',
              'tokenCount': 1,
            },
            'vector': <Object?>[10, 0],
          },
          <String, Object?>{
            'chunk': <String, Object?>{
              'documentId': 'b',
              'id': '0',
              'text': 'unit',
              'tokenCount': 1,
            },
            'vector': <Object?>[0, 1],
          },
        ],
      });

      final results = index.search(Float32List.fromList(<double>[1, 0]));

      expect(results.first.score, 1);
      expect(results.first.score, greaterThan(results.last.score));
    });
  });

  group('RAG prompt builder', () {
    test('stops at the token budget', () {
      final prompt = const RagPromptBuilder().buildContext(const <TextChunk>[
        TextChunk(documentId: 'a', id: '0', text: 'one two', tokenCount: 2),
        TextChunk(documentId: 'b', id: '0', text: 'three', tokenCount: 1),
      ], maxContextTokens: 2);

      expect(prompt.usedTokens, 2);
      expect(prompt.includedChunks, hasLength(1));
      expect(prompt.citations.single.documentId, 'a');
      expect(prompt.citations.single.chunkId, '0');
      expect(
        () => prompt.includedChunks.add(
          const TextChunk(
            documentId: 'c',
            id: '0',
            text: 'extra',
            tokenCount: 1,
          ),
        ),
        throwsUnsupportedError,
      );
      expect(() => prompt.citations.clear(), throwsUnsupportedError);
      expect(prompt.context, contains('[source: a#0]'));
      expect(prompt.context, isNot(contains('three')));
    });

    test(
      'tokenizer-backed context budgets headers and source labels',
      () async {
        Future<List<int>> tokenize(String text) async {
          final count = text.trim().split(RegExp(r'\s+')).length;
          return List<int>.generate(count, (index) => index);
        }

        final prompt = await const RagPromptBuilder().buildContextWithTokenizer(
          const <TextChunk>[
            TextChunk(documentId: 'a', id: '0', text: 'one two', tokenCount: 2),
            TextChunk(documentId: 'b', id: '0', text: 'three', tokenCount: 1),
          ],
          maxContextTokens: 9,
          tokenize: tokenize,
        );

        expect(prompt.includedChunks.single.documentId, 'a');
        expect(prompt.context, isNot(contains('three')));
        expect(prompt.usedTokens, (await tokenize(prompt.context)).length);
        expect(prompt.usedTokens, 7);
        await expectLater(
          const RagPromptBuilder().buildContextWithTokenizer(
            const <TextChunk>[
              TextChunk(documentId: 'a', id: '0', text: 'one', tokenCount: 1),
            ],
            maxContextTokens: 8,
            tokenize: (_) async => const <int>[],
          ),
          throwsArgumentError,
        );
        await expectLater(
          const RagPromptBuilder().buildContextWithTokenizer(
            const <TextChunk>[
              TextChunk(documentId: 'a', id: '0', text: 'one', tokenCount: 1),
            ],
            maxContextTokens: 8,
            tokenize: (_) async => const <int>[-1],
          ),
          throwsArgumentError,
        );
      },
    );

    test('chat prompt budget counts the full formatted message list', () async {
      Future<int> countTokens(List<ChatMessage> messages) async {
        var count = 0;
        for (final message in messages) {
          count += 1 + message.text.trim().split(RegExp(r'\s+')).length;
        }
        return count;
      }

      final prompt = await const RagPromptBuilder().buildChatPrompt(
        question: 'What?',
        systemPrompt: 'Policy',
        chunks: const <TextChunk>[
          TextChunk(documentId: 'a', id: '0', text: 'one two', tokenCount: 2),
          TextChunk(documentId: 'b', id: '0', text: 'three', tokenCount: 1),
        ],
        maxPromptTokens: 14,
        countTokens: countTokens,
      );

      expect(prompt.usedTokens, await countTokens(prompt.messages));
      expect(prompt.usedTokens, 12);
      expect(prompt.includedChunks.single.documentId, 'a');
      expect(prompt.citations.single.chunkId, '0');
      expect(prompt.messages.map((message) => message.role), <ChatRole>[
        ChatRole.system,
        ChatRole.system,
        ChatRole.user,
      ]);
      expect(prompt.messages[1].text, isNot(contains('three')));
      expect(() => prompt.messages.clear(), throwsUnsupportedError);
      expect(() => prompt.includedChunks.clear(), throwsUnsupportedError);
      expect(() => prompt.citations.clear(), throwsUnsupportedError);

      await expectLater(
        const RagPromptBuilder().buildChatPrompt(
          question: 'What?',
          systemPrompt: 'Policy',
          chunks: const <TextChunk>[],
          maxPromptTokens: 3,
          countTokens: countTokens,
        ),
        throwsArgumentError,
      );
      await expectLater(
        const RagPromptBuilder().buildChatPrompt(
          question: 'What?',
          chunks: const <TextChunk>[],
          maxPromptTokens: 3,
          countTokens: (_) async => 0,
        ),
        throwsArgumentError,
      );
    });

    test('skips oversized chunks and keeps later fitting chunks', () {
      final prompt = const RagPromptBuilder().buildContext(const <TextChunk>[
        TextChunk(documentId: 'a', id: '0', text: 'too large', tokenCount: 4),
        TextChunk(documentId: 'b', id: '0', text: 'fits', tokenCount: 1),
      ], maxContextTokens: 2);

      expect(prompt.usedTokens, 1);
      expect(prompt.includedChunks.single.documentId, 'b');
      expect(prompt.context, contains('fits'));
      expect(prompt.context, isNot(contains('too large')));
    });

    test('builds chat messages from retrieved context', () {
      final messages = const RagPromptBuilder().buildChatMessages(
        question: 'What fits?',
        systemPrompt: 'Answer from local context.',
        chunks: <TextChunk>[
          TextChunk(documentId: 'a', id: '0', text: 'too large', tokenCount: 4),
          TextChunk(documentId: 'b', id: '0', text: 'fits', tokenCount: 1),
        ],
        maxContextTokens: 2,
      );

      expect(messages.map((message) => message.role), <ChatRole>[
        ChatRole.system,
        ChatRole.system,
        ChatRole.user,
      ]);
      expect(messages[1].text, contains('[source: b#0]'));
      expect(messages.last.text, 'What fits?');
      expect(
        () => const RagPromptBuilder().buildChatMessages(
          question: '',
          chunks: <TextChunk>[],
          maxContextTokens: 1,
        ),
        throwsArgumentError,
      );
    });

    test('preserves citation source spans', () {
      final sourceUri = Uri.parse('file:///fixture.txt');
      final prompt = const RagPromptBuilder().buildContext(<TextChunk>[
        TextChunk(
          documentId: 'doc',
          id: '0',
          text: 'alpha',
          tokenCount: 1,
          metadata: <String, Object?>{
            'start': 2,
            'end': 7,
            'tokenStart': 1,
            'tokenEnd': 2,
          },
          sourceUri: sourceUri,
        ),
      ], maxContextTokens: 1);

      final citation = prompt.citations.single;
      expect(citation.documentId, 'doc');
      expect(citation.chunkId, '0');
      expect(citation.sourceUri, sourceUri);
      expect(citation.start, 2);
      expect(citation.end, 7);
      expect(citation.tokenStart, 1);
      expect(citation.tokenEnd, 2);
    });

    test('snapshots included chunk metadata', () {
      final tags = <Object?>['fixture'];
      final metadata = <String, Object?>{'title': 'fixture', 'tags': tags};
      final prompt = const RagPromptBuilder().buildContext(<TextChunk>[
        TextChunk(
          documentId: 'doc',
          id: '0',
          text: 'alpha',
          tokenCount: 1,
          metadata: metadata,
        ),
      ], maxContextTokens: 1);
      metadata['title'] = 'changed';
      tags[0] = 'changed';

      expect(prompt.includedChunks.single.metadata['title'], 'fixture');
      final storedTags =
          prompt.includedChunks.single.metadata['tags']! as List<Object?>;
      expect(storedTags.single, 'fixture');
      expect(
        () => prompt.includedChunks.single.metadata['title'] = 'changed',
        throwsUnsupportedError,
      );
      expect(() => storedTags[0] = 'changed', throwsUnsupportedError);

      expect(
        () => const RagPromptBuilder().buildContext(<TextChunk>[
          TextChunk(
            documentId: 'doc',
            id: '0',
            text: 'alpha',
            tokenCount: 1,
            metadata: <String, Object?>{'bad': Object()},
          ),
        ], maxContextTokens: 1),
        throwsArgumentError,
      );

      final cyclicMetadata = <String, Object?>{};
      cyclicMetadata['self'] = cyclicMetadata;
      expect(
        () => const RagPromptBuilder().buildContext(<TextChunk>[
          TextChunk(
            documentId: 'doc',
            id: '0',
            text: 'alpha',
            tokenCount: 1,
            metadata: cyclicMetadata,
          ),
        ], maxContextTokens: 1),
        throwsArgumentError,
      );
    });

    test('rejects non-positive chunk token counts', () {
      expect(
        () => const RagPromptBuilder().buildContext(const <TextChunk>[
          TextChunk(documentId: 'a', id: '0', text: 'bad', tokenCount: -1),
        ], maxContextTokens: 2),
        throwsArgumentError,
      );
      expect(
        () => const RagPromptBuilder().buildContext(const <TextChunk>[
          TextChunk(documentId: 'a', id: '0', text: 'bad', tokenCount: 0),
        ], maxContextTokens: 2),
        throwsArgumentError,
      );
    });

    test('rejects empty chunk source ids', () {
      expect(
        () => const RagPromptBuilder().buildContext(const <TextChunk>[
          TextChunk(documentId: '', id: '0', text: 'bad', tokenCount: 1),
        ], maxContextTokens: 2),
        throwsArgumentError,
      );
      expect(
        () => const RagPromptBuilder().buildContext(const <TextChunk>[
          TextChunk(documentId: 'a\nb', id: '0', text: 'bad', tokenCount: 1),
        ], maxContextTokens: 2),
        throwsArgumentError,
      );
      expect(
        () => const RagPromptBuilder().buildContext(const <TextChunk>[
          TextChunk(documentId: 'a', id: '0', text: 'bad\u0000', tokenCount: 1),
        ], maxContextTokens: 2),
        throwsArgumentError,
      );
      expect(
        () => const RagPromptBuilder().buildContext(const <TextChunk>[
          TextChunk(documentId: 'a', id: '0', text: ' ', tokenCount: 1),
        ], maxContextTokens: 2),
        throwsArgumentError,
      );
      expect(
        () => const RagPromptBuilder().buildContext(<TextChunk>[
          TextChunk(
            documentId: 'a',
            id: '0',
            text: 'ok',
            tokenCount: 1,
            sourceUri: Uri(),
          ),
        ], maxContextTokens: 2),
        throwsArgumentError,
      );
      expect(
        () => RagPromptBuilder().buildContext(<TextChunk>[
          TextChunk(
            documentId: 'a',
            id: '0',
            text: 'ok',
            tokenCount: 1,
            sourceUri: Uri.parse('file:///bad%00path'),
          ),
        ], maxContextTokens: 2),
        throwsArgumentError,
      );
    });

    test('rejects invalid prompt headers', () {
      expect(
        () => const RagPromptBuilder(header: '  \n\t').buildContext(
          const <TextChunk>[
            TextChunk(documentId: 'a', id: '0', text: 'ok', tokenCount: 1),
          ],
          maxContextTokens: 2,
        ),
        throwsArgumentError,
      );
      expect(
        () => const RagPromptBuilder(header: 'bad\u0000header').buildContext(
          const <TextChunk>[
            TextChunk(documentId: 'a', id: '0', text: 'ok', tokenCount: 1),
          ],
          maxContextTokens: 2,
        ),
        throwsArgumentError,
      );
    });

    test('rejects invalid citation spans', () {
      expect(
        () => const RagPromptBuilder().buildContext(const <TextChunk>[
          TextChunk(
            documentId: 'a',
            id: '0',
            text: 'bad',
            tokenCount: 1,
            metadata: <String, Object?>{'start': '0'},
          ),
        ], maxContextTokens: 2),
        throwsArgumentError,
      );
      expect(
        () => const RagPromptBuilder().buildContext(const <TextChunk>[
          TextChunk(
            documentId: 'a',
            id: '0',
            text: 'bad',
            tokenCount: 1,
            metadata: <String, Object?>{'start': 0},
          ),
        ], maxContextTokens: 2),
        throwsArgumentError,
      );
      expect(
        () => const RagPromptBuilder().buildContext(const <TextChunk>[
          TextChunk(
            documentId: 'a',
            id: '0',
            text: 'bad',
            tokenCount: 1,
            metadata: <String, Object?>{'tokenEnd': 1},
          ),
        ], maxContextTokens: 2),
        throwsArgumentError,
      );
      expect(
        () => const RagPromptBuilder().buildContext(const <TextChunk>[
          TextChunk(
            documentId: 'a',
            id: '0',
            text: 'bad',
            tokenCount: 1,
            metadata: <String, Object?>{'start': 2, 'end': 1},
          ),
        ], maxContextTokens: 2),
        throwsArgumentError,
      );
      expect(
        () => const RagPromptBuilder().buildContext(const <TextChunk>[
          TextChunk(
            documentId: 'a',
            id: '0',
            text: 'bad',
            tokenCount: 1,
            metadata: <String, Object?>{'start': 1, 'end': 1},
          ),
        ], maxContextTokens: 2),
        throwsArgumentError,
      );
      expect(
        () => const RagPromptBuilder().buildContext(const <TextChunk>[
          TextChunk(
            documentId: 'a',
            id: '0',
            text: 'bad',
            tokenCount: 1,
            metadata: <String, Object?>{'tokenStart': 1, 'tokenEnd': 1},
          ),
        ], maxContextTokens: 2),
        throwsArgumentError,
      );
    });
  });
}

final class _FakeEmbeddingModel implements EmbeddingModel {
  const _FakeEmbeddingModel(this.embedding);

  final Float32List embedding;

  @override
  Future<Float32List> embedText(String text) async => embedding;
}

final class _ThrowingEmbeddingModel implements EmbeddingModel {
  const _ThrowingEmbeddingModel();

  @override
  Future<Float32List> embedText(String text) async {
    throw StateError('embedding should not run');
  }
}

final class _MalformedVectorIndex implements VectorIndex {
  const _MalformedVectorIndex({
    this.dimensions = 2,
    this.isEmpty = false,
    this.resultCount = 1,
    this.score = double.infinity,
    this.duplicateChunks = false,
  });

  @override
  final int dimensions;

  @override
  final bool isEmpty;

  final int resultCount;
  final double score;
  final bool duplicateChunks;

  @override
  void add(List<TextChunk> chunks, Iterable<Float32List> vectors) {
    throw UnsupportedError('not needed');
  }

  @override
  void clear() {}

  @override
  Future<void> persist(String path) async {}

  @override
  bool remove(String chunkId, {String? documentId}) => false;

  @override
  void removeDocument(String documentId) {}

  @override
  List<VectorSearchResult> search(
    Float32List query, {
    int topK = 5,
    double minScore = double.negativeInfinity,
  }) {
    return List<VectorSearchResult>.unmodifiable(<VectorSearchResult>[
      for (var i = 0; i < resultCount; i += 1)
        VectorSearchResult(
          chunk: TextChunk(
            documentId: 'a',
            id: duplicateChunks ? '0' : '$i',
            text: 'alpha',
            tokenCount: 1,
          ),
          score: score,
        ),
    ]);
  }
}

final class _ChangingDimensionsVectorIndex implements VectorIndex {
  int _reads = 0;
  int searchQueryLength = 0;

  @override
  int get dimensions {
    _reads += 1;
    return _reads == 1 ? 2 : 3;
  }

  @override
  bool get isEmpty => false;

  @override
  void add(List<TextChunk> chunks, Iterable<Float32List> vectors) {
    throw UnsupportedError('not needed');
  }

  @override
  void clear() {}

  @override
  Future<void> persist(String path) async {}

  @override
  bool remove(String chunkId, {String? documentId}) => false;

  @override
  void removeDocument(String documentId) {}

  @override
  List<VectorSearchResult> search(
    Float32List query, {
    int topK = 5,
    double minScore = double.negativeInfinity,
  }) {
    searchQueryLength = query.length;
    return const <VectorSearchResult>[
      VectorSearchResult(
        chunk: TextChunk(
          documentId: 'a',
          id: '0',
          text: 'alpha',
          tokenCount: 1,
        ),
        score: 1,
      ),
    ];
  }
}

final class _MutableMetadataVectorIndex implements VectorIndex {
  const _MutableMetadataVectorIndex(this.metadata);

  final Map<String, Object?> metadata;

  @override
  int get dimensions => 2;

  @override
  bool get isEmpty => false;

  @override
  void add(List<TextChunk> chunks, Iterable<Float32List> vectors) {
    throw UnsupportedError('not needed');
  }

  @override
  void clear() {}

  @override
  Future<void> persist(String path) async {}

  @override
  bool remove(String chunkId, {String? documentId}) => false;

  @override
  void removeDocument(String documentId) {}

  @override
  List<VectorSearchResult> search(
    Float32List query, {
    int topK = 5,
    double minScore = double.negativeInfinity,
  }) {
    return <VectorSearchResult>[
      VectorSearchResult(
        chunk: TextChunk(
          documentId: 'a',
          id: '0',
          text: 'alpha',
          tokenCount: 1,
          metadata: metadata,
        ),
        score: 1,
      ),
    ];
  }
}

final class _ReverseReranker implements Reranker {
  const _ReverseReranker();

  @override
  Future<List<VectorSearchResult>> rerank(
    String query,
    List<VectorSearchResult> results,
  ) async {
    return results.reversed.toList();
  }
}

final class _UnknownChunkReranker implements Reranker {
  const _UnknownChunkReranker();

  @override
  Future<List<VectorSearchResult>> rerank(
    String query,
    List<VectorSearchResult> results,
  ) async {
    return const <VectorSearchResult>[
      VectorSearchResult(
        chunk: TextChunk(
          documentId: 'other',
          id: '0',
          text: 'gamma',
          tokenCount: 1,
        ),
        score: 1,
      ),
    ];
  }
}

final class _DuplicateReranker implements Reranker {
  const _DuplicateReranker();

  @override
  Future<List<VectorSearchResult>> rerank(
    String query,
    List<VectorSearchResult> results,
  ) async {
    return <VectorSearchResult>[results.first, results.first];
  }
}

final class _DroppingReranker implements Reranker {
  const _DroppingReranker();

  @override
  Future<List<VectorSearchResult>> rerank(
    String query,
    List<VectorSearchResult> results,
  ) async {
    return <VectorSearchResult>[results.first];
  }
}

final class _ThrowingReranker implements Reranker {
  const _ThrowingReranker();

  @override
  Future<List<VectorSearchResult>> rerank(
    String query,
    List<VectorSearchResult> results,
  ) async {
    throw StateError('reranker should not run');
  }
}

final class _NonFiniteReranker implements Reranker {
  const _NonFiniteReranker();

  @override
  Future<List<VectorSearchResult>> rerank(
    String query,
    List<VectorSearchResult> results,
  ) async {
    return <VectorSearchResult>[
      VectorSearchResult(chunk: results.single.chunk, score: double.nan),
    ];
  }
}

final class _DeleteFailsFile implements File {
  const _DeleteFailsFile(this._delegate);

  final File _delegate;

  @override
  String get path => _delegate.path;

  @override
  Future<int> length() => _delegate.length();

  @override
  Future<RandomAccessFile> open({FileMode mode = FileMode.read}) {
    return _delegate.open(mode: mode);
  }

  @override
  Stream<List<int>> openRead([int? start, int? end]) {
    return _delegate.openRead(start, end);
  }

  @override
  Future<FileSystemEntity> delete({bool recursive = false}) {
    return Future<FileSystemEntity>.error(const FileSystemException('blocked'));
  }

  @override
  dynamic noSuchMethod(Invocation invocation) => super.noSuchMethod(invocation);
}

List<int> _snapshotTokenHistory(Uint8List snapshot) {
  if (snapshot.length < 64 ||
      ascii.decode(snapshot.sublist(0, 8)) != 'FLLAMERS') {
    throw const FormatException('Expected a versioned state snapshot.');
  }
  final data = ByteData.sublistView(snapshot);
  final targetSize = data.getUint64(24, Endian.little);
  final draftSize = data.getUint64(32, Endian.little);
  final speculativeSize = data.getUint64(40, Endian.little);
  final tokenCount = data.getUint64(48, Endian.little);
  final offset = 64 + targetSize + draftSize + speculativeSize;
  if (offset + tokenCount * 4 != snapshot.length) {
    throw const FormatException('State snapshot sections are malformed.');
  }
  return List<int>.generate(
    tokenCount,
    (index) => data.getInt32(offset + index * 4, Endian.little),
    growable: false,
  );
}

List<String> _vocabGoldenInputs(String raw) {
  const separator = '\n__ggml_vocab_test__\n';
  final inputs = <String>[];
  var offset = 0;
  while (offset < raw.length) {
    final next = raw.indexOf(separator, offset);
    if (next < 0) {
      inputs.add(raw.substring(offset));
      break;
    }
    inputs.add(raw.substring(offset, next));
    offset = next + separator.length;
  }
  return inputs;
}

List<List<int>> _vocabGoldenTokens(List<String> lines) {
  return <List<int>>[
    for (final line in lines)
      if (line.trim().isEmpty)
        const <int>[]
      else
        <int>[
          for (final token in line.trim().split(RegExp(r'\s+')))
            int.parse(token),
        ],
  ];
}

Future<String?> _buildAbiMismatchBridge() async {
  if (Platform.isWindows) {
    return null;
  }
  final temp = await Directory.systemTemp.createTemp('fllamer_bad_abi_');
  final source = File('${temp.path}${Platform.pathSeparator}bad_abi.c');
  final output =
      '${temp.path}${Platform.pathSeparator}libbad_abi$_nativeBridgeExtension';
  try {
    await source.writeAsString('''
#include <stdint.h>
uint32_t llama_dart_abi_version(void) { return 0; }
''');
    final args = Platform.isMacOS || Platform.isIOS
        ? <String>['-dynamiclib', source.path, '-o', output]
        : <String>['-shared', '-fPIC', source.path, '-o', output];
    final result = await Process.run('cc', args);
    await _requireSuccessfulFakeBridgeBuild(result, temp, source.path);
    addTearDown(() async {
      if (await temp.exists()) {
        await temp.delete(recursive: true);
      }
    });
    return output;
  } on ProcessException {
    await temp.delete(recursive: true);
    return null;
  }
}

Future<String?> _buildAbiOnlyBridge() async {
  if (Platform.isWindows) {
    return null;
  }
  final temp = await Directory.systemTemp.createTemp('fllamer_abi_only_');
  final source = File('${temp.path}${Platform.pathSeparator}abi_only.c');
  final output =
      '${temp.path}${Platform.pathSeparator}libabi_only'
      '$_nativeBridgeExtension';
  try {
    await source.writeAsString('''
#include "llama_dart.h"

LLAMA_DART_EXPORT uint32_t llama_dart_abi_version(void) {
  return LLAMA_DART_ABI_VERSION;
}
''');
    final include = Directory('native/llama_dart_bridge/include').absolute.path;
    final args = Platform.isMacOS || Platform.isIOS
        ? <String>['-dynamiclib', source.path, '-I', include, '-o', output]
        : <String>[
            '-shared',
            '-fPIC',
            source.path,
            '-I',
            include,
            '-o',
            output,
          ];
    final result = await Process.run('cc', args);
    await _requireSuccessfulFakeBridgeBuild(result, temp, source.path);
    addTearDown(() async {
      if (await temp.exists()) {
        await temp.delete(recursive: true);
      }
    });
    return output;
  } on ProcessException {
    await temp.delete(recursive: true);
    return null;
  }
}

Future<String?> _buildOversizedMetadataBridge() async {
  if (Platform.isWindows) {
    return null;
  }
  final temp = await Directory.systemTemp.createTemp('fllamer_metadata_cap_');
  final source = File('${temp.path}${Platform.pathSeparator}metadata_cap.c');
  final output =
      '${temp.path}${Platform.pathSeparator}libmetadata_cap'
      '$_nativeBridgeExtension';
  try {
    await source.writeAsString(r'''
#include "llama_dart.h"
#include <stdint.h>

static uintptr_t fake_model_storage;
static const char *last_error = "";

LLAMA_DART_EXPORT uint32_t llama_dart_abi_version(void) {
  return LLAMA_DART_ABI_VERSION;
}

LLAMA_DART_EXPORT llama_dart_result llama_dart_model_load(
    const llama_dart_model_load_config *config, llama_dart_model **out_model) {
  if (config == NULL || out_model == NULL) {
    last_error = "invalid model load arguments";
    return LLAMA_DART_ERROR_INVALID_ARGUMENT;
  }
  *out_model = (llama_dart_model *) &fake_model_storage;
  last_error = "";
  return LLAMA_DART_SUCCESS;
}

LLAMA_DART_EXPORT void llama_dart_model_free(llama_dart_model *model) {
  (void) model;
  last_error = "";
}

LLAMA_DART_EXPORT llama_dart_result llama_dart_model_metadata_count(
    const llama_dart_model *model, size_t *out_count) {
  if (model == NULL || out_count == NULL) {
    last_error = "invalid metadata count arguments";
    return LLAMA_DART_ERROR_INVALID_ARGUMENT;
  }
  *out_count = 65537;
  last_error = "";
  return LLAMA_DART_SUCCESS;
}

LLAMA_DART_EXPORT const char *llama_dart_last_error_message(void) {
  return last_error;
}
''');
    final include = Directory('native/llama_dart_bridge/include').absolute.path;
    final args = Platform.isMacOS || Platform.isIOS
        ? <String>['-dynamiclib', source.path, '-I', include, '-o', output]
        : <String>[
            '-shared',
            '-fPIC',
            source.path,
            '-I',
            include,
            '-o',
            output,
          ];
    final result = await Process.run('cc', args);
    await _requireSuccessfulFakeBridgeBuild(result, temp, source.path);
    addTearDown(() async {
      if (await temp.exists()) {
        await temp.delete(recursive: true);
      }
    });
    return output;
  } on ProcessException {
    await temp.delete(recursive: true);
    return null;
  }
}

Future<({String libraryPath, String markerPath})?>
_buildStreamingCaptureBridge() async {
  if (Platform.isWindows) {
    return null;
  }
  final temp = await Directory.systemTemp.createTemp('fllamer_streaming_');
  final source = File('${temp.path}${Platform.pathSeparator}streaming.c');
  final output =
      '${temp.path}${Platform.pathSeparator}libstreaming'
      '$_nativeBridgeExtension';
  final marker = File('${temp.path}${Platform.pathSeparator}steps.bin');
  try {
    await marker.writeAsBytes(const <int>[]);
    await source.writeAsString(r'''
#include "llama_dart.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static const char *last_error = "";
static uintptr_t fake_model_storage;
static uintptr_t fake_context_storage;
static uintptr_t fake_generation_storage;
static char marker_path[4096];
static char selected_chat_template[256];
static uint32_t generated_tokens;
static uint32_t generation_limit;
static int generation_active;
static int cancelled;
static uint8_t requested_load_mtp;
static uint8_t requested_reuse_prompt_prefix;

static llama_dart_result fail(llama_dart_result result, const char *message) {
  last_error = message;
  return result;
}

LLAMA_DART_EXPORT uint32_t llama_dart_abi_version(void) {
  return LLAMA_DART_ABI_VERSION;
}

LLAMA_DART_EXPORT const char *llama_dart_multimodal_marker(void) {
  return "<__media__>";
}

LLAMA_DART_EXPORT llama_dart_result llama_dart_model_load(
    const llama_dart_model_load_config *config, llama_dart_model **out_model) {
  if (config == NULL || out_model == NULL || config->model_path_data == NULL ||
      config->model_path_size == 0 ||
      config->model_path_size >= sizeof(marker_path)) {
    return fail(LLAMA_DART_ERROR_MODEL_LOAD, "invalid marker path");
  }
  memcpy(marker_path, config->model_path_data, config->model_path_size);
  marker_path[config->model_path_size] = '\0';
  const char *embedded_template = "fake-embedded-template";
  const size_t template_size = config->chat_template_size == 0
      ? strlen(embedded_template)
      : config->chat_template_size;
  if (template_size >= sizeof(selected_chat_template)) {
    return fail(LLAMA_DART_ERROR_MODEL_LOAD, "chat template is too large");
  }
  memcpy(selected_chat_template,
         config->chat_template_size == 0
             ? (const uint8_t *)embedded_template
             : config->chat_template_data,
         template_size);
  selected_chat_template[template_size] = '\0';
  requested_load_mtp = config->load_mtp;
  *out_model = (llama_dart_model *)&fake_model_storage;
  last_error = "";
  return LLAMA_DART_SUCCESS;
}

LLAMA_DART_EXPORT void llama_dart_model_free(llama_dart_model *model) {
  (void)model;
  last_error = "";
}

LLAMA_DART_EXPORT llama_dart_result llama_dart_model_get_chat_template(
    const llama_dart_model *model, llama_dart_buffer *out_template) {
  (void)model;
  const size_t size = strlen(selected_chat_template);
  out_template->data = (uint8_t *)malloc(size);
  if (out_template->data == NULL) {
    return fail(LLAMA_DART_ERROR_INTERNAL, "native allocation failed");
  }
  memcpy(out_template->data, selected_chat_template, size);
  out_template->size = size;
  last_error = "";
  return LLAMA_DART_SUCCESS;
}

LLAMA_DART_EXPORT llama_dart_result llama_dart_model_apply_chat_template(
    const llama_dart_model *model, const llama_dart_chat_message *messages,
    size_t message_count, uint8_t add_assistant_prompt,
    llama_dart_buffer *out_prompt) {
  (void)model;
  (void)add_assistant_prompt;
  if (messages == NULL || message_count != 1 || out_prompt == NULL) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT, "invalid chat messages");
  }
  out_prompt->size = messages[0].content_size;
  out_prompt->data = (uint8_t *)malloc(out_prompt->size);
  if (out_prompt->data == NULL) {
    return fail(LLAMA_DART_ERROR_INTERNAL, "native allocation failed");
  }
  memcpy(out_prompt->data, messages[0].content_data, out_prompt->size);
  last_error = "";
  return LLAMA_DART_SUCCESS;
}

LLAMA_DART_EXPORT llama_dart_result llama_dart_context_create(
    llama_dart_model *model, const llama_dart_context_config *config,
    llama_dart_context **out_context) {
  (void)model;
  if (out_context == NULL) {
    return fail(LLAMA_DART_ERROR_CONTEXT_CREATE, "missing context output");
  }
  const uint8_t expected_load_mtp =
      config->speculative_type == LLAMA_DART_SPECULATIVE_MTP &&
              config->speculative_model_path_size == 0
          ? 1
          : 0;
  if (requested_load_mtp != expected_load_mtp) {
    return fail(LLAMA_DART_ERROR_CONTEXT_CREATE,
                "integrated MTP load intent was not forwarded");
  }
  *out_context = (llama_dart_context *)&fake_context_storage;
  last_error = "";
  return LLAMA_DART_SUCCESS;
}

LLAMA_DART_EXPORT void llama_dart_context_free(llama_dart_context *context) {
  (void)context;
  usleep(50000);
  generation_active = 0;
  cancelled = 0;
  last_error = "";
}

LLAMA_DART_EXPORT llama_dart_result llama_dart_context_reset(
    llama_dart_context *context) {
  (void)context;
  if (generation_active) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "generation is still active");
  }
  usleep(50000);
  generated_tokens = 0;
  cancelled = 0;
  last_error = "";
  return LLAMA_DART_SUCCESS;
}

LLAMA_DART_EXPORT llama_dart_result llama_dart_context_cancel(
    llama_dart_context *context) {
  (void)context;
  cancelled = 1;
  last_error = "";
  return LLAMA_DART_SUCCESS;
}

LLAMA_DART_EXPORT llama_dart_result llama_dart_context_get_info(
    const llama_dart_context *context, llama_dart_context_info *out_info) {
  (void)context;
  if (cancelled) {
    return fail(LLAMA_DART_ERROR_CANCELLED, "stale cancellation");
  }
  memset(out_info, 0, sizeof(*out_info));
  out_info->struct_size = sizeof(*out_info);
  out_info->context_size = 128;
  out_info->sequence_context_size = 128;
  out_info->batch_size = 16;
  out_info->ubatch_size = 16;
  out_info->max_sequences = 1;
  out_info->used_tokens = generated_tokens;
  last_error = "";
  return LLAMA_DART_SUCCESS;
}

LLAMA_DART_EXPORT llama_dart_result llama_dart_generation_start(
    llama_dart_context *context, const llama_dart_completion_config *config,
    llama_dart_generation **out_generation) {
  (void)context;
  if (config == NULL || out_generation == NULL || config->max_tokens == 0 ||
      generation_active) {
    return fail(LLAMA_DART_ERROR_GENERATION, "invalid generation start");
  }
  generated_tokens = 0;
  generation_limit = config->max_tokens;
  generation_active = 1;
  cancelled = 0;
  requested_reuse_prompt_prefix = config->reuse_prompt_prefix;
  *out_generation = (llama_dart_generation *)&fake_generation_storage;
  last_error = "";
  return LLAMA_DART_SUCCESS;
}

LLAMA_DART_EXPORT llama_dart_result llama_dart_generation_next(
    llama_dart_generation *generation, llama_dart_buffer *out_text,
    llama_dart_completion_stats *out_stats, uint8_t *out_done) {
  (void)generation;
  if (!generation_active || out_text == NULL || out_stats == NULL ||
      out_done == NULL) {
    return fail(LLAMA_DART_ERROR_GENERATION, "invalid generation step");
  }
  if (cancelled) {
    return fail(LLAMA_DART_ERROR_CANCELLED, "generation cancelled");
  }
  FILE *marker = fopen(marker_path, "ab");
  if (marker == NULL) {
    return fail(LLAMA_DART_ERROR_GENERATION, "marker could not be opened");
  }
  const int base = requested_reuse_prompt_prefix != 0 ? 'A' : 'a';
  fputc(base + (int)generated_tokens, marker);
  fclose(marker);

  out_text->data = (uint8_t *)malloc(1);
  if (out_text->data == NULL) {
    return fail(LLAMA_DART_ERROR_INTERNAL, "native allocation failed");
  }
  out_text->data[0] = (uint8_t)(base + (int)generated_tokens);
  out_text->size = 1;
  ++generated_tokens;
  memset(out_stats, 0, sizeof(*out_stats));
  out_stats->struct_size = sizeof(*out_stats);
  out_stats->prompt_tokens = 2;
  out_stats->generated_tokens = generated_tokens;
  *out_done = generated_tokens >= generation_limit ? 1 : 0;
  if (*out_done != 0) {
    out_stats->stop_reason = LLAMA_DART_STOP_REASON_MAX_TOKENS;
  }
  last_error = "";
  return LLAMA_DART_SUCCESS;
}

LLAMA_DART_EXPORT void
llama_dart_generation_free(llama_dart_generation *generation) {
  (void)generation;
  generation_active = 0;
  cancelled = 0;
  last_error = "";
}

LLAMA_DART_EXPORT void llama_dart_buffer_free(uint8_t *data) {
  free(data);
  last_error = "";
}

LLAMA_DART_EXPORT const char *llama_dart_last_error_message(void) {
  return last_error;
}

LLAMA_DART_EXPORT void llama_dart_clear_last_error(void) {
  last_error = "";
}
''');
    final include = Directory('native/llama_dart_bridge/include').absolute.path;
    final args = Platform.isMacOS || Platform.isIOS
        ? <String>['-dynamiclib', source.path, '-I', include, '-o', output]
        : <String>[
            '-shared',
            '-fPIC',
            source.path,
            '-I',
            include,
            '-o',
            output,
          ];
    final result = await Process.run('cc', args);
    await _requireSuccessfulFakeBridgeBuild(result, temp, source.path);
    addTearDown(() async {
      if (await temp.exists()) {
        await temp.delete(recursive: true);
      }
    });
    return (libraryPath: output, markerPath: marker.path);
  } on ProcessException {
    await temp.delete(recursive: true);
    return null;
  }
}

Future<String?> _buildMultimodalCaptureBridge() async {
  if (Platform.isWindows) {
    return null;
  }
  final temp = await Directory.systemTemp.createTemp('fllamer_multimodal_');
  final source = File('${temp.path}${Platform.pathSeparator}multimodal.c');
  final output =
      '${temp.path}${Platform.pathSeparator}libmultimodal'
      '$_nativeBridgeExtension';
  try {
    await source.writeAsString(r'''
#include "llama_dart.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static const char *last_error = "";
static uintptr_t fake_model_storage;
static uintptr_t fake_context_storage;
static uintptr_t fake_generation_storage;
static uintptr_t fake_lora_storage;
static int warmed;
static int cancelled;
static uint32_t used_tokens;
static uint32_t prefill_calls;
static float active_lora_scale = -1.0f;

static llama_dart_result fail(llama_dart_result result, const char *message) {
  last_error = message;
  return result;
}

static size_t marker_count(const uint8_t *data, size_t size) {
  static const char marker[] = "<__media__>";
  const size_t marker_size = sizeof(marker) - 1;
  size_t count = 0;
  for (size_t i = 0; i + marker_size <= size; ++i) {
    if (memcmp(data + i, marker, marker_size) == 0) {
      ++count;
      i += marker_size - 1;
    }
  }
  return count;
}

LLAMA_DART_EXPORT uint32_t llama_dart_abi_version(void) {
  return LLAMA_DART_ABI_VERSION;
}

LLAMA_DART_EXPORT const char *llama_dart_multimodal_marker(void) {
  return "<__media__>";
}

LLAMA_DART_EXPORT llama_dart_result llama_dart_model_load(
    const llama_dart_model_load_config *config, llama_dart_model **out_model) {
  if (config->gpu_backend != LLAMA_DART_GPU_BACKEND_AUTO ||
      config->n_gpu_layers != -1) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT, "invalid GPU config");
  }
  last_error = "";
  *out_model = (llama_dart_model *)&fake_model_storage;
  return LLAMA_DART_SUCCESS;
}

LLAMA_DART_EXPORT void llama_dart_model_free(llama_dart_model *model) {
  (void)model;
  last_error = "";
}

LLAMA_DART_EXPORT llama_dart_result llama_dart_context_create(
    llama_dart_model *model, const llama_dart_context_config *config,
    llama_dart_context **out_context) {
  static const char expected[] = "mmproj.gguf";
  (void)model;
  if (config->mmproj_path_size != sizeof(expected) - 1 ||
      memcmp(config->mmproj_path_data, expected, sizeof(expected) - 1) != 0 ||
      config->mmproj_use_gpu != 1 ||
      config->kv_cache_key_type != LLAMA_DART_KV_CACHE_Q4_0 ||
      config->kv_cache_value_type != LLAMA_DART_KV_CACHE_Q8_0 ||
      config->flash_attention != LLAMA_DART_FLASH_ATTENTION_ENABLED ||
      config->kv_cache_offload != 0 || config->swa_full != 0 ||
      config->kv_unified != 1) {
    return fail(LLAMA_DART_ERROR_CONTEXT_CREATE, "invalid mmproj config");
  }
  last_error = "";
  *out_context = (llama_dart_context *)&fake_context_storage;
  return LLAMA_DART_SUCCESS;
}

LLAMA_DART_EXPORT void llama_dart_context_free(llama_dart_context *context) {
  (void)context;
  last_error = "";
}

LLAMA_DART_EXPORT llama_dart_result llama_dart_context_reset(
    llama_dart_context *context) {
  (void)context;
  cancelled = 0;
  used_tokens = 0;
  prefill_calls = 0;
  last_error = "";
  return LLAMA_DART_SUCCESS;
}

LLAMA_DART_EXPORT llama_dart_result llama_dart_context_cancel(
    llama_dart_context *context) {
  (void)context;
  cancelled = 1;
  last_error = "";
  return LLAMA_DART_SUCCESS;
}

LLAMA_DART_EXPORT llama_dart_result llama_dart_context_warm_up(
    llama_dart_context *context) {
  (void)context;
  warmed = 1;
  last_error = "";
  return LLAMA_DART_SUCCESS;
}

LLAMA_DART_EXPORT llama_dart_result llama_dart_lora_load(
    llama_dart_model *model, const llama_dart_lora_load_config *config,
    llama_dart_lora_adapter **out_adapter) {
  (void)model;
  (void)config;
  *out_adapter = (llama_dart_lora_adapter *)&fake_lora_storage;
  last_error = "";
  return LLAMA_DART_SUCCESS;
}

LLAMA_DART_EXPORT void llama_dart_lora_free(
    llama_dart_lora_adapter *adapter) {
  (void)adapter;
  last_error = "";
}

LLAMA_DART_EXPORT llama_dart_result llama_dart_context_set_lora_adapters(
    llama_dart_context *context, llama_dart_lora_adapter **adapters,
    const float *scales, size_t adapter_count) {
  (void)context;
  if (adapter_count == 0) {
    active_lora_scale = -1.0f;
  } else if (adapter_count == 1 &&
             adapters[0] == (llama_dart_lora_adapter *)&fake_lora_storage) {
    active_lora_scale = scales[0];
  } else {
    return fail(LLAMA_DART_ERROR_LORA, "invalid LoRA selection");
  }
  last_error = "";
  return LLAMA_DART_SUCCESS;
}

LLAMA_DART_EXPORT llama_dart_result llama_dart_context_get_info(
    const llama_dart_context *context, llama_dart_context_info *out_info) {
  (void)context;
  if (cancelled) {
    return fail(LLAMA_DART_ERROR_CANCELLED, "stale cancellation");
  }
  memset(out_info, 0, sizeof(*out_info));
  out_info->struct_size = sizeof(*out_info);
  out_info->context_size = 128;
  out_info->sequence_context_size = 128;
  out_info->batch_size = 16;
  out_info->ubatch_size = 16;
  out_info->max_sequences = 1;
  out_info->supports_vision = 1;
  out_info->supports_audio = 1;
  out_info->used_tokens = used_tokens;
  out_info->kv_cache_key_type = LLAMA_DART_KV_CACHE_Q4_0;
  out_info->kv_cache_value_type = LLAMA_DART_KV_CACHE_Q8_0;
  out_info->flash_attention = LLAMA_DART_FLASH_ATTENTION_ENABLED;
  out_info->kv_cache_offload = 0;
  out_info->swa_full = 0;
  out_info->kv_unified = 1;
  last_error = "";
  return LLAMA_DART_SUCCESS;
}

LLAMA_DART_EXPORT llama_dart_result llama_dart_model_tokenize(
    const llama_dart_model *model, const uint8_t *text_data, size_t text_size,
    int32_t *tokens, size_t tokens_capacity, size_t *out_token_count,
    uint8_t add_special, uint8_t parse_special) {
  (void)model;
  const size_t required = text_size + (add_special != 0 ? 1 : 0) +
      (parse_special != 0 ? 1 : 0);
  if (out_token_count == NULL || (text_size != 0 && text_data == NULL)) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT, "invalid tokenize input");
  }
  *out_token_count = required;
  if (tokens_capacity < required || (required != 0 && tokens == NULL)) {
    last_error = "token buffer too small";
    return LLAMA_DART_ERROR_BUFFER_TOO_SMALL;
  }
  size_t offset = 0;
  if (add_special != 0) {
    tokens[offset++] = 256;
  }
  if (parse_special != 0) {
    tokens[offset++] = 257;
  }
  for (size_t i = 0; i < text_size; ++i) {
    tokens[offset + i] = (int32_t)text_data[i];
  }
  last_error = "";
  return LLAMA_DART_SUCCESS;
}

LLAMA_DART_EXPORT llama_dart_result llama_dart_model_detokenize(
    const llama_dart_model *model, const int32_t *tokens, size_t token_count,
    uint8_t *text_data, size_t text_capacity, size_t *out_text_size,
    uint8_t remove_special, uint8_t unparse_special) {
  (void)model;
  if (out_text_size == NULL || (token_count != 0 && tokens == NULL)) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT, "invalid detokenize input");
  }
  size_t offset = 0;
  if (remove_special != 0 && token_count != 0 && tokens[offset] == 256) {
    ++offset;
  }
  if (unparse_special != 0 && offset < token_count &&
      tokens[offset] == 257) {
    ++offset;
  }
  const size_t required = token_count - offset;
  *out_text_size = required;
  if (text_capacity < required || (required != 0 && text_data == NULL)) {
    last_error = "text buffer too small";
    return LLAMA_DART_ERROR_BUFFER_TOO_SMALL;
  }
  for (size_t i = 0; i < required; ++i) {
    text_data[i] = (uint8_t)(tokens[offset + i] & 0xff);
  }
  last_error = "";
  return LLAMA_DART_SUCCESS;
}

LLAMA_DART_EXPORT llama_dart_result
llama_dart_model_get_chat_template_capabilities(
    const llama_dart_model *model,
    llama_dart_chat_template_capabilities *out_capabilities) {
  (void)model;
  if (out_capabilities == NULL) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT,
                "missing chat capabilities output");
  }
  memset(out_capabilities, 0, sizeof(*out_capabilities));
  out_capabilities->struct_size = sizeof(*out_capabilities);
  out_capabilities->supports_tools = 1;
  out_capabilities->supports_tool_calls = 1;
  out_capabilities->supports_parallel_tool_calls = 0;
  last_error = "";
  return LLAMA_DART_SUCCESS;
}

LLAMA_DART_EXPORT llama_dart_result llama_dart_model_apply_chat_template(
    const llama_dart_model *model, const llama_dart_chat_message *messages,
    size_t message_count, uint8_t add_assistant_prompt,
    llama_dart_buffer *out_prompt) {
  (void)model;
  (void)add_assistant_prompt;
  if (message_count != 1 || messages == NULL) {
    return fail(LLAMA_DART_ERROR_INVALID_ARGUMENT, "invalid messages");
  }
  out_prompt->size = messages[0].content_size;
  out_prompt->data = (uint8_t *)malloc(out_prompt->size);
  if (out_prompt->data == NULL) {
    return fail(LLAMA_DART_ERROR_INTERNAL, "native allocation failed");
  }
  memcpy(out_prompt->data, messages[0].content_data, out_prompt->size);
  last_error = "";
  return LLAMA_DART_SUCCESS;
}

LLAMA_DART_EXPORT llama_dart_result llama_dart_context_complete(
    llama_dart_context *context, const llama_dart_completion_config *config,
    llama_dart_buffer *out_text, llama_dart_completion_stats *out_stats) {
  (void)context;
  if (config->max_tokens != 0 || config->media_input_count != 0 ||
      config->prompt_size == 0 || out_text == NULL || out_stats == NULL) {
    return fail(LLAMA_DART_ERROR_GENERATION, "invalid prefill request");
  }
  if ((prefill_calls == 0 &&
       (config->add_special != LLAMA_DART_ADD_SPECIAL_IF_CONTEXT_EMPTY ||
        config->parse_special != 0)) ||
      (prefill_calls == 1 &&
       (config->add_special != LLAMA_DART_ADD_SPECIAL_NEVER ||
        config->parse_special != 1))) {
    return fail(LLAMA_DART_ERROR_GENERATION, "invalid prefill tokenization");
  }
  const uint32_t prompt_tokens = (uint32_t)config->prompt_size +
      ((config->add_special == LLAMA_DART_ADD_SPECIAL_ALWAYS ||
        (config->add_special == LLAMA_DART_ADD_SPECIAL_IF_CONTEXT_EMPTY &&
         used_tokens == 0)) ? 1u : 0u);
  used_tokens += prompt_tokens;
  ++prefill_calls;
  out_text->data = NULL;
  out_text->size = 0;
  memset(out_stats, 0, sizeof(*out_stats));
  out_stats->struct_size = sizeof(*out_stats);
  out_stats->prompt_tokens = prompt_tokens;
  out_stats->prompt_eval_ms = 2.5;
  out_stats->total_ms = 3.0;
  last_error = "";
  return LLAMA_DART_SUCCESS;
}

LLAMA_DART_EXPORT llama_dart_result llama_dart_generation_start(
    llama_dart_context *context, const llama_dart_completion_config *config,
    llama_dart_generation **out_generation) {
  static const uint8_t image[] = {1, 2, 3};
  static const char audio_path[] = "clip.wav";
  (void)context;
  if (!warmed || config->media_input_count != 2 ||
      config->add_special != LLAMA_DART_ADD_SPECIAL_IF_CONTEXT_EMPTY ||
      config->parse_special != 1 ||
      marker_count(config->prompt_data, config->prompt_size) != 2) {
    return fail(LLAMA_DART_ERROR_GENERATION, "invalid media count or prompt");
  }
  const llama_dart_media_input *media = config->media_inputs;
  if (media[0].type != LLAMA_DART_MEDIA_IMAGE ||
      media[0].content_size != sizeof(image) ||
      memcmp(media[0].content_data, image, sizeof(image)) != 0 ||
      media[0].path_data != NULL ||
      media[1].type != LLAMA_DART_MEDIA_AUDIO ||
      media[1].path_size != sizeof(audio_path) - 1 ||
      memcmp(media[1].path_data, audio_path, sizeof(audio_path) - 1) != 0 ||
      media[1].content_data != NULL) {
    return fail(LLAMA_DART_ERROR_GENERATION, "invalid media payload");
  }
  last_error = "";
  *out_generation = (llama_dart_generation *)&fake_generation_storage;
  return LLAMA_DART_SUCCESS;
}

LLAMA_DART_EXPORT llama_dart_result llama_dart_generation_next(
    llama_dart_generation *generation, llama_dart_buffer *out_text,
    llama_dart_completion_stats *out_stats, uint8_t *out_done) {
  if (active_lora_scale == 0.5f) {
    return fail(LLAMA_DART_ERROR_CANCELLED, "generation cancelled");
  }
  const char *text = active_lora_scale == 0.75f
      ? "override-ok"
      : active_lora_scale == 0.25f ? "global-ok" : "disabled-ok";
  const size_t text_size = strlen(text);
  (void)generation;
  out_text->size = text_size;
  out_text->data = (uint8_t *)malloc(out_text->size);
  if (out_text->data == NULL) {
    return fail(LLAMA_DART_ERROR_INTERNAL, "native allocation failed");
  }
  memcpy(out_text->data, text, text_size);
  memset(out_stats, 0, sizeof(*out_stats));
  out_stats->struct_size = sizeof(*out_stats);
  out_stats->prompt_tokens = 42;
  out_stats->generated_tokens = 1;
  *out_done = 1;
  last_error = "";
  return LLAMA_DART_SUCCESS;
}

LLAMA_DART_EXPORT void
llama_dart_generation_free(llama_dart_generation *generation) {
  (void)generation;
  cancelled = 0;
  last_error = "";
}

LLAMA_DART_EXPORT void llama_dart_buffer_free(uint8_t *data) {
  free(data);
  last_error = "";
}

LLAMA_DART_EXPORT const char *llama_dart_last_error_message(void) {
  return last_error;
}

LLAMA_DART_EXPORT void llama_dart_clear_last_error(void) {
  last_error = "";
}
''');
    final include = Directory('native/llama_dart_bridge/include').absolute.path;
    final args = Platform.isMacOS || Platform.isIOS
        ? <String>['-dynamiclib', source.path, '-I', include, '-o', output]
        : <String>[
            '-shared',
            '-fPIC',
            source.path,
            '-I',
            include,
            '-o',
            output,
          ];
    final result = await Process.run('cc', args);
    await _requireSuccessfulFakeBridgeBuild(result, temp, source.path);
    addTearDown(() async {
      if (await temp.exists()) {
        await temp.delete(recursive: true);
      }
    });
    return output;
  } on ProcessException {
    await temp.delete(recursive: true);
    return null;
  }
}

Future<String?> _buildContextFreeFailureBridge() async {
  if (Platform.isWindows) {
    return null;
  }
  final temp = await Directory.systemTemp.createTemp('fllamer_context_free_');
  final source = File('${temp.path}${Platform.pathSeparator}context_free.c');
  final output =
      '${temp.path}${Platform.pathSeparator}libcontext_free'
      '$_nativeBridgeExtension';
  try {
    await source.writeAsString('''
#include "llama_dart.h"

static const char *last_error = "";
static uintptr_t fake_model_storage;
static uintptr_t fake_context_storage;

LLAMA_DART_EXPORT uint32_t llama_dart_abi_version(void) {
  return LLAMA_DART_ABI_VERSION;
}

LLAMA_DART_EXPORT llama_dart_result llama_dart_model_load(
    const llama_dart_model_load_config *config, llama_dart_model **out_model) {
  (void)config;
  last_error = "";
  *out_model = (llama_dart_model *)&fake_model_storage;
  return LLAMA_DART_SUCCESS;
}

LLAMA_DART_EXPORT void llama_dart_model_free(llama_dart_model *model) {
  (void)model;
  last_error = "";
}

LLAMA_DART_EXPORT llama_dart_result llama_dart_context_create(
    llama_dart_model *model, const llama_dart_context_config *config,
    llama_dart_context **out_context) {
  (void)model;
  (void)config;
  last_error = "";
  *out_context = (llama_dart_context *)&fake_context_storage;
  return LLAMA_DART_SUCCESS;
}

LLAMA_DART_EXPORT llama_dart_result llama_dart_context_cancel(
    llama_dart_context *context) {
  (void)context;
  last_error = "";
  return LLAMA_DART_SUCCESS;
}

LLAMA_DART_EXPORT llama_dart_result llama_dart_context_shift(
    llama_dart_context *context, uint32_t keep_tokens,
    uint32_t discard_tokens, uint32_t *out_discarded_tokens) {
  (void)context;
  (void)keep_tokens;
  *out_discarded_tokens = discard_tokens == 0 ? 2 : discard_tokens;
  last_error = "";
  return LLAMA_DART_SUCCESS;
}

LLAMA_DART_EXPORT void llama_dart_context_free(llama_dart_context *context) {
  (void)context;
  last_error = "context free blocked";
}

LLAMA_DART_EXPORT const char *llama_dart_last_error_message(void) {
  return last_error;
}

LLAMA_DART_EXPORT void llama_dart_clear_last_error(void) {
  last_error = "";
}
''');
    final include = Directory('native/llama_dart_bridge/include').absolute.path;
    final args = Platform.isMacOS || Platform.isIOS
        ? <String>['-dynamiclib', source.path, '-I', include, '-o', output]
        : <String>[
            '-shared',
            '-fPIC',
            source.path,
            '-I',
            include,
            '-o',
            output,
          ];
    final result = await Process.run('cc', args);
    await _requireSuccessfulFakeBridgeBuild(result, temp, source.path);
    addTearDown(() async {
      if (await temp.exists()) {
        await temp.delete(recursive: true);
      }
    });
    return output;
  } on ProcessException {
    await temp.delete(recursive: true);
    return null;
  }
}

Future<String?> _buildModelFreeFailureBridge() async {
  if (Platform.isWindows) {
    return null;
  }
  final temp = await Directory.systemTemp.createTemp('fllamer_model_free_');
  final source = File('${temp.path}${Platform.pathSeparator}model_free.c');
  final output =
      '${temp.path}${Platform.pathSeparator}libmodel_free'
      '$_nativeBridgeExtension';
  try {
    await source.writeAsString('''
#include "llama_dart.h"
#include <string.h>

static const char *last_error = "";
static uintptr_t fake_model_storage;

LLAMA_DART_EXPORT uint32_t llama_dart_abi_version(void) {
  return LLAMA_DART_ABI_VERSION;
}

LLAMA_DART_EXPORT llama_dart_result llama_dart_model_load(
    const llama_dart_model_load_config *config, llama_dart_model **out_model) {
  (void)config;
  last_error = "";
  *out_model = (llama_dart_model *)&fake_model_storage;
  return LLAMA_DART_SUCCESS;
}

LLAMA_DART_EXPORT void llama_dart_model_free(llama_dart_model *model) {
  (void)model;
  last_error = "model free blocked";
}

LLAMA_DART_EXPORT llama_dart_result llama_dart_model_get_info(
    const llama_dart_model *model, llama_dart_model_info *out_info) {
  (void)model;
  last_error = "";
  memset(out_info, 0, sizeof(*out_info));
  out_info->struct_size = sizeof(*out_info);
  return LLAMA_DART_SUCCESS;
}

LLAMA_DART_EXPORT llama_dart_result llama_dart_model_get_description(
    const llama_dart_model *model, char *buffer, size_t buffer_size,
    size_t *out_size) {
  const char *description = "fake model";
  const size_t size = strlen(description);
  (void)model;
  last_error = "";
  *out_size = size;
  if (buffer_size <= size) {
    return LLAMA_DART_ERROR_BUFFER_TOO_SMALL;
  }
  memcpy(buffer, description, size + 1);
  return LLAMA_DART_SUCCESS;
}

LLAMA_DART_EXPORT const char *llama_dart_model_file_type_name(int32_t ftype) {
  (void)ftype;
  return "unknown";
}

LLAMA_DART_EXPORT const char *llama_dart_last_error_message(void) {
  return last_error;
}

LLAMA_DART_EXPORT void llama_dart_clear_last_error(void) {
  last_error = "";
}
''');
    final include = Directory('native/llama_dart_bridge/include').absolute.path;
    final args = Platform.isMacOS || Platform.isIOS
        ? <String>['-dynamiclib', source.path, '-I', include, '-o', output]
        : <String>[
            '-shared',
            '-fPIC',
            source.path,
            '-I',
            include,
            '-o',
            output,
          ];
    final result = await Process.run('cc', args);
    await _requireSuccessfulFakeBridgeBuild(result, temp, source.path);
    addTearDown(() async {
      if (await temp.exists()) {
        await temp.delete(recursive: true);
      }
    });
    return output;
  } on ProcessException {
    await temp.delete(recursive: true);
    return null;
  }
}

Future<void> _requireSuccessfulFakeBridgeBuild(
  ProcessResult result,
  Directory temporaryDirectory,
  String sourcePath,
) async {
  if (result.exitCode == 0) {
    return;
  }
  var diagnostics = '${result.stdout}\n${result.stderr}'.trim();
  if (diagnostics.length > 8192) {
    diagnostics = '${diagnostics.substring(0, 8192)}\n[output truncated]';
  }
  try {
    await temporaryDirectory.delete(recursive: true);
  } on FileSystemException {
    // Preserve the compiler failure, which is the actionable test result.
  }
  throw StateError(
    'Fake bridge compilation failed for $sourcePath '
    '(exit ${result.exitCode}):\n$diagnostics',
  );
}

String get _nativeBridgePath {
  if (Platform.isMacOS || Platform.isIOS) {
    return 'build/native/libllama_dart_bridge.dylib';
  }
  if (Platform.isWindows) {
    return r'build\native\llama_dart_bridge.dll';
  }
  return 'build/native/libllama_dart_bridge.so';
}

String get _nativeBridgeExtension {
  if (Platform.isMacOS || Platform.isIOS) {
    return '.dylib';
  }
  if (Platform.isWindows) {
    return '.dll';
  }
  return '.so';
}

double _squaredMagnitude(Float32List vector) {
  var result = 0.0;
  for (final value in vector) {
    result += value * value;
  }
  return result;
}

double _dotProduct(Float32List left, Float32List right) {
  if (left.length != right.length) {
    throw ArgumentError('vectors must have equal lengths');
  }
  var result = 0.0;
  for (var i = 0; i < left.length; i += 1) {
    result += left[i] * right[i];
  }
  return result;
}

double _maxAbsoluteDifference(Float32List left, Float32List right) {
  if (left.length != right.length) {
    throw ArgumentError('vectors must have equal lengths');
  }
  var result = 0.0;
  for (var i = 0; i < left.length; i += 1) {
    final difference = (left[i] - right[i]).abs();
    if (difference > result) {
      result = difference;
    }
  }
  return result;
}

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

import 'src/native_test_support.dart';
import '../hook/build.dart' as build_hook;

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
      final path = nativeBridgePath;
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
      final bridgePath = nativeBridgePath;
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
      final bridgePath = nativeBridgePath;
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
      final path = nativeBridgePath;
      if (!File(path).existsSync()) {
        markTestSkipped('native bridge has not been built at $path');
      }

      final source = File(path);
      final copy = File(
        '${source.parent.path}${Platform.pathSeparator}'
        'libllama_dart_bridge_path_test_'
        '${DateTime.now().microsecondsSinceEpoch}'
        '$nativeBridgeExtension ',
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
        'third_party/llama.cpp/vendor/hash/rotate-bits/LICENSE.md',
        'third_party/llama.cpp/vendor/hash/sha256/LICENSE',
        'third_party/llama.cpp/vendor/hash/xxhash/LICENSE',
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
      final path = nativeBridgePath;
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
          throwsUnsupportedFeature,
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
        final chunks = await request
            .where((chunk) => chunk.generatedTokens == null)
            .toList();

        expect(chunks, hasLength(1));
        expect(chunks.single.text, 'override-ok');
        expect(chunks.single.isDone, isTrue);
        expect(chunks.single.telemetry?.promptTokens, 42);
        await Future<void>.delayed(Duration.zero);
        expect((await engine.contextInfo()).contextSize, 128);
        expect((await engine.loraAdapters()).single.scale, 0.25);

        final global = await engine
            .chat(messages: messages)
            .where((chunk) => chunk.generatedTokens == null)
            .toList();
        expect(global.single.text, 'global-ok');
        final disabled = await engine
            .chat(
              messages: messages,
              config: const GenerationConfig(loraScales: <int, double>{}),
            )
            .where((chunk) => chunk.generatedTokens == null)
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
              .where((chunk) => chunk.generatedTokens == null)
              .toList(),
          throwsA(isA<CancelledException>()),
        );
        final restoredAfterCancellation = await engine
            .chat(messages: messages)
            .where((chunk) => chunk.generatedTokens == null)
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
              .where((chunk) => chunk.generatedTokens == null)
              .toList(),
          throwsA(isA<LoraException>()),
        );
        final restored = await engine
            .chat(messages: messages)
            .where((chunk) => chunk.generatedTokens == null)
            .toList();
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

        expect(
          reused.where((chunk) => chunk.generatedTokens == null).single.text,
          'A',
        );
        expect(
          reset.where((chunk) => chunk.generatedTokens == null).single.text,
          'a',
        );
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
                  if (chunk.generatedTokens != null) return;
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

    test(
      'completed native steps report progress before coalesced UTF-8 text',
      () async {
        final fixture = await _buildStreamingCaptureBridge(delayedUtf8: true);
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
          final chunks = <GenerationChunk>[];
          final firstProgress = Completer<void>();
          final done = Completer<void>();
          late final StreamSubscription<GenerationChunk> subscription;
          subscription = engine
              .complete(
                prompt: 'go',
                config: const GenerationConfig(
                  maxTokens: 6,
                  streamChunkTokens: 3,
                ),
              )
              .listen(
                (chunk) {
                  chunks.add(chunk);
                  if (chunk.generatedTokens == 1) {
                    subscription.pause();
                    firstProgress.complete();
                  }
                },
                onError: done.completeError,
                onDone: done.complete,
              );
          await firstProgress.future;
          expect(chunks.single.text, isEmpty);
          expect(chunks.single.isDone, isFalse);
          expect(chunks.single.telemetry, isNull);
          expect(chunks.single.assistantMessage, isNull);
          await Future<void>.delayed(const Duration(milliseconds: 80));
          // The second FFI call is blocked. No synthetic heartbeat or extra
          // stream request may appear while it has not completed.
          expect(chunks, hasLength(1));
          expect(await File(fixture.markerPath).length(), 1);
          // Resuming during an outstanding native batch must not request a
          // second batch. Pause again before the delayed call completes.
          subscription.resume();
          subscription.pause();
          await Future<void>.delayed(const Duration(milliseconds: 350));
          expect(await File(fixture.markerPath).length(), 3);
          expect(chunks, hasLength(1));
          subscription.resume();
          await done.future;
          expect(chunks.map((chunk) => chunk.generatedTokens), <int?>[
            1,
            2,
            3,
            null,
            4,
            5,
            6,
            null,
          ]);
          expect(chunks.map((chunk) => chunk.text), <String>[
            '',
            '',
            '',
            '€',
            '',
            '',
            '',
            '€',
          ]);
          expect(chunks.last.isDone, isTrue);
          await subscription.cancel();
        } finally {
          await engine.close();
        }
      },
    );

    test('unchanged native token counts are not progress', () async {
      final fixture = await _buildStreamingCaptureBridge(
        unchangedMiddleCount: true,
      );
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
        final chunks = await engine
            .complete(
              prompt: 'go',
              config: const GenerationConfig(
                maxTokens: 3,
                streamChunkTokens: 3,
              ),
            )
            .toList();
        expect(chunks.map((chunk) => chunk.generatedTokens), <int?>[
          1,
          3,
          null,
        ]);
        expect(chunks.last.text, 'abc');
      } finally {
        await engine.close();
      }
    });

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
                expect(chunk.generatedTokens, 1);
                expect(chunk.text, isEmpty);
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
      final path = nativeBridgePath;
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
      final path = nativeBridgePath;
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
      final bridgePath = nativeBridgePath;
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
      expect(info.maximumTokenPieceBytes, greaterThan(0));
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
      final bridgePath = nativeBridgePath;
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
      final path = nativeBridgePath;
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
      final path = nativeBridgePath;
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
        final bridgePath = nativeBridgePath;
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
      final bridgePath = nativeBridgePath;
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
      final bridgePath = nativeBridgePath;
      if (!File(bridgePath).existsSync()) {
        markTestSkipped('native bridge has not been built at $bridgePath');
      }
      final modelConfig = LlamaModelConfig(
        modelPath: modelPath,
        chatTemplate: 'chatml',
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
        final info = await engine.modelInfo();
        expect(info.maximumTokenPieceBytes, greaterThan(0));
        expect(
          (await engine.modelInfo()).maximumTokenPieceBytes,
          info.maximumTokenPieceBytes,
        );
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
        expect(
          baselineText.length,
          lessThanOrEqualTo(
            generation.maxTokens * info.maximumTokenPieceBytes!,
          ),
        );
        expect(
          baseline.last.telemetry?.generatedTokens,
          lessThanOrEqualTo(generation.maxTokens),
        );
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
        final steppedText = stepped
            .where((chunk) => chunk.generatedTokens == null)
            .toList();
        expect(steppedText, hasLength(2));
        expect(steppedText.first.isDone, isFalse);
        expect(steppedText.first.text, isNotEmpty);
        final progress = stepped
            .map((chunk) => chunk.generatedTokens)
            .whereType<int>()
            .toList();
        expect(progress, orderedEquals(<int>[1, 2]));
        expect(stepped.last.isDone, isTrue);
        expect(stepped.last.telemetry?.generatedTokens, inInclusiveRange(1, 2));
        final maximumPromptBytes =
            modelConfig.contextSize * info.maximumTokenPieceBytes!;
        for (final mode in <bool>[false, true]) {
          await engine.reset();
          final messages = <ChatMessage>[ChatMessage.user('Hello')];
          final count = await engine.countChatTokens(
            messages,
            enableThinking: mode,
            maximumPromptBytes: maximumPromptBytes,
          );
          final formatted = await engine.formatChat(
            messages,
            enableThinking: mode,
            maximumPromptBytes: maximumPromptBytes,
          );
          expect(
            count,
            await engine.countTokens(
              formatted,
              addSpecial: true,
              parseSpecial: true,
            ),
          );
          final reply = await engine
              .chat(
                messages: messages,
                maximumPromptBytes: maximumPromptBytes,
                config: GenerationConfig(
                  maxTokens: 1,
                  enableThinking: mode,
                  temperature: 0,
                ),
              )
              .toList();
          expect(reply.last.telemetry!.promptTokens, count);
        }
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
        final bridgePath = nativeBridgePath;
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
      final bridgePath = nativeBridgePath;
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
        final bridgePath = nativeBridgePath;
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
      final bridgePath = nativeBridgePath;
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
        final bridgePath = nativeBridgePath;
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
      final bridgePath = nativeBridgePath;
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
      final bridgePath = nativeBridgePath;
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
      final bridgePath = nativeBridgePath;
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
        final bridgePath = nativeBridgePath;
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
      final bridgePath = nativeBridgePath;
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
      final bridgePath = nativeBridgePath;
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
      final bridgePath = nativeBridgePath;
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
      final bridgePath = nativeBridgePath;
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
      final bridgePath = nativeBridgePath;
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
      final bridgePath = nativeBridgePath;
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
      final bridgePath = nativeBridgePath;
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
      final bridgePath = nativeBridgePath;
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
      final bridgePath = nativeBridgePath;
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

    test(
      'explicit thinking counts match native plan rendering for vocabulary fixtures',
      () async {
        if (!File(nativeBridgePath).existsSync()) {
          markTestSkipped(
            'native bridge has not been built at $nativeBridgePath',
          );
          return;
        }
        for (final vocab in <String>['gpt-2', 'gemma-4', 'qwen35']) {
          final path = 'third_party/llama.cpp/models/ggml-vocab-$vocab.gguf';
          if (!File(path).existsSync()) {
            markTestSkipped('vocab fixture is missing at $path');
            return;
          }
        }
        for (final vocab in <String>['gpt-2', 'gemma-4', 'qwen35']) {
          final config = LlamaModelConfig(
            modelPath: 'third_party/llama.cpp/models/ggml-vocab-$vocab.gguf',
            nativeLibraryPath: nativeBridgePath,
            chatTemplate: vocab == 'gpt-2' ? 'chatml' : null,
          );
          for (final mode in <bool>[false, true]) {
            final messages = <ChatMessage>[
              ChatMessage.system('Use the selected language.'),
              ChatMessage.user('Explain季節の変化 with an example.'),
            ];
            final formatted = await LlamaChatTemplate.format(
              config,
              messages,
              enableThinking: mode,
            );
            final count = await LlamaChatTemplate.countTokens(
              config,
              messages,
              enableThinking: mode,
            );
            expect(
              count,
              await LlamaTokenizer.countTokens(
                config,
                formatted,
                addSpecial: true,
                parseSpecial: true,
              ),
              reason: '$vocab thinking=$mode',
            );
          }
        }
      },
    );

    test('counts chat-template tokens with the upstream tokenizer', () async {
      final bridgePath = nativeBridgePath;
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
      '${temp.path}${Platform.pathSeparator}libbad_abi$nativeBridgeExtension';
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
      '$nativeBridgeExtension';
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
      '$nativeBridgeExtension';
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
_buildStreamingCaptureBridge({
  bool delayedUtf8 = false,
  bool unchangedMiddleCount = false,
}) async {
  if (Platform.isWindows) {
    return null;
  }
  final temp = await Directory.systemTemp.createTemp('fllamer_streaming_');
  final source = File('${temp.path}${Platform.pathSeparator}streaming.c');
  final output =
      '${temp.path}${Platform.pathSeparator}libstreaming'
      '$nativeBridgeExtension';
  final marker = File('${temp.path}${Platform.pathSeparator}steps.bin');
  try {
    await marker.writeAsBytes(const <int>[]);
    await source.writeAsString(
      '#define DELAYED_UTF8 ${delayedUtf8 ? 1 : 0}\n'
      '#define UNCHANGED_MIDDLE_COUNT ${unchangedMiddleCount ? 1 : 0}\n'
      r'''
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
  if (DELAYED_UTF8 && generated_tokens == 1) usleep(300000);
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
  static const uint8_t utf8_pieces[] = {0xe2, 0x82, 0xac};
  out_text->data[0] = DELAYED_UTF8
      ? utf8_pieces[generated_tokens % 3]
      : (uint8_t)(base + (int)generated_tokens);
  out_text->size = 1;
  ++generated_tokens;
  memset(out_stats, 0, sizeof(*out_stats));
  out_stats->struct_size = sizeof(*out_stats);
  out_stats->prompt_tokens = 2;
  out_stats->generated_tokens =
      UNCHANGED_MIDDLE_COUNT && generated_tokens == 2 ? 1 : generated_tokens;
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
''',
    );
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
      '$nativeBridgeExtension';
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
      '$nativeBridgeExtension';
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
      '$nativeBridgeExtension';
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
  out_info->maximum_token_piece_bytes = 256;
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

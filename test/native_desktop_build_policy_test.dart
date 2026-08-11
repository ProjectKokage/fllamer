import 'dart:io';

import 'package:code_assets/code_assets.dart';
import 'package:hooks/hooks.dart';
import 'package:test/test.dart';

import '../hook/build.dart' as build_hook;

void main() {
  group('Vulkan build policy', () {
    test('is strict-by-default only on Linux and Windows', () {
      expect(
        build_hook.vulkanEnabledForNativeAssetsBuild(OS.linux, null),
        isTrue,
      );
      expect(
        build_hook.vulkanEnabledForNativeAssetsBuild(OS.windows, null),
        isTrue,
      );
      expect(
        build_hook.vulkanEnabledForNativeAssetsBuild(OS.macOS, null),
        isFalse,
      );
      expect(
        build_hook.vulkanEnabledForNativeAssetsBuild(OS.android, null),
        isFalse,
      );
    });

    test('supports an explicit reproducible CPU-only build', () {
      expect(
        build_hook.vulkanEnabledForNativeAssetsBuild(OS.linux, false),
        isFalse,
      );
      expect(
        build_hook.vulkanEnabledForNativeAssetsBuild(OS.windows, false),
        isFalse,
      );
      expect(
        build_hook.vulkanEnabledForNativeAssetsBuild(OS.windows, true),
        isTrue,
      );
      expect(
        build_hook.vulkanEnabledForNativeAssetsBuild(OS.android, false),
        isFalse,
      );
      expect(
        build_hook.vulkanEnabledForNativeAssetsBuild(OS.android, true),
        isTrue,
      );
    });

    test('rejects non-boolean Vulkan policy values', () {
      expect(
        () => build_hook.vulkanEnabledForNativeAssetsBuild(OS.windows, 'true'),
        throwsA(
          isA<BuildError>().having(
            (error) => error.message,
            'message',
            contains('must be a boolean'),
          ),
        ),
      );
    });

    test('passes the effective policy and optional SDK root to CMake', () {
      final sdk = Uri.directory('/opt/vulkan-sdk');
      final overlay = Uri.directory('/tmp/fllamer-vulkan-overlay');

      expect(
        build_hook.vulkanCmakeArgsForNativeAssetsBuild(enableVulkan: true),
        contains('-DLLAMA_DART_ENABLE_VULKAN=ON'),
      );
      expect(
        build_hook.vulkanCmakeArgsForNativeAssetsBuild(
          enableVulkan: false,
          android: true,
        ),
        containsAll(<String>[
          '-DLLAMA_DART_ENABLE_VULKAN=OFF',
          '-ULLAMA_DART_ANDROID_VULKAN_HOST_ROOT',
          '-ULLAMA_DART_ANDROID_VULKAN_SHADER_OVERLAY_DIR',
          '-UVulkan_GLSLC_EXECUTABLE',
        ]),
      );
      expect(
        build_hook.vulkanCmakeArgsForNativeAssetsBuild(
          enableVulkan: true,
          vulkanSdk: sdk,
        ),
        contains('-DVulkan_ROOT=/opt/vulkan-sdk/'),
      );
      expect(
        build_hook.vulkanCmakeArgsForNativeAssetsBuild(
          enableVulkan: true,
          vulkanSdk: sdk,
          androidVulkanGlslc: sdk.resolve('bin/glslc'),
          androidVulkanShaderOverlay: overlay,
          android: true,
        ),
        containsAll([
          '-DLLAMA_DART_ANDROID_VULKAN_HOST_ROOT=/opt/vulkan-sdk/',
          '-DVulkan_GLSLC_EXECUTABLE=/opt/vulkan-sdk/bin/glslc',
          '-DLLAMA_DART_ANDROID_VULKAN_SHADER_OVERLAY_DIR='
              '/tmp/fllamer-vulkan-overlay/',
        ]),
      );
      expect(
        build_hook.vulkanCmakeArgsForNativeAssetsBuild(
          enableVulkan: true,
          vulkanSdk: sdk,
          androidVulkanGlslc: sdk.resolve('bin/glslc'),
          androidVulkanShaderOverlay: overlay,
          android: true,
        ),
        isNot(contains('-DVulkan_ROOT=/opt/vulkan-sdk/')),
      );

      expect(
        () => build_hook.vulkanCmakeArgsForNativeAssetsBuild(
          enableVulkan: true,
          vulkanSdk: sdk,
          androidVulkanGlslc: sdk.resolve('bin/glslc'),
          android: true,
        ),
        throwsA(
          isA<BuildError>().having(
            (error) => error.message,
            'message',
            contains('shader overlay'),
          ),
        ),
      );
      expect(
        () => build_hook.vulkanCmakeArgsForNativeAssetsBuild(
          enableVulkan: false,
          androidVulkanShaderOverlay: overlay,
          android: true,
        ),
        throwsA(isA<BuildError>()),
      );
      expect(
        () => build_hook.vulkanCmakeArgsForNativeAssetsBuild(
          enableVulkan: true,
          androidVulkanShaderOverlay: overlay,
        ),
        throwsA(isA<BuildError>()),
      );
    });

    test('resolves an absolute Windows SDK user-define as a file URI', () {
      final sdk = build_hook.resolveVulkanSdkUserDefine(
        r'C:\VulkanSDK\1.4.321.0',
        Uri.parse('c:/VulkanSDK/1.4.321.0'),
        OS.windows,
      );

      expect(sdk!.isScheme('file'), isTrue);
      expect(sdk.toFilePath(windows: true), endsWith(r'VulkanSDK\1.4.321.0\'));

      final androidSdk = build_hook.resolveVulkanSdkUserDefine(
        r'C:\VulkanSDK\1.4.321.0',
        Uri.parse('c:/VulkanSDK/1.4.321.0'),
        OS.android,
      );
      expect(
        build_hook.vulkanCmakeArgsForNativeAssetsBuild(
          enableVulkan: true,
          vulkanSdk: androidSdk,
          androidVulkanGlslc: androidSdk!.resolve('Bin/glslc.exe'),
          androidVulkanShaderOverlay: androidSdk.resolve(
            'generated/android-vulkan-shaders/',
          ),
          android: true,
        ),
        contains(
          '-DLLAMA_DART_ANDROID_VULKAN_HOST_ROOT='
          'C:/VulkanSDK/1.4.321.0/',
        ),
      );
    });

    test('validates an explicitly configured SDK directory', () async {
      final temp = await Directory.systemTemp.createTemp('fllamer_vulkan_sdk_');
      try {
        final header = File('${temp.path}/include/vulkan/vulkan.h');
        final hppHeader = File('${temp.path}/include/vulkan/vulkan.hpp');
        final videoHeader = File(
          '${temp.path}/include/vk_video/vulkan_video_codecs_common.h',
        );
        final glslc = File('${temp.path}/bin/glslc');
        final loader = File('${temp.path}/lib/libvulkan.so');
        final spirvConfig = File(
          '${temp.path}/share/cmake/SPIRV-Headers/'
          'SPIRV-HeadersConfig.cmake',
        );
        final unusedDoc = File('${temp.path}/Documentation/guide.pdf');
        for (final file in [
          header,
          hppHeader,
          videoHeader,
          glslc,
          loader,
          spirvConfig,
          unusedDoc,
        ]) {
          await file.create(recursive: true);
        }

        expect(
          build_hook.vulkanSdkForNativeAssetsBuild(
            temp.path,
            temp.uri,
            enabled: true,
            targetOS: OS.linux,
          ),
          temp.uri,
        );
        expect(
          build_hook.vulkanSdkForNativeAssetsBuild(
            temp.path,
            temp.uri,
            enabled: false,
            targetOS: OS.linux,
          ),
          isNull,
        );
        expect(
          build_hook.vulkanSdkForNativeAssetsBuild(
            temp.path,
            temp.uri,
            enabled: true,
            targetOS: OS.android,
          ),
          temp.uri,
        );
        expect(
          () => build_hook.vulkanSdkForNativeAssetsBuild(
            1,
            null,
            enabled: true,
            targetOS: OS.linux,
          ),
          throwsA(isA<BuildError>()),
        );
        final dependencies = build_hook
            .vulkanSdkBuildDependenciesForNativeAssetsBuild(temp.uri, OS.linux);
        expect(
          dependencies,
          containsAll([header.uri, glslc.uri, loader.uri, spirvConfig.uri]),
        );
        expect(dependencies, isNot(contains(unusedDoc.uri)));
        final androidDependencies = build_hook
            .vulkanSdkBuildDependenciesForNativeAssetsBuild(
              temp.uri,
              OS.android,
            );
        expect(
          androidDependencies,
          containsAll([header.uri, hppHeader.uri, videoHeader.uri]),
        );
        expect(androidDependencies, isNot(contains(glslc.uri)));
        expect(androidDependencies, isNot(contains(loader.uri)));
        expect(androidDependencies, isNot(contains(spirvConfig.uri)));
      } finally {
        await temp.delete(recursive: true);
      }
    });

    test('uses bundled Android headers without a workspace path', () async {
      final temp = await Directory.systemTemp.createTemp(
        'fllamer_android_vulkan_headers_',
      );
      try {
        await File(
          '${temp.path}/include/vulkan/vulkan.hpp',
        ).create(recursive: true);
        expect(
          build_hook.vulkanSdkForNativeAssetsBuild(
            null,
            null,
            enabled: true,
            targetOS: OS.android,
            bundledAndroidVulkanHeaders: temp.uri,
          ),
          temp.uri,
        );
      } finally {
        await temp.delete(recursive: true);
      }
    });

    test('pins the bundled Android Vulkan-Headers version and notice', () {
      final root = Directory(build_hook.bundledAndroidVulkanHeadersPath);
      final core = File('${root.path}/include/vulkan/vulkan_core.h');
      final hpp = File('${root.path}/include/vulkan/vulkan.hpp');
      final video = File(
        '${root.path}/include/vk_video/vulkan_video_codec_av1std_decode.h',
      );
      final license = File('${root.path}/LICENSE.md');

      expect(root.existsSync(), isTrue);
      expect(hpp.existsSync(), isTrue);
      expect(video.existsSync(), isTrue);
      expect(
        core.readAsStringSync(),
        contains('#define VK_HEADER_VERSION 357'),
      );
      expect(
        core.readAsStringSync(),
        contains('VK_MAKE_API_VERSION(0, 1, 4, VK_HEADER_VERSION)'),
      );
      expect(license.readAsStringSync(), contains('Apache-2.0'));
    });

    test('requires Vulkan-Hpp input for enabled Android builds', () async {
      expect(
        () => build_hook.vulkanSdkForNativeAssetsBuild(
          null,
          null,
          enabled: true,
          targetOS: OS.android,
        ),
        throwsA(
          isA<BuildError>().having(
            (error) => error.message,
            'message',
            contains('Bundled Android Vulkan headers'),
          ),
        ),
      );

      final temp = await Directory.systemTemp.createTemp(
        'fllamer_android_vulkan_sdk_',
      );
      try {
        expect(
          () => build_hook.vulkanSdkForNativeAssetsBuild(
            temp.path,
            temp.uri,
            enabled: true,
            targetOS: OS.android,
          ),
          throwsA(
            isA<BuildError>().having(
              (error) => error.message,
              'message',
              contains('vulkan/vulkan.hpp'),
            ),
          ),
        );

        final hppHeader = File('${temp.path}/include/vulkan/vulkan.hpp');
        await hppHeader.create(recursive: true);
        expect(
          build_hook.vulkanSdkForNativeAssetsBuild(
            temp.path,
            temp.uri,
            enabled: true,
            targetOS: OS.android,
          ),
          temp.uri,
        );
      } finally {
        await temp.delete(recursive: true);
      }
    });

    test('uses glslc from the Flutter-selected Android NDK', () async {
      final temp = await Directory.systemTemp.createTemp(
        'fllamer_android_ndk_glslc_',
      );
      try {
        await File(
          '${temp.path}/build/cmake/android.toolchain.cmake',
        ).create(recursive: true);
        final macosGlslc = File(
          '${temp.path}/shader-tools/darwin-x86_64/glslc',
        );
        final linuxGlslc = File('${temp.path}/shader-tools/linux-x86_64/glslc');
        final windowsGlslc = File(
          '${temp.path}/shader-tools/windows-x86_64/glslc.exe',
        );
        for (final glslc in [macosGlslc, linuxGlslc, windowsGlslc]) {
          await glslc.create(recursive: true);
        }

        expect(
          build_hook.androidVulkanGlslcForNdk(
            temp.path,
            hostOperatingSystem: 'macos',
          ),
          macosGlslc.uri,
        );
        expect(
          build_hook.androidVulkanGlslcForNdk(
            temp.path,
            hostOperatingSystem: 'linux',
          ),
          linuxGlslc.uri,
        );
        expect(
          build_hook.androidVulkanGlslcForNdk(
            temp.path,
            hostOperatingSystem: 'windows',
          ),
          windowsGlslc.uri,
        );
        await linuxGlslc.delete();
        expect(
          () => build_hook.androidVulkanGlslcForNdk(
            temp.path,
            hostOperatingSystem: 'linux',
          ),
          throwsA(
            isA<BuildError>().having(
              (error) => error.message,
              'message',
              contains('shader-tools/linux-x86_64/glslc'),
            ),
          ),
        );
      } finally {
        await temp.delete(recursive: true);
      }
    });

    test('requires CMake 3.19 or newer for Vulkan builds', () {
      expect(
        () => build_hook.validateCmakeVersionForVulkanBuild(
          'cmake version 3.18.6',
        ),
        throwsA(
          isA<BuildError>().having(
            (error) => error.message,
            'message',
            contains('3.19 or newer'),
          ),
        ),
      );
      expect(
        () => build_hook.validateCmakeVersionForVulkanBuild(
          'cmake version 3.19.0',
        ),
        returnsNormally,
      );
      expect(
        () => build_hook.validateCmakeVersionForVulkanBuild(
          'cmake version 4.0.0',
        ),
        returnsNormally,
      );
    });

    test('gates the safe K-quant runtime marker to Android Vulkan', () {
      final cmake = File(
        'native/llama_dart_bridge/CMakeLists.txt',
      ).readAsStringSync();
      const defaultMarker = 'set(LLAMA_DART_ANDROID_VULKAN_SAFE_K_QUANT 0)';
      const enabledMarker = 'set(LLAMA_DART_ANDROID_VULKAN_SAFE_K_QUANT 1)';
      const runtimeMarker =
          r'GGML_VULKAN_ANDROID_SAFE_K_QUANT=${LLAMA_DART_ANDROID_VULKAN_SAFE_K_QUANT}';
      final androidVulkanStart = cmake.indexOf(
        'if(ANDROID AND LLAMA_DART_ENABLE_VULKAN)',
      );
      final androidVulkanEnd = cmake.indexOf(
        'elseif(LLAMA_DART_ANDROID_VULKAN_SHADER_OVERLAY_DIR)',
        androidVulkanStart,
      );

      expect(_occurrences(cmake, defaultMarker), 1);
      expect(_occurrences(cmake, enabledMarker), 1);
      expect(_occurrences(cmake, runtimeMarker), 1);
      expect(cmake.indexOf(defaultMarker), lessThan(androidVulkanStart));
      expect(cmake.indexOf(enabledMarker), greaterThan(androidVulkanStart));
      expect(cmake.indexOf(enabledMarker), lessThan(androidVulkanEnd));
    });
  });

  group('Windows CMake toolchain policy', () {
    final toolchain = CCompilerConfig(
      compiler: Uri.file(
        r'C:\Visual Studio\VC\bin\Hostx64\x64\cl.exe',
        windows: true,
      ),
      linker: Uri.file(
        r'C:\Visual Studio\VC\bin\Hostx64\x64\link.exe',
        windows: true,
      ),
      archiver: Uri.file(
        r'C:\Visual Studio\VC\bin\Hostx64\x64\lib.exe',
        windows: true,
      ),
      windows: WindowsCCompilerConfig(),
    );

    test('uses the native-assets compiler through Ninja', () {
      final args = build_hook.windowsCmakeToolchainArgsForNativeAssetsBuild(
        Architecture.x64,
        toolchain,
        hostArchitecture: Architecture.x64,
      );

      expect(args, containsAllInOrder(['-G', 'Ninja']));
      expect(args, contains('-DCMAKE_SYSTEM_PROCESSOR=AMD64'));
      expect(
        args,
        contains(
          '-DCMAKE_CXX_COMPILER=C:/Visual Studio/VC/bin/Hostx64/x64/cl.exe',
        ),
      );
      expect(
        args,
        contains('-DCMAKE_LINKER=C:/Visual Studio/VC/bin/Hostx64/x64/link.exe'),
      );
      expect(
        args,
        contains('-DCMAKE_AR=C:/Visual Studio/VC/bin/Hostx64/x64/lib.exe'),
      );
    });

    test('rejects missing and cross-architecture toolchains', () {
      expect(
        () => build_hook.windowsCmakeToolchainArgsForNativeAssetsBuild(
          Architecture.x64,
          null,
          hostArchitecture: Architecture.x64,
        ),
        throwsA(
          isA<BuildError>().having(
            (error) => error.message,
            'message',
            contains('flutter doctor -v'),
          ),
        ),
      );
      expect(
        () => build_hook.windowsCmakeToolchainArgsForNativeAssetsBuild(
          Architecture.arm64,
          toolchain,
          hostArchitecture: Architecture.x64,
        ),
        throwsA(
          isA<BuildError>().having(
            (error) => error.message,
            'message',
            contains('Cross-architecture'),
          ),
        ),
      );
    });

    test('parses developer environment output without pseudo variables', () {
      expect(
        build_hook.parseWindowsBuildEnvironment(
          'Path=C:\\Windows\\System32\r\n'
          'INCLUDE=C:\\SDK\\Include\r\n'
          '=C:=C:\\workspace\r\n'
          'INVALID\r\n',
        ),
        <String, String>{
          'Path': r'C:\Windows\System32',
          'INCLUDE': r'C:\SDK\Include',
        },
      );
    });
  });
}

int _occurrences(String source, String pattern) =>
    source.split(pattern).length - 1;

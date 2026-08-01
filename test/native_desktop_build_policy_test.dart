import 'dart:io';

import 'package:code_assets/code_assets.dart';
import 'package:hooks/hooks.dart';
import 'package:test/test.dart';

import '../hook/build.dart' as build_hook;

void main() {
  group('desktop Vulkan build policy', () {
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

    test('supports an explicit reproducible CPU-only desktop build', () {
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

      expect(
        build_hook.vulkanCmakeArgsForNativeAssetsBuild(enableVulkan: true),
        contains('-DLLAMA_DART_ENABLE_VULKAN=ON'),
      );
      expect(
        build_hook.vulkanCmakeArgsForNativeAssetsBuild(enableVulkan: false),
        contains('-DLLAMA_DART_ENABLE_VULKAN=OFF'),
      );
      expect(
        build_hook.vulkanCmakeArgsForNativeAssetsBuild(
          enableVulkan: true,
          vulkanSdk: sdk,
        ),
        contains('-DVulkan_ROOT=/opt/vulkan-sdk/'),
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
    });

    test('validates an explicitly configured SDK directory', () async {
      final temp = await Directory.systemTemp.createTemp('fllamer_vulkan_sdk_');
      try {
        final header = File('${temp.path}/include/vulkan/vulkan.h');
        final glslc = File('${temp.path}/bin/glslc');
        final loader = File('${temp.path}/lib/libvulkan.so');
        final spirvConfig = File(
          '${temp.path}/share/cmake/SPIRV-Headers/'
          'SPIRV-HeadersConfig.cmake',
        );
        final unusedDoc = File('${temp.path}/Documentation/guide.pdf');
        for (final file in [header, glslc, loader, spirvConfig, unusedDoc]) {
          await file.create(recursive: true);
        }

        expect(
          build_hook.vulkanSdkForNativeAssetsBuild(
            temp.path,
            temp.uri,
            enabled: true,
          ),
          temp.uri,
        );
        expect(
          build_hook.vulkanSdkForNativeAssetsBuild(
            temp.path,
            temp.uri,
            enabled: false,
          ),
          isNull,
        );
        expect(
          () =>
              build_hook.vulkanSdkForNativeAssetsBuild(1, null, enabled: true),
          throwsA(isA<BuildError>()),
        );
        final dependencies = build_hook
            .vulkanSdkBuildDependenciesForNativeAssetsBuild(temp.uri, OS.linux);
        expect(
          dependencies,
          containsAll([header.uri, glslc.uri, loader.uri, spirvConfig.uri]),
        );
        expect(dependencies, isNot(contains(unusedDoc.uri)));
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

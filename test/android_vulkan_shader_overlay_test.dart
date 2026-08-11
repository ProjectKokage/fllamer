import 'dart:convert';
import 'dart:io';

import 'package:crypto/crypto.dart';
import 'package:test/test.dart';

import '../hook/src/android_vulkan_shader_overlay.dart';

void main() {
  final shaderDirectory = Directory(
    'third_party/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders',
  );
  final dequantFile = File('${shaderDirectory.path}/dequant_funcs.glsl');
  final mulMmFile = File('${shaderDirectory.path}/mul_mm_funcs.glsl');
  final vulkanFile = File(
    'third_party/llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp',
  );

  group('Android Vulkan safe quant shader transform', () {
    test('accepts only the pinned dequant source and emits exact output', () {
      final source = dequantFile.readAsStringSync();
      final patched = patchPinnedAndroidVulkanDequantFuncs(source);

      expect(_sha256(source), pinnedDequantFuncsSha256);
      expect(_sha256(patched), patchedDequantFuncsSha256);
      expect(patched, contains('uint(data_a[a_offset + ib].qs[iqs + 1])'));
      expect(patched, contains('int(data_a[a_offset + ib].qs[iqs + 3])'));
      expect(
        patched,
        contains(
          'const uint vui = '
          'uint(data_a_packed16[a_offset + ib].qs[iqs/2]);\n'
          '    return (vec4(vui & 0xF, (vui >> 4) & 0xF, '
          '(vui >> 8) & 0xF, vui >> 12) - 8.0f);',
        ),
        reason: 'The Q4_0 matvec path is intentionally unchanged.',
      );
    });

    test('accepts only the pinned matrix source and preserves scales', () {
      final source = mulMmFile.readAsStringSync();
      final patched = patchPinnedAndroidVulkanMulMmFuncs(source);

      expect(_sha256(source), pinnedMulMmFuncsSha256);
      expect(_sha256(patched), patchedMulMmFuncsSha256);
      expect(
        patched,
        contains('const float d = float(data_a_packed16[ib].d);'),
      );
      expect(
        patched,
        contains('const vec2 dm = vec2(data_a_packed32[ib].dm);'),
      );
      expect(
        RegExp(
          r'const uvec4 q = uvec4\(data_a\[ib\]\.qs\[qsi',
        ).allMatches(patched),
        hasLength(2),
      );
      expect(patched, contains('int(data_a[ib].qs[qsi + 3])'));
      expect(
        patched,
        isNot(contains('vec2(float(data_a[ib].d), float(data_a[ib].m))')),
      );
    });

    test('rejects mutation and already-patched input', () {
      final source = dequantFile.readAsStringSync();
      final patched = patchPinnedAndroidVulkanDequantFuncs(source);

      expect(
        () => patchPinnedAndroidVulkanDequantFuncs('$source\n'),
        throwsA(isA<AndroidVulkanShaderOverlayException>()),
      );
      expect(
        () => patchPinnedAndroidVulkanDequantFuncs(patched),
        throwsA(isA<AndroidVulkanShaderOverlayException>()),
      );
    });

    test('patches only the proprietary Qualcomm K-quant policy', () {
      final source = vulkanFile.readAsStringSync();
      final patched = patchPinnedAndroidGgmlVulkan(source);

      expect(_sha256(source), pinnedGgmlVulkanSha256);
      expect(_sha256(patched), patchedGgmlVulkanSha256);
      expect(
        patched,
        contains(
          'device->vendor_id == VK_VENDOR_ID_QUALCOMM &&\n'
          '           device->driver_id == '
          'vk::DriverId::eQualcommProprietary;',
        ),
      );
      expect(
        patched,
        contains(
          'if (!ggml_vk_is_qualcomm_proprietary(device)) {\n'
          '                ggml_vk_create_pipeline(device, '
          'device->pipeline_dequant_mul_mat_vec_f32_f32[w]'
          '[GGML_TYPE_Q4_K][i]',
        ),
      );
      expect(
        patched,
        contains(
          'if (!ggml_vk_is_qualcomm_proprietary(device)) {\n'
          '                ggml_vk_create_pipeline(device, '
          'device->pipeline_dequant_mul_mat_vec_f16_f32[w]'
          '[GGML_TYPE_Q4_K][i]',
        ),
      );
      expect(
        patched,
        contains(
          'if (ggml_vk_is_qualcomm_proprietary(ctx->device) && '
          'src0->type == GGML_TYPE_Q4_K) {\n'
          '        mmp = nullptr;\n'
          '        quantize_y = false;',
        ),
      );
      expect(
        patched,
        contains(
          '} else if (!(ggml_vk_is_qualcomm_proprietary(ctx->device) && '
          'src0->type == GGML_TYPE_Q4_K) &&',
        ),
      );
      expect(
        patched,
        contains(
          'ggml_vk_is_qualcomm_proprietary(ctx->device) && '
          'mul->src[0]->type == GGML_TYPE_Q4_K',
        ),
      );

      final cpuFallback = RegExp(
        r'if \(ggml_vk_is_qualcomm_proprietary\(device\) &&\s*'
        r'\((src0_type == [^)]+)\)\) \{\s*return false;',
      ).firstMatch(patched);
      expect(cpuFallback, isNotNull);
      final fallbackTypes = RegExp(r'GGML_TYPE_[A-Z0-9_]+')
          .allMatches(cpuFallback!.group(1)!)
          .map((match) => match.group(0))
          .toList();
      expect(fallbackTypes, ['GGML_TYPE_Q5_K', 'GGML_TYPE_Q6_K']);

      expect(
        _typesOnLinesContaining(patched, 'ggml_vk_create_pipeline'),
        _typesOnLinesContaining(source, 'ggml_vk_create_pipeline'),
        reason: 'The complete pipeline registration type set must be retained.',
      );
      expect(
        _typesOnLinesContaining(patched, 'CREATE_MM('),
        _typesOnLinesContaining(source, 'CREATE_MM('),
        reason: 'The complete matrix and ID registration type set must remain.',
      );

      for (final preserved in [
        'pipeline_dequant_mul_mat_vec_q8_1',
        'pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_K]',
        'pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_K]',
        'pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q6_K]',
      ]) {
        expect(
          _occurrences(patched, preserved),
          _occurrences(source, preserved),
          reason: '$preserved registration count must stay unchanged.',
        );
      }
    });

    test('normalizes native CRLF and rejects mutated or patched input', () {
      final source = vulkanFile.readAsStringSync();
      final patched = patchPinnedAndroidGgmlVulkan(source);

      expect(
        patchPinnedAndroidGgmlVulkan(source.replaceAll('\n', '\r\n')),
        patched,
      );
      expect(
        () => patchPinnedAndroidGgmlVulkan('$source\n'),
        throwsA(isA<AndroidVulkanShaderOverlayException>()),
      );
      expect(
        () => patchPinnedAndroidGgmlVulkan(patched),
        throwsA(isA<AndroidVulkanShaderOverlayException>()),
      );
    });

    test('normalizes a pinned Windows checkout to exact LF output', () {
      final source = dequantFile.readAsStringSync();
      final crlfSource = source.replaceAll('\n', '\r\n');

      expect(
        patchPinnedAndroidVulkanDequantFuncs(crlfSource),
        patchPinnedAndroidVulkanDequantFuncs(source),
      );
      expect(
        () => patchPinnedAndroidVulkanDequantFuncs(
          source.replaceFirst('\n', '\r'),
        ),
        throwsA(isA<AndroidVulkanShaderOverlayException>()),
      );
    });

    test(
      'preparation accepts CRLF and rejects mutated or patched native input',
      () async {
        final temp = await Directory.systemTemp.createTemp(
          'fllamer_android_vulkan_native_preparation_',
        );
        final source = Directory('${temp.path}/shaders');
        final nativeSource = File('${temp.path}/native/ggml-vulkan.cpp');
        final output = Directory('${temp.path}/output');
        try {
          await source.create();
          await dequantFile.copy('${source.path}/dequant_funcs.glsl');
          await mulMmFile.copy('${source.path}/mul_mm_funcs.glsl');
          final pristine = await vulkanFile.readAsString();
          await nativeSource.create(recursive: true);
          await nativeSource.writeAsString(pristine.replaceAll('\n', '\r\n'));

          await prepareAndroidVulkanShaderOverlay(
            sourceDirectory: source,
            vulkanSourceFile: nativeSource,
            outputDirectory: output,
          );
          final settled = File('${output.path}/ggml-vulkan.cpp');
          expect(
            _sha256(await settled.readAsString()),
            patchedGgmlVulkanSha256,
          );

          await nativeSource.writeAsString('$pristine\n');
          await expectLater(
            prepareAndroidVulkanShaderOverlay(
              sourceDirectory: source,
              vulkanSourceFile: nativeSource,
              outputDirectory: output,
            ),
            throwsA(isA<AndroidVulkanShaderOverlayException>()),
          );
          expect(
            _sha256(await settled.readAsString()),
            patchedGgmlVulkanSha256,
          );

          await nativeSource.writeAsString(
            patchPinnedAndroidGgmlVulkan(pristine),
          );
          await expectLater(
            prepareAndroidVulkanShaderOverlay(
              sourceDirectory: source,
              vulkanSourceFile: nativeSource,
              outputDirectory: output,
            ),
            throwsA(isA<AndroidVulkanShaderOverlayException>()),
          );
          expect(
            _sha256(await settled.readAsString()),
            patchedGgmlVulkanSha256,
          );
        } finally {
          await temp.delete(recursive: true);
        }
      },
    );

    test('serializes isolated build-output preparation', () async {
      final temp = await Directory.systemTemp.createTemp(
        'fllamer_android_vulkan_overlay_',
      );
      final source = Directory('${temp.path}/source');
      final output = Directory('${temp.path}/output');
      try {
        await source.create();
        final sourceDequant = File('${source.path}/dequant_funcs.glsl');
        final sourceMulMm = File('${source.path}/mul_mm_funcs.glsl');
        final unrelated = File('${source.path}/nested/unrelated.comp');
        await sourceDequant.writeAsString(await dequantFile.readAsString());
        await sourceMulMm.writeAsString(await mulMmFile.readAsString());
        await unrelated.create(recursive: true);
        await unrelated.writeAsBytes([0, 1, 2, 255]);
        await output.create();
        await File('${output.path}/stale.comp').writeAsString('stale');

        await Future.wait([
          prepareAndroidVulkanShaderOverlay(
            sourceDirectory: source,
            vulkanSourceFile: vulkanFile,
            outputDirectory: output,
          ),
          prepareAndroidVulkanShaderOverlay(
            sourceDirectory: source,
            vulkanSourceFile: vulkanFile,
            outputDirectory: output,
          ),
        ]);

        expect(
          _sha256(await sourceDequant.readAsString()),
          pinnedDequantFuncsSha256,
        );
        expect(
          _sha256(await sourceMulMm.readAsString()),
          pinnedMulMmFuncsSha256,
        );
        expect(
          _sha256(
            await File('${output.path}/dequant_funcs.glsl').readAsString(),
          ),
          patchedDequantFuncsSha256,
        );
        expect(
          _sha256(
            await File('${output.path}/mul_mm_funcs.glsl').readAsString(),
          ),
          patchedMulMmFuncsSha256,
        );
        expect(
          _sha256(await File('${output.path}/ggml-vulkan.cpp').readAsString()),
          patchedGgmlVulkanSha256,
        );
        expect(
          await File('${output.path}/nested/unrelated.comp').readAsBytes(),
          [0, 1, 2, 255],
        );
        expect(File('${output.path}/stale.comp').existsSync(), isFalse);
      } finally {
        await temp.delete(recursive: true);
      }
    });

    test('accepts a hook output URI with a trailing separator', () async {
      final temp = await Directory.systemTemp.createTemp(
        'fllamer_android_vulkan_trailing_separator_',
      );
      final output = Directory('${temp.path}/output/');
      try {
        await prepareAndroidVulkanShaderOverlay(
          sourceDirectory: shaderDirectory,
          vulkanSourceFile: vulkanFile,
          outputDirectory: output,
        );

        expect(File('${output.path}dequant_funcs.glsl').existsSync(), isTrue);
        expect(File('${temp.path}/output.lock').existsSync(), isTrue);
        expect(File('${output.path}.lock').existsSync(), isFalse);
      } finally {
        await temp.delete(recursive: true);
      }
    });

    test('rejects an output nested inside the source', () async {
      final temp = await Directory.systemTemp.createTemp(
        'fllamer_android_vulkan_overlap_',
      );
      try {
        await expectLater(
          prepareAndroidVulkanShaderOverlay(
            sourceDirectory: temp,
            vulkanSourceFile: vulkanFile,
            outputDirectory: Directory('${temp.path}/overlay'),
          ),
          throwsA(
            isA<AndroidVulkanShaderOverlayException>().having(
              (error) => error.message,
              'message',
              contains('overlap'),
            ),
          ),
        );
      } finally {
        await temp.delete(recursive: true);
      }
    });
  });

  group('Android Vulkan CMake overlay', () {
    test(
      'swaps exactly one native source and watches all overlay inputs',
      () async {
        final temp = await Directory.systemTemp.createTemp(
          'fllamer_android_vulkan_cmake_',
        );
        try {
          final overlay = Directory('${temp.path}/overlay');
          await prepareAndroidVulkanShaderOverlay(
            sourceDirectory: shaderDirectory,
            vulkanSourceFile: vulkanFile,
            outputDirectory: overlay,
          );
          final project = await _writeCmakeOverlayFixture(
            temp,
            overlay,
            duplicateVulkanSource: false,
          );
          final build = Directory('${temp.path}/build');
          final configure = await Process.run('cmake', [
            '-S',
            project.path,
            '-B',
            build.path,
          ]);
          expect(
            configure.exitCode,
            0,
            reason: '${configure.stdout}\n${configure.stderr}',
          );

          final sources = File(
            '${build.path}/sources.txt',
          ).readAsStringSync().split(';');
          final vulkanSources = sources
              .where((source) => source.endsWith('/ggml-vulkan.cpp'))
              .toList();
          expect(vulkanSources, [
            _cmakePath('${overlay.path}/ggml-vulkan.cpp'),
          ]);
          expect(sources, contains(endsWith('/unrelated.cpp')));

          final configureDepends = File(
            '${build.path}/configure-depends.txt',
          ).readAsStringSync().split(';').toSet();
          expect(configureDepends, {
            _cmakePath('${overlay.path}/dequant_funcs.glsl'),
            _cmakePath('${overlay.path}/mul_mm_funcs.glsl'),
            _cmakePath('${overlay.path}/ggml-vulkan.cpp'),
          });

          await File(
            '${overlay.path}/ggml-vulkan.cpp',
          ).writeAsString('// mutation\n', mode: FileMode.append, flush: true);
          final rebuild = await Process.run('cmake', [
            '--build',
            build.path,
            '--target',
            'ggml-vulkan',
          ]);
          expect(rebuild.exitCode, isNot(0));
          expect(
            '${rebuild.stdout}\n${rebuild.stderr}',
            contains('Android Vulkan native source overlay is not exact'),
          );
        } finally {
          await temp.delete(recursive: true);
        }
      },
    );

    test('rejects a target with duplicate ggml-vulkan sources', () async {
      final temp = await Directory.systemTemp.createTemp(
        'fllamer_android_vulkan_cmake_duplicate_',
      );
      try {
        final overlay = Directory('${temp.path}/overlay');
        await prepareAndroidVulkanShaderOverlay(
          sourceDirectory: shaderDirectory,
          vulkanSourceFile: vulkanFile,
          outputDirectory: overlay,
        );
        final project = await _writeCmakeOverlayFixture(
          temp,
          overlay,
          duplicateVulkanSource: true,
        );
        final configure = await Process.run('cmake', [
          '-S',
          project.path,
          '-B',
          '${temp.path}/build',
        ]);

        expect(configure.exitCode, isNot(0));
        expect(
          '${configure.stdout}\n${configure.stderr}',
          contains('Pinned ggml-vulkan.cpp target source shape changed'),
        );
      } finally {
        await temp.delete(recursive: true);
      }
    });
  });
}

String _sha256(String source) => sha256.convert(utf8.encode(source)).toString();

int _occurrences(String source, String pattern) =>
    source.split(pattern).length - 1;

List<String> _typesOnLinesContaining(String source, String marker) => source
    .split('\n')
    .where((line) => line.contains(marker))
    .expand(
      (line) => RegExp(
        r'GGML_TYPE_[A-Z0-9_]+',
      ).allMatches(line).map((match) => match.group(0)!),
    )
    .toList();

Future<Directory> _writeCmakeOverlayFixture(
  Directory root,
  Directory overlay, {
  required bool duplicateVulkanSource,
}) async {
  final project = Directory('${root.path}/project');
  final original = Directory('${project.path}/original');
  final second = Directory('${project.path}/second');
  final sourceNames = [
    'ggml-vulkan.cpp',
    'ggml-vulkan-shaders.hpp',
    'copy_from_quant.comp.cpp',
    'get_rows_quant.comp.cpp',
    'mul_mat_vec.comp.cpp',
    'mul_mm.comp.cpp',
    'unrelated.cpp',
  ];
  for (final name in sourceNames) {
    final file = File('${original.path}/$name');
    await file.create(recursive: true);
    await file.writeAsString('// fixture: $name\n');
  }
  if (duplicateVulkanSource) {
    final duplicate = File('${second.path}/ggml-vulkan.cpp');
    await duplicate.create(recursive: true);
    await duplicate.writeAsString('// duplicate fixture\n');
  }
  final glslc = File('${project.path}/glslc');
  await glslc.create(recursive: true);

  final targetSources = <String>[
    for (final name in sourceNames)
      '  "${_cmakePath('${original.path}/$name')}"',
    if (duplicateVulkanSource)
      '  "${_cmakePath('${second.path}/ggml-vulkan.cpp')}"',
  ].join('\n');
  final overlayCmake = File(
    'native/llama_dart_bridge/cmake/android-vulkan-shader-overlay.cmake',
  ).absolute;
  final cmakeLists = File('${project.path}/CMakeLists.txt');
  await cmakeLists.writeAsString('''cmake_minimum_required(VERSION 3.19)
project(fllamer_android_vulkan_overlay_fixture LANGUAGES CXX)
set(Vulkan_GLSLC_EXECUTABLE "${_cmakePath(glslc.path)}")
add_custom_target(vulkan-shaders-gen)
add_library(ggml-vulkan STATIC
$targetSources
)
include("${_cmakePath(overlayCmake.path)}")
fllamer_apply_android_vulkan_shader_overlay(
  ggml-vulkan
  "${_cmakePath(overlay.path)}"
)
get_target_property(_sources ggml-vulkan SOURCES)
get_property(_configure_depends DIRECTORY PROPERTY CMAKE_CONFIGURE_DEPENDS)
file(WRITE "\${CMAKE_BINARY_DIR}/sources.txt" "\${_sources}")
file(WRITE "\${CMAKE_BINARY_DIR}/configure-depends.txt" "\${_configure_depends}")
''');
  return project;
}

String _cmakePath(String path) => path.replaceAll('\\', '/');

import 'dart:convert';
import 'dart:io';

import 'package:crypto/crypto.dart';

const pinnedDequantFuncsSha256 =
    '3b7a5edf6072ee8e44b4a9ddeb89b099d1fb64901d9be419c8ed9701d2f1d68b';
const patchedDequantFuncsSha256 =
    'bcaf084f68df8da15dcf2df3d15d45d6a443eccd9b53f26c78792e368079c7a8';
const pinnedMulMmFuncsSha256 =
    '71f86758ba309b36a9af0420e08abef386a3b3b08545e096cad10e16186ef61e';
const patchedMulMmFuncsSha256 =
    '64ae33bf3fa95dfbed3fc17e2ed1e2e5405d3db193ad8a604b0e84f5536d665c';
const pinnedMulMmCompSha256 =
    'abed36cd38b5edc3f0f365fbc71a672576e7cd8a703131e297b416f23c3b1221';
const patchedMulMmCompSha256 =
    '0a96c140a076875c6dc0e3fd0f29e5caceba7df8219da2b2067eb2cc1162af15';
const pinnedGgmlVulkanSha256 =
    'e43a39ce1e9443f7c1c00e7bb9f13d56ab5a01e2355973ee740d4324e0e0698b';
const patchedGgmlVulkanSha256 =
    '5c3d8a468f7cd742ff4a3ef513d648f350bfd3ee53dbce773895e0531714d3fc';

const _pinnedQ4_1Dequantize4 =
    '''vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const uint vui = uint(data_a_packed16[a_offset + ib].qs[iqs/2]);
    return vec4(vui & 0xF, (vui >> 4) & 0xF, (vui >> 8) & 0xF, vui >> 12);
}''';

const _patchedQ4_1Dequantize4 =
    '''vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const uint q0 = uint(data_a[a_offset + ib].qs[iqs    ]);
    const uint q1 = uint(data_a[a_offset + ib].qs[iqs + 1]);
    return vec4(q0 & 0xF, q0 >> 4, q1 & 0xF, q1 >> 4);
}''';

const _pinnedQ8_0Dequantize4 =
    '''vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    const i8vec2 v0 = unpack8(int32_t(data_a_packed16[a_offset + ib].qs[iqs/2])).xy; // vec4 used due to #12147
    const i8vec2 v1 = unpack8(int32_t(data_a_packed16[a_offset + ib].qs[iqs/2 + 1])).xy;
    return vec4(v0.x, v0.y, v1.x, v1.y);
}''';

const _patchedQ8_0Dequantize4 =
    '''vec4 dequantize4(uint ib, uint iqs, uint a_offset) {
    return vec4(int(data_a[a_offset + ib].qs[iqs    ]),
                int(data_a[a_offset + ib].qs[iqs + 1]),
                int(data_a[a_offset + ib].qs[iqs + 2]),
                int(data_a[a_offset + ib].qs[iqs + 3]));
}''';

const _pinnedQ4_0MatrixLoad =
    '''        const float d = float(a_q4_0_p16.data[ib].d);
        const uint vui = uint(a_q4_0_p16.data[ib].qs[2*iqs]) | (uint(a_q4_0_p16.data[ib].qs[2*iqs + 1]) << 16);
        const vec4 v0 = (vec4(unpack8(vui & 0x0F0F0F0F)) - 8.0f) * d;
        const vec4 v1 = (vec4(unpack8((vui >> 4) & 0x0F0F0F0F)) - 8.0f) * d;''';

const _patchedQ4_0MatrixLoad =
    '''        const float d = float(a_q4_0_p16.data[ib].d);
        const uint qsi = 4 * iqs;
        const uvec4 q = uvec4(a_q4_0.data[ib].qs[qsi    ],
                              a_q4_0.data[ib].qs[qsi + 1],
                              a_q4_0.data[ib].qs[qsi + 2],
                              a_q4_0.data[ib].qs[qsi + 3]);
        const vec4 v0 = (vec4(q & 0x0Fu) - 8.0f) * d;
        const vec4 v1 = (vec4(q >> 4) - 8.0f) * d;''';

const _pinnedQ4_1MatrixLoad =
    '''        const vec2 dm = vec2(a_q4_1_p32.data[ib].dm);
        const uint vui = a_q4_1_p32.data[ib].qs[iqs];
        const vec4 v0 = vec4(unpack8(vui & 0x0F0F0F0F)) * dm.x + dm.y;
        const vec4 v1 = vec4(unpack8((vui >> 4) & 0x0F0F0F0F)) * dm.x + dm.y;''';

const _patchedQ4_1MatrixLoad =
    '''        const vec2 dm = vec2(a_q4_1_p32.data[ib].dm);
        const uint qsi = 4 * iqs;
        const uvec4 q = uvec4(a_q4_1.data[ib].qs[qsi    ],
                              a_q4_1.data[ib].qs[qsi + 1],
                              a_q4_1.data[ib].qs[qsi + 2],
                              a_q4_1.data[ib].qs[qsi + 3]);
        const vec4 v0 = vec4(q & 0x0Fu) * dm.x + dm.y;
        const vec4 v1 = vec4(q >> 4) * dm.x + dm.y;''';

const _pinnedQ8_0MatrixLoad =
    '''        const float d = float(a_q8_0_p16.data[ib].d);
        const i8vec2 v0 = unpack8(int32_t(a_q8_0_p16.data[ib].qs[2*iqs])).xy; // vec4 used due to #12147
        const i8vec2 v1 = unpack8(int32_t(a_q8_0_p16.data[ib].qs[2*iqs + 1])).xy;
        const vec4 v = vec4(v0.x, v0.y, v1.x, v1.y) * d;''';

const _patchedQ8_0MatrixLoad =
    '''        const float d = float(a_q8_0_p16.data[ib].d);
        const uint qsi = 4 * iqs;
        const vec4 v = vec4(int(a_q8_0.data[ib].qs[qsi    ]),
                            int(a_q8_0.data[ib].qs[qsi + 1]),
                            int(a_q8_0.data[ib].qs[qsi + 2]),
                            int(a_q8_0.data[ib].qs[qsi + 3])) * d;''';

const _pinnedMatrixByteViews =
    '''layout (binding = 0) readonly buffer BUF_TQ2_0    { block_tq2_0 data[];    } a_tq2_0;
// Packed16 aliases''';

const _patchedMatrixByteViews =
    '''layout (binding = 0) readonly buffer BUF_TQ2_0    { block_tq2_0 data[];    } a_tq2_0;
layout (binding = 0) readonly buffer BUF_Q4_0     { block_q4_0 data[];     } a_q4_0;
layout (binding = 0) readonly buffer BUF_Q4_1     { block_q4_1 data[];     } a_q4_1;
layout (binding = 0) readonly buffer BUF_Q8_0     { block_q8_0 data[];     } a_q8_0;
// Packed16 aliases''';

const _pinnedQualcommPredicateInsertion = '''    return false;
#endif
}

void ggml_vk_load_shaders(vk_device& device, vk_pipeline requested) {''';

const _patchedQualcommPredicateInsertion = '''    return false;
#endif
}

static bool ggml_vk_is_qualcomm_proprietary(const vk_device & device) {
    return device->vendor_id == VK_VENDOR_ID_QUALCOMM &&
           device->driver_id == vk::DriverId::eQualcommProprietary;
}

void ggml_vk_load_shaders(vk_device& device, vk_pipeline requested) {''';

const _pinnedQ4KMatVecF32Registration =
    '''            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q4_K][i], "mul_mat_vec_q4_k_f32_f32", arr_dmmv_q4_k_f32_f32_len[reduc16], arr_dmmv_q4_k_f32_f32_data[reduc16], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);''';

const _patchedQ4KMatVecF32Registration =
    '''            if (!ggml_vk_is_qualcomm_proprietary(device)) {
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q4_K][i], "mul_mat_vec_q4_k_f32_f32", arr_dmmv_q4_k_f32_f32_len[reduc16], arr_dmmv_q4_k_f32_f32_data[reduc16], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            }''';

const _pinnedQ4KMatVecF16Registration =
    '''            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q4_K][i], "mul_mat_vec_q4_k_f16_f32", arr_dmmv_q4_k_f16_f32_len[reduc16], arr_dmmv_q4_k_f16_f32_data[reduc16], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);''';

const _patchedQ4KMatVecF16Registration =
    '''            if (!ggml_vk_is_qualcomm_proprietary(device)) {
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q4_K][i], "mul_mat_vec_q4_k_f16_f32", arr_dmmv_q4_k_f16_f32_len[reduc16], arr_dmmv_q4_k_f16_f32_data[reduc16], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            }''';

const _pinnedQ4KForcedDequant =
    '''    const std::vector<vk_matmul_pipeline_pair>* mmp_map = quantize_y ? ggml_vk_get_mul_mat_mat_pipeline_map(ctx, src0->type, GGML_TYPE_Q8_1, (ggml_prec)dst->op_params[0]) : nullptr;
    if (mmp_map == nullptr) {
        quantize_y = false;
    }

    const bool y_non_contig = (ctx->device->coopmat2 && src1->type == GGML_TYPE_F32) ||
                              // coopmat1: force f32->f16 conversion so the f16 B-type quant pipeline is
                              // used, but only when the int8 MMQ path above is not taken.
                              (ctx->device->coopmat_support && !ctx->device->coopmat2 && !quantize_y &&
                               ggml_is_quantized(src0->type) && src1->type == GGML_TYPE_F32) ||
                              (src0->type == GGML_TYPE_BF16 && src1->type != GGML_TYPE_BF16) ||
                              !ggml_vk_dim01_contiguous(src1);

    const bool y_f32_kernel = src1->type == GGML_TYPE_F32 && !y_non_contig;

    if (mmp_map == nullptr) {
        // Fall back to f16 dequant mul mat
        mmp_map = ggml_vk_get_mul_mat_mat_pipeline_map(ctx, src0->type, y_non_contig ? f16_type : src1->type, (ggml_prec)dst->op_params[0]);
    }

    const bool qx_needs_dequant = mmp_map == nullptr || x_non_contig;''';

const _patchedQ4KForcedDequant =
    '''    const std::vector<vk_matmul_pipeline_pair>* mmp_map = quantize_y ? ggml_vk_get_mul_mat_mat_pipeline_map(ctx, src0->type, GGML_TYPE_Q8_1, (ggml_prec)dst->op_params[0]) : nullptr;
    if (ggml_vk_is_qualcomm_proprietary(ctx->device) && src0->type == GGML_TYPE_Q4_K) {
        mmp_map = nullptr;
    }
    if (mmp_map == nullptr) {
        quantize_y = false;
    }

    const bool y_non_contig = (ctx->device->coopmat2 && src1->type == GGML_TYPE_F32) ||
                              // coopmat1: force f32->f16 conversion so the f16 B-type quant pipeline is
                              // used, but only when the int8 MMQ path above is not taken.
                              (ctx->device->coopmat_support && !ctx->device->coopmat2 && !quantize_y &&
                               ggml_is_quantized(src0->type) && src1->type == GGML_TYPE_F32) ||
                              (src0->type == GGML_TYPE_BF16 && src1->type != GGML_TYPE_BF16) ||
                              !ggml_vk_dim01_contiguous(src1);

    const bool y_f32_kernel = src1->type == GGML_TYPE_F32 && !y_non_contig;

    if (mmp_map == nullptr) {
        // Fall back to f16 dequant mul mat
        mmp_map = ggml_vk_get_mul_mat_mat_pipeline_map(ctx, src0->type, y_non_contig ? f16_type : src1->type, (ggml_prec)dst->op_params[0]);
    }

    if (ggml_vk_is_qualcomm_proprietary(ctx->device) && src0->type == GGML_TYPE_Q4_K) {
        mmp_map = nullptr;
    }

    const bool qx_needs_dequant = mmp_map == nullptr || x_non_contig;''';

const _pinnedQ4KMatVecDispatch =
    '''    } else if ((dst->ne[1] == 1 || (dst->ne[1] <= mul_mat_vec_max_cols && src1->ne[2] * src1->ne[3] == 1)) &&
               (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16 || ggml_is_quantized(src0->type))) {''';

const _patchedQ4KMatVecDispatch =
    '''    } else if (!(ggml_vk_is_qualcomm_proprietary(ctx->device) && src0->type == GGML_TYPE_Q4_K) &&
               (dst->ne[1] == 1 || (dst->ne[1] <= mul_mat_vec_max_cols && src1->ne[2] * src1->ne[3] == 1)) &&
               (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16 || ggml_is_quantized(src0->type))) {''';

const _pinnedQ4KMulMatAddFusion =
    '''    if ((ops.size() == 2 || ops.size() == 3) && ops.begin()[0] == GGML_OP_MUL_MAT && ops.begin()[1] == GGML_OP_ADD) {
        // additional constraints specific to this fusion
        const ggml_tensor *mul = cgraph->nodes[node_idx];
        const ggml_tensor *add = cgraph->nodes[node_idx + 1];''';

const _patchedQ4KMulMatAddFusion =
    '''    if ((ops.size() == 2 || ops.size() == 3) && ops.begin()[0] == GGML_OP_MUL_MAT && ops.begin()[1] == GGML_OP_ADD) {
        // additional constraints specific to this fusion
        const ggml_tensor *mul = cgraph->nodes[node_idx];
        if (ggml_vk_is_qualcomm_proprietary(ctx->device) && mul->src[0]->type == GGML_TYPE_Q4_K) {
            return false;
        }
        const ggml_tensor *add = cgraph->nodes[node_idx + 1];''';

const _pinnedQ5KQ6KSupportsOp =
    '''                ggml_type src0_type = op->src[0]->type;
                if (op->op == GGML_OP_MUL_MAT_ID) {''';

const _patchedQ5KQ6KSupportsOp =
    '''                ggml_type src0_type = op->src[0]->type;
                if (ggml_vk_is_qualcomm_proprietary(device) &&
                    (src0_type == GGML_TYPE_Q5_K || src0_type == GGML_TYPE_Q6_K)) {
                    return false;
                }
                if (op->op == GGML_OP_MUL_MAT_ID) {''';

final class AndroidVulkanShaderOverlayException implements Exception {
  const AndroidVulkanShaderOverlayException(this.message);

  final String message;

  @override
  String toString() => message;
}

final class _ShaderReplacement {
  const _ShaderReplacement(
    this.pinned,
    this.patched, {
    this.retainsPinnedSubstring = false,
  });

  final String pinned;
  final String patched;
  final bool retainsPinnedSubstring;
}

String patchPinnedAndroidVulkanDequantFuncs(String source) =>
    _patchPinnedShader(
      source,
      name: 'dequant_funcs.glsl',
      expectedSourceSha256: pinnedDequantFuncsSha256,
      expectedPatchedSha256: patchedDequantFuncsSha256,
      replacements: const [
        _ShaderReplacement(_pinnedQ4_1Dequantize4, _patchedQ4_1Dequantize4),
        _ShaderReplacement(_pinnedQ8_0Dequantize4, _patchedQ8_0Dequantize4),
      ],
    );

String patchPinnedAndroidVulkanMulMmFuncs(String source) => _patchPinnedShader(
  source,
  name: 'mul_mm_funcs.glsl',
  expectedSourceSha256: pinnedMulMmFuncsSha256,
  expectedPatchedSha256: patchedMulMmFuncsSha256,
  replacements: const [
    _ShaderReplacement(_pinnedQ4_0MatrixLoad, _patchedQ4_0MatrixLoad),
    _ShaderReplacement(_pinnedQ4_1MatrixLoad, _patchedQ4_1MatrixLoad),
    _ShaderReplacement(_pinnedQ8_0MatrixLoad, _patchedQ8_0MatrixLoad),
  ],
);

String patchPinnedAndroidVulkanMulMmComp(String source) => _patchPinnedShader(
  source,
  name: 'mul_mm.comp',
  expectedSourceSha256: pinnedMulMmCompSha256,
  expectedPatchedSha256: patchedMulMmCompSha256,
  replacements: const [
    _ShaderReplacement(_pinnedMatrixByteViews, _patchedMatrixByteViews),
  ],
);

String patchPinnedAndroidGgmlVulkan(String source) => _patchPinnedShader(
  source,
  name: 'ggml-vulkan.cpp',
  expectedSourceSha256: pinnedGgmlVulkanSha256,
  expectedPatchedSha256: patchedGgmlVulkanSha256,
  replacements: const [
    _ShaderReplacement(
      _pinnedQualcommPredicateInsertion,
      _patchedQualcommPredicateInsertion,
      retainsPinnedSubstring: true,
    ),
    _ShaderReplacement(
      _pinnedQ4KMatVecF32Registration,
      _patchedQ4KMatVecF32Registration,
      retainsPinnedSubstring: true,
    ),
    _ShaderReplacement(
      _pinnedQ4KMatVecF16Registration,
      _patchedQ4KMatVecF16Registration,
      retainsPinnedSubstring: true,
    ),
    _ShaderReplacement(_pinnedQ4KForcedDequant, _patchedQ4KForcedDequant),
    _ShaderReplacement(_pinnedQ4KMatVecDispatch, _patchedQ4KMatVecDispatch),
    _ShaderReplacement(_pinnedQ4KMulMatAddFusion, _patchedQ4KMulMatAddFusion),
    _ShaderReplacement(_pinnedQ5KQ6KSupportsOp, _patchedQ5KQ6KSupportsOp),
  ],
);

String _patchPinnedShader(
  String source, {
  required String name,
  required String expectedSourceSha256,
  required String expectedPatchedSha256,
  required List<_ShaderReplacement> replacements,
}) {
  final normalizedSource = _normalizeLineEndings(source, name);
  final sourceSha256 = _sha256(normalizedSource);
  if (sourceSha256 != expectedSourceSha256) {
    throw AndroidVulkanShaderOverlayException(
      'Refusing to patch unrecognized pinned $name.',
    );
  }

  var result = normalizedSource;
  for (final replacement in replacements) {
    if (_occurrences(result, replacement.pinned) != 1 ||
        result.contains(replacement.patched)) {
      throw AndroidVulkanShaderOverlayException(
        'Pinned shader block in $name is not exact.',
      );
    }
    result = result.replaceFirst(replacement.pinned, replacement.patched);
    if ((!replacement.retainsPinnedSubstring &&
            result.contains(replacement.pinned)) ||
        _occurrences(result, replacement.patched) != 1) {
      throw AndroidVulkanShaderOverlayException(
        'Shader transform in $name did not settle exactly once.',
      );
    }
  }

  if (_sha256(result) != expectedPatchedSha256) {
    throw AndroidVulkanShaderOverlayException(
      'Patched $name did not match the validated output.',
    );
  }
  return result;
}

String _normalizeLineEndings(String source, String name) {
  final normalized = source.replaceAll('\r\n', '\n');
  if (normalized.contains('\r')) {
    throw AndroidVulkanShaderOverlayException(
      'Refusing to patch $name with unsupported line endings.',
    );
  }
  return normalized;
}

final _overlayPreparationTails = <String, Future<void>>{};

Future<void> prepareAndroidVulkanShaderOverlay({
  required Directory sourceDirectory,
  required File vulkanSourceFile,
  required Directory outputDirectory,
}) {
  final targetDirectory = Directory(
    outputDirectory.path.replaceFirst(RegExp(r'[\\/]+$'), ''),
  );
  final key = _normalizedFileSystemPath(targetDirectory.path);
  final previous = _overlayPreparationTails[key] ?? Future<void>.value();
  final current = () async {
    try {
      await previous;
    } on Object {
      // A failed predecessor must not poison a later, independently validated
      // preparation attempt for the same build output.
    }
    await _prepareAndroidVulkanShaderOverlay(
      sourceDirectory: sourceDirectory,
      vulkanSourceFile: vulkanSourceFile,
      outputDirectory: targetDirectory,
    );
  }();
  _overlayPreparationTails[key] = current;
  return current.whenComplete(() {
    if (identical(_overlayPreparationTails[key], current)) {
      _overlayPreparationTails.remove(key);
    }
  });
}

Future<void> _prepareAndroidVulkanShaderOverlay({
  required Directory sourceDirectory,
  required File vulkanSourceFile,
  required Directory outputDirectory,
}) async {
  final targetDirectory = outputDirectory;
  if (!sourceDirectory.existsSync()) {
    throw const AndroidVulkanShaderOverlayException(
      'Pinned Android Vulkan shader source directory is missing.',
    );
  }
  if (!vulkanSourceFile.existsSync()) {
    throw const AndroidVulkanShaderOverlayException(
      'Pinned Android Vulkan native source is missing.',
    );
  }
  try {
    if (FileSystemEntity.typeSync(targetDirectory.path, followLinks: false) ==
        FileSystemEntityType.link) {
      throw const AndroidVulkanShaderOverlayException(
        'Android Vulkan shader overlay output must not be a link.',
      );
    }
    await targetDirectory.parent.create(recursive: true);
    await _rejectOverlappingDirectories(sourceDirectory, targetDirectory);
    await _rejectOverlappingDirectories(
      vulkanSourceFile.parent,
      targetDirectory,
    );
  } on AndroidVulkanShaderOverlayException {
    rethrow;
  } on FileSystemException catch (error) {
    throw AndroidVulkanShaderOverlayException(
      'Could not validate the Android Vulkan shader overlay directories: '
      '${_boundedOsError(error)}.',
    );
  }

  RandomAccessFile? lockHandle;
  try {
    lockHandle = await File(
      '${targetDirectory.path}.lock',
    ).open(mode: FileMode.append);
    await lockHandle.lock(FileLock.exclusive);
  } on FileSystemException catch (error) {
    await _releaseOverlayLock(lockHandle);
    throw AndroidVulkanShaderOverlayException(
      'Could not lock the Android Vulkan shader overlay output: '
      '${_boundedOsError(error)}.',
    );
  }

  final stagingDirectory = Directory(
    '${targetDirectory.path}.staging-$pid-'
    '${DateTime.now().microsecondsSinceEpoch}',
  );
  try {
    await stagingDirectory.create(recursive: true);
    await _copyShaderTree(sourceDirectory, stagingDirectory);
    await _patchOverlayFile(
      stagingDirectory,
      'dequant_funcs.glsl',
      patchPinnedAndroidVulkanDequantFuncs,
    );
    await _patchOverlayFile(
      stagingDirectory,
      'mul_mm_funcs.glsl',
      patchPinnedAndroidVulkanMulMmFuncs,
    );
    await _patchOverlayFile(
      stagingDirectory,
      'mul_mm.comp',
      patchPinnedAndroidVulkanMulMmComp,
    );
    await _patchExternalOverlayFile(
      vulkanSourceFile,
      stagingDirectory,
      'ggml-vulkan.cpp',
      patchPinnedAndroidGgmlVulkan,
    );

    if (targetDirectory.existsSync()) {
      await targetDirectory.delete(recursive: true);
    } else if (FileSystemEntity.typeSync(
          targetDirectory.path,
          followLinks: false,
        ) !=
        FileSystemEntityType.notFound) {
      throw const AndroidVulkanShaderOverlayException(
        'Android Vulkan shader overlay output is not a directory.',
      );
    }
    await stagingDirectory.rename(targetDirectory.path);
  } on AndroidVulkanShaderOverlayException {
    rethrow;
  } on FileSystemException catch (error) {
    throw AndroidVulkanShaderOverlayException(
      'Could not prepare the Android Vulkan shader overlay: '
      '${_boundedOsError(error)}.',
    );
  } finally {
    if (stagingDirectory.existsSync()) {
      await stagingDirectory.delete(recursive: true);
    }
    await _releaseOverlayLock(lockHandle);
  }
}

Future<void> _releaseOverlayLock(RandomAccessFile? handle) async {
  if (handle == null) {
    return;
  }
  try {
    await handle.unlock();
  } on FileSystemException {
    // Closing the descriptor below also releases the process-owned lock.
  }
  try {
    await handle.close();
  } on FileSystemException {
    // The overlay is already settled; do not replace its build result with a
    // cleanup-only failure.
  }
}

Future<void> _rejectOverlappingDirectories(
  Directory sourceDirectory,
  Directory outputDirectory,
) async {
  final sourcePath = await sourceDirectory.resolveSymbolicLinks();
  final outputPath = outputDirectory.existsSync()
      ? await outputDirectory.resolveSymbolicLinks()
      : await _resolvedMissingDirectoryPath(outputDirectory);

  final normalizedSource = _normalizedFileSystemPath(sourcePath);
  final normalizedOutput = _normalizedFileSystemPath(outputPath);
  if (_pathContains(normalizedSource, normalizedOutput) ||
      _pathContains(normalizedOutput, normalizedSource)) {
    throw const AndroidVulkanShaderOverlayException(
      'Android Vulkan shader source and output directories overlap.',
    );
  }
}

Future<String> _resolvedMissingDirectoryPath(Directory directory) async {
  final outputName = directory.absolute.uri.pathSegments
      .where((segment) => segment.isNotEmpty)
      .last;
  final parentPath = await directory.parent.resolveSymbolicLinks();
  return Directory(
    '$parentPath${Platform.pathSeparator}$outputName',
  ).absolute.path;
}

String _normalizedFileSystemPath(String path) {
  final absolute = Directory(path).absolute.path;
  return Platform.isWindows ? absolute.toLowerCase() : absolute;
}

bool _pathContains(String parent, String child) =>
    parent == child || child.startsWith('$parent${Platform.pathSeparator}');

Future<void> _copyShaderTree(Directory source, Directory output) async {
  final sourcePrefix = source.path.endsWith(Platform.pathSeparator)
      ? source.path
      : '${source.path}${Platform.pathSeparator}';
  await for (final entity in source.list(recursive: true, followLinks: false)) {
    if (!entity.path.startsWith(sourcePrefix)) {
      throw const AndroidVulkanShaderOverlayException(
        'Android Vulkan shader source escaped its directory.',
      );
    }
    final relativePath = entity.path.substring(sourcePrefix.length);
    final destinationPath =
        '${output.path}${Platform.pathSeparator}$relativePath';
    if (entity is Directory) {
      await Directory(destinationPath).create(recursive: true);
    } else if (entity is File) {
      await File(destinationPath).parent.create(recursive: true);
      await entity.copy(destinationPath);
    } else {
      throw const AndroidVulkanShaderOverlayException(
        'Android Vulkan shader source contains an unsupported entry.',
      );
    }
  }
}

Future<void> _patchOverlayFile(
  Directory output,
  String relativePath,
  String Function(String) transform,
) async {
  final file = File(
    '${output.path}${Platform.pathSeparator}'
    '${relativePath.replaceAll('/', Platform.pathSeparator)}',
  );
  final patched = transform(await file.readAsString());
  await file.writeAsString(patched, flush: true);
}

Future<void> _patchExternalOverlayFile(
  File source,
  Directory output,
  String relativePath,
  String Function(String) transform,
) async {
  final file = File(
    '${output.path}${Platform.pathSeparator}'
    '${relativePath.replaceAll('/', Platform.pathSeparator)}',
  );
  final patched = transform(await source.readAsString());
  await file.writeAsString(patched, flush: true);
}

int _occurrences(String source, String pattern) =>
    source.split(pattern).length - 1;

String _sha256(String source) => sha256.convert(utf8.encode(source)).toString();

String _boundedOsError(FileSystemException error) {
  final raw = (error.osError?.message ?? error.message)
      .replaceAll(RegExp(r'[\r\n]+'), ' ')
      .trim();
  if (raw.isEmpty) {
    return 'filesystem operation failed';
  }
  const maximumLength = 160;
  return raw.length <= maximumLength
      ? raw
      : '${raw.substring(0, maximumLength)}…';
}

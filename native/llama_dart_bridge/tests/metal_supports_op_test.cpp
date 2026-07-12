#include "ggml-backend.h"
#include "ggml-metal-device.h"
#include "ggml-metal.h"
#include "ggml.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <array>
#include <cassert>

namespace {

bool supports_op(ggml_backend_dev_t device, enum ggml_op operation,
                 enum ggml_type source_type) {
  ggml_tensor source{};
  source.type = source_type;

  ggml_tensor op{};
  op.type = GGML_TYPE_F32;
  op.op = operation;
  op.src[0] = &source;

  return ggml_backend_dev_supports_op(device, &op);
}

}  // namespace

int main() {
  ggml_backend_reg_t registry = ggml_backend_metal_reg();
  if (registry == nullptr || ggml_backend_reg_dev_count(registry) == 0) {
    return 0;
  }

  ggml_backend_dev_t device = ggml_backend_reg_dev_get(registry, 0);
  assert(device != nullptr);

  constexpr std::array unsupported_types{
      GGML_TYPE_TQ1_0,
      GGML_TYPE_TQ2_0,
      GGML_TYPE_Q8_K,
      GGML_TYPE_Q2_0,
      GGML_TYPE_NVFP4,
      GGML_TYPE_I8,
      GGML_TYPE_F64,
  };
  for (const enum ggml_type type : unsupported_types) {
    assert(!ggml_metal_supports_mul_mat_type(type));
    assert(!ggml_metal_supports_get_rows_type(type));
    assert(!supports_op(device, GGML_OP_MUL_MAT, type));
    assert(!supports_op(device, GGML_OP_MUL_MAT_ID, type));
    assert(!supports_op(device, GGML_OP_GET_ROWS, type));
  }

  // These kernels are present on every Metal target.  This guards against an
  // accidentally over-restrictive capability filter.
  assert(ggml_metal_supports_mul_mat_type(GGML_TYPE_F32));
  assert(ggml_metal_supports_mul_mat_type(GGML_TYPE_Q4_0));
  assert(ggml_metal_supports_get_rows_type(GGML_TYPE_F32));
  assert(ggml_metal_supports_get_rows_type(GGML_TYPE_Q4_0));
  assert(ggml_metal_supports_get_rows_type(GGML_TYPE_I32));
  assert(supports_op(device, GGML_OP_GET_ROWS, GGML_TYPE_F32));
  assert(supports_op(device, GGML_OP_GET_ROWS, GGML_TYPE_Q4_0));

  return 0;
}

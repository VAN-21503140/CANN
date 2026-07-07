// Host-side tiling and operator registration.
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/fast_gelu_tiling.h"
#include "../op_kernel/tiling_key_fast_gelu.h"

constexpr uint32_t TILE_ELEM_NUM = 2048;
constexpr uint32_t LARGE_CORE_THRESHOLD = TILE_ELEM_NUM * 4;

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        ge::DataType dtype_x = tensor_x->GetDataType();
        uint32_t length_x = static_cast<uint32_t>(tensor_x->GetShapeSize());

        uint32_t DT_X = static_cast<uint32_t>(dtype_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);

        FastGeluTilingData *tiling = context->GetTilingData<FastGeluTilingData>();
        tiling->length = length_x;

        uint32_t block_dim = 1U;
        if (length_x > 0) {
            uint32_t max_core_num = num_cores_aiv > 0 ? static_cast<uint32_t>(num_cores_aiv) : 1U;
            uint32_t needed_core_num = (length_x + TILE_ELEM_NUM - 1) / TILE_ELEM_NUM;
            if (length_x > LARGE_CORE_THRESHOLD) {
                block_dim = max_core_num;
                if (length_x < block_dim) {
                    block_dim = length_x;
                }
            } else {
                block_dim = needed_core_num < max_core_num ? needed_core_num : max_core_num;
            }
        }
        context->SetBlockDim(block_dim);

        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *x_shape = context->GetInputShape(0);
        gert::Shape *y_shape = context->GetOutputShape(0);
        *y_shape = *x_shape;
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        return ge::GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
    class FastGelu : public OpDef {
    public:
        explicit FastGelu(const char *name) : OpDef(name) {
            this->Input("x")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("y")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(FastGelu);
}  // namespace ops
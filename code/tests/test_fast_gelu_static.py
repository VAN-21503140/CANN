import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
HOST = ROOT / "op_host" / "fast_gelu.cpp"
KERNEL = ROOT / "op_kernel" / "fast_gelu.cpp"
TILING = ROOT / "op_kernel" / "fast_gelu_tiling.h"


def read(path):
    return path.read_text(encoding="utf-8")


class FastGeluStaticTests(unittest.TestCase):
    def test_host_infers_output_shape_and_dtype_from_input(self):
        host = read(HOST)
        self.assertIn("GetInputShape", host)
        self.assertIn("GetOutputShape", host)
        self.assertIn("*y_shape = *x_shape", host)
        self.assertIn('Input("x")', host)
        self.assertIn('Output("y")', host)
        self.assertGreaterEqual(host.count("ge::DT_FLOAT16"), 2)
        self.assertGreaterEqual(host.count("ge::DT_FLOAT"), 2)

    def test_tiling_keeps_total_length_for_dynamic_kernel(self):
        tiling = read(TILING)
        self.assertIn("uint32_t totalLength", tiling)
        self.assertIn("uint32_t smallCoreDataNum", tiling)
        self.assertIn("uint32_t bigCoreDataNum", tiling)
        self.assertIn("uint32_t tileDataNum", tiling)
        self.assertIn("uint32_t tailBlockNum", tiling)
        self.assertIn("FastGeluTilingData", tiling)

    def test_host_can_cap_aiv_core_count_for_experiments(self):
        host = read(HOST)
        self.assertIn("AIV_CORE_CAP", host)
        self.assertIn("if (max_core_num > AIV_CORE_CAP)", host)
        self.assertIn("max_core_num = AIV_CORE_CAP", host)

    def test_kernel_splits_work_by_core_and_handles_empty_cores(self):
        kernel = read(KERNEL)
        self.assertIn("GetBlockIdx()", kernel)
        self.assertIn("coreDataNum_", kernel)
        self.assertIn("realCoreDataNum_", kernel)
        self.assertIn("tileNum_", kernel)
        self.assertIn("== 0", kernel)
        self.assertIn("return;", kernel)

    def test_kernel_uses_tail_safe_padded_copies(self):
        kernel = read(KERNEL)
        self.assertIn("DataCopyPad", kernel)
        self.assertIn("DataCopyExtParams", kernel)
        self.assertIn("DataCopyPadExtParams", kernel)

    def test_kernel_implements_fast_gelu_vector_formula(self):
        kernel = read(KERNEL)
        for token in [
            "Muls(",
            "Sigmoid(",
            "Mul(",
            "1.702",
        ]:
            self.assertIn(token, kernel)

    def test_kernel_uses_balanced_queues_and_local_buffers(self):
        kernel = read(KERNEL)
        for token in [
            "TPipe",
            "TQue",
            "AllocTensor",
            "EnQue",
            "DeQue",
            "FreeTensor",
            "TBuf",
        ]:
            self.assertIn(token, kernel)


if __name__ == "__main__":
    unittest.main()

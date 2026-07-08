# Generalized Tiling Strategy

Use this reference for generalized tiling, 泛化 tiling, tilling, tiling设计, 通用算子, 算子策略, 对齐原则, 多核均衡, and 访存优化 in Huawei Ascend C / CANN operators.

Primary source:

- `cann-learning-hub/tutorials/ascendc_operator_development/03_intermediate_vector_operator_development/03.04_generalized_tiling_design.ipynb`

## Core Idea

A generalized Ascend C operator supports a class of valid dtypes, shapes, and hardware variants. Kernel-side code defines the compute logic; Host-side Tiling defines how runtime data is split across AI Cores and UB tiles. A generalized operator needs generalized Tiling because fixed shapes, fixed tile sizes, and hardcoded `blockDim` break once input size or hardware changes.

## Design Principles

1. **32B alignment / 对齐原则**
   - Treat 32 bytes as the minimum Tiling granularity for UB-related data movement.
   - Convert element count to bytes using dtype size.
   - Round total byte length up to a multiple of 32B before calculating block distribution.

2. **Memory-access optimization / 访存优化**
   - Use as much UB as safely possible per tile to reduce GM-to-UB copy frequency.
   - Account for all live buffers. For Add-like operators, UB is split across `x`, `y`, and `z`.
   - Compute tile capacity from UB size, buffer count, 32B granularity, and dtype size.

3. **Multi-core balance / 多核均衡**
   - Select actual `blockDim` from available AI Core count and aligned 32B block count.
   - Avoid assigning work to cores that would receive no 32B block.
   - Split remainder blocks across the first `tailBlockNum` cores.

4. **Tail-core and tail-tile handling**
   - Big cores process one more 32B block than small cores.
   - Each core may need multiple UB tiles.
   - The last tile on each core uses `bigTailDataNum` or `smallTailDataNum`.

## Four Partition Cases

| Case | Meaning | Required Handling |
| --- | --- | --- |
| core-even, tile-even | cores equal, tiles equal | simplest loop |
| core-even, tile-uneven | cores equal, last tile differs | tail-tile handling |
| core-uneven, tile-even | big/small cores differ, tiles equal | tail-core offset handling |
| core-uneven, tile-uneven | big/small cores differ and last tiles differ | tail-core plus tail-tile handling |

## Worked Pattern

Example from the source notebook:

- input shape: `(1, 660)`
- dtype: `half`
- dtype size: 2B
- raw length: `660 * 2 = 1320B`
- aligned length: `1344B`
- aligned elements: 672 half elements
- 32B blocks: 42
- AI Cores: 4

Core distribution:

- base blocks per core: `42 // 4 = 10`
- remainder blocks: `42 % 4 = 2`
- first 2 cores are big cores: 11 blocks each
- remaining 2 cores are small cores: 10 blocks each

If example UB capacity is 768B and the operator has 3 buffers (`x`, `y`, `z`):

- single-buffer UB capacity: `768 / 3 = 256B`
- tile capacity: 8 blocks of 32B
- big core tiles: `8 + 3` blocks
- small core tiles: `8 + 2` blocks

## Tiling Data Fields

Reusable field set for generalized elementwise operators:

```cpp
struct TilingDataTemplate {
  uint32_t smallCoreDataNum;
  uint32_t bigCoreDataNum;
  uint32_t finalBigTileNum;
  uint32_t finalSmallTileNum;
  uint32_t tileDataNum;
  uint32_t smallTailDataNum;
  uint32_t bigTailDataNum;
  uint32_t tailBlockNum;
};
```

Field intent:

- `smallCoreDataNum`: valid elements processed by small cores.
- `bigCoreDataNum`: valid elements processed by big cores.
- `finalSmallTileNum`: loop count for small cores.
- `finalBigTileNum`: loop count for big cores.
- `tileDataNum`: normal per-tile element count.
- `smallTailDataNum`: last-tile element count for small cores.
- `bigTailDataNum`: last-tile element count for big cores.
- `tailBlockNum`: number of big cores.

## Host-Side Checklist

In `TilingFunc`:

1. Build platform object with `platform_ascendc::PlatformAscendC(context->GetPlatformInfo())`.
2. Get available AI Core count with `GetCoreNum()`.
3. Get input element count with `context->GetInputShape(0)->GetStorageShape().GetShapeSize()`.
4. Get dtype byte length with `ge::TypeUtils::GetDataTypeLength(...)`.
5. Convert element count to bytes and align to 32B.
6. Choose actual `coreNum`, then call `context->SetBlockDim(coreNum)`.
7. Compute base 32B blocks per core and `tailBlockNum`.
8. Get UB size with `GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize)`.
9. Compute `tileBlockNum`, `tileDataNum`, big/small tile counts, and big/small tail element counts.
10. Write all fields to `context->GetTilingData<TilingDataTemplate>()`.

## Kernel-Side Checklist

In Kernel `Init`:

1. Use `GetBlockIdx()` to determine current core.
2. Compare block index with `tailBlockNum`.
3. For big cores, use `bigCoreDataNum`, `finalBigTileNum`, and `bigTailDataNum`.
4. For small cores, use `smallCoreDataNum`, `finalSmallTileNum`, and `smallTailDataNum`.
5. Calculate GM offset so small cores start after all big-core regions and prior small-core regions.
6. Bind `GlobalTensor` ranges with the selected per-core length.
7. Initialize `TPipe` / `TQue` buffers with `tileDataNum`.

In Kernel `Process`:

1. Loop `tileNum` times.
2. Use `tileDataNum` for normal iterations.
3. Use `tailDataNum` for the final iteration.
4. Keep queue flow balanced: `AllocTensor`, `EnQue`, `DeQue`, `FreeTensor`.

In kernel entry:

```cpp
REGISTER_TILING_DEFAULT(TilingDataTemplate);
GET_TILING_DATA_WITH_STRUCT(TilingDataTemplate, tiling_data, tiling);
```

Pass the extracted fields into the operator class `Init`.

## Training Exercises

Good AI training tasks from this pattern:

- Calculate generalized Tiling fields for a new shape, dtype, core count, and UB size.
- Fill missing Host-side `TilingFunc` code.
- Fill missing Kernel `Init` offset logic for big and small cores.
- Fix a bug where `blockDim` exceeds aligned 32B block count.
- Fix a bug where small-core offset ignores prior big-core extra blocks.
- Transfer generalized Add to generalized Sub while preserving Tiling logic.

## Common Mistakes

- Treating `tilling` as just a misspelling and missing the user's intent: they mean Tiling design.
- Using all available cores even when input has fewer 32B blocks.
- Mixing byte counts and element counts in the same field.
- Forgetting dtype size when converting aligned bytes back to elements.
- Hardcoding `blockDim`, tile size, or dtype.
- Handling tail tile but forgetting tail core.
- Assuming padded aligned length means real GM contains safe extra data without verifying the framework/test setup.

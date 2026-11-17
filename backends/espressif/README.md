# ESP-NN backend for Espressif chips

This backend is a **prototype under active development**.

This backend runs int8-quantized PyTorch models on Espressif microcontrollers.
It is tuned for the **ESP32-S3** and also runs on the original ESP32. It
delegates quantized subgraphs to [ESP-NN](https://github.com/espressif/esp-nn),
Espressif's kernel library. ESP-NN has SIMD kernels for the ESP32-S3 and
portable C kernels for the other chips.

A lowered model is a normal `.pte` program. `EspNNBackend` runs the int8
partitions with ESP-NN, and ExecuTorch's portable and quantized kernels run
everything else (float ops, unsupported layers, pre- and post-processing).
Each delegate payload is a compact int8 graph (XTS3) that the delegate's
interpreter executes. A host reference interpreter reproduces the device
arithmetic, so fully delegated programs are checked bit-exactly.

The Fused Quant **operator-library flow is under development** alongside the
delegate. The two flows have distinct quantizers:

| Quantizer | Export flow |
|---|---|
| `EspNNQuantizer` | Existing delegate: PT2E Q/DQ graphs partitioned by `EspNNPartitioner` into `EspNNBackend` delegates |
| `EspFusedQuantizer` | Operator library: shared Fused Quant annotation and fusion, followed by `EspPassManager` lowering to direct ESP-NN operators or portable fallback |

Both flows use ESP-NN 1.5.0 kernels. The operator library currently accelerates
int8 tensor add with equal input shapes and multiply with equal shapes or a
contiguous trailing block broadcast across the full tensor, including
one-element tensors. Sigmoid, tanh, GELU, SiLU, HardSwish, HardSigmoid, ReLU,
and HardTanh (including ReLU6) support static, contiguous int8 tensors with
constant per-tensor input and output qparams. Export evaluates all 256 input
codes with the FQ reference, using calibrated qparams, and stores the results
as a 256-byte constant table. The `quantized_lut_int8` operator applies that
table using ESP-NN's logistic lookup kernel, without scratch storage.
It also accelerates values-only 4-D int8 MaxPool2d with matching input
and output qparams, unit dilation, floor mode, and channel counts divisible by
four. AvgPool2d uses the same shape and channel restrictions, with shared
input/output qparams, floor mode, and no divisor override. Padding is supported
when `count_include_pad=False`.
Its integer kernel rounds ties away from zero; the FQ reference rounds ties to
even, so results can differ by one int8 step.
Linear supports static, contiguous int8 inputs of rank one or higher, constant
per-output-channel symmetric int8 weights, and optional constant int32 bias.
`EspFusedQuantizer` derives the bias scales from input and weight scales.
Leading input dimensions are treated as batch rows by ESP-NN's batched fully
connected API. Unsupported linear calls fail at export with the reason;
portable linear fallback is not available yet, even when CPU fallback is enabled.
Conv2d supports static 4-D int8 inputs, constant per-output-channel symmetric
weights, optional derived int32 bias, and unit dilation. Ordinary convolution
uses `groups=1`; depthwise uses `groups=input_channels` and supports positive
integer channel multipliers within the vendor's dimension and indexing bounds.
Batches, rectangular kernels, stride and padding are supported. ESP-NN selects
SIMD or vendor C kernels for each shape; constant depthwise weights are packed
into its `1HWO` order at export. Other convolution configurations retain portable
execution.

Set the recipe's `EspTarget` and `EspKernelMode` to match the device's ESP-NN
configuration when planning scratch:

```python
recipe = get_esp_fused_quant_recipe(
    target=EspTarget.ESP32P4, kernel_mode=EspKernelMode.OPTIMIZED
)
```

The defaults are `EspTarget.ESP32S3` and `EspKernelMode.OPTIMIZED`.
Accelerated Conv2d export requires the optional [scratch-size bindings](#scratch-size-queries).
If the selected binding or query is unavailable, convolution retains portable
execution; strict export rejects that fallback. Other operator exports do not
require the bindings.

Lowering adds a scratch Tensor to each convolution call, and ordinary memory
planning assigns its storage and 16-byte alignment. The device checks its
required size before using the buffer. Scratch follows the planned arena's
memory placement. ESP-NN uses a global scratch pointer: applications must
serialize operator-library inference and must not run it concurrently with
the delegate or other ESP-NN calls.

The shared Fused Quant layout pass introduces contiguous NHWC tensors for
spatial operators; model inputs and outputs retain their original layout.
Required layout copies are visible in the operator-library report and accepted
in strict mode. Unsupported pooling configurations retain portable execution.
Its shared quantization rules also cover operators that
still execute through portable fallback. The delegate's supported operators
and export example are described below.

## Scratch-size queries

Optional Python bindings expose ESP-NN 1.5.0's convolution, depthwise, softmax,
mean, hard-swish, batched fully connected, and logistic scratch-size queries:

```bash
CMAKE_ARGS="-DEXECUTORCH_BUILD_ESP_NN_PYBINDS=ON" pip install -e . --no-build-isolation
```

The host libraries retain only sizing code. CMake fetches the pinned ESP-NN
sources; `ESP_NN_LOCAL_PATH` can reuse an existing 1.5.0 checkout or ESP-IDF
managed component.

Select the device's `EspTarget` and `EspKernelMode` to match its ESP-NN build:

```python
from executorch.backends.espressif import EspKernelMode, EspTarget
from executorch.backends.espressif.scratch import get_scratch_queries

queries = get_scratch_queries(EspTarget.ESP32P4, EspKernelMode.OPTIMIZED)
if queries is not None:
    scratch_bytes = queries.conv2d_scratch_size(
        input_dims=(8, 8, 16),
        filter_dims=(3, 3, 16),
        output_dims=(8, 8, 32),
        stride=(1, 1),
        padding=(1, 1),
    )
```

Convolution queries use `(width, height, channels)` for dimensions,
`(width, height)` for stride and padding, and unit dilation. Dimensions,
strides, and depthwise channel multipliers must be in `[1, 65535]`; padding must be in
`[0, 65535]`. Invalid values raise `ValueError`. Sizes are in bytes,
including zero for kernels that need no scratch. `get_scratch_queries` returns
`None` when the selected binding is unavailable.

Optimized mode selects S3 queries for ESP32-S3, shared PIE queries for
ESP32-P4/ESP32-S31, and generic queries for other `EspTarget` values. ANSI mode
selects ANSI queries for every chip. Query results are checked on the host for
all configurations; device execution tests currently cover ESP32-S3.

## Contents

```
backends/espressif/
  espnn_preprocess.py        EspNNBackend: BackendDetails.preprocess -> XTS3 graph
  partition/                 EspNNPartitioner: int8 regions -> delegate, rest -> CPU
  quantizer/                 EspNNQuantizer (delegate), EspFusedQuantizer (operator library)
  lowering.py                delegate lowering, Fused Quant operator-library recipe + report
  serialization/             XTS3 format writer and parser, op lowering rules
  reference/                 bit-exact host model of the ESP-NN kernels and the CPU fallback
  espnn_runtime/             ESP-IDF component: XTS3 interpreter + ExecuTorch BackendInterface
  test/                      pytest suite (lowering, operators, CPU fallback, parser)
examples/espressif/
  quantize_and_lower.py           minimal quantize -> lower -> .pte example
  export_espnn_test_programs.py   the test models as BundleIO programs for the runner
  camera_stream/                  live camera classification on the ExecuTorch runtime
```

Programs run on the board with the existing
[`examples/espressif/project`](../../examples/espressif/project) runner, built
with `-DET_ESPNN_BACKEND=ON`.

## Supported operators

All delegated operators are int8 with per-tensor activations and per-channel
symmetric weights, as annotated by `EspNNQuantizer`.

| Group | Operators |
|---|---|
| Convolution | `conv2d` (any stride and padding, groups 1 or depthwise, dilation through phase decomposition), `conv_transpose2d`, `linear` |
| Activations | `relu`, `relu6`, `hardtanh`, plus table lookups for `hardswish`, `hardsigmoid`, `sigmoid`, `silu`, `tanh`, `gelu`, `elu`, `leaky_relu` |
| Elementwise | `add`, `mul` (with broadcasting), `softmax` (last dim) |
| Pooling | `max_pool2d`, `avg_pool2d`, `adaptive_avg_pool2d`, `mean` over H and W |
| Resize | `upsample_nearest2d`, `upsample_bilinear2d` (`align_corners` either way) |
| Data movement | `reshape`, `view`, `flatten`, `unflatten`, `permute`, `transpose`, `squeeze`, `unsqueeze`, `slice`, `select`, `expand`, `pad`, `cat`, `stack`, `split`, `chunk`, `unbind` |

Batch norm that follows a convolution is folded into it by PT2E
(`prepare_pt2e` / `prepare_qat_pt2e`); dropout in an exported eval-mode
model is a no-op.
Partitions can have several inputs and outputs. Anything else stays in the
program and runs on the CPU, including float ops, LSTM, layer norm and
attention matmuls. The partitioner reports what it delegated and what it left
behind. Pass `allow_cpu_fallback=False` to make leftover operators an error.

## Quantize and lower with the delegate

[`examples/espressif/quantize_and_lower.py`](../../examples/espressif/quantize_and_lower.py)
contains the whole flow:

```python
import torch
import torchvision
from executorch.backends.espressif import EspNNQuantizer, to_espnn_edge
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

model = torchvision.models.mobilenet_v2(weights="DEFAULT").eval()
example_inputs = (torch.randn(1, 3, 224, 224),)

# 1. Capture the model and insert observers for the ESP-NN int8 scheme
exported = torch.export.export(model, example_inputs).module()
prepared = prepare_pt2e(exported, EspNNQuantizer())

# 2. Calibrate (use real samples from your dataset here)
with torch.no_grad():
    for images in calibration_batches:
        prepared(images)
quantized = convert_pt2e(prepared)

# 3. Lower: int8 regions -> EspNNBackend delegates, the rest -> CPU kernels
edge, report = to_espnn_edge(torch.export.export(quantized, example_inputs))
print(report["partitions"], report["delegated_ops"], report["cpu_ops"])

# 4. Serialize the ExecuTorch program
with open("mobilenet_v2_espnn.pte", "wb") as f:
    f.write(edge.to_executorch().buffer)
```

```
$ python examples/espressif/quantize_and_lower.py
1 EspNN partition(s), 101 delegated operators, 0 operators left for the CPU
wrote mobilenet_v2_espnn.pte
```

Calibrate on a few hundred real samples; random data gives useless scales.

## Export with the Fused Quant operator library

Use the shared export API with `get_esp_fused_quant_recipe()`. The recipe
configures `EspFusedQuantizer`, calibrates the model, fuses Q/DQ into Fused
Quant operators, and runs `EspPassManager`. Unsupported operators decompose
to portable kernels before serialization. No delegate partitions are created.

```python
from executorch.backends.espressif import (
    get_esp_fused_quant_recipe,
    get_operator_library_report,
)
from executorch.export import export

session = export(
    model.eval(),
    example_inputs=[example_inputs],
    export_recipe=get_esp_fused_quant_recipe(
        calibration_inputs_fn=lambda: calibration_samples,
    ),
)
edge = session.get_edge_program_manager()
print(get_operator_library_report(edge))
with open("model_esp_ops.pte", "wb") as f:
    f.write(session.get_executorch_program_manager().buffer)
```

Each calibration sample is a tuple matching the model's inputs. Pass
`allow_cpu_fallback=False` to the recipe to reject remaining portable compute;
Q/DQ boundary operators are allowed. The report lists portable compute and
counts accelerated operators across all methods.

For this flow, build the existing runner with `-DET_ESPRESSIF_OPS=ON` and
`-DET_ESPNN_BACKEND=OFF`, using `model_esp_ops.pte` as the model. The runtime
still needs the portable fallback operators and Q/DQ kernels described below.
`to_espnn_edge()` is the separate delegate entry point; its former
`use_operator_library` option is replaced by this recipe.

## Running on the ESP32-S3

You need ESP-IDF and an activated ESP-IDF shell.
[`examples/espressif/setup.sh`](../../examples/espressif/setup.sh) installs a
matching ESP-IDF and the emulator.

### 1. Cross-compile ExecuTorch

Build the runtime with the `esp-baremetal` preset. Select the portable CPU
operators your program needs. The quantized kernels provide the
quantize/dequantize operators at the delegate boundaries. The program lists
its operators:

```python
plan = edge.to_executorch().executorch_program.execution_plan[0]
print(",".join(sorted(
    f"{op.name}.{op.overload}" for op in plan.operators
    if not op.name.startswith("quantized_decomposed::")
)))
```

```bash
cmake --preset esp-baremetal -G Ninja -B cmake-out-esp \
    -DCMAKE_TOOLCHAIN_FILE="${IDF_PATH}/tools/cmake/toolchain-esp32s3.cmake" \
    -DCMAKE_BUILD_TYPE=Release \
    -DEXECUTORCH_BUILD_DEVTOOLS=ON \
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
    -DEXECUTORCH_SELECT_OPS_LIST=<operators printed above> \
    -DMAX_KERNEL_NUM=256 \
    -DEXECUTORCH_BUILD_PTHREADPOOL=OFF -DEXECUTORCH_BUILD_CPUINFO=OFF
cmake --build cmake-out-esp --target install
```

- `quantized_ops_lib` registers kernels beyond the selected list, so
  `MAX_KERNEL_NUM` is set explicitly instead of being sized from the
  selection.
- pthreadpool needs POSIX calls that ESP-IDF doesn't provide. With it off,
  the quantized kernels run single-threaded.
- The runner links `executorch_selected_kernels`, which only exists when a
  selection is given. A fully delegated program with no CPU operators
  still needs at least one selected operator, such as `aten::add.out`.

### 2. Build the runner with the ESP-NN backend

```bash
idf.py -C examples/espressif/project -B build-espnn \
    -DIDF_TARGET=esp32s3 \
    -DET_ESPNN_BACKEND=ON \
    -DET_BUILD_DIR_PATH=$PWD/cmake-out-esp \
    -DET_PTE_FILE_PATH=$PWD/mobilenet_v2_espnn.pte \
    build flash monitor
```

`ET_ESPNN_BACKEND` adds
[`backends/espressif/espnn_runtime`](espnn_runtime) to the project. The
component pulls in `espressif/esp-nn` from the component registry, links
against the ExecuTorch targets that `executor_runner` imports, and registers
`EspNNBackend` at startup. It defines no platform layer; the runner's
`esp_pal.cpp` provides it.

ESP-NN defaults to `CONFIG_NN_OPTIMIZED=y` and selects its implementation from
ESP-IDF's `IDF_TARGET`. For `esp32s3`, this enables the S3 assembly kernels.
`CONFIG_NN_ANSI_C=y` selects reference C kernels for verification and debugging.

For a BundleIO program (`.bpte`), add `-DET_BUNDLE_IO=ON` and the tolerances
`-DET_ATOL=... -DET_RTOL=...`. The runner then checks the outputs against the
bundled reference and logs `Test_result: PASS` or `FAIL`.

Other options, in menuconfig under *ExecuTorch ESP-NN backend*:

| Kconfig | Default | Effect |
|---|---|---|
| `ESPNN_DELEGATE_CONSTANTS_TO_PSRAM` | y | Copies weights from memory-mapped flash into PSRAM when they fit (faster than flash cache misses on large models) |
| `ESPNN_DELEGATE_DUAL_CORE` | n | Splits large convolutions across both cores |
| `ESPNN_DELEGATE_PROFILE` | n | Prints a per-operator time table after a delegate's first run |

### The original ESP32

The delegate also runs on the ESP32. ESP-NN has no SIMD kernels for it and
uses portable C, about 13 million int8 multiply-accumulates per second at
240 MHz, roughly an order of magnitude slower than the S3. Most ESP32 modules
have no PSRAM, so activations must fit in about 280 KB of internal RAM.

## Testing

```bash
pytest backends/espressif/test
```

The tests export small models for every operator group, lower them, run the
XTS3 graphs through the reference interpreter and compare the result against
the quantized PyTorch model. `test_xts3_parser_host.py` builds the device
parser for the host and feeds it corrupted graphs.

End to end, `.ci/scripts/test_espressif.sh` runs the same models through the
ExecuTorch runtime on an emulated ESP32-S3.
`examples/espressif/export_espnn_test_programs.py` exports each one as a
BundleIO program with the host reference's outputs. CI builds the runner with
`-DET_ESPNN_BACKEND=ON` for each program and runs it in `esp-emu`. Fully
delegated programs must match to float precision, which means bit-exact int8
results. Programs with CPU regions get a tolerance for the portable kernels'
float rounding. Separate CI jobs run ESP-NN's optimized S3 kernels and its
ANSI C reference kernels against the same bundled outputs. Both jobs target
ESP32-S3. Logs include the kernel configuration in their filenames; artifacts
retain the bundled fixtures, SDK configuration, and resolved component versions.

The code follows the ExecuTorch coding style and passes the repository's
`lintrunner` checks:

```bash
lintrunner --paths-cmd "git ls-files backends/espressif examples/espressif"
```

## Limitations

- Only int8 is delegated. Float32 operators run on ExecuTorch's portable
  kernels, which are slow on these chips.
- Activations are per-tensor and weights are per-channel symmetric. Dynamic
  shapes are not supported, so each program has a fixed input shape.
- Attention matmuls, LSTM/GRU and normalization layers other than folded batch
  norm fall back to the CPU.
- Dual-core execution only helps large convolutions, and it is off by default.
- The delegate handle comes from ExecuTorch's runtime allocator, but each
  delegate allocates its activation arena, scratch buffers and optional PSRAM
  weight copy with ESP-IDF's `heap_caps` allocator at `init()`, so it can
  choose between internal RAM and PSRAM. Nothing is allocated during
  `execute()`.

## License

Every source file in this backend and its examples is released under the
BSD 3-Clause license (`SPDX-License-Identifier: BSD-3-Clause`), the same
license as ExecuTorch. See the `LICENSE` file at the repository root. ESP-NN
is Apache-2.0 and is fetched from the Espressif component registry, not
vendored.

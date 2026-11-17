# Copyright 2026 Mwaniki Njagi.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import operator
from collections.abc import Callable, Iterable

import executorch.kernels.quantized  # noqa: F401  registers quantize/dequantize out variants

from executorch.backends.espressif.partition.espnn_partitioner import EspNNPartitioner
from executorch.backends.espressif.passes import EspPassManager
from executorch.backends.espressif.quantizer import EspFusedQuantizer
from executorch.backends.espressif.target import EspKernelMode, EspTarget
from executorch.exir import (
    EdgeCompileConfig,
    EdgeProgramManager,
    to_edge_transform_and_lower,
)
from executorch.exir.backend.canonical_partitioners.duplicate_dequant_node_pass import (
    DuplicateDequantNodePass,
)
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.memory import alloc
from executorch.export import (
    ExportRecipe,
    LoweringRecipe,
    QuantizationRecipe,
    StageType,
)
from executorch.frontends.fused_quant.quantizer.quantizer import FusedQuantQuantizer
from torch._ops import OpOverload


def to_espnn_edge(program, allow_cpu_fallback: bool = True):
    """Lower a PT2E-quantized program to EspNNBackend delegates and portable ops.

    Returns (edge_manager, report). For the Fused Quant operator-library flow,
    use get_esp_fused_quant_recipe() with executorch.export.export().
    """
    partitioner = EspNNPartitioner(allow_cpu_fallback=allow_cpu_fallback)
    edge = to_edge_transform_and_lower(
        program,
        transform_passes=[DuplicateDequantNodePass()],
        partitioner=[partitioner],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    return edge, partitioner.report


def get_operator_library_report(edge: EdgeProgramManager) -> dict:
    """Count direct ESP-NN operators and list portable compute across methods."""
    esp_nn_ops = 0
    cpu_ops = []
    layout_ops = []
    for method in sorted(edge.methods):
        for node in edge.exported_program(method).graph.nodes:
            if node.op != "call_function" or node.target in (operator.getitem, alloc):
                continue
            target = node.target
            if isinstance(target, (EdgeOpOverload, OpOverload)):
                if target._schema.name == "aten::permute_copy":
                    dims = list(node.args[1])
                    neighbors = (
                        [node.args[0]]
                        if dims == [0, 3, 1, 2]
                        else list(node.users) if dims == [0, 2, 3, 1] else []
                    )
                    if any(
                        isinstance(neighbor.target, (EdgeOpOverload, OpOverload))
                        and neighbor.target._schema.name
                        in {
                            "esp_nn::quantized_max_pool2d_int8",
                            "esp_nn::quantized_avg_pool2d_int8",
                            "esp_nn::quantized_conv2d_int8",
                        }
                        for neighbor in neighbors
                    ):
                        layout_ops.append(
                            {
                                "method": method,
                                "name": node.name,
                                "target": target.name(),
                            }
                        )
                        continue
                if target._schema.name in {
                    "quantized_decomposed::quantize_per_tensor",
                    "quantized_decomposed::dequantize_per_tensor",
                    "quantized_decomposed::quantize_per_channel",
                    "quantized_decomposed::dequantize_per_channel",
                }:
                    continue
                if target.namespace == "esp_nn":
                    esp_nn_ops += 1
                    continue
                target = target.name()
            cpu_ops.append({"method": method, "name": node.name, "target": str(target)})
    return {"esp_nn_ops": esp_nn_ops, "cpu_ops": cpu_ops, "layout_ops": layout_ops}


def _require_no_portable_compute(edge: EdgeProgramManager) -> list:
    report = get_operator_library_report(edge)
    if report["cpu_ops"]:
        raise RuntimeError(
            "The ESP-NN operator library left these operators on the CPU: "
            + ", ".join(
                f"{op['method']}.{op['name']} ({op['target']})"
                for op in report["cpu_ops"]
            )
        )
    return []


def get_esp_fused_quant_recipe(
    calibration_inputs_fn: Callable[[], Iterable[tuple]] | None = None,
    *,
    quantizer: FusedQuantQuantizer | None = None,
    allow_cpu_fallback: bool = True,
    target: EspTarget = EspTarget.ESP32S3,
    kernel_mode: EspKernelMode = EspKernelMode.OPTIMIZED,
) -> ExportRecipe:
    """Quantize and lower a float model for the Espressif operator library.

    Use with executorch.export.export(). Supported Fused Quant operators become
    direct ESP-NN calls; other operators decompose to portable kernels. Setting
    allow_cpu_fallback=False rejects remaining compute, excluding Q/DQ and required NHWC layout copies.
    Unsupported linear calls currently raise even when CPU fallback is enabled.
    Set target and kernel_mode to match the device's ESP-NN configuration
    when planning scratch.
    The EspNNBackend delegate continues to use EspNNQuantizer and to_espnn_edge.
    """
    quantizer = quantizer if quantizer is not None else EspFusedQuantizer()

    def fuse(graph_module):
        quantizer.fuse(graph_module)
        return graph_module

    transforms = [lambda _: EspPassManager(target=target, kernel_mode=kernel_mode)]
    if not allow_cpu_fallback:
        transforms.append(_require_no_portable_compute)
    return ExportRecipe(
        quantization_recipe=QuantizationRecipe(
            quantizers=[quantizer],
            preserve_ops=quantizer.preserved_ops(),
            calibration_inputs_fn=calibration_inputs_fn,
            post_convert_passes=[fuse],
        ),
        lowering_recipe=LoweringRecipe(
            edge_compile_config=EdgeCompileConfig(
                _skip_dim_order=True,
                _core_aten_ops_exception_list=quantizer.preserved_ops(),
            ),
            edge_manager_transform_passes=transforms,
        ),
        pipeline_stages=[
            StageType.QUANTIZE,
            StageType.TORCH_EXPORT,
            StageType.TO_EDGE,
            StageType.EDGE_PROGRAM_MANAGER_TRANSFORM,
            StageType.TO_EXECUTORCH,
        ],
    )


__all__ = ["to_espnn_edge", "get_esp_fused_quant_recipe", "get_operator_library_report"]

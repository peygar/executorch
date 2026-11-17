# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from executorch.backends.espressif import (
    get_esp_fused_quant_recipe,
    get_operator_library_report,
)
from executorch.backends.espressif.test.tester import (
    EspTestCase,
    EspTester,
    ramp_tensor,
)
from executorch.backends.test.suite.operators.test_sigmoid import Model
from executorch.backends.transforms.permute_pass_utils import get_arg
from executorch.backends.transforms.utils import get_constant, get_fqn
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
)
from executorch.exir._serialize._program import deserialize_pte_binary
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.scalar_type import ScalarType
from executorch.exir.schema import KernelCall
from executorch.export import export
from executorch.frontends.fused_quant.quantizer.quantizer import (
    FusedQuantQuantizer,
    OpQuantizer,
)

_FQ_SIGMOID = exir_ops.edge.fused_quant.sigmoid.default
_ESP_LUT = exir_ops.edge.esp_nn.quantized_lut_int8.default
_SIGMOID = exir_ops.edge.aten.sigmoid.default


class EspLut(torch.nn.Module):
    def __init__(self, activation, fused_op):
        super().__init__()
        self.activation = activation
        self.ops_before_transforms = {fused_op: 1}
        self.ops_after_transforms = {fused_op: 0, _ESP_LUT: 1}

    def forward(self, x):
        return self.activation(x)


class EspSigmoid(Model):
    ops_before_transforms = {_FQ_SIGMOID: 1}
    ops_after_transforms = {_FQ_SIGMOID: 0, _ESP_LUT: 1, _SIGMOID: 0}


class EspSigmoidFloatUsers(EspSigmoid):
    def forward(self, x):
        y = torch.sigmoid(x)
        return y, torch.sin(x), torch.sin(y)


class EspSigmoidChain(EspSigmoid):
    ops_before_transforms = {_FQ_SIGMOID: 2}
    ops_after_transforms = {_FQ_SIGMOID: 0, _ESP_LUT: 2, _SIGMOID: 0}

    def forward(self, x):
        return torch.sigmoid(torch.sigmoid(x))


class EspSigmoidTwoScales(EspSigmoidChain):
    def forward(self, x, y):
        return torch.sigmoid(x), torch.sigmoid(y)


activation_cases = {
    name: EspTestCase(EspLut(activation, fused_op), (ramp_tensor(-16, 16, (4097,)),))
    for name, activation, fused_op in (
        ("sigmoid", torch.nn.Sigmoid(), _FQ_SIGMOID),
        ("tanh", torch.nn.Tanh(), exir_ops.edge.fused_quant.tanh.default),
        ("gelu", torch.nn.GELU(), exir_ops.edge.fused_quant.gelu.default),
        (
            "gelu_tanh",
            torch.nn.GELU(approximate="tanh"),
            exir_ops.edge.fused_quant.gelu.default,
        ),
        ("silu", torch.nn.SiLU(), exir_ops.edge.fused_quant.silu.default),
        (
            "hardswish",
            torch.nn.Hardswish(),
            exir_ops.edge.fused_quant.hardswish.default,
        ),
        (
            "hardsigmoid",
            torch.nn.Hardsigmoid(),
            exir_ops.edge.fused_quant.hardsigmoid.default,
        ),
        ("relu", torch.nn.ReLU(), exir_ops.edge.fused_quant.relu.default),
        ("relu6", torch.nn.ReLU6(), exir_ops.edge.fused_quant.hard_tanh.default),
        (
            "hardtanh",
            torch.nn.Hardtanh(min_val=-2.0, max_val=3.0),
            exir_ops.edge.fused_quant.hard_tanh.default,
        ),
    )
}


test_cases = {
    **activation_cases,
    "scalar": EspTestCase(EspSigmoid(), (torch.tensor(-1.0),)),
    "vector_tail": EspTestCase(EspSigmoid(), (ramp_tensor(-3, 5, (17,)),)),
    "matrix": EspTestCase(EspSigmoid(), (ramp_tensor(-3, 5, (3, 19)),)),
    "rank_four": EspTestCase(EspSigmoid(), (ramp_tensor(-3, 5, (2, 3, 5, 7)),)),
    "positive": EspTestCase(EspSigmoid(), (ramp_tensor(0.5, 6, (17,)),)),
    "negative": EspTestCase(EspSigmoid(), (ramp_tensor(-6, -0.5, (17,)),)),
    "zero": EspTestCase(EspSigmoid(), (torch.zeros(17),)),
    "float_users": EspTestCase(EspSigmoidFloatUsers(), (ramp_tensor(-3, 5, (2, 17)),)),
    "chain": EspTestCase(EspSigmoidChain(), (ramp_tensor(-3, 5, (17,)),)),
    "two_scales": EspTestCase(
        EspSigmoidTwoScales(),
        (ramp_tensor(-0.5, 0.5, (17,)), ramp_tensor(-8, 8, (17,))),
    ),
    "saturation": EspTestCase(
        EspSigmoid(),
        (ramp_tensor(-50, 50, (257,)),),
        [(ramp_tensor(-2, 2, (257,)),)],
    ),
}


@pytest.mark.parametrize("case", test_cases.values(), ids=test_cases.keys())
def test_dialect_lut(case):
    tester = EspTester(case.model, case.example_inputs, case.calibration_samples)
    tester.test_dialect(
        case.model.ops_before_transforms,
        case.model.ops_after_transforms,
        atol=0,
        rtol=0,
    )
    program = tester.get_artifact().exported_program()
    for node in program.graph.find_nodes(op="call_function", target=_ESP_LUT):
        table = get_constant(program, get_arg(node, "table"))
        assert table.dtype == torch.int8
        assert table.shape == (256,)
    tester.to_executorch().serialize()
    plan = deserialize_pte_binary(tester.get_artifact()).program.execution_plan[0]
    assert not plan.delegates
    assert all(not op.name.startswith("fused_quant::") for op in plan.operators)
    for chain in plan.chains:
        for instruction in chain.instructions:
            call = instruction.instr_args
            if (
                isinstance(call, KernelCall)
                and plan.operators[call.op_index].name == "esp_nn::quantized_lut_int8"
            ):
                table = plan.values[call.args[1]].val
                assert table.scalar_type == ScalarType.CHAR
                assert table.data_buffer_idx > 0
                assert table.allocation_info is None


@pytest.mark.parametrize("case", test_cases.values(), ids=test_cases.keys())
def test_implementation_lut(case, pytestconfig):
    EspTester(
        case.model, case.example_inputs, case.calibration_samples
    ).test_implementation(
        pytestconfig.rootpath, rtol=torch.finfo(torch.float32).eps, num_inferences=2
    )


def test_implementation_lut_undersized_table(pytestconfig):
    if not os.environ.get("ESPRESSIF_BUILD_DIR"):
        pytest.skip("Set ESPRESSIF_BUILD_DIR to the ESP32-S3 build directory")
    case = test_cases["zero"]
    tester = (
        EspTester(case.model, case.example_inputs)
        .export_fused_quant()
        .to_edge()
        .run_passes()
    )
    program = tester.get_artifact().exported_program()
    (node,) = program.graph.find_nodes(op="call_function", target=_ESP_LUT)
    table = get_arg(node, "table")
    program.constants[get_fqn(program, table)] = get_constant(program, table)[
        :255
    ].clone()
    table.meta["val"] = table.meta["val"].new_empty((255,))
    with pytest.raises(AssertionError):
        tester.test_implementation(pytestconfig.rootpath)
    worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
    log = (
        Path(os.environ["ESPRESSIF_BUILD_DIR"])
        / f"operator-library-{worker}-esp32s3-optimized"
        / "emulator.log"
    )
    assert "table.nbytes()" in log.read_text()


@pytest.mark.parametrize("case", activation_cases.values(), ids=activation_cases.keys())
def test_lut_covers_all_input_codes(case):
    tester = EspTester(case.model, case.example_inputs).test_dialect(
        case.model.ops_before_transforms,
        case.model.ops_after_transforms,
        atol=0,
        rtol=0,
    )
    module = tester.get_artifact().exported_program().module()
    interpreter = torch.fx.Interpreter(module, garbage_collect_values=False)
    interpreter.run(*case.example_inputs)
    (node,) = module.graph.find_nodes(op="call_function", target=_ESP_LUT)
    codes = interpreter.env[get_arg(node, "input")]
    torch.testing.assert_close(
        codes.unique().to(torch.int32),
        torch.arange(-128, 128, dtype=torch.int32),
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize(
    "float_edge",
    ["input_activation", "output_activation"],
    ids=["float_input", "float_output"],
)
def test_sigmoid_unsupported_qparams_stay_portable(float_edge):
    config = replace(get_symmetric_quantization_config(), **{float_edge: None})
    quantizer = FusedQuantQuantizer(
        [
            OpQuantizer(
                torch.ops.aten.sigmoid.default,
                torch.ops.fused_quant.sigmoid.default,
                config,
            )
        ]
    )
    case = test_cases["float_users"]
    tester = (
        EspTester(case.model, case.example_inputs)
        .export_fused_quant(quantizer)
        .to_edge()
    )
    expected = tester.get_artifact().exported_program().module()(*case.example_inputs)
    tester.run_passes().check_node_count({_FQ_SIGMOID: 0, _ESP_LUT: 0, _SIGMOID: 1})
    torch.testing.assert_close(
        tester.get_artifact().exported_program().module()(*case.example_inputs),
        expected,
        atol=0,
        rtol=0,
    )
    tester.to_executorch().serialize()


def test_public_recipe_lut_without_sizing(monkeypatch):
    def unexpected_sizing(*args):
        pytest.fail("A constant lookup table does not require a sizing binding")

    monkeypatch.setattr(
        "executorch.backends.espressif.passes.fused_quant_to_esp_pass.get_scratch_queries",
        unexpected_sizing,
    )
    case = test_cases["two_scales"]
    session = export(
        case.model,
        example_inputs=[case.example_inputs],
        export_recipe=get_esp_fused_quant_recipe(allow_cpu_fallback=False),
    )
    assert get_operator_library_report(session.get_edge_program_manager()) == {
        "esp_nn_ops": 2,
        "cpu_ops": [],
        "layout_ops": [],
    }
    assert session.get_executorch_program_manager().buffer

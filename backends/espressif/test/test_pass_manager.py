# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math

import pytest
import torch
from executorch.backends.espressif import (
    EspNNQuantizer,
    get_esp_fused_quant_recipe,
    get_operator_library_report,
    to_espnn_edge,
)
from executorch.backends.espressif.passes import EspPassManager
from executorch.backends.espressif.test.tester import EspRunPasses, EspTester
from executorch.backends.test.harness.stages import Quantize
from executorch.exir import to_edge, to_edge_transform_and_lower
from executorch.exir._serialize._program import deserialize_pte_binary
from executorch.export import export


class Multiply(torch.nn.Module):
    def forward(self, x, y):
        return x * y


class MultiplyAndFloatUsers(torch.nn.Module):
    def forward(self, x, y):
        product = x * y
        return product, torch.sin(x), torch.sin(product)


def _output_scale(program):
    output = next(node for node in program.graph.nodes if node.op == "output")
    return output.args[0][0].args[1]


@pytest.mark.parametrize("entry_point", ["separate", "combined"])
def test_entry_points_preserve_program_behavior(entry_point):
    inputs = (
        torch.linspace(-1, 1, 6).reshape(2, 3),
        torch.linspace(-2, 2, 6).reshape(2, 3),
    )
    tester = EspTester(MultiplyAndFloatUsers(), inputs).export_fused_quant()
    reference = tester.get_artifact()
    if entry_point == "combined":
        tester.to_edge_transform_and_lower()
    else:
        tester.to_edge().run_passes()
    torch.testing.assert_close(
        tester.get_artifact().exported_program().module()(*inputs),
        reference.module()(*inputs),
        atol=_output_scale(reference),
        rtol=0,
    )
    edge = tester.get_artifact()
    assert edge.compile_config._check_ir_validity
    edge.exported_program().validate()
    tester.to_executorch().serialize()
    program = deserialize_pte_binary(tester.get_artifact()).program
    assert program.execution_plan[0].name == "forward"
    assert not program.execution_plan[0].delegates


def test_public_entry_points_keep_delegate_and_operator_library_separate():
    inputs = (torch.linspace(-1, 1, 6), torch.linspace(-2, 2, 6))
    program = (
        EspTester(Multiply(), inputs)
        .quantize(Quantize(EspNNQuantizer()))
        .export()
        .get_artifact()
    )
    delegated, delegate_report = to_espnn_edge(program, False)
    assert delegate_report["partitions"] == 1
    assert delegate_report["delegated_ops"] == 1
    assert not delegate_report["cpu_ops"]
    assert delegated.to_executorch().executorch_program.execution_plan[0].delegates

    session = export(
        Multiply(),
        example_inputs=[inputs],
        export_recipe=get_esp_fused_quant_recipe(allow_cpu_fallback=False),
    )
    edge = session.get_edge_program_manager()
    assert get_operator_library_report(edge) == {
        "esp_nn_ops": 1,
        "cpu_ops": [],
        "layout_ops": [],
    }
    assert edge.compile_config._check_ir_validity
    serialized = deserialize_pte_binary(
        session.get_executorch_program_manager().buffer
    ).program
    assert not serialized.execution_plan[0].delegates


def test_operator_library_strict_mode_rejects_remaining_compute():
    inputs = (torch.linspace(-1, 1, 6), torch.linspace(-2, 2, 6))
    with pytest.raises(RuntimeError, match=r"forward\.\w+ \(aten::sin\)"):
        export(
            MultiplyAndFloatUsers(),
            example_inputs=[inputs],
            export_recipe=get_esp_fused_quant_recipe(allow_cpu_fallback=False),
        )


def test_operator_library_strict_mode_accepts_no_compute():
    session = export(
        torch.nn.Identity(),
        example_inputs=[(torch.ones(3),)],
        export_recipe=get_esp_fused_quant_recipe(allow_cpu_fallback=False),
    )
    assert get_operator_library_report(session.get_edge_program_manager()) == {
        "esp_nn_ops": 0,
        "cpu_ops": [],
        "layout_ops": [],
    }


def test_operator_library_uses_standard_edge_decompositions():
    inputs = (torch.linspace(-1, 1, 6).reshape(2, 3),)
    program = torch.export.export(torch.nn.Linear(3, 2), inputs)
    edge = to_edge_transform_and_lower(program, transform_passes=EspPassManager())
    report = get_operator_library_report(edge)
    targets = {op["target"] for op in report["cpu_ops"]}
    assert "aten::linear" not in targets
    assert "aten::addmm" in targets
    torch.testing.assert_close(
        edge.exported_program().module()(*inputs), program.module()(*inputs)
    )
    serialized = deserialize_pte_binary(edge.to_executorch().buffer).program
    assert not serialized.execution_plan[0].delegates


@pytest.mark.parametrize("entry_point", ["separate", "combined"])
def test_passes_preserve_methods_and_constants(entry_point):
    class Scale(torch.nn.Module):
        def __init__(self, scale, offset):
            super().__init__()
            self.register_buffer("scale", scale)
            self.offset = offset

        def forward(self, x):
            return x * self.scale + self.offset

    inputs = (torch.tensor([-1.0, 2.0, 3.0]),)
    models = {
        "forward": Scale(torch.tensor([2.0, 3.0, 4.0]), torch.tensor([0.5, 1.0, 1.5])),
        "other": Scale(torch.tensor([5.0, 6.0, 7.0]), torch.tensor([1.5, 2.0, 2.5])),
    }
    programs = {
        name: torch.export.export(model, inputs) for name, model in models.items()
    }
    if entry_point == "combined":
        transformed = to_edge_transform_and_lower(
            programs, transform_passes=EspPassManager(), constant_methods={"version": 1}
        )
    else:
        stage = EspRunPasses()
        stage.run(to_edge(programs, constant_methods={"version": 1}))
        transformed = stage.artifact
    assert transformed.methods == {"forward", "other"}
    assert transformed.config_methods == {"version"}
    for name, model in models.items():
        program = transformed.exported_program(name)
        program.validate()
        torch.testing.assert_close(
            program.module()(*inputs), model(*inputs), atol=0, rtol=0
        )
    serialized = deserialize_pte_binary(transformed.to_executorch().buffer).program
    assert {plan.name for plan in serialized.execution_plan} == {
        "forward",
        "other",
        "version",
    }
    assert all(not plan.delegates for plan in serialized.execution_plan)


def test_operator_library_reports_all_methods():
    inputs = (torch.linspace(-1, 1, 6), torch.linspace(-2, 2, 6))
    session = export(
        {name: Multiply() for name in ("forward", "other")},
        example_inputs={name: [inputs] for name in ("forward", "other")},
        export_recipe=get_esp_fused_quant_recipe(),
        constant_methods={"version": 1},
    )
    edge = session.get_edge_program_manager()
    assert get_operator_library_report(edge) == {
        "esp_nn_ops": 2,
        "cpu_ops": [],
        "layout_ops": [],
    }
    serialized = deserialize_pte_binary(
        session.get_executorch_program_manager().buffer
    ).program
    assert {plan.name for plan in serialized.execution_plan} == {
        "forward",
        "other",
        "version",
    }
    assert all(not plan.delegates for plan in serialized.execution_plan)


@pytest.mark.parametrize(
    "model, inputs",
    [
        (MultiplyAndFloatUsers(), (torch.linspace(-1, 1, 6), torch.linspace(-2, 2, 6))),
        (torch.nn.MaxPool2d(2), (torch.linspace(-1, 1, 140).reshape(1, 4, 5, 7),)),
    ],
    ids=["multiply", "maxpool"],
)
def test_lowering_preserves_fused_quant_source_program(model, inputs):
    tester = EspTester(model, inputs).export_fused_quant().to_edge()
    reference = tester.get_artifact().exported_program()
    graph_before = str(reference.graph)
    constants_before = reference.constants.copy()
    expected = reference.module()(*inputs)
    tester.run_passes().to_executorch().serialize()
    assert str(reference.graph) == graph_before
    assert reference.constants.keys() == constants_before.keys()
    assert all(
        reference.constants[key] is value for key, value in constants_before.items()
    )
    reference.validate()
    torch.testing.assert_close(reference.module()(*inputs), expected, atol=0, rtol=0)


@pytest.mark.parametrize("pool", [torch.nn.MaxPool2d, torch.nn.AvgPool2d])
def test_strict_export_accepts_and_reports_required_layout_copies(pool):
    session = export(
        pool(2),
        example_inputs=[(torch.linspace(-1, 1, 100).reshape(1, 4, 5, 5),)],
        export_recipe=get_esp_fused_quant_recipe(allow_cpu_fallback=False),
    )
    report = get_operator_library_report(session.get_edge_program_manager())
    assert report["esp_nn_ops"] == 1
    assert report["cpu_ops"] == []
    assert len(report["layout_ops"]) == 2
    assert all(op["target"] == "aten::permute_copy.out" for op in report["layout_ops"])
    assert session.get_executorch_program_manager().buffer


def test_strict_export_rejects_user_permute():
    class PermuteAndPool(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.max_pool2d(x.permute(0, 1, 3, 2), 2)

    with pytest.raises(RuntimeError, match=r"aten::permute_copy"):
        export(
            PermuteAndPool(),
            example_inputs=[(torch.linspace(-1, 1, 140).reshape(1, 4, 5, 7),)],
            export_recipe=get_esp_fused_quant_recipe(allow_cpu_fallback=False),
        )


@pytest.mark.parametrize(
    "model, shape",
    [
        (torch.nn.Conv1d(4, 8, 3), (1, 4, 9)),
        (torch.nn.Conv2d(4, 8, 3), (1, 4, 7, 9)),
    ],
    ids=["conv1d", "conv2d"],
)
def test_other_spatial_operators_remain_portable(model, shape):
    inputs = (torch.linspace(-1, 1, math.prod(shape)).reshape(shape),)
    tester = EspTester(model, inputs).export_fused_quant().to_edge()
    expected = tester.get_artifact().exported_program().module()(*inputs)
    tester.run_passes()
    assert get_operator_library_report(tester.get_artifact())["esp_nn_ops"] == 0
    torch.testing.assert_close(
        tester.get_artifact().exported_program().module()(*inputs),
        expected,
        atol=0,
        rtol=0,
    )
    tester.to_executorch().serialize()

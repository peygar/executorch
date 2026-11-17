# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys

import pytest
from executorch.backends.espressif.scratch import get_scratch_queries
from executorch.backends.espressif.target import EspKernelMode, EspTarget


@pytest.mark.parametrize("target", list(EspTarget))
@pytest.mark.parametrize("kernel_mode", list(EspKernelMode))
def test_vendor_scratch_queries(target, kernel_mode):
    queries = get_scratch_queries(target, kernel_mode)
    if queries is None:
        pytest.skip("Build with EXECUTORCH_BUILD_ESP_NN_PYBINDS=ON")
    if kernel_mode == EspKernelMode.ANSI:
        expected = (0, 0, 0, 0, 0, 0, 256)
    elif target == EspTarget.ESP32S3:
        expected = (4928, 32, 2064, 0, 512, 704, 256)
    elif target in (EspTarget.ESP32P4, EspTarget.ESP32S31):
        expected = (1808, 0, 2064, 0, 0, 800, 256)
    else:
        expected = (0, 0, 128, 0, 0, 0, 256)
    conv_args = {
        "input_dims": (11, 7, 16),
        "filter_dims": (3, 2, 16),
        "output_dims": (6, 4, 32),
        "stride": (2, 2),
        "padding": (1, 1),
    }
    assert (
        queries.conv2d_scratch_size(**conv_args),
        queries.depthwise_conv2d_scratch_size(**conv_args, channel_multiplier=2),
        queries.softmax_scratch_size(width=32, height=8),
        queries.mean_scratch_size(height=8, width=8, channels=16),
        queries.hard_swish_scratch_size(),
        queries.fc_batch_scratch_size(batches=8, row_len=16, out_ch=32),
        queries.logistic_scratch_size(),
    ) == expected


@pytest.mark.parametrize("dimension", ["row_len", "out_ch"])
@pytest.mark.parametrize("value", [-1, 65536])
def test_fc_dimension_bounds(dimension, value):
    queries = get_scratch_queries(EspTarget.ESP32S3, EspKernelMode.OPTIMIZED)
    if queries is None:
        pytest.skip("Build with EXECUTORCH_BUILD_ESP_NN_PYBINDS=ON")
    args = {"batches": 8, "row_len": 16, "out_ch": 32}
    args[dimension] = value
    with pytest.raises(TypeError):
        queries.fc_batch_scratch_size(**args)


@pytest.fixture(
    params=[
        (EspTarget.ESP32S3, EspKernelMode.OPTIMIZED),
        (EspTarget.ESP32P4, EspKernelMode.OPTIMIZED),
        (EspTarget.ESP32, EspKernelMode.OPTIMIZED),
        (EspTarget.ESP32S3, EspKernelMode.ANSI),
    ],
    ids=["s3", "pie", "generic", "ansi"],
)
def scratch_queries(request):
    queries = get_scratch_queries(*request.param)
    if queries is None:
        pytest.skip("Build with EXECUTORCH_BUILD_ESP_NN_PYBINDS=ON")
    return queries


@pytest.mark.parametrize("depthwise", [False, True], ids=["conv", "depthwise"])
@pytest.mark.parametrize(
    "argument,value",
    [
        (argument, value)
        for argument in ("input_dims", "filter_dims", "output_dims", "stride")
        for value in (-1, 0, 65536)
    ]
    + [("padding", -1), ("padding", 65536)],
)
def test_convolution_dimension_bounds(scratch_queries, depthwise, argument, value):
    args = {
        "input_dims": (8, 8, 3),
        "filter_dims": (1, 1, 3),
        "output_dims": (8, 8, 16),
        "stride": (1, 1),
        "padding": (0, 0),
    }
    query = scratch_queries.conv2d_scratch_size
    if depthwise:
        query = scratch_queries.depthwise_conv2d_scratch_size
        args["output_dims"] = (8, 8, 3)
        args["channel_multiplier"] = 1
    for axis in range(len(args[argument])):
        invalid = list(args[argument])
        invalid[axis] = value
        with pytest.raises(ValueError, match=argument):
            query(**{**args, argument: tuple(invalid)})


def test_convolution_dimension_boundaries(scratch_queries):
    assert (
        scratch_queries.conv2d_scratch_size(
            input_dims=(65535, 1, 1),
            filter_dims=(1, 1, 1),
            output_dims=(65535, 1, 1),
            stride=(1, 1),
            padding=(0, 0),
        )
        >= 0
    )
    assert (
        scratch_queries.depthwise_conv2d_scratch_size(
            input_dims=(1, 1, 1),
            filter_dims=(1, 1, 1),
            output_dims=(1, 1, 65535),
            stride=(1, 1),
            padding=(0, 0),
            channel_multiplier=65535,
        )
        >= 0
    )


@pytest.mark.parametrize("value", [-1, 0, 65536])
def test_depthwise_multiplier_bounds(scratch_queries, value):
    with pytest.raises(ValueError, match="channel_multiplier"):
        scratch_queries.depthwise_conv2d_scratch_size(
            input_dims=(8, 8, 3),
            filter_dims=(1, 1, 3),
            output_dims=(8, 8, 3),
            stride=(1, 1),
            padding=(0, 0),
            channel_multiplier=value,
        )


def test_missing_binding(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "executorch.backends.espressif._esp_nn_scratch_esp32s3", None
    )
    assert get_scratch_queries(EspTarget.ESP32S3, EspKernelMode.OPTIMIZED) is None


def test_invalid_kernel_mode():
    with pytest.raises(ValueError, match="EspKernelMode"):
        get_scratch_queries(EspTarget.ESP32S3, "unknown")


def test_invalid_target():
    with pytest.raises(ValueError, match="EspTarget"):
        get_scratch_queries("unknown", EspKernelMode.OPTIMIZED)

# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch

from torch_mlir_e2e_test.torchscript.framework import TestUtils
from torch_mlir_e2e_test.torchscript.registry import register_test_case
from torch_mlir_e2e_test.torchscript.annotations import annotate_args, export

# ==============================================================================

class AdaptiveAvgPool2dModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.aap2d = torch.nn.AdaptiveAvgPool2d((1, 1))

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return self.aap2d(x)


@register_test_case(module_factory=lambda: AdaptiveAvgPool2dModule())
def AdaptiveAvgPool2dModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(10, 3, 8, 9))

# ==============================================================================

class MaxPool2dModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mp2d = torch.nn.MaxPool2d(kernel_size=[6, 8],
                                       stride=[2, 2],
                                       padding=[3, 4],
                                       dilation=2)

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return self.mp2d(x)


@register_test_case(module_factory=lambda: MaxPool2dModule())
def MaxPool2dModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 1, 20, 20) - 0.5)


class MaxPool2dStaticModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mp2d = torch.nn.MaxPool2d(kernel_size=[3, 3],
                                       stride=[2, 2],
                                       padding=[1, 1],
                                       dilation=[1, 1])

    @export
    @annotate_args([
        None,
        ([1, 64, 112, 112], torch.float32, True),
    ])
    def forward(self, x):
        return self.mp2d(x)


@register_test_case(module_factory=lambda: MaxPool2dStaticModule())
def MaxPool2dStaticModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 64, 112, 112))

# ==============================================================================

class MaxPool2dWithIndicesModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.max_pool2d_with_indices(x,
                                                      kernel_size=[2, 2],
                                                      stride=[1, 1],
                                                      padding=[0, 0],
                                                      dilation=[1, 1])

@register_test_case(module_factory=lambda: MaxPool2dWithIndicesModule())
def MaxPool2dWithIndicesModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 1, 8, 8, low=0.5, high=1.0))


class MaxPool2dWithIndicesFullSizeKernelModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.max_pool2d_with_indices(x,
                                                      kernel_size=[4, 4],
                                                      stride=1,
                                                      padding=0,
                                                      dilation=1)

@register_test_case(module_factory=lambda: MaxPool2dWithIndicesFullSizeKernelModule())
def MaxPool2dWithIndicesFullSizeKernelModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 3, 4, 4, low=0.5, high=1.0))


class MaxPool2dWithIndicesNonDefaultPaddingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.max_pool2d_with_indices(x,
                                                      kernel_size=[4, 8],
                                                      stride=[1, 1],
                                                      padding=[2, 4],
                                                      dilation=1)

@register_test_case(module_factory=lambda: MaxPool2dWithIndicesNonDefaultPaddingModule())
def MaxPool2dWithIndicesNonDefaultPaddingModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(2, 4, 16, 16, low=-1.5, high=1.0))


class MaxPool2dWithIndicesNonDefaultStrideModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.max_pool2d_with_indices(x,
                                                      kernel_size=[4,4],
                                                      stride=[1, 2],
                                                      padding=0,
                                                      dilation=1)

@register_test_case(module_factory=lambda: MaxPool2dWithIndicesNonDefaultStrideModule())
def MaxPool2dWithIndicesNonDefaultStrideModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 4, 16, 80, low=0.5, high=2.0))


class MaxPool2dWithIndicesNonDefaultDilationModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.max_pool2d_with_indices(x,
                                                      kernel_size=[4,4],
                                                      stride=[1, 1],
                                                      padding=0,
                                                      dilation=[2, 2])

@register_test_case(module_factory=lambda: MaxPool2dWithIndicesNonDefaultDilationModule())
def MaxPool2dWithIndicesNonDefaultDilationModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 4, 16, 80, low=0.5, high=2.0))


class MaxPool2dWithIndicesNonDefaultParamsModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.ops.aten.max_pool2d_with_indices(x,
                                                      kernel_size=[8,4],
                                                      stride=[2, 2],
                                                      padding=[1, 2],
                                                      dilation=[2, 2])

@register_test_case(module_factory=lambda: MaxPool2dWithIndicesNonDefaultParamsModule())
def MaxPool2dWithIndicesNonDefaultParamsModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(1, 4, 16, 80, low=-0.5, high=4.0))

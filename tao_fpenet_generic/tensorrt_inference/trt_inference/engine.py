# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import logging
import os
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

logger = logging.getLogger(__name__)

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)


# Array of TensorRT loggers. We need to keep global references to
# the TensorRT loggers that we create to prevent them from being
# garbage collected as those are referenced from C++ code without
# Python knowing about it.


tensorrt_loggers = []


def _create_tensorrt_logger(verbose=False):
    """Create a TensorRT logger.

    Args:
        verbose (bool): whether to make the logger verbose.
    """
    if verbose:
        # trt_verbosity = trt.Logger.Severity.INFO
        trt_verbosity = trt.Logger.Severity.VERBOSE
    else:
        trt_verbosity = trt.Logger.Severity.WARNING
    tensorrt_logger = trt.Logger(trt_verbosity)
    tensorrt_loggers.append(tensorrt_logger)
    return tensorrt_logger


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem, binding_name, shape=None):
        self.host = host_mem
        self.device = device_mem
        self.binding_name = binding_name
        self.shape = shape

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine, context):

    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        binding_id = engine.get_binding_index(str(binding))
        size = trt.volume(context.get_binding_shape(binding_id)) * engine.max_batch_size
        print("{}:{}".format(binding, size))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem, binding))
        else:
            output_shape = engine.get_binding_shape(binding)
            if len(output_shape) == 3:
                dims = trt.Dims3(engine.get_binding_shape(binding))
                output_shape = (engine.max_batch_size, dims[0], dims[1], dims[2])
            elif len(output_shape) == 2:
                dims = trt.Dims2(output_shape)
                output_shape = (engine.max_batch_size, dims[0], dims[1])
            outputs.append(HostDeviceMem(host_mem, device_mem, binding, output_shape))

    return inputs, outputs, bindings, stream


def do_inference(batch, context, bindings, inputs, outputs, stream):
    batch_size = batch.shape[0]
    assert len(inputs) == 1
    inputs[0].host = np.ascontiguousarray(batch, dtype=np.float32)
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    stream.synchronize()

    outputs_dict = {}
    outputs_shape = {}
    for out in outputs:
        outputs_dict[out.binding_name] = np.reshape(out.host, out.shape)
        outputs_shape[out.binding_name] = out.shape

    return outputs_shape, outputs_dict


def load_tensorrt_engine(filename, verbose=False):
    tensorrt_logger = _create_tensorrt_logger(verbose)

    if not os.path.exists(filename):
        raise ValueError("{} does not exits".format(filename))

    with trt.Runtime(tensorrt_logger) as runtime, open(filename, "rb") as f:
        trt_engine = runtime.deserialize_cuda_engine(f.read())

    return trt_engine

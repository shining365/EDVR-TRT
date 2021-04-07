#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import numpy as np

src_onnx = 'edvr.onnx'
dst_onnx = 'edvr_surgeon.onnx'

import onnx_graphsurgeon as gs
import onnx
import numpy as np

graph = gs.import_onnx(onnx.load(src_onnx))

for node in graph.nodes:
    if node.op == 'Resize':
    # actually not used in this sample
        node_concat = node.i(2, 0)
        node_concat.i(0, 0).attrs['value'] = gs.Constant('', np.concatenate((node_concat.i(0, 0).attrs['value'].values, node_concat.i(1, 0).attrs['value'].values)))
        node.inputs[2] = node_concat.inputs[0]
        node_concat.outputs.clear()

    if node.op == 'Clip':
        node_cast0 = node.i(1, 0)
        node_cast1 = node.i(2, 0)
        #change data type to fp32
        node_cast0.i(0, 0).attrs['value'] = gs.Constant('', np.asarray([-1.0], dtype=np.float32))
        node_cast1.i(0, 0).attrs['value'] = gs.Constant('', np.asarray([1.0], dtype=np.float32))
        #skip cast
        node.inputs = [node.inputs[0], node_cast0.inputs[0], node_cast1.inputs[0]]
        #cleanup cast
        node_cast0.outputs.clear()
        node_cast1.outputs.clear()

    if node.op == 'grid_sampler':
        #cleanup 3 unused inputs
        for i in [4, 3, 2]:
            node.i(i, 0).outputs.clear()
            del node.inputs[i]

graph.cleanup()
onnx.save(gs.export_onnx(graph), dst_onnx)

model = onnx.load(dst_onnx)

# May not work with non-standard ONNX op
#onnx.checker.check_model(model)
#print(onnx.helper.printable_graph(model.graph))

#/usr/local/tensorrt7/bin/trtexec --onnx=edvr_surgeon.onnx --saveEngine=edvr.trt --plugins=./DeformConvPlugin.so --verbose

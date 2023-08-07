
/* coding=utf-8
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// #include <cuda_fp16.h>
#include <hip/hip_fp16.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>
#include <vector>

#include "type_shim.h"

void tensorMult_gpu_launch(int sq, int b, int np, int hn, at::Tensor* A,
                           at::Tensor* B, at::Tensor* C, at::Tensor* res);
void tensorMult_backward_gpu_launch(int sq, int b, int np, int hn,
                                    at::Tensor* grad_out, at::Tensor* B,
                                    at::Tensor* C, at::Tensor* res_grad);

at::Tensor forward_gpu(int sq, int b, int np, int hn, at::Tensor tensor_A,
                       at::Tensor tensor_B, at::Tensor tensor_C) {
    at::Tensor tensor_res =
        at::empty({sq, b, np, hn},
                  at::device(torch::kCUDA).dtype(tensor_B.scalar_type()));
    tensorMult_gpu_launch(sq, b, np, hn, &tensor_A, &tensor_B, &tensor_C,
                          &tensor_res);
    return tensor_res;
}

at::Tensor backward_gpu(int sq, int b, int np, int hn,
                        at::Tensor tensor_grad_out, at::Tensor tensor_B,
                        at::Tensor tensor_C) {
    at::Tensor tensor_res =
        at::empty({sq, b, np, hn},
                  at::device(torch::kCUDA).dtype(tensor_B.scalar_type()));
    tensorMult_backward_gpu_launch(sq, b, np, hn, &tensor_grad_out, &tensor_B,
                                   &tensor_C, &tensor_res);
    return tensor_res;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_gpu, "forward for gpu");
    m.def("backward", &backward_gpu, "backward for gpu");
}

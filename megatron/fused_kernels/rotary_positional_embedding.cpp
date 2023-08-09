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

void RoPE_gpu_launch(int sq, int b, int np, int hn, at::Tensor* Q,
                     at::Tensor* cos, at::Tensor* sin, at::Tensor* q_emb);
void RoPE_backward_gpu_launch(int sq, int b, int np, int hn,
                              at::Tensor* grad_out, at::Tensor* cos,
                              at::Tensor* sin, at::Tensor* res_grad);

at::Tensor RoPE_forward_gpu(int sq, int b, int np, int hn, at::Tensor tensor_q,
                            at::Tensor tensor_cos, at::Tensor tensor_sin) {
    at::Tensor tensor_q_emb =
        at::empty({sq, b, np, hn},
                  at::device(torch::kCUDA).dtype(tensor_cos.scalar_type()));    
               
    RoPE_gpu_launch(sq, b, np, hn, &tensor_q, &tensor_cos, &tensor_sin, &tensor_q_emb);
    return tensor_q_emb;
}

at::Tensor RoPE_backward_gpu(int sq, int b, int np, int hn,
                             at::Tensor tensor_grad_out, at::Tensor tensor_cos,
                             at::Tensor tensor_sin) {
    at::Tensor tensor_res =
        at::empty({sq, b, np, hn},
                  at::device(torch::kCUDA).dtype(tensor_cos.scalar_type()));
    RoPE_backward_gpu_launch(sq, b, np, hn, &tensor_grad_out, &tensor_cos,
                             &tensor_sin, &tensor_res);
    return tensor_res;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &RoPE_forward_gpu, "rotary positional embedding forward for gpu");
    m.def("backward", &RoPE_backward_gpu, "rotary positional embedding backward for gpu");
}

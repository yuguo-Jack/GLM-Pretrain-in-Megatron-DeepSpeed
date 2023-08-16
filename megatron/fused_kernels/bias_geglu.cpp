/* coding=utf-8
 * Copyright (c) 2023, SUGON CORPORATION.  All rights reserved.
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

#include <hip/hip_fp16.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>
#include <vector>
#include "type_shim.h"

void FusedBiasGeGLU_gpu_launch(int elem_cnt, at::Tensor* in, at::Tensor* out, at::Tensor* bias);
void FusedBiasGeGLU_backward_gpu_launch(int elem_cnt, int dim, at::Tensor* out_grad, at::Tensor* in, at::Tensor* bias, at::Tensor* in_grad, at::Tensor* bias_grad);

at::Tensor FusedBiasGeGLU_forward_gpu(at::Tensor in, at::Tensor bias) {
    int elem_cnt = in.numel();
    at::Tensor out =
        at::empty({elem_cnt / 2},
                  at::device(torch::kCUDA).dtype(in.scalar_type()));    
               
    FusedBiasGeGLU_gpu_launch(elem_cnt, &in, &out, &bias);
    return out;
}

std::vector<at::Tensor> FusedBiasGeGLU_backward_gpu(at::Tensor out_grad, at::Tensor in, at::Tensor bias) {
    int elem_cnt = in.numel();
    int dim = bias.numel();
    at::Tensor in_grad =
        at::empty({elem_cnt},
                  at::device(torch::kCUDA).dtype(in.scalar_type()));
    at::Tensor bias_grad =
        at::empty({dim},
                  at::device(torch::kCUDA).dtype(in.scalar_type()));
    FusedBiasGeGLU_backward_gpu_launch(elem_cnt, dim, &out_grad, &in, &bias, &in_grad, &bias_grad);
    return {in_grad, bias_grad};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &FusedBiasGeGLU_forward_gpu, "fused bias geglu mul for gpu");
    m.def("backward", &FusedBiasGeGLU_backward_gpu, "fused bias geglu mul for gpu");
}


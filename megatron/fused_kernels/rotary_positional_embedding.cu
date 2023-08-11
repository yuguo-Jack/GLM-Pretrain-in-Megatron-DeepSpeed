
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

#include <limits>
#include <stdint.h>
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/CUDAGeneratorImpl.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <c10/macros/Macros.h>
#include <hiprand_kernel.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <thrust/pair.h>
#include <THC/THCGeneral.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/autograd.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "type_shim.h"

#define Q(sq_id, b_id, head_id, hn_id) (Q + sq_id * b * np * hn + b_id * np * hn + head_id * hn + hn_id)
#define cos(sq_id, b_id, hn_id) (cos + sq_id * b * hn + b_id * hn + hn_id)
#define sin(sq_id, b_id, hn_id) (sin + sq_id * b * hn + b_id * hn + hn_id)
#define q_emb(sq_id, b_id, head_id, hn_id) (q_emb + sq_id * b * np * hn + b_id * np * hn + head_id * hn + hn_id)

#define res_grad(sq_id, b_id, head_id, hn_id) \
        (res_grad + sq_id * b * np * hn + b_id * np * hn + head_id * hn + hn_id)
#define grad_out(sq_id, b_id, head_id, hn_id) \
        (grad_out + sq_id * b * np * hn + b_id * np * hn + head_id * hn + hn_id)



template <typename scalar_t, int BLOCK_NP, int BLOCK_HN, int VEC>
__global__ void RoPE_gpu_kernel(int sq, int b, int np, int hn,
                                const scalar_t* Q, const scalar_t* cos, const scalar_t* sin,
                                scalar_t* q_emb) {
    using LoadT = at::native::memory::aligned_vector<scalar_t, VEC>;
    int sq_id = blockIdx.x, b_id = blockIdx.y;
    int np_id = threadIdx.y, hn_cid = threadIdx.x * VEC;
    scalar_t v_cos[VEC];
    scalar_t v_sin[VEC];
    scalar_t v_q[VEC];
    scalar_t v_q_rotate[VEC];
    scalar_t v_q_emb[VEC];
    #pragma unroll
    for (int head_id = np_id; head_id < np; head_id += BLOCK_NP) {
        #pragma unroll
        for(int hn_id = hn_cid; hn_id < hn; hn_id += BLOCK_HN * VEC) {
            *(LoadT*)v_cos = *(LoadT*)cos(sq_id, b_id, hn_id);
            *(LoadT*)v_sin = *(LoadT*)sin(sq_id, b_id, hn_id);
            *(LoadT*)v_q = *(LoadT*)Q(sq_id, b_id, head_id, hn_id);
            *(LoadT*)v_q_rotate = (hn_id + hn / 2 < hn) ? *(LoadT*)Q(sq_id, b_id, head_id, hn_id + hn / 2) : *(LoadT*)Q(sq_id, b_id, head_id, hn_id + hn / 2 - hn);
            #pragma unroll
            for (int vec_id = 0; vec_id < VEC; vec_id++) {
                v_q_emb[vec_id] = v_q[vec_id] * v_cos[vec_id] + ((hn_id + hn / 2 < hn) ? -v_q_rotate[vec_id] : v_q_rotate[vec_id]) * v_sin[vec_id];
            }
            *(LoadT*)q_emb(sq_id, b_id, head_id, hn_id) = *(LoadT*)v_q_emb;
        }
    }
}

template <typename scalar_t>
void host_apply_RoPE_gpu(int sq, int b, int np, int hn,
                         const scalar_t* Q, const scalar_t* cos,
                         const scalar_t* sin, scalar_t* q_emb) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    dim3 blocks(sq, b);

    if(np <= 16) {
        dim3 threads(16, 16);
        RoPE_gpu_kernel<scalar_t, 16, 16, 4><<<blocks, threads, 0, stream>>>(sq, b, np, hn, Q, cos, sin, q_emb); // defaut float, (16, 16, 1) for half
    } else {
        dim3 threads(32, 32);
        RoPE_gpu_kernel<scalar_t, 32, 32, 4><<<blocks, threads, 0, stream>>>(sq, b, np, hn, Q, cos, sin, q_emb); // defaut float, (32, 32, 8) for half
    }

    
}

void RoPE_gpu_launch(int sq, int b, int np, int hn, at::Tensor* Q,
                     at::Tensor* cos, at::Tensor* sin, at::Tensor* q_emb) {
    DISPATCH_FLOAT_HALF_AND_BFLOAT_TYPES(
        cos->scalar_type(), "RoPE_forward_gpu",
        host_apply_RoPE_gpu(
            sq, b, np, hn, Q->data_ptr<scalar_t>(),
            cos->data_ptr<scalar_t>(), sin->data_ptr<scalar_t>(), q_emb->data_ptr<scalar_t>());)
}

template <typename scalar_t, int BLOCK_NP, int BLOCK_HN, int VEC>
__global__ void RoPE_backward_gpu_kernel(int sq, int b, int np,
                                         int hn, const scalar_t* grad_out,
                                         const scalar_t* cos, const scalar_t* sin,
                                         scalar_t* res_grad) {
    using LoadT = at::native::memory::aligned_vector<scalar_t, VEC>;
    int sq_id = blockIdx.x, b_id = blockIdx.y;
    int np_id = threadIdx.y, hn_cid = threadIdx.x * VEC;
    scalar_t v_cos[VEC];
    scalar_t v_sin[VEC];
    scalar_t v_grad_out[VEC];
    scalar_t v_grad_out_rotate[VEC];
    scalar_t v_res_grad[VEC];
    #pragma unroll
    for (int head_id = np_id; head_id < np; head_id += BLOCK_NP) {
        #pragma unroll
        for(int hn_id = hn_cid; hn_id < hn; hn_id += BLOCK_HN * VEC) {
            *(LoadT*)v_cos = *(LoadT*)cos(sq_id, b_id, hn_id);
            *(LoadT*)v_sin = (hn_id + hn / 2 < hn) ? *(LoadT*)sin(sq_id, b_id, hn_id + hn / 2) : *(LoadT*)sin(sq_id, b_id, hn_id + hn / 2 - hn);
            *(LoadT*)v_grad_out = *(LoadT*)grad_out(sq_id, b_id, head_id, hn_id);
            *(LoadT*)v_grad_out_rotate = (hn_id + hn / 2 < hn) ? *(LoadT*)grad_out(sq_id, b_id, head_id, hn_id + hn / 2) : *(LoadT*)grad_out(sq_id, b_id, head_id, hn_id + hn / 2 - hn);
            #pragma unroll
            for (int vec_id = 0; vec_id < VEC; vec_id++) {
                v_res_grad[vec_id] = v_grad_out[vec_id] * v_cos[vec_id] + v_grad_out_rotate[vec_id] * ((hn_id + hn / 2 < hn) ? v_sin[vec_id] : -v_sin[vec_id]);
            }
            *(LoadT*)res_grad(sq_id, b_id, head_id, hn_id) = *(LoadT*)v_res_grad;
        }
    }
}

template <typename scalar_t>
void host_apply_RoPE_backward_gpu(int sq, int b, int np, int hn,
                                  const scalar_t* grad_out, const scalar_t* cos,
                                  const scalar_t* sin, scalar_t* res_grad) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    dim3 blocks(sq, b);

    if(np <= 16) {
        dim3 threads(16, 16);
        RoPE_backward_gpu_kernel<scalar_t, 16, 16, 4><<<blocks, threads, 0, stream>>>(sq, b, np, hn, grad_out, cos, sin, res_grad); // defaut float, (16, 16, 1) for half
    } else {
        dim3 threads(32, 32);
        RoPE_backward_gpu_kernel<scalar_t, 32, 32, 4><<<blocks, threads, 0, stream>>>(sq, b, np, hn, grad_out, cos, sin, res_grad); // defaut float, (32, 32, 8) for half
    }
}

void RoPE_backward_gpu_launch(int sq, int b, int np, int hn,
                              at::Tensor* grad_out, at::Tensor* cos,
                              at::Tensor* sin, at::Tensor* res_grad) {
    DISPATCH_FLOAT_HALF_AND_BFLOAT_TYPES(
        cos->scalar_type(), "RoPE_backward_gpu",
        host_apply_RoPE_backward_gpu(
            sq, b, np, hn, grad_out->data_ptr<scalar_t>(),
            cos->data_ptr<scalar_t>(), sin->data_ptr<scalar_t>(),
            res_grad->data_ptr<scalar_t>());)
}

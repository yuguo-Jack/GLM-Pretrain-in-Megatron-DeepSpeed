
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

#define Q(sq_id, b_id, head_id, hn_id) Q[sq_id * b * np * hn + b_id * np * hn + head_id * hn + hn_id]
#define cos(sq_id, b_id, hn_id) cos[sq_id * b * hn + b_id * hn + hn_id]
#define sin(sq_id, b_id, hn_id) sin[sq_id * b * hn + b_id * hn + hn_id]
#define q_emb(sq_id, b_id, head_id, hn_id) q_emb[sq_id * b * np * hn + b_id * np * hn + head_id * hn + hn_id]

#define res_grad(sq_id, b_id, head_id, hn_id) \
        res_grad[sq_id * b * np * hn + b_id * np * hn + head_id * hn + hn_id]
#define grad_out(sq_id, b_id, head_id, hn_id) \
        grad_out[sq_id * b * np * hn + b_id * np * hn + head_id * hn + hn_id]

#define BLOCK_NP 16
#define BLOCK_HN 64 // BLOCK_HN * CYCLE = HN (128) = HIDDEN / NHEADS

template <typename scalar_t>
__global__ void RoPE_gpu_kernel(int sq, int b, int np, int hn,
                                const scalar_t* Q, const scalar_t* cos, const scalar_t* sin,
                                scalar_t* q_emb) {
    int sq_id = blockIdx.x, b_id = blockIdx.y;
    int np_id = threadIdx.y, hn_cid = threadIdx.x;
    __shared__ scalar_t cos_shared[BLOCK_HN];
    __shared__ scalar_t sin_shared[BLOCK_HN];
    __shared__ scalar_t q_shared[BLOCK_NP][BLOCK_HN];
    __shared__ scalar_t q_shared_rotate[BLOCK_NP][BLOCK_HN];
    #pragma unroll
    for (int head_id = np_id; head_id < np; head_id += BLOCK_NP) {
      #pragma unroll
        for(int hn_id = hn_cid; hn_id < hn; hn_id += BLOCK_HN) {
            cos_shared[hn_cid] = cos(sq_id, b_id, hn_id);
            sin_shared[hn_cid] = sin(sq_id, b_id, hn_id);
            q_shared[np_id][hn_cid] = Q(sq_id, b_id, head_id, hn_id);
            q_shared_rotate[np_id][hn_cid] = (hn_id + hn / 2 < hn) ? -Q(sq_id, b_id, head_id, hn_id + hn / 2) : Q(sq_id, b_id, head_id, hn_id + hn / 2 - hn);
            q_emb(sq_id, b_id, head_id, hn_id) = q_shared[np_id][hn_cid] * cos_shared[hn_cid] + q_shared_rotate[np_id][hn_cid] * sin_shared[hn_cid];
        }
    }
}

template <typename scalar_t>
void host_apply_RoPE_gpu(int sq, int b, int np, int hn,
                         const scalar_t* Q, const scalar_t* cos,
                         const scalar_t* sin, scalar_t* q_emb) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    dim3 blocks(sq, b);
    dim3 threads(BLOCK_HN, BLOCK_NP);

    RoPE_gpu_kernel<scalar_t><<<blocks, threads, 0, stream>>>(sq, b, np, hn, Q, cos, sin, q_emb);
}

void RoPE_gpu_launch(int sq, int b, int np, int hn, at::Tensor* Q,
                     at::Tensor* cos, at::Tensor* sin, at::Tensor* q_emb) {
    DISPATCH_FLOAT_HALF_AND_BFLOAT_TYPES(
        cos->scalar_type(), "RoPE_forward_gpu",
        host_apply_RoPE_gpu(
            sq, b, np, hn, Q->data_ptr<scalar_t>(),
            cos->data_ptr<scalar_t>(), sin->data_ptr<scalar_t>(), q_emb->data_ptr<scalar_t>());)
}

template <typename scalar_t>
__global__ void RoPE_backward_gpu_kernel(int sq, int b, int np,
                                         int hn, const scalar_t* grad_out,
                                         const scalar_t* cos, const scalar_t* sin,
                                         scalar_t* res_grad) {
    int sq_id = blockIdx.x, b_id = blockIdx.y;
    int np_id = threadIdx.y, hn_cid = threadIdx.x;
    __shared__ scalar_t cos_shared[BLOCK_HN];
    __shared__ scalar_t sin_shared[BLOCK_HN];
    __shared__ scalar_t grad_out_shared[BLOCK_NP][BLOCK_HN];
    __shared__ scalar_t grad_out_shared_rotate[BLOCK_NP][BLOCK_HN];
    #pragma unroll
    for (int head_id = np_id; head_id < np; head_id += BLOCK_NP) {
        #pragma unroll
        for(int hn_id = hn_cid; hn_id < hn; hn_id += BLOCK_HN) {
            cos_shared[hn_cid] = cos(sq_id, b_id, hn_id);
            sin_shared[hn_cid] = (hn_id + hn / 2 < hn) ? sin(sq_id, b_id, hn_id + hn / 2) : -sin(sq_id, b_id, hn_id + hn / 2 - hn);
            grad_out_shared[np_id][hn_cid] = grad_out(sq_id, b_id, head_id, hn_id);
            grad_out_shared_rotate[np_id][hn_cid] = (hn_id + hn / 2 < hn) ? grad_out(sq_id, b_id, head_id, hn_id + hn / 2) : grad_out(sq_id, b_id, head_id, hn_id + hn / 2 - hn);
            res_grad(sq_id, b_id, head_id, hn_id) = grad_out_shared[np_id][hn_cid] * cos_shared[hn_cid] + grad_out_shared_rotate[np_id][hn_cid] * sin_shared[hn_cid];
        }
    }
}

template <typename scalar_t>
void host_apply_RoPE_backward_gpu(int sq, int b, int np, int hn,
                                  const scalar_t* grad_out, const scalar_t* cos,
                                  const scalar_t* sin, scalar_t* res_grad) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    dim3 blocks(sq, b);
    dim3 threads(BLOCK_HN, BLOCK_NP);

    RoPE_backward_gpu_kernel<scalar_t><<<blocks, threads, 0, stream>>>(sq, b, np, hn, grad_out, cos, sin, res_grad);
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

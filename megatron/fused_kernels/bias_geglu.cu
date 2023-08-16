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
#include <hip/hip_fp16.h>
#include "type_shim.h"

static constexpr float alpha = 0.7978845608028654F;
static constexpr float beta = 0.044714998453855515F;
static constexpr float h = 0.5F;
static constexpr float one = 1.0F;
static constexpr float one_h = 1.5F;

constexpr int VEC_n = 8;
constexpr int kColwiseReduceTileSize=32;
constexpr int REDUCE_BLOCK_SIZE=256;

// fast gelu or gelu
template <typename acc_t>
__device__ acc_t gelu_mul(acc_t x, acc_t m) {
    // acc_t tanh_in = alpha * (x + beta * x * x * x);
    // return h * x * (one + tanhf(tanh_in)) * m; 
    return c10::cuda::compat::normcdf(x) * x * m;
}

template <typename scalar_t, typename acc_t, int VEC>
__global__ void FusedBiasGeGLU_gpu_kernel(int elem_cnt, scalar_t* out, const scalar_t* in, const scalar_t* bias, int dim) {
    using LoadT = at::native::memory::aligned_vector<scalar_t, VEC>;
    scalar_t v_in[VEC];
    scalar_t v_multiplier[VEC];
    scalar_t v_bias_in[VEC];
    scalar_t v_bias_multiplier[VEC];
    scalar_t v_out[VEC];
    int idx = ((int)blockIdx.x * blockDim.x + threadIdx.x) * VEC;
    if(idx < elem_cnt / 2){
        int row_id = idx / (dim / 2);
        int col_id = idx % (dim / 2);
        *(LoadT*)(v_in) = *(LoadT*)(in + row_id * dim + col_id);
        *(LoadT*)(v_bias_in) = *(LoadT*)(bias + col_id);
        *(LoadT*)(v_multiplier) = *(LoadT*)(in + row_id * dim + col_id + dim / 2);
        *(LoadT*)(v_bias_multiplier) = *(LoadT*)(bias + col_id + dim / 2);
        #pragma unroll
        for(int i = 0; i < VEC; i++){
            v_out[i] = static_cast<scalar_t>(gelu_mul<acc_t>(static_cast<acc_t>(v_in[i] + v_bias_in[i]), 
                                             static_cast<acc_t>(v_multiplier[i] + v_bias_multiplier[i])));
        }
        *(LoadT*)(out + row_id * dim / 2 + col_id) = *(LoadT*)v_out;
    }
}

template <typename scalar_t>
void host_apply_FusedBiasGeGLU_gpu(int elem_cnt, scalar_t* out, const scalar_t* in, const scalar_t* bias, int dim) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int ThreadsPerBlock = REDUCE_BLOCK_SIZE;
    int gridsize=(elem_cnt/2/VEC_n-1)/ThreadsPerBlock+1;
    using T_ACC = at::acc_type<scalar_t, true>;
    FusedBiasGeGLU_gpu_kernel<scalar_t, T_ACC, VEC_n><<<gridsize, ThreadsPerBlock, 0, stream>>>(elem_cnt, out, in, bias, dim); 
}

void FusedBiasGeGLU_gpu_launch(int elem_cnt, at::Tensor* in, at::Tensor* out, at::Tensor* bias) {
    int dim = bias->numel();
    DISPATCH_FLOAT_HALF_AND_BFLOAT_TYPES(
        in->scalar_type(), "FusedBiasGeGLU_forward_gpu",
        host_apply_FusedBiasGeGLU_gpu<scalar_t>(elem_cnt,
                                       out->data_ptr<scalar_t>(), in->data_ptr<scalar_t>(), bias->data_ptr<scalar_t>(), dim);
        )
}

// fast gelu or gelu
template <typename acc_t>
__device__ void gelu_mul_backward(acc_t& x_diff, acc_t& m_diff, acc_t dy, acc_t x, acc_t m) {
    // acc_t pow3 = x * x * x;
    // acc_t tanh_out = tanhf(alpha * (x + beta * pow3));
    // m_diff = h * x * (one + tanh_out) * dy;
    // acc_t dtanh = alpha * (h * x + beta * one_h * pow3);
    // x_diff = (h + h * tanh_out + dtanh * (one - tanh_out * tanh_out)) * m * dy;
    m_diff = c10::cuda::compat::normcdf(x) * x * dy;
    constexpr acc_t kBeta = M_2_SQRTPI * M_SQRT1_2 * 0.5;
    const acc_t cdf = c10::cuda::compat::normcdf(x);
    const acc_t pdf = c10::cuda::compat::exp((acc_t)-0.5 * x * x) * kBeta;
    x_diff = dy * m * (cdf + x * pdf);
}

template <typename scalar_t, typename acc_t, int VEC>
__global__ void FusedBiasGeGLU_backward_gpu_kernel(int elem_cnt, scalar_t* in_grad, const scalar_t* out_grad, const scalar_t* in, const scalar_t* bias, int dim) {
    using LoadT = at::native::memory::aligned_vector<scalar_t, VEC>;
    scalar_t v_in_grad[VEC];
    scalar_t v_multiplier_grad[VEC];
    scalar_t v_out_grad[VEC];
    scalar_t v_in[VEC];
    scalar_t v_multiplier[VEC];
    scalar_t v_bias_in[VEC];
    scalar_t v_bias_multiplier[VEC];
    int idx = ((int)blockIdx.x * blockDim.x + threadIdx.x) * VEC;
    if(idx < elem_cnt / 2){
        int row_id = idx / (dim / 2);
        int col_id = idx % (dim / 2);
        *(LoadT*)(v_in) = *(LoadT*)(in + row_id * dim + col_id);
        *(LoadT*)(v_bias_in) = *(LoadT*)(bias + col_id);
        *(LoadT*)(v_multiplier) = *(LoadT*)(in + row_id * dim + col_id + dim / 2);
        *(LoadT*)(v_bias_multiplier) = *(LoadT*)(bias + col_id + dim / 2);
        *(LoadT*)(v_out_grad) = *(LoadT*)(out_grad + row_id * dim / 2 + col_id);
        #pragma unroll
        for(int i = 0; i < VEC; i++){
            acc_t x_diff;
            acc_t m_diff;
            gelu_mul_backward<acc_t>(x_diff, m_diff, static_cast<acc_t>(v_out_grad[i]), static_cast<acc_t>(v_in[i] + v_bias_in[i]), 
                                          static_cast<acc_t>(v_multiplier[i] + v_bias_multiplier[i]));
            v_in_grad[i] = static_cast<scalar_t>(x_diff);
            v_multiplier_grad[i] = static_cast<scalar_t>(m_diff);
        }
        *(LoadT*)(in_grad + row_id * dim + col_id) = *(LoadT*)v_in_grad;
        *(LoadT*)(in_grad + row_id * dim + col_id + dim / 2) = *(LoadT*)v_multiplier_grad;
    }
}

template <typename T>
__inline__ __device__ T WarpReduceSum(T val, int max = 32) {
  for (int offset = max; offset > 0; offset >>= 1) {
    val += __shfl_down(val, offset);
  }
  return val;
}

template <typename scalar_t, typename acc_t>
__global__ void col_wise_reduce(scalar_t *dst, const scalar_t *src, int M, int N) {
    __shared__ acc_t g_shared[kColwiseReduceTileSize][kColwiseReduceTileSize];
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    acc_t grad_sum = 0;
    if (j < N) {
        for (int i = threadIdx.y; i < M; i += blockDim.y) {
            grad_sum += static_cast<acc_t>(src[i * N + j]);
        }
    }
    g_shared[threadIdx.y][threadIdx.x] = grad_sum;
    __syncthreads();
    acc_t sum = g_shared[threadIdx.x][threadIdx.y];
    sum = WarpReduceSum<acc_t>(sum, kColwiseReduceTileSize / 2);
    if (threadIdx.x == 0) {
        const int j = blockIdx.x * blockDim.x + threadIdx.y;
        if (j < N) {
            dst[j] = static_cast<scalar_t>(sum);
        }
   }
}

template <typename scalar_t>
void host_apply_FusedBiasGeGLU_backward_gpu(cudaStream_t stream, int elem_cnt, scalar_t* in_grad, scalar_t* bias_grad, const scalar_t* out_grad, const scalar_t* in, const scalar_t* bias, int dim) {
    const int ThreadsPerBlock = REDUCE_BLOCK_SIZE;
    int gridsize=(elem_cnt/2/VEC_n-1)/ThreadsPerBlock+1;
    int M = elem_cnt / dim;
    int B =(dim - 1) / kColwiseReduceTileSize + 1;
    using T_ACC = at::acc_type<scalar_t, true>;
    FusedBiasGeGLU_backward_gpu_kernel<scalar_t, T_ACC, VEC_n><<<gridsize, ThreadsPerBlock, 0, stream>>>(elem_cnt, in_grad, out_grad, in, bias, dim); 
    col_wise_reduce<scalar_t, T_ACC><<<B, dim3(kColwiseReduceTileSize, kColwiseReduceTileSize), 0, stream>>>(bias_grad, in_grad, M, dim);
}

void FusedBiasGeGLU_backward_gpu_launch(int elem_cnt, int dim, at::Tensor* out_grad, at::Tensor* in, at::Tensor* bias, at::Tensor* in_grad, at::Tensor* bias_grad) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    DISPATCH_FLOAT_HALF_AND_BFLOAT_TYPES(
        in->scalar_type(), "FusedBiasGeGLU_backward_gpu",
        host_apply_FusedBiasGeGLU_backward_gpu<scalar_t>(stream, elem_cnt, in_grad->data_ptr<scalar_t>(), bias_grad->data_ptr<scalar_t>(),
                                                           out_grad->data_ptr<scalar_t>(), in->data_ptr<scalar_t>(), bias->data_ptr<scalar_t>(), dim);)
}

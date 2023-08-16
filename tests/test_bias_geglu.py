import random
import torch
from torch.nn import functional as F
from megatron.model.glu_activations import geglu
import os
import time
import pathlib
import subprocess

from torch.utils import cpp_extension

cpp_extension.COMMON_HIPCC_FLAGS = ['--gpu-max-threads-per-block=1024']

def load():
    arch_list = os.environ.get('TORCH_CUDA_ARCH_LIST', None)
    if arch_list is None:
        os.environ["TORCH_CUDA_ARCH_LIST"] = ""

    # Build path
    srcpath = pathlib.Path('../megatron/fused_kernels').absolute()
    buildpath = srcpath / 'build'
    buildpath.mkdir(parents=True, exist_ok=True)

    # Helper function to build the kernels.
    def _cpp_extention_load_helper(name, sources, extra_cuda_flags):
        return cpp_extension.load(
            name=name,
            sources=sources,
            build_directory=buildpath,
            extra_cflags=['-O3',],
            extra_cuda_cflags=['-O3'] + extra_cuda_flags,
            verbose=True
        )

    extra_cuda_flags = []
    sources=[srcpath / 'bias_geglu.cpp',
             srcpath / 'bias_geglu.cu']
    rotary_positional_embedding_cuda = _cpp_extention_load_helper(
        "bias_geglu_cuda", sources, extra_cuda_flags)
    
class FusedBiasGeGLUFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, bias):
        import bias_geglu_cuda
        input_ = input.contiguous()
        bias_ = bias.contiguous()
        output = bias_geglu_cuda.forward(input_, bias_)
        ctx.save_for_backward(input_, bias_, input)

        return output.view(*tuple(input.shape[:-1]), input.shape[-1] // 2)

    @staticmethod
    def backward(ctx, grad_output):
        import bias_geglu_cuda
        input_, bias_, input = ctx.saved_tensors
        input_grad, bias_grad = bias_geglu_cuda.backward(grad_output, input_, bias_)

        return input_grad.view(input.shape), bias_grad, None, None
    
class FuseBiasGeGLU(torch.nn.Module):

    def __init__(self):
        super(FuseBiasGeGLU, self).__init__()
        
    def forward(self, input, bias):
        return FusedBiasGeGLUFunction.apply(input, bias)

if __name__ == '__main__':
    load()

    RUNS = 10

    batch_size = 2
    seq_len = 2048
    num_channels = 1024
    x = torch.randn(batch_size, seq_len, num_channels).cuda().half()
    bias = torch.randn(num_channels).cuda().half()
    x.requires_grad_(True)
    bias.requires_grad_(True)
    # glu should halve the last dimension
    output_shape = (batch_size, seq_len, num_channels // 2)
    grad = torch.randn(output_shape).cuda().half()
    fuse_bias_geglu = FuseBiasGeGLU().cuda().half()

    for _ in range(RUNS):
        fused = fuse_bias_geglu(x, bias)
        x_in = x + bias
        input, mul = x_in.chunk(2, dim=-1)
        expected = mul * F.gelu(input)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(RUNS):
        x.grad = None
        bias.grad = None
        fused = fuse_bias_geglu(x, bias)
        fused.backward(grad)
    torch.cuda.synchronize()
    end.record()
    x_grad_fused, bias_grad_fused = x.grad, bias.grad

    print(f"custom_kernel time {start.elapsed_time(end) / RUNS * 1000} us")
    torch.cuda.synchronize()

    start.record()
    for _ in range(RUNS):
        x.grad = None
        bias.grad = None
        x_in = x + bias
        input, mul = x_in.chunk(2, dim=-1)
        expected = mul * F.gelu(input)
        expected.backward(grad)
    torch.cuda.synchronize()
    end.record()
    x_grad_expected, bias_grad_expected = x.grad, bias.grad
    
    print(f"jit time {start.elapsed_time(end) / RUNS * 1000} us")
    torch.cuda.synchronize()

    def rerr(x1, x2):
        return ((x1.float() - x2.float()) / (x1.float() + x2.float() + 1e-6)).abs().max()

    print(((expected - fused).abs().max()))
    print(((x_grad_expected - x_grad_fused).abs().max()))
    print(((bias_grad_expected - bias_grad_fused).abs().max()))

    print(
        f"element-wise relative error max: output: {rerr(expected, fused)}")
    print(
        f"element-wise relative error max: x.grad: {rerr(x_grad_expected, x_grad_fused)}, bias.grad: {rerr(bias_grad_expected, bias_grad_fused)}")
 

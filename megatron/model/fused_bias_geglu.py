import torch
import importlib

global bias_geglu_cuda
bias_geglu_cuda = None

class FusedBiasGeGLUFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, bias):
        import bias_geglu_cuda
        ctx.input_shape = input.shape
        input_ = input.contiguous()
        bias_ = bias.contiguous()
        output = bias_geglu_cuda.forward(input_, bias_)
        ctx.save_for_backward(input_, bias_)

        return output.view(*tuple(input.shape[:-1]), input.shape[-1] // 2)

    @staticmethod
    def backward(ctx, grad_output):
        import bias_geglu_cuda
        input_, bias_ = ctx.saved_tensors
        input_grad, bias_grad = bias_geglu_cuda.backward(grad_output, input_, bias_)

        return input_grad.view(ctx.input_shape), bias_grad, None, None
    
class FuseBiasGeGLU(torch.nn.Module):

    def __init__(self):
        super(FuseBiasGeGLU, self).__init__()
        global bias_geglu_cuda
        bias_geglu_cuda = importlib.import_module("bias_geglu_cuda")
        
    def forward(self, input, bias):
        return FusedBiasGeGLUFunction.apply(input, bias)
    
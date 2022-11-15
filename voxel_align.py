
import torch
from . import bev_pool_ext

__all__ = ["voxel_pool","voxel_align"]




class VoxelAlignFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor_volume, tensor_geom, bev_h, bev_w):
        tensor_volume = tensor_volume.contiguous()
        tensor_geom = tensor_geom.contiguous()
        ctx.mark_non_differentiable(tensor_geom)

        out = bev_pool_ext.voxel_align_forward(
            tensor_volume, tensor_geom, bev_h, bev_w
        )
        ctx.save_for_backward(tensor_geom)
        return out

    @staticmethod
    def backward(ctx, bev_grad):
        bev_grad = bev_grad.contiguous()
        (tensor_geom,) = ctx.saved_tensors
        x_grad = bev_pool_ext.voxel_align_backward(
           bev_grad, tensor_geom
        )
        return x_grad

class VoxelPoolFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor_volume, tensor_geom, bev_h, bev_w):
        tensor_volume = tensor_volume.contiguous()
        tensor_geom = tensor_geom.int().contiguous()
        ctx.mark_non_differentiable(tensor_geom)

        out = bev_pool_ext.voxel_pool_forward(
            tensor_volume, tensor_geom, bev_h, bev_w
        )
        ctx.save_for_backward(tensor_geom)
        return out

    @staticmethod
    def backward(ctx, bev_grad):
        bev_grad = bev_grad.contiguous()
        (tensor_geom,) = ctx.saved_tensors

        x_grad = bev_pool_ext.voxel_pool_backward(
           bev_grad, tensor_geom
        )
        return x_grad



def check_tensor_shape(volume, geom):
    # (n,sn,d,fh,fw,c),
    #   (sn,d,fh,fw)
    if len(volume.shape) != 6 or len(geom.shape) != 5 or geom.shape[-1]!=3:
        return False
    for (i,j) in zip(volume.shape[1:-1],geom.shape[:-1]):
        if i!=j: return False
    return True

def voxel_pool(volume, geom, bev_h, bev_w):
    assert check_tensor_shape(volume, geom)
    return VoxelPoolFunc.apply(volume, geom, bev_h, bev_w)
def voxel_align(volume,geom, bev_h, bev_w):
    assert check_tensor_shape(volume, geom)
    return VoxelAlignFunc.apply(volume, geom, bev_h, bev_w)






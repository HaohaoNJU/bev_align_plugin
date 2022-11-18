'''
Author: Abraham423 christian.wong423@gmail.com
Date: 2022-11-15 15:39:41
LastEditors: Abraham423 christian.wong423@gmail.com
LastEditTime: 2022-11-18 12:13:51
FilePath: /projects/BEVDet/mmdet3d/ops/bev_pool/voxel_align.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

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
        return x_grad,None,None,None

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
        return x_grad,None,None,None



def check_tensor_shape(volume, geom):
    # (n,sn,d,fh,fw,c),
    # (n,sn,d,fh,fw,3)
    if len(volume.shape) != 6 or len(geom.shape) != 6 or geom.shape[-1]!=3:
        return False
    for (i,j) in zip(volume.shape[:-1],geom.shape[:-1]):
        if i!=j: return False
    return True

def voxel_pool(volume, geom, bev_h, bev_w):
    assert check_tensor_shape(volume, geom)
    return VoxelPoolFunc.apply(volume, geom, bev_h, bev_w)
def voxel_align(volume,geom, bev_h, bev_w):
    assert check_tensor_shape(volume, geom)
    return VoxelAlignFunc.apply(volume, geom, bev_h, bev_w)






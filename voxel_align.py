'''
Author: wanghao christian.wong423@gmail.com
Date: 2022-11-15 15:39:41
LastEditTime: 2023-02-22 12:37:22
'''

import torch
from . import bev_pool_ext

__all__ = ["voxel_pool","voxel_align"]



class VoxelAlignFastFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor_geom, tensor_depth, tensor_imgfeat, bev_h, bev_w):
        tensor_depth = tensor_depth.contiguous()
        tensor_imgfeat = tensor_imgfeat.contiguous()
        tensor_geom = tensor_geom.contiguous()

        ctx.mark_non_differentiable(tensor_geom)
        batch_size = tensor_imgfeat.shape[0]
        feat_channel = tensor_imgfeat.shape[2]
        bev_out = tensor_depth.new_zeros((batch_size, feat_channel, bev_h, bev_w))
        bev_pool_ext.voxel_align_fast_forward(
            tensor_geom, tensor_depth, tensor_imgfeat, bev_out, bev_h, bev_w
        )
        ctx.save_for_backward(tensor_geom, tensor_depth, tensor_imgfeat)
        return bev_out

    @staticmethod
    def backward(ctx, bev_grad):
        bev_grad = bev_grad.contiguous()
        (tensor_geom,tensor_depth, tensor_imgfeat) = ctx.saved_tensors
        depth_grad = tensor_depth.new_zeros(tensor_depth.shape)
        imgfeat_grad = tensor_imgfeat.new_zeros(tensor_imgfeat.shape)

        bev_pool_ext.voxel_align_fast_backward(
           tensor_geom,  tensor_depth, tensor_imgfeat,
           bev_grad, depth_grad, imgfeat_grad)
        return None, depth_grad, imgfeat_grad, None,None


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

def check_tensor_shape1(depth, img_feat, geom):
    # (n,sn,d,fh,fw),
    # (n,sn,c,fh,fw),
    # (n,sn,d,fh,fw,3)
    if len(depth.shape) != 5 or len(img_feat.shape) != 5 or len(geom.shape) != 6 or geom.shape[-1]!=3:
        return False
    if depth[:,:,0].shape != img_feat[:,:,0].shape: return False
    if depth.shape != geom[...,0].shape: return False
    return True

def voxel_pool(volume, geom, bev_h, bev_w):
    assert check_tensor_shape(volume, geom)
    return VoxelPoolFunc.apply(volume, geom, bev_h, bev_w)

def voxel_align(volume,geom, bev_h, bev_w):
    assert check_tensor_shape(volume, geom)
    return VoxelAlignFunc.apply(volume, geom, bev_h, bev_w)


#  tensor_geom, tensor_depth, tensor_imgfeat, bev_h, bev_w):
def voxel_align_fast(depth, img_feat, geom, bev_h, bev_w):
    check_tensor_shape1(depth, img_feat, geom)
    return VoxelAlignFastFunc.apply(geom, depth, img_feat, bev_h, bev_w)




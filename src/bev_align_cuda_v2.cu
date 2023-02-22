#include <stdio.h>
#include <stdlib.h>
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))
#define THREAD_NUM 256






__global__ void voxel_align_v2_kernel(const float* dev_geom, const float* dev_depth, const float* dev_imgfeat,  float* bev_feat, 
const int bs, const int sn, 
const int fh, const int fw, const int c, const int d, 
const int bev_h, const int bev_w)
// threads : (bs, d, fh, fw, c)
// dev_depth : (bs, sn, d, fh, fw),
// dev_imgfeat : (bs,sn, c, fh, fw),
// dev_geom size : (sn, d, fh, fw ,  3),
// bev_feat : (bs, c, bev_h, bev_w)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t batch_idx = idx / (d * fh * fw * c);
    size_t depth_idx = idx / (fh * fw * c) % (d);
    size_t hw_idx = (idx / c) % (fh * fw);
    size_t channel_idx = idx % c;
    if (idx < (bs * d * fh * fw * c))
    // framsform dev_geom to bev_idx
    {
      for (int sensor_idx=0;sensor_idx < sn; sensor_idx ++)
      {
        float bev_idx_x = dev_geom[(batch_idx * sn * d * fh * fw + sensor_idx * d * fh * fw + depth_idx * fh * fw + hw_idx) * 3 + 0];
        float bev_idx_y = dev_geom[(batch_idx * sn * d * fh * fw + sensor_idx * d * fh * fw + depth_idx * fh * fw + hw_idx) * 3 + 1];
        float bev_idx_z = dev_geom[(batch_idx * sn * d * fh * fw + sensor_idx * d * fh * fw + depth_idx * fh * fw + hw_idx) * 3 + 2];
        if (bev_idx_z >=0 && bev_idx_z <1)
        // cur feature is located in the valid bev area 
        { 
              // dev_depth : (bs, sn, d, fh, fw),
            float cur_depth = dev_depth[batch_idx*sn*d*fh*fw + sensor_idx*d*fh*fw + depth_idx*fh*fw + hw_idx];
              
              // dev_imgfeat :(bs, sn, c, fh, fw)
            float cur_imgfeat = dev_imgfeat[batch_idx*sn*c*fh*fw+ sensor_idx*c*fh*fw + channel_idx*fh*fw + hw_idx];

            float cur_feat_channel_value = cur_imgfeat * cur_depth;

            int x0 = __float2int_rd(bev_idx_x);
            int y0 = __float2int_rd(bev_idx_y);
            int x1 = x0+1; int y1=y0+1;
            float wa = (x1-bev_idx_x) * (y1-bev_idx_y);
            float wb = (x1-bev_idx_x) * (bev_idx_y-y0);
            float wc = (bev_idx_x-x0) * (y1-bev_idx_y);
            float wd = (bev_idx_x-x0) * (bev_idx_y-y0);


            // top left 
            if (x0 >= 0 && x0 < bev_w && y0 >=0 && y0 < bev_h)
            { 
              atomicAdd( bev_feat + batch_idx * bev_h * bev_w * c + 
                        channel_idx * bev_h * bev_w + y0 * bev_w + x0 ,
                        cur_feat_channel_value * wa);
            }
            __threadfence();

            // bottom left 
            if (x0 >= 0 && x0 < bev_w && y1 >=0 && y1 < bev_h)
            { 
              atomicAdd( bev_feat + batch_idx * bev_h * bev_w * c + 
                        channel_idx * bev_h * bev_w + y1 * bev_w + x0 ,
                        cur_feat_channel_value * wb);
            }
            __threadfence();

            // top right 
            if (x1 >= 0 && x1 < bev_w && y0 >=0 && y0 < bev_h)
            { 
              atomicAdd( bev_feat + batch_idx * bev_h * bev_w * c + 
                        channel_idx * bev_h * bev_w + y0 * bev_w + x1 ,
                        cur_feat_channel_value * wc);
            }
            __threadfence();

            // bottom right 
            if (x1 >= 0 && x1 < bev_w && y1 >=0 && y1 < bev_h)
            { 
              atomicAdd( bev_feat + batch_idx * bev_h * bev_w * c + 
                        channel_idx * bev_h * bev_w + y1 * bev_w + x1 ,
                        cur_feat_channel_value * wd);
            }
            __threadfence();
        }
      }
    }
}


__global__ void voxel_align_v2_grad_kernel(
    const float* dev_geom, const float* dev_depth, const float* dev_imgfeat,
    const float* bev_feat_grad, float* dev_depth_grad, float* dev_imgfeat_grad,  
    const int bs, const int sn, 
    const int fh, const int fw, const int c, const int d, 
    const int bev_h, const int bev_w)
// threads : (bs, d, fh, fw, c)
// dev_geom size : (sn, d, fh, fw ,  3),
// dev_depth : (bs, sn, d, fh, fw),
// dev_imgfeat : (bs,sn, c, fh, fw),
// bev_feat_grad : (bs, c, bev_h, bev_w),
// dev_depth_grad : (bs, sn, d, fh, fw),
// dev_imgfeat_grad :  (bs,sn, c, fh, fw),
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t batch_idx = idx / (d * fh * fw * c);
    size_t depth_idx = idx / (fh * fw * c) % (d);
    size_t hw_idx = (idx / c) % (fh * fw);
    size_t channel_idx = idx % c;

    if (idx < (bs * d * fh * fw * c))
    // framsform dev_geom to bev_idx
    {
      for (int sensor_idx=0;sensor_idx < sn; sensor_idx ++)
      {
        float bev_idx_x = dev_geom[(batch_idx * sn * d * fh * fw + sensor_idx * d * fh * fw + depth_idx * fh * fw + hw_idx) * 3 + 0];
        float bev_idx_y = dev_geom[(batch_idx * sn * d * fh * fw + sensor_idx * d * fh * fw + depth_idx * fh * fw + hw_idx) * 3 + 1];
        float bev_idx_z = dev_geom[(batch_idx * sn * d * fh * fw + sensor_idx * d * fh * fw + depth_idx * fh * fw + hw_idx) * 3 + 2];
        size_t depthfeat_idx = batch_idx*sn*d*fh*fw + sensor_idx*d*fh*fw + depth_idx*fh*fw + hw_idx;
        size_t imgfeat_idx =  batch_idx*sn*c*fh*fw + sensor_idx*c*fh*fw + channel_idx*fh*fw + hw_idx;
        
        if (bev_idx_z >=0 && bev_idx_z <1)
        // cur feature is located in the valid bev area 
        { 
            int x0 = __float2int_rd(bev_idx_x);
            int y0 = __float2int_rd(bev_idx_y);
            int x1 = x0+1; int y1=y0+1;
            float wa = (x1-bev_idx_x) * (y1-bev_idx_y);
            float wb = (x1-bev_idx_x) * (bev_idx_y-y0);
            float wc = (bev_idx_x-x0) * (y1-bev_idx_y);
            float wd = (bev_idx_x-x0) * (bev_idx_y-y0);

            float grad_value = 0;
            // top left 
            if (x0 >= 0 && x0 < bev_w && y0 >=0 && y0 < bev_h)
            { 
              grad_value += bev_feat_grad[batch_idx * bev_h * bev_w * c + channel_idx * bev_h * bev_w + y0 * bev_w + x0] * wa;
            }

            // bottom left 
            if (x0 >= 0 && x0 < bev_w && y1 >=0 && y1 < bev_h)
            { 
              grad_value += bev_feat_grad[batch_idx * bev_h * bev_w * c + channel_idx * bev_h * bev_w + y1 * bev_w + x0] * wb;
            }

            // top right 
            if (x1 >= 0 && x1 < bev_w && y0 >=0 && y0 < bev_h)
            { 
              grad_value += bev_feat_grad[batch_idx * bev_h * bev_w * c + channel_idx * bev_h * bev_w + y0 * bev_w + x1] * wc;

            }

            // bottom right 
            if (x1 >= 0 && x1 < bev_w && y1 >=0 && y1 < bev_h)
            { 
              grad_value += bev_feat_grad[batch_idx * bev_h * bev_w * c + channel_idx * bev_h * bev_w + y1 * bev_w + x1] * wd;
            }

            // atomicAdd(dev_volume_grad+batch_idx*sn*d*fh*fw*c+ sensor_idx*d*fh*fw*c+dhwc_idx, grad_value);

            atomicAdd(dev_depth_grad+depthfeat_idx, grad_value * dev_imgfeat[imgfeat_idx]);
            __threadfence();
            atomicAdd(dev_imgfeat_grad+imgfeat_idx, grad_value * dev_depth[depthfeat_idx]);
            __threadfence();

        }
      }
    }
}





////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void voxel_align_v2(const float* dev_geom, const float* dev_depth, const float* dev_imgfeat,  float* bev_feat,  
                   const int bs, const int sn, 
                   const int fh, const int fw, const int c, const int d, 
                   const int bev_h, const int bev_w)
{
  size_t block_num = DIVUP(bs * d * fh * fw * c, THREAD_NUM);
  voxel_align_v2_kernel<<<block_num, THREAD_NUM>>>(dev_geom, dev_depth, dev_imgfeat,  bev_feat,
  bs, sn,
  fh, fw, c, d,
  bev_h, bev_w);
}


void voxel_align_v2_grad(const float* dev_geom, const float* dev_depth, const float* dev_imgfeat,
                        const float* bev_feat_grad, float* dev_depth_grad, float* dev_imgfeat_grad,  
                        const int bs, const int sn, 
                        const int fh, const int fw, const int c, const int d, 
                        const int bev_h, const int bev_w)
{
  size_t block_num = DIVUP(bs * d * fh * fw * c, THREAD_NUM);
  voxel_align_v2_grad_kernel<<<block_num, THREAD_NUM>>>(
    dev_geom, dev_depth,  dev_imgfeat,
    bev_feat_grad, dev_depth_grad, dev_imgfeat_grad,
    bs, sn,
    fh, fw, c, d,
    bev_h, bev_w);
}





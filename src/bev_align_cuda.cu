#include <stdio.h>
#include <stdlib.h>
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))
#define THREAD_NUM 256

__global__ void voxel_pooling_kernel(const int* dev_geom, const float* dev_volume, float* bev_feat,  
const int bs, const int sn, 
const int fh, const int fw, const int c, const int d, 
const int bev_h, const int bev_w)
// threads : (bs, d, fh, fw, c)
// dev_volume :  (bs, sn, d, fh, fw, c),
// dev_geom size : (sn, d, fh, fw ,  3),
// bev_feat : (bs, c, bev_h, bev_w)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t batch_idx = idx / (d * fh * fw * c);
    size_t dhwc_idx = idx % (d * fh * fw * c);
    size_t depth_idx = dhwc_idx / (fh * fw * c);
    size_t hw_idx = (idx / c) % (fh * fw);
    size_t channel_idx = idx % c;
    if (idx < (bs * d * fh * fw * c))
    // framsform dev_geom to bev_idx
    {
      for (int sensor_idx=0;sensor_idx < sn; sensor_idx ++)
      {
        int bev_idx_x = dev_geom[(batch_idx * sn * d * fh * fw + sensor_idx * d * fh * fw + depth_idx * fh * fw + hw_idx) * 3 + 0];
        int bev_idx_y = dev_geom[(batch_idx * sn * d * fh * fw + sensor_idx * d * fh * fw + depth_idx * fh * fw + hw_idx) * 3 + 1];
        int bev_idx_z = dev_geom[(batch_idx * sn * d * fh * fw + sensor_idx * d * fh * fw + depth_idx * fh * fw + hw_idx) * 3 + 2];
        if (bev_idx_x >=0 && bev_idx_x < bev_w && bev_idx_y >=0 && bev_idx_y < bev_h && bev_idx_z >=0 && bev_idx_z <1)
        // cur feature is located in the valid bev area 
        {
            float cur_feat_channel_value = dev_volume[batch_idx*sn*d*fh*fw*c+ sensor_idx*d*fh*fw*c+dhwc_idx];
              // NEED to add pillar-wise or pillar-channel-wise max pooling, 
              // Note that for (H,W) matrix, W is along y-axis, H is along x-axis)

            atomicAdd( bev_feat + batch_idx * bev_h * bev_w * c + 
                        channel_idx * bev_h * bev_w + bev_idx_y * bev_w + bev_idx_x ,
                        cur_feat_channel_value);
            __threadfence();
        }
      }
    }
}




__global__ void voxel_align_kernel(const float* dev_geom, const float* dev_volume, float* bev_feat, 
const int bs, const int sn, 
const int fh, const int fw, const int c, const int d, 
const int bev_h, const int bev_w)
// threads : (bs, d, fh, fw, c)
// dev_volume :  (bs,sn, d, fh, fw, c),
// dev_geom size : (sn, d, fh, fw ,  3),
// bev_feat : (bs, c, bev_h, bev_w)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t batch_idx = idx / (d * fh * fw * c);
    size_t dhwc_idx = idx % (d * fh * fw * c);
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
            float cur_feat_channel_value = dev_volume[batch_idx*sn*d*fh*fw*c+ sensor_idx*d*fh*fw*c+dhwc_idx];
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


__global__ void voxel_pooling_grad_kernel(const int* dev_geom, const float* bev_feat_grad,  float* dev_volume_grad,
const int bs, const int sn, 
const int fh, const int fw, const int c, const int d, 
const int bev_h, const int bev_w)
// threads : (bs, d, fh, fw, c)
// dev_volume_grad :  (bs, sn, d, fh, fw, c),
// dev_geom size : (sn, d, fh, fw ,  3),
// bev_feat_grad : (bs, c, bev_h, bev_w)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t batch_idx = idx / (d * fh * fw * c);
    size_t dhwc_idx = idx % (d * fh * fw * c);
    size_t depth_idx = idx / (fh * fw * c) % (d);
    size_t hw_idx = (idx / c) % (fh * fw);
    size_t channel_idx = idx % c;
    if (idx < (bs * d * fh * fw * c))
    // framsform dev_geom to bev_idx
    {
      for (int sensor_idx=0;sensor_idx < sn; sensor_idx ++)
      {
        int bev_idx_x = dev_geom[(batch_idx * sn * d * fh * fw + sensor_idx * d * fh * fw + depth_idx * fh * fw + hw_idx) * 3 + 0];
        int bev_idx_y = dev_geom[(batch_idx * sn * d * fh * fw + sensor_idx * d * fh * fw + depth_idx * fh * fw + hw_idx) * 3 + 1];
        int bev_idx_z = dev_geom[(batch_idx * sn * d * fh * fw + sensor_idx * d * fh * fw + depth_idx * fh * fw + hw_idx) * 3 + 2];
        if (bev_idx_z >=0 && bev_idx_z <1 && bev_idx_x >= 0 && bev_idx_x < bev_w && bev_idx_y >=0 && bev_idx_y < bev_h)
        // cur feature is located in the valid bev area 
        { 
            float grad_value = bev_feat_grad[bev_idx_x + bev_idx_y * bev_w + channel_idx * bev_h * bev_w + batch_idx * c * bev_h * bev_w];
            atomicAdd(dev_volume_grad+batch_idx*sn*d*fh*fw*c+ sensor_idx*d*fh*fw*c+dhwc_idx, 
                      grad_value);
            __threadfence();
        }
      }
    }
}


__global__ void voxel_align_grad_kernel(const float* dev_geom, const float* bev_feat_grad,  float* dev_volume_grad,
const int bs, const int sn, 
const int fh, const int fw, const int c, const int d, 
const int bev_h, const int bev_w)
// threads : (bs, d, fh, fw, c)
// dev_volume_grad :  (bs, sn, d, fh, fw, c),
// dev_geom size : (sn, d, fh, fw ,  3),
// bev_feat_grad : (bs, c, bev_h, bev_w)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t batch_idx = idx / (d * fh * fw * c);
    size_t dhwc_idx = idx % (d * fh * fw * c);
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


            atomicAdd(dev_volume_grad+batch_idx*sn*d*fh*fw*c+ sensor_idx*d*fh*fw*c+dhwc_idx, 
                      grad_value);
            __threadfence();
        }
      }
    }
}





////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void voxel_pool(const int* dev_geom, const float* dev_volume, float* bev_feat,  
const int bs, const int sn, 
const int fh, const int fw, const int c, const int d, 
const int bev_h, const int bev_w)
{
  size_t block_num = DIVUP(bs * d * fh * fw * c, THREAD_NUM);
  voxel_pooling_kernel<<<block_num, THREAD_NUM>>>(dev_geom, dev_volume, bev_feat,
  bs, sn,
  fh, fw, c, d,
  bev_h, bev_w);
}


void voxel_align(const float* dev_geom, const float* dev_volume, float* bev_feat,  
const int bs, const int sn, 
const int fh, const int fw, const int c, const int d, 
const int bev_h, const int bev_w)
{
  size_t block_num = DIVUP(bs * d * fh * fw * c, THREAD_NUM);
  voxel_align_kernel<<<block_num, THREAD_NUM>>>(dev_geom, dev_volume, bev_feat,
  bs, sn,
  fh, fw, c, d,
  bev_h, bev_w);
}

void voxel_pool_grad(const int* dev_geom, const float* bev_feat_grad, float* dev_volume_grad,  
const int bs, const int sn, 
const int fh, const int fw, const int c, const int d, 
const int bev_h, const int bev_w)
{
  size_t block_num = DIVUP(bs * d * fh * fw * c, THREAD_NUM);
  voxel_pooling_grad_kernel<<<block_num, THREAD_NUM>>>(dev_geom, bev_feat_grad, dev_volume_grad,
  bs, sn,
  fh, fw, c, d,
  bev_h, bev_w);
}


void voxel_align_grad(const float* dev_geom, const float* bev_feat_grad, float* dev_volume_grad,  
const int bs, const int sn, 
const int fh, const int fw, const int c, const int d, 
const int bev_h, const int bev_w)
{
  size_t block_num = DIVUP(bs * d * fh * fw * c, THREAD_NUM);
  voxel_align_grad_kernel<<<block_num, THREAD_NUM>>>(dev_geom, bev_feat_grad, dev_volume_grad,
  bs, sn,
  fh, fw, c, d,
  bev_h, bev_w);
}



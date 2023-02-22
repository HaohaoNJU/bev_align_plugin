#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>

// CUDA function declarations
void bev_pool(int b, int d, int h, int w, int n, int c, int n_intervals, const float* x,
    const int* geom_feats, const int* interval_starts, const int* interval_lengths, float* out);

void bev_pool_grad(int b, int d, int h, int w, int n, int c, int n_intervals, const float* out_grad,
  const int* geom_feats, const int* interval_starts, const int* interval_lengths, float* x_grad);


// add by wanghao 
void voxel_pool(const int* dev_geom, const float* dev_volume, float* bev_feat,  
const int batch_size, const int sensor_num, 
const int img_output_h, const int img_output_w, const int img_output_c, const int img_output_d, 
const int bev_h, const int bev_w);

void voxel_pool_grad(const int* dev_geom, const float* bev_feat_grad, float* dev_volume_grad,  
const int bs, const int sn, 
const int fh, const int fw, const int c, const int d, 
const int bev_h, const int bev_w);

void voxel_align(const float* dev_geom, const float* dev_volume, float* bev_feat,  
const int batch_size, const int sensor_num, 
const int img_output_h, const int img_output_w, const int img_output_c, const int img_output_d, 
const int bev_h, const int bev_w);

void voxel_align_grad(const float* dev_geom, const float* bev_feat_grad, float* dev_volume_grad,  
const int batch_size, const int sensor_num, 
const int img_output_h, const int img_output_w, const int img_output_c, const int img_output_d, 
const int bev_h, const int bev_w);

// For efficient bev align, wo input volume
void voxel_align_v2(const float* dev_geom, const float* dev_depth, const float* dev_imgfeat,  float* bev_feat,  
                   const int bs, const int sn, 
                   const int fh, const int fw, const int c, const int d, 
                   const int bev_h, const int bev_w);

void voxel_align_v2_grad(const float* dev_geom, const float* dev_depth, const float* dev_imgfeat,
                        const float* bev_feat_grad, float* dev_depth_grad, float* dev_imgfeat_grad,  
                        const int bs, const int sn, 
                        const int fh, const int fw, const int c, const int d, 
                        const int bev_h, const int bev_w);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// BEV ALIGN SOURCE CODE


/*
  Function: pillar voxel fast align (forward, cuda)
  Args:
    tensor_depth     :  (batch_size, sensor_num, img_output_d, img_output_h, img_output_w),
    tensor_imgfeat   :  (batch_size, sensor_num, img_output_c, img_output_h, img_output_w),
    tensor_geom size :  (batch_size, sensor_num, img_output_d, img_output_h, img_output_w ,  3),
  Return:
    out               :  (batch_size, img_output_c, bev_h, bev_w)
*/

void voxel_align_v2_forward(
  const at::Tensor tensor_geom,
  const at::Tensor tensor_depth,
  const at::Tensor tensor_imgfeat,
  at::Tensor bev_out,
  const int bev_h,
  const int bev_w)
{
  int n = tensor_geom.size(0);
  int sn = tensor_geom.size(1);
  int d = tensor_geom.size(2);
  int fh = tensor_geom.size(3);
  int fw = tensor_geom.size(4);
  int c = tensor_imgfeat.size(2);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(tensor_depth));
  const float* depth_ = tensor_depth.data_ptr<float>();
  const float* imgfeat_ = tensor_imgfeat.data_ptr<float>();
  const float* geom_ = tensor_geom.data_ptr<float>();
  float* out_ = bev_out.data_ptr<float>();
  voxel_align_v2(
    geom_, depth_, imgfeat_, out_,  n,sn,   fh,fw,c,d,   bev_h, bev_w
  );
}

/*
  Function: pillar pooling (backward, cuda)
  Args:
    tensor_geom      :  input coordinates, FloatTensor[n, sn, d, fh, fw, 3],
    tensor_depth     :  (batch_size, sensor_num, img_output_d, img_output_h, img_output_w),
    tensor_imgfeat   :  (batch_size, sensor_num, img_output_c, img_output_h, img_output_w),
    bev_grad         :  input features, FloatTensor[n, c, bev_h, bev_w]

  Return:
    depth_grad       :  (batch_size, sensor_num, img_output_d, img_output_h, img_output_w),
    imgfeat_grad     :  (batch_size, sensor_num, img_output_c, img_output_h, img_output_w),
*/

void voxel_align_v2_backward(
  const at::Tensor tensor_geom,
  const at::Tensor tensor_depth,
  const at::Tensor tensor_imgfeat,
  const at::Tensor bev_grad,
  at::Tensor depth_grad,
  at::Tensor imgfeat_grad
) 
{
  int n = bev_grad.size(0);
  int c = bev_grad.size(1);
  int bev_h = bev_grad.size(2);
  int bev_w = bev_grad.size(3);

  int sn = tensor_geom.size(1);
  int d = tensor_geom.size(2);
  int fh = tensor_geom.size(3);
  int fw = tensor_geom.size(4);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(bev_grad));
  const float* bev_grad_ = bev_grad.data_ptr<float>();
  const float* geom_ = tensor_geom.data_ptr<float>();
  const float* depth_ = tensor_depth.data_ptr<float>();
  const float* imgfeat_ = tensor_imgfeat.data_ptr<float>();

  float* depth_grad_ = depth_grad.data_ptr<float>();
  float* imgfeat_grad_ = imgfeat_grad.data_ptr<float>();

  voxel_align_v2_grad(
    geom_, depth_, imgfeat_, 
    bev_grad_, depth_grad_, imgfeat_grad_,

    n, sn, fh, fw, c, d,   
    bev_h,bev_w);  
}



/*
  Function: pillar voxel align (forward, cuda)
  Args:
    tensor_volume     :  (batch_size, sensor_num, img_output_d, img_output_h, img_output_w,   img_output_c),
    tensor_geom size  :  (batch_size, sensor_num, img_output_d, img_output_h, img_output_w ,  3),
  Return:
    out               :  (batch_size, img_output_c, bev_h, bev_w)
*/

at::Tensor voxel_align_forward(
  const at::Tensor tensor_volume,
  const at::Tensor tensor_geom,
  const int bev_h,
  const int bev_w)
{
  int n = tensor_volume.size(0);
  int sn = tensor_volume.size(1);
  int d = tensor_volume.size(2);
  int fh = tensor_volume.size(3);
  int fw = tensor_volume.size(4);
  int c = tensor_volume.size(5);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(tensor_volume));
  const float* volume_ = tensor_volume.data_ptr<float>();
  const float* geom_ = tensor_geom.data_ptr<float>();
  auto options = torch::TensorOptions().dtype(tensor_volume.dtype()).device(tensor_volume.device());
  at::Tensor out = torch::zeros({n, c, bev_h, bev_w}, options);
  float* out_ = out.data_ptr<float>();
  voxel_align(
    geom_, volume_, out_,  n,sn,   fh,fw,c,d,   bev_h, bev_w
  );
  return out;
}

/*
  Function: pillar pooling (backward, cuda)
  Args:
    bev_grad         : input features, FloatTensor[n, c, bev_h, bev_w]
    geom_feats       : input coordinates, FloatTensor[n, sn, d, fh, fw, 3]
  Return:
    x_grad           : output features, FloatTensor[n, sn, d, fh, fw, c]
*/
at::Tensor voxel_align_backward(
  const at::Tensor bev_grad,
  const at::Tensor tensor_geom) 
{
  int n = bev_grad.size(0);
  int c = bev_grad.size(1);
  int bev_h = bev_grad.size(2);
  int bev_w = bev_grad.size(3);

  int sn = tensor_geom.size(1);
  int d = tensor_geom.size(2);
  int fh = tensor_geom.size(3);
  int fw = tensor_geom.size(4);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(bev_grad));
  const float* bev_grad_ = bev_grad.data_ptr<float>();
  const float* geom_ = tensor_geom.data_ptr<float>();

  auto options = torch::TensorOptions().dtype(bev_grad.dtype()).device(bev_grad.device());
  at::Tensor volume_grad = torch::zeros({n,sn,d,fh,fw,c}, options);
  float* volume_grad_ = volume_grad.data_ptr<float>();
  
  voxel_align_grad(geom_, bev_grad_, volume_grad_,
  n,sn,  fh,fw,c,d,   bev_h,bev_w);
  
  return volume_grad;
}


/*
  Function: pillar voxel pooling (forward, cuda)
  Args:
    tensor_volume     :  (batch_size, sensor_num, img_output_d, img_output_h, img_output_w,   img_output_c),
    tensor_geom size  :  (batch_size, sensor_num, img_output_d, img_output_h, img_output_w ,  3),
  Return:
    out               :  (batch_size, img_output_c, bev_h, bev_w)
*/

at::Tensor voxel_pool_forward(
  const at::Tensor tensor_volume,
  const at::Tensor tensor_geom,
  const int bev_h,
  const int bev_w)
{
  int n = tensor_volume.size(0);
  int sn = tensor_volume.size(1);
  int d = tensor_volume.size(2);
  int fh = tensor_volume.size(3);
  int fw = tensor_volume.size(4);
  int c = tensor_volume.size(5);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(tensor_volume));
  const float* volume_ = tensor_volume.data_ptr<float>();
  const int* geom_ = tensor_geom.data_ptr<int>();
  auto options = torch::TensorOptions().dtype(tensor_volume.dtype()).device(tensor_volume.device());
  at::Tensor out = torch::zeros({n, c, bev_h, bev_w}, options);
  float* out_ = out.data_ptr<float>();
  voxel_pool(
    geom_, volume_, out_,  n,sn,   fh,fw,c,d,   bev_h, bev_w
  );
  return out;
}

/*
  Function: pillar pooling (backward, cuda)
  Args:
    bev_grad         : input features, FloatTensor[n, c, bev_h, bev_w]
    geom_feats       : input coordinates, FloatTensor[n, sn, d, fh, fw, 3]
  Return:
    x_grad           : output features, FloatTensor[n, sn, d, fh, fw, c]
*/
at::Tensor voxel_pool_backward(
  const at::Tensor bev_grad,
  const at::Tensor tensor_geom) 
{
  int n = bev_grad.size(0);
  int c = bev_grad.size(1);
  int bev_h = bev_grad.size(2);
  int bev_w = bev_grad.size(3);

  int sn = tensor_geom.size(1);
  int d = tensor_geom.size(2);
  int fh = tensor_geom.size(3);
  int fw = tensor_geom.size(4);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(bev_grad));
  const float* bev_grad_ = bev_grad.data_ptr<float>();
  const int* geom_ = tensor_geom.data_ptr<int>();

  auto options = torch::TensorOptions().dtype(bev_grad.dtype()).device(bev_grad.device());
  at::Tensor volume_grad = torch::zeros({n,sn,d,fh,fw,c}, options);
  float* volume_grad_ = volume_grad.data_ptr<float>();
  
  voxel_pool_grad(geom_, bev_grad_, volume_grad_,
  n,sn,  fh,fw,c,d,   bev_h,bev_w);
  
  return volume_grad;
}





///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


/*
  Function: pillar pooling (forward, cuda)
  Args:
    x                : input features, FloatTensor[n, c]
    geom_feats       : input coordinates, IntTensor[n, 4]
    interval_lengths : starting position for pooled point, IntTensor[n_intervals]
    interval_starts  : how many points in each pooled point, IntTensor[n_intervals]
  Return:
    out              : output features, FloatTensor[b, d, h, w, c]
*/
at::Tensor bev_pool_forward(
  const at::Tensor _x,
  const at::Tensor _geom_feats, 
  const at::Tensor _interval_lengths, 
  const at::Tensor _interval_starts,
  int b, int d, int h, int w
) 
{
  int n = _x.size(0);
  int c = _x.size(1);
  int n_intervals = _interval_lengths.size(0);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(_x));
  const float* x = _x.data_ptr<float>();
  const int* geom_feats = _geom_feats.data_ptr<int>();
  const int* interval_lengths = _interval_lengths.data_ptr<int>();
  const int* interval_starts = _interval_starts.data_ptr<int>();
  
  auto options =
      torch::TensorOptions().dtype(_x.dtype()).device(_x.device());
  at::Tensor _out = torch::zeros({b, d, h, w, c}, options);
  float* out = _out.data_ptr<float>();
  bev_pool(
    b, d, h, w, n, c, n_intervals, x,
    geom_feats, interval_starts, interval_lengths, out
  );
  return _out;
}


/*
  Function: pillar pooling (backward, cuda)
  Args:
    out_grad         : input features, FloatTensor[b, d, h, w, c]
    geom_feats       : input coordinates, IntTensor[n, 4]
    interval_lengths : starting position for pooled point, IntTensor[n_intervals]
    interval_starts  : how many points in each pooled point, IntTensor[n_intervals]
  Return:
    x_grad           : output features, FloatTensor[n, 4]
*/
at::Tensor bev_pool_backward(
  const at::Tensor _out_grad,
  const at::Tensor _geom_feats, 
  const at::Tensor _interval_lengths, 
  const at::Tensor _interval_starts,
  int b, int d, int h, int w
) {
  int n = _geom_feats.size(0);
  int c = _out_grad.size(4);
  int n_intervals = _interval_lengths.size(0);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(_out_grad));
  const float* out_grad = _out_grad.data_ptr<float>();
  const int* geom_feats = _geom_feats.data_ptr<int>();
  const int* interval_lengths = _interval_lengths.data_ptr<int>();
  const int* interval_starts = _interval_starts.data_ptr<int>();

  auto options =
      torch::TensorOptions().dtype(_out_grad.dtype()).device(_out_grad.device());
  at::Tensor _x_grad = torch::zeros({n, c}, options);
  float* x_grad = _x_grad.data_ptr<float>();
  
  bev_pool_grad(
    b, d, h, w, n, c, n_intervals, out_grad,
    geom_feats, interval_starts, interval_lengths, x_grad
  );
  
  return _x_grad;
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bev_pool_forward", &bev_pool_forward,
        "bev_pool_forward");
  m.def("bev_pool_backward", &bev_pool_backward,
        "bev_pool_backward");

  // wanghao add 
  m.def("voxel_align_fast_forward", &voxel_align_v2_forward,
        "voxel_align_fast_forward");
  m.def("voxel_align_fast_backward", &voxel_align_v2_backward,
        "voxel_align_fast_backward");

  m.def("voxel_align_forward", &voxel_align_forward,
        "voxel_align_forward");
  m.def("voxel_align_backward", &voxel_align_backward,
        "voxel_align_backward");

  m.def("voxel_pool_forward", &voxel_pool_forward,
        "voxel_pool_forward");
  m.def("voxel_pool_backward", &voxel_pool_backward,
        "voxel_pool_backward");
}

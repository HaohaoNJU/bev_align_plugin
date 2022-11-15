# BEV Pooling and BEV Align Source Code 

## Illustration
see readme.pdf 

## Configure
Please move this project to BEVDet, replace the old one 
```
rm -rf ${THIS_PROJECT} ${BEVDet_Path}/mmdet3d/ops/bev_pool
mv ${THIS_PROJECT} ${BEVDet_Path}/mmdet3d/ops/bev_pool
```


Add import item in `${BEVDet_Path}/mmdet3d/ops/__init__.py` :
```
from .bev_pool.voxel_align import voxel_pool,voxel_align

__all__.extend(['voxel_align','voxel_pool'])
```

Add source file in `${BEVDet_Path}/setup.py` cuda extend item , replace the old one with : 
```
make_cuda_ext(
    name="bev_pool_ext",
    module="mmdet3d.ops.bev_pool",
    sources=[
        "src/bev_pool.cpp",
        "src/bev_pool_cuda.cu",
        "src/bev_align_cuda.cu"
    ],
),
```

## Install 

Then recompiling the bevdet source code 

```
cd ${THIS_PROJECT} ${BEVDet_Path}
pip install -v -e .
```

## Example
```
from mmdet3d.ops import voxel_pool, voxel_align
bev_h = 64
bev_w = 128
img_output_h=16 # img feature map height
img_output_w=44 # img feature map width
img_output_c=64 # img feature map feature channels
img_output_d=59 # img feature map depth channels
batch_size = 1 # batch size 
sensor_num = 1 # how many sensors
volume = torch.randn([batch_size, sensor_num, img_output_d, img_output_h, img_output_w, img_output_c]).float()
geom_feats = torch.randn([sensor_num,img_output_d, img_output_h, img_output_w,3]) * 30
# please strictly follow the above data shape format and data dtype
output_pool = voxel_pool(volume.cuda(), geom_feats.cuda(), bev_h,bev_w)
output_align = voxel_align(volume.cuda(), geom_feats.cuda(), bev_h,bev_w)
print(output_pool.shape)
print(output_align.shape)
```



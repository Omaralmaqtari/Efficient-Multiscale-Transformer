import torchvision.models as models
import torch
from ptflops import get_model_complexity_info
import re
import time

#Model thats already available
model = models.densenet161()
macs, params = get_model_complexity_info(model, (3, 256, 256), as_strings=True, print_per_layer_stat=False, verbose=True)
# Extract the numerical value
flops = eval(re.findall(r'([\d.]+)', macs)[0])*2
# Extract the unit
flops_unit = re.findall(r'([A-Za-z]+)', macs)[0][0]

print('Computational complexity: {:<8}'.format(macs))
print('Computational complexity: {} {}Flops'.format(flops, flops_unit))
print('Number of parameters: {:<8}'.format(params))


x = torch.rand(1,3,256,256).to('cuda')
models = [LCT([[96,96], [96,128,128], [128,192,192,192], [128,128,128]], 1), TransMUNet(), SegViTv2(CONFIGS['R50-ViT-B_16'], img_size=256, num_classes=1), SwinTransformerSeg(), DeepLabv3plus(1), EfficientNetSeg(efficientnets[7], 32, 160, 640, 1), MobileNetV3Seg(40, 112, 960, 1, 'large'), ShuffleNetV2Seg(shufflenets[3], 244, 488, 976, 1)]
for i in models:
    model = i.to('cuda')
    time1 = 0.
    print(type(model))
    for i in range(20):
        t = time.time()
        y = model(x)
        time1 += (time.time() - t)
        
    print()
    print((time1/20.))
    print()
    print()
    print()
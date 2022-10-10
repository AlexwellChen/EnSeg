from tools import ADE20KDataset, inference_img
from mmseg.apis import init_segmentor
from torchvision.transforms import Compose
import torchvision.transforms as transforms
from tqdm import tqdm
import torch

'''
-------------------------------------------
                  加载模型
-------------------------------------------
'''
pspnet_config_file = '../configs/pspnet/pspnet_r50-d8_512x512_80k_ade20k.py'
pspnet_checkpoint_file = '../checkpoints/pspnet_r50-d8_512x512_80k_ade20k_20200615_014128-15a8b914.pth'

deeplabv3plus_config_file = '../configs/deeplabv3plus/deeplabv3plus_r50-d8_512x512_80k_ade20k.py'
deeplabv3plus_checkpoint_file = '../checkpoints/deeplabv3plus_r50-d8_512x512_80k_ade20k_20200614_185028-bf1400d8.pth'

fcn_config_file = '../configs/fcn/fcn_r50-d8_512x512_80k_ade20k.py'
fcn_checkpoint_file = '../checkpoints/fcn_r50-d8_512x512_80k_ade20k_20200614_144016-f8ac5082.pth'

pspnet_model = init_segmentor(pspnet_config_file, pspnet_checkpoint_file, device='cuda:0')
deeplabv3plus_model = init_segmentor(deeplabv3plus_config_file, deeplabv3plus_checkpoint_file, device='cuda:0')
fcn_model = init_segmentor(fcn_config_file, fcn_checkpoint_file, device='cuda:0')

'''
-------------------------------------------
                  加载数据集
-------------------------------------------
'''
train_path = '../../ADEChallengeData2016/images/training/'
train_label_path = '../../ADEChallengeData2016/annotations/training/'
val_path = '../../ADEChallengeData2016/images/validation/'
val_label_path = '../../ADEChallengeData2016/annotations/validation/'

transform = Compose([transforms.ToTensor()])
ADE20K_Dataset = ADE20KDataset(train_path, train_label_path, transform)

'''
-------------------------------------------
                推理并保存Tensor
-------------------------------------------
'''
inference_num = 10 # 推理数量
img_path = [ADE20K_Dataset.get_img_path(i) for i in range(inference_num)]
img_name = [ADE20K_Dataset.get_img_name(i) for i in range(inference_num)]
netdisk_path = "/root/Desktop/我的网盘/inference_tensor/"
for i in tqdm(range(inference_num)):
    pspnet_model_res = inference_img(pspnet_model, img_path[i])[0]
    torch.save(pspnet_model_res, netdisk_path + "pspnet_model/" + img_name[i] + ".pt")
    
for i in tqdm(range(inference_num)):
    deeplabv3plus_model_res = inference_img(deeplabv3plus_model, img_path[i])[0]
    torch.save(deeplabv3plus_model_res, netdisk_path + "deeplabv3p_model/" + img_name[i] + ".pt")
    
for i in tqdm(range(inference_num)):
    fcn_model_res = inference_img(fcn_model, img_path[i])[0]
    torch.save(fcn_model_res, netdisk_path + "fcn_model/" + img_name[i] + ".pt")
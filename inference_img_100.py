from tools import ADE20KDataset, inference_img
from mmseg.apis import init_segmentor
from torchvision.transforms import Compose
import torchvision.transforms as transforms
from tqdm import tqdm
import torch

pspnet_inference_flag = True
fcn_inference_flag = False
deeplabv3p_inference_flag = False

'''
-------------------------------------------
                  加载模型
-------------------------------------------
'''
if pspnet_inference_flag:
    pspnet_config_file = 'configs/pspnet/pspnet_r50-d8_512x1024_40k_ade20k.py'
    pspnet_checkpoint_file = 'work_dirs/pspnet_r50-d8_512x1024_40k_ade20k/pspnet_r50-d8_512x1024_40k_ade20k_20200629_143358-1e7e6a4a.pth'
    pspnet_model = init_segmentor(pspnet_config_file, pspnet_checkpoint_file, device='cuda:0')
    
if fcn_inference_flag:
    fcn_config_file = 'configs/fcn/fcn_r50-d8_512x1024_40k_ade20k.py'
    fcn_checkpoint_file = 'work_dirs/fcn_r50-d8_512x1024_40k_ade20k/fcn_r50-d8_512x1024_40k_ade20k_20200629_143358-1e7e6a4a.pth'
    fcn_model = init_segmentor(fcn_config_file, fcn_checkpoint_file, device='cuda:0')

if deeplabv3p_inference_flag:
    deeplabv3p_config_file = 'configs/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_40k_ade20k.py'
    deeplabv3p_checkpoint_file = 'work_dirs/deeplabv3plus_r50-d8_512x1024_40k_ade20k/deeplabv3plus_r50-d8_512x1024_40k_ade20k_20200629_143358-1e7e6a4a.pth'
    deeplabv3plus_model = init_segmentor(deeplabv3p_config_file, deeplabv3p_checkpoint_file, device='cuda:0')

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
# Dataset Generator Config
Train_num = 50
Val_num = 10
Test_num = 10

# Train[0, 50)
train_start = 0 
train_end = 50

# Val[50, 60)
val_start = train_end
val_end = val_start + Val_num

# Test[60, 70)
test_start = val_end
test_end = test_start + Test_num

train_img_path = [ADE20K_Dataset.get_img_path(i) for i in range(train_start, train_end)]
train_img_name = [ADE20K_Dataset.get_img_name(i) for i in range(train_start, train_end)]

val_img_path = [ADE20K_Dataset.get_img_path(i) for i in range(val_start, val_end)]
val_img_name = [ADE20K_Dataset.get_img_name(i) for i in range(val_start, val_end)]

test_img_path = [ADE20K_Dataset.get_img_path(i) for i in range(test_start, test_end)]
test_img_name = [ADE20K_Dataset.get_img_name(i) for i in range(test_start, test_end)]

netdisk_train_path = "/root/Desktop/我的网盘/inference_tensor_train/"
netdisk_val_path = "/root/Desktop/我的网盘/inference_tensor_val/"
netdisk_test_path = "/root/Desktop/我的网盘/inference_tensor_test/"

'''
-------------------PSPnet------------------------
'''
if pspnet_inference_flag:
    print("PSPnet Inference...")

    print("Train Inference...")
    for i in tqdm(range(train_start, train_end)):
        pspnet_model_res = inference_img(pspnet_model, train_img_path[i])[0]
        torch.save(pspnet_model_res, netdisk_train_path + "pspnet_model/" + train_img_name[i] + ".pt")

    print("Val Inference...")
    for i in tqdm(range(val_start, val_end)):
        pspnet_model_res = inference_img(pspnet_model, val_img_path[i])[0]
        torch.save(pspnet_model_res, netdisk_val_path + "pspnet_model/" + val_img_name[i] + ".pt")

    print("Test Inference...")
    for i in tqdm(range(test_start, test_end)):
        pspnet_model_res = inference_img(pspnet_model, test_img_path[i])[0]
        torch.save(pspnet_model_res, netdisk_test_path + "pspnet_model/" + test_img_name[i] + ".pt")

'''
-------------------Deeplabv3plus------------------------
'''
if deeplabv3p_inference_flag:
    print("Deeplabv3plus Inference...")

    print("Train Inference...")
    for i in tqdm(range(train_start, train_end)):
        deeplabv3plus_model_res = inference_img(deeplabv3plus_model, train_img_path[i])[0]
        torch.save(deeplabv3plus_model_res, netdisk_train_path + "deeplabv3plus_model/" + train_img_name[i] + ".pt")

    print("Val Inference...")
    for i in tqdm(range(val_start, val_end)):
        deeplabv3plus_model_res = inference_img(deeplabv3plus_model, val_img_path[i])[0]
        torch.save(deeplabv3plus_model_res, netdisk_val_path + "deeplabv3plus_model/" + val_img_name[i] + ".pt")

    print("Test Inference...")
    for i in tqdm(range(test_start, test_end)):
        deeplabv3plus_model_res = inference_img(deeplabv3plus_model, test_img_path[i])[0]
        torch.save(deeplabv3plus_model_res, netdisk_test_path + "deeplabv3plus_model/" + test_img_name[i] + ".pt")

'''
-------------------FCN------------------------
'''
if fcn_inference_flag:
    print("FCN Inference...")

    print("Train Inference...")
    for i in tqdm(range(train_start, train_end)):
        fcn_model_res = inference_img(fcn_model, train_img_path[i])[0]
        torch.save(fcn_model_res, netdisk_train_path + "fcn_model/" + train_img_name[i] + ".pt")

    print("Val Inference...")
    for i in tqdm(range(val_start, val_end)):
        fcn_model_res = inference_img(fcn_model, val_img_path[i])[0]
        torch.save(fcn_model_res, netdisk_val_path + "fcn_model/" + val_img_name[i] + ".pt")

    print("Test Inference...")
    for i in tqdm(range(test_start, test_end)):
        fcn_model_res = inference_img(fcn_model, test_img_path[i])[0]
        torch.save(fcn_model_res, netdisk_test_path + "fcn_model/" + test_img_name[i] + ".pt")
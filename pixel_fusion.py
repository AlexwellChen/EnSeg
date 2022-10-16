from configparser import Interpolation
from dataset_tools import ADE20KDataset
from torchvision.transforms import Compose
import torchvision.transforms as transforms
import torchvision.transforms.functional as f
from tqdm import tqdm
import torch

pspnet_inference_flag = True
fcn_inference_flag = True
deeplabv3p_inference_flag = True

Train_Flag = True
Val_Flag = True
Test_Flag = True

'''
-------------------------------------------
                  加载数据集
-------------------------------------------
'''
train_path = '../../ADEChallengeData2016/images/training/'
train_label_path = '../../ADEChallengeData2016/annotations/training/'
val_path = '../../ADEChallengeData2016/images/validation/'
val_label_path = '../../ADEChallengeData2016/annotations/validation/'

# 统一尺寸
transform = Compose([transforms.ToTensor(), transforms.Resize([512, 512], interpolation = f._interpolation_modes_from_int(0))])
ADE20K_Dataset = ADE20KDataset(train_path, train_label_path, transform)

'''
-------------------------------------------
                推理并保存Tensor
-------------------------------------------
'''
# Dataset Generator Config
Train_num = 100
Val_num = 10
Test_num = 100

# Train[0, 50)
train_start = 0 
train_end = Train_num

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

    if Train_Flag:
        print("Train Inference...")
        for i in tqdm(range(train_start, train_end)):
            idx = i - train_start
            pspnet_model_res = inference_img(pspnet_model, train_img_path[idx])[0]
            torch.save(pspnet_model_res, netdisk_train_path + "pspnet_model/" + train_img_name[idx] + ".pt")

    if Val_Flag:
        print("Val Inference...")
        for i in tqdm(range(val_start, val_end)):
            idx = i - val_start
            pspnet_model_res = inference_img(pspnet_model, val_img_path[idx])[0]
            torch.save(pspnet_model_res, netdisk_val_path + "pspnet_model/" + val_img_name[idx] + ".pt")

    if Test_Flag:
        print("Test Inference...")
        for i in tqdm(range(test_start, test_end)):
            idx = i - test_start
            pspnet_model_res = inference_img(pspnet_model, test_img_path[idx])[0]
            torch.save(pspnet_model_res, netdisk_test_path + "pspnet_model/" + test_img_name[idx] + ".pt")

'''
-------------------Deeplabv3plus------------------------
'''
if deeplabv3p_inference_flag:
    print("Deeplabv3plus Inference...")

    if Train_Flag:
        print("Train Inference...")
        for i in tqdm(range(train_start, train_end)):
            idx = i - train_start
            deeplabv3plus_model_res = inference_img(deeplabv3plus_model, train_img_path[idx])[0]
            torch.save(deeplabv3plus_model_res, netdisk_train_path + "deeplabv3plus_model/" + train_img_name[idx] + ".pt")

    if Val_Flag:
        print("Val Inference...")
        for i in tqdm(range(val_start, val_end)):
            idx = i - val_start
            deeplabv3plus_model_res = inference_img(deeplabv3plus_model, val_img_path[idx])[0]
            torch.save(deeplabv3plus_model_res, netdisk_val_path + "deeplabv3plus_model/" + val_img_name[idx] + ".pt")

    if Test_Flag:
        print("Test Inference...")
        for i in tqdm(range(test_start, test_end)):
            idx = i - test_start
            deeplabv3plus_model_res = inference_img(deeplabv3plus_model, test_img_path[idx])[0]
            torch.save(deeplabv3plus_model_res, netdisk_test_path + "deeplabv3plus_model/" + test_img_name[idx] + ".pt")

'''
-------------------FCN------------------------
'''
if fcn_inference_flag:
    print("FCN Inference...")

    if Train_Flag:
        print("Train Inference...")
        for i in tqdm(range(train_start, train_end)):
            idx = i - train_start
            fcn_model_res = inference_img(fcn_model, train_img_path[idx])[0]
            torch.save(fcn_model_res, netdisk_train_path + "fcn_model/" + train_img_name[idx] + ".pt")

    if Val_Flag:
        print("Val Inference...")
        for i in tqdm(range(val_start, val_end)):
            idx = i - val_start
            fcn_model_res = inference_img(fcn_model, val_img_path[idx])[0]
            torch.save(fcn_model_res, netdisk_val_path + "fcn_model/" + val_img_name[idx] + ".pt")

    if Test_Flag:
        print("Test Inference...")
        for i in tqdm(range(test_start, test_end)):
            idx = i - test_start
            fcn_model_res = inference_img(fcn_model, test_img_path[idx])[0]
            torch.save(fcn_model_res, netdisk_test_path + "fcn_model/" + test_img_name[idx] + ".pt")




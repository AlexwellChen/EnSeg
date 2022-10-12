from torch import nn
import torch.nn.functional as F
import torch
import numpy as np
from dataset_tools import TensorDataset
from train_tools import training_loop
from torch.utils.data import DataLoader

class FusionModel(nn.Module):
    '''
    输入: 三个模型的分割结果
    输出: 融合后的分割结果
    说明:
        每一个分割结果对应一个长度为150的向量, 每个向量的第i个元素表示第i类的权重
        三个分割结果的权重向量相加, 然后softmax, 得到最终的分割结果
    '''
    def __init__(self, class_num):
        super().__init__()
        # class_num: 150
        vec_1 = torch.from_numpy(np.random.rand(class_num)).reshape(class_num, 1, 1)
        vec_2 = torch.from_numpy(np.random.rand(class_num)).reshape(class_num, 1, 1)
        vec_3 = torch.from_numpy(np.random.rand(class_num)).reshape(class_num, 1, 1)
        self.w1 = torch.nn.Parameter(vec_1) 
        self.w2 = torch.nn.Parameter(vec_2)
        self.w3 = torch.nn.Parameter(vec_3)
        self.Softmax = nn.Softmax()
        
    def forward(self, input_seg1, input_seg2, input_seg3):
        # input_seg size: (150, H, W)
        res_1 = input_seg1 * self.w1
        res_2 = input_seg2 * self.w2
        res_3 = input_seg3 * self.w3
        res = res_1 + res_2 + res_3
        # TODO: Softmax dim BUG HERE
        # res = self.Softmax(res, dim=0) 
        return res

'''
--------------------- 以下是测试代码 ---------------------
'''
# device = "cpu"

# model = FusionModel(150,device)
# model.to(device)

'''
数据准备
Size: (150, H, W)
Name:
    deeplabv3p_logits_res
    pspnet_logits_res
    fcn_logits_res
来源: ./inference_tensor
Type: torch.Tensor
'''
'''
预测范围: 0-149, 对应1-150类别
Label范围: 1-150, 0是背景
'''
# device = "cpu"
# # 模型定义
# model = FusionModel(150)
# model.to(device)
# Tensor_Dataset = TensorDataset(root='./inference_tensor/', label_root='../../ADEChallengeData2016/annotations/training', device=device)
# tensor_dict, annotation_tensor = Tensor_Dataset[0]
# annotation_tensor = annotation_tensor - 1 # 忽略背景类，将label范围从1-150变为0-149

# deeplabv3p_logits_res = tensor_dict['deeplabv3p'].to(device)
# pspnet_logits_res = tensor_dict['pspnet'].to(device)
# fcn_logits_res = tensor_dict['fcn'].to(device)

# res = model.forward(deeplabv3p_logits_res, pspnet_logits_res, fcn_logits_res)

# loss = nn.CrossEntropyLoss(ignore_index=-1)
# res = res.unsqueeze(0)
# pspnet_logits_res = pspnet_logits_res.unsqueeze(0)
# annotation_tensor = annotation_tensor.unsqueeze(0)
# ce_loss = loss(pspnet_logits_res, annotation_tensor)
# print(ce_loss)

'''
--------------------- 训练 ---------------------
'''

netdisk_train_path = "/root/Desktop/我的网盘/inference_tensor_train/"
netdisk_val_path = "/root/Desktop/我的网盘/inference_tensor_val/"
netdisk_test_path = "/root/Desktop/我的网盘/inference_tensor_test/"
netdisk_label_train_path = "/root/Desktop/我的网盘/Label/train/"
netdisk_label_val_path = "/root/Desktop/我的网盘/Label/val/"

device = "cuda:0"
# # 模型定义
# model = FusionModel(150)
# model.to(device)

# 数据准备
Train_tensor = TensorDataset(root=netdisk_train_path, label_root=netdisk_label_train_path, device=device)
Val_tensor = TensorDataset(root=netdisk_val_path, label_root=netdisk_label_val_path, device=device)

train_dataloader = DataLoader(Train_tensor, batch_size=1, shuffle=True)
val_dataloader = DataLoader(Val_tensor, batch_size=1, shuffle=True)

criterion = nn.CrossEntropyLoss(ignore_index=-1)
model = FusionModel(150)
epochs_num = 5
opt = torch.optim.Adam(model.parameters(),
                lr=0.001,
                betas=(0.9, 0.999),
                eps=1e-08)

trained_model, train_losses_vgg, train_accs_vgg, val_losses_vgg, val_accs_vgg= training_loop(model, optimizer=opt, 
                                                                     loss_fn=criterion, train_loader=train_dataloader, 
                                                                     val_loader = val_dataloader, 
                                                                     num_epochs=epochs_num, print_every=5)
                                                
# 保存模型
model_save_path = "/root/Desktop/我的网盘/"
torch.save(trained_model, model_save_path + "fusion_model.pth")


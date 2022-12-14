from torch import nn
import torch.nn.functional as F
import torch
import numpy as np
from dataset_tools import TensorDataset
from train_tools import training_loop
from torch.utils.data import DataLoader

train_flag = False
multi_GPU = False

class FusionModel_512(nn.Module):
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
        self.w1 = torch.nn.Parameter(torch.from_numpy(np.random.rand(class_num*512*512)).reshape(class_num, 512, 512))  
        self.w2 = torch.nn.Parameter(torch.from_numpy(np.random.rand(class_num*512*512)).reshape(class_num, 512, 512)) 
        self.w3 = torch.nn.Parameter(torch.from_numpy(np.random.rand(class_num*512*512)).reshape(class_num, 512, 512)) 
        
    def forward(self, input_seg1, input_seg2, input_seg3):
        # input_seg size: (150, 512, 512)
        # TODO: Softmax dim BUG HERE
        # res = self.Softmax(res, dim=0) 
        return input_seg1 * self.w1 + input_seg2 * self.w2 + input_seg3 * self.w3

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
if train_flag:
    netdisk_train_path = "/root/Desktop/我的网盘/inference_tensor_train/"
    netdisk_val_path = "/root/Desktop/我的网盘/inference_tensor_val/"
    netdisk_test_path = "/root/Desktop/我的网盘/inference_tensor_test/"
    netdisk_label_train_path = "/root/Desktop/我的网盘/Label/train/"
    netdisk_label_val_path = "/root/Desktop/我的网盘/Label/val/"
    device_ids = [0, 1]
    if multi_GPU:
        model = nn.DataParallel(FusionModel_512(150), device_ids=device_ids)
    else:
        # 模型定义
        model = FusionModel_512(150)
    device = "cuda:0"

    # 数据准备
    Train_tensor = TensorDataset(root=netdisk_train_path, label_root=netdisk_label_train_path, device=device)
    Val_tensor = TensorDataset(root=netdisk_val_path, label_root=netdisk_label_val_path, device=device)


    if multi_GPU:
        train_dataloader = DataLoader(Train_tensor, batch_size=2, shuffle=True)
        val_dataloader = DataLoader(Val_tensor, batch_size=2, shuffle=True)
    else:
        # 由于每张图像Tensor的H和W不一致, 因此batch_size必须为1
        train_dataloader = DataLoader(Train_tensor, batch_size=1, shuffle=True)
        val_dataloader = DataLoader(Val_tensor, batch_size=1, shuffle=True)

    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    epochs_num = 5
    opt = torch.optim.Adam(model.parameters(),
                    lr=0.001,
                    betas=(0.9, 0.999),
                    eps=1e-08)
    if multi_GPU:
        opt = nn.DataParallel(opt, device_ids=device_ids)
    
    trained_model, train_losses, train_IoU, val_losses, val_IoU= training_loop(model, optimizer=opt, 
                                                                        loss_fn=criterion, train_loader=train_dataloader, 
                                                                        val_loader = val_dataloader, 
                                                                        num_epochs=epochs_num, print_every=20)
                                                    
    # 保存模型
    model_save_path = "/root/Desktop/我的网盘/"
    torch.save(trained_model, model_save_path + "fusion_model_100_fix_512_Epoch_5.pth")

    # 保存训练过程中的loss和IoU
    train_data_path = "/root/Desktop/我的网盘/train_data/"
    data_dic = {'train_losses': train_losses, 'train_IoU': train_IoU, 'val_losses': val_losses, 'val_IoU': val_IoU}
    np.save(model_save_path + 'data_dic_100_fix_512_Epoch_5.npy', data_dic)



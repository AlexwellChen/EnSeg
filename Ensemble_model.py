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
    def __init__(self, class_num, device):
        super().__init__()
        # class_num: 150
        self.vec_1 = torch.from_numpy(np.random.rand(class_num)).reshape(class_num, 1, 1).to(device)
        self.vec_2 = torch.from_numpy(np.random.rand(class_num)).reshape(class_num, 1, 1).to(device)
        self.vec_3 = torch.from_numpy(np.random.rand(class_num)).reshape(class_num, 1, 1).to(device)
        
        self.Softmax = nn.Softmax()
        
    def forward(self, input_seg1, input_seg2, input_seg3):
        # input_seg size: (150, H, W)
        res_1 = input_seg1 * self.vec_1
        res_2 = input_seg2 * self.vec_2
        res_3 = input_seg3 * self.vec_3
        res = res_1 + res_2 + res_3
        # TODO: Softmax dim BUG HERE
        # res = self.Softmax(res, dim=0) 
        return res

'''
-------------------- Loss Function --------------------
'''
class SoftIoULoss(nn.Module):
    def __init__(self, n_classes):
        super(SoftIoULoss, self).__init__()
        self.n_classes = n_classes

    @staticmethod
    def to_one_hot(tensor, n_classes):
        h, w = tensor.size()
        one_hot = torch.zeros(n_classes, h, w).scatter_(1, tensor.view(1, h, w), 1)
        return one_hot

    def forward(self, input, target):
        # logit => Classes x H x W
        # target => H x W

        N = len(input)

        pred = F.softmax(input, dim=0)
        target_onehot = self.to_one_hot(target, self.n_classes)

        # Numerator Product
        inter = pred * target_onehot
        # Sum over all pixels C x H x W => C
        inter = inter.view(self.n_classes, -1).sum(1)

        # Denominator
        union = pred + target_onehot - (pred * target_onehot)
        # Sum over all pixels C x H x W => C
        union = union.view(self.n_classes, -1).sum(1)

        loss = inter / (union + 1e-16)

        # Return average loss over classes and batch
        return -loss.mean()


class CrossEntropy2d(nn.Module):
    def __init__(self, ignore_label=0):
        super().__init__()
        self.ignore_label = ignore_label

    def forward(self, predict, target):
        """
        :param predict: [batch, num_class, height, width]
        :param target: [batch, height, width]
        :return: entropy loss
        """
        target_mask = target != self.ignore_label  # [batch, height, width]筛选出所有需要训练的像素点标签
        target = target[target_mask]  # [num_pixels]
        batch, num_class, height, width = predict.size()
        predict = predict.permute(0, 2, 3, 1)  # [batch, height, width, num_class]
        predict = predict[target_mask.unsqueeze(-1).repeat(1, 1, 1, num_class)].view(-1, num_class)
        loss = F.cross_entropy(predict, target)
        return loss	

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

device = "cuda:0"
# 模型定义
model = FusionModel(150,device)
model.to(device)

# 数据准备
Train_tensor = TensorDataset(root=netdisk_train_path, label_root='../../ADEChallengeData2016/annotations/training', device=device)
Val_tensor = TensorDataset(root=netdisk_val_path, label_root='../../ADEChallengeData2016/annotations/validation', device=device)

train_dataloader = DataLoader(Train_tensor, batch_size=1, shuffle=True)
val_dataloader = DataLoader(Val_tensor, batch_size=1, shuffle=True)

criterion = nn.CrossEntropyLoss(ignore_index=-1)
model = FusionModel(150,device)
epochs_num = 5
opt = torch.optim.Adam(model.parameters(),
                lr=0.001,
                betas=(0.9, 0.999),
                eps=1e-08)

trained_model, train_losses_vgg, train_accs_vgg, val_losses_vgg, val_accs_vgg= training_loop(model, optimizer=opt, 
                                                                     loss_fn=criterion, train_loader=train_dataloader, 
                                                                     val_loader = val_dataloader, 
                                                                     num_epochs=epochs_num, print_every=5)

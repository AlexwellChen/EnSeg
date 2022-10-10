from torch import nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset
from itertools import chain
from pathlib import Path
from PIL import Image
import numpy as np
from dataset_tools import TensorDataset

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
        return res.argmax(0)


'''
--------------------- 以下是测试代码 ---------------------
'''
device = "cpu"

model = FusionModel(150,device)
model.to(device)

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

Tensor_Dataset = TensorDataset(root='./inference_tensor/', label_root='../../ADEChallengeData2016/annotations/training', device=device)
tensor_dict, annotation_tensor = Tensor_Dataset[2]
deeplabv3p_logits_res = tensor_dict['deeplabv3p'].to(device)
pspnet_logits_res = tensor_dict['pspnet'].to(device)
fcn_logits_res = tensor_dict['fcn'].to(device)
res = model.forward(deeplabv3p_logits_res, pspnet_logits_res, fcn_logits_res)
print(res.shape)
print(res)
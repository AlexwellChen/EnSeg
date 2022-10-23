from itertools import chain
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torch
import numpy as np
from cv2 import imread, IMREAD_GRAYSCALE
import torchvision.transforms as transforms
import torchvision.transforms.functional as f

resize_flag = False

'''
Load ADE20K dataset
'''
class ADE20KDataset(Dataset):
    def __init__(self, root, label_root, transforms=None):
        self.root = Path(root)
        self.label_root = Path(label_root)
        self.transforms = transforms
        self.imgs = list(sorted(chain(self.root.glob('*.jpg'), self.root.glob('*.png'))))
        self.labels = list(sorted(chain(self.label_root.glob('*.jpg'), self.label_root.glob('*.png'))))
    
    def __getitem__(self, idx):
        '''
        ADE20KDataset: return tuple, first is image, second is label

        return (img (Tensor.float), label (Tensor.int32))
        '''
        img_path = self.imgs[idx]
        label_path = self.labels[idx]
        img = Image.open(img_path).convert("RGB")
        label = Image.open(label_path)
        if self.transforms is not None:
            img = self.transforms(img)
            if resize_flag:
                resize_module = transforms.Resize([512, 512], interpolation = f._interpolation_modes_from_int(0))
                label = resize_module(label)
            label = np.array(label, dtype = np.uint8)
            label = torch.Tensor(label).int()
        return img, label
    
    def __len__(self):
        return len(self.imgs)

    def get_img_path(self, idx):
        return str(self.imgs[idx])
    
    def get_label_path(self, idx):
        return str(self.labels[idx])

    def get_img_as_ndarray(self, idx):
        return imread(self.get_img_path(idx))
    
    def get_label_as_ndarray(self, idx):
        return imread(self.get_label_path(idx), IMREAD_GRAYSCALE)

    def get_img_as_PIL(self, idx):
        return Image.open(self.imgs[idx]).convert("RGB")
    
    def get_label_as_PIL(self, idx):
        return Image.open(self.labels[idx])

    def get_img_name(self, idx):
        return str(Path(self.get_img_path(idx)).stem)


'''
------------- Inference Tensor DataLoader -------------
'''
class TensorDataset(Dataset):
    def __init__(self, root, label_root, device):
        self.device = device
        self.root = Path(root)
        self.label_root = Path(label_root)
        deeplabv3p_tensor_path = Path(root + 'deeplabv3plus_model')
        fcn_tensor_path = Path(root + 'fcn_model')
        pspnet_tensor_path = Path(root + 'pspnet_model')
        self.deeplabv3p_tensor = list(sorted(chain(deeplabv3p_tensor_path.glob('*.pt'))))
        self.fcn_tensor = list(sorted(chain(fcn_tensor_path.glob('*.pt'))))
        self.pspnet_tensor = list(sorted(chain(pspnet_tensor_path.glob('*.pt'))))
        self.labels = list(sorted(chain(self.label_root.glob('*.jpg'), self.label_root.glob('*.png'))))

    
    def __getitem__(self, idx):
        '''
        Input: idx
        Output: {'deeplabv3p': Tensor, 'fcn': Tensor, 'pspnet': Tensor}, annotation_tensor
        '''
        deeplabv3p_tensor = torch.load(self.deeplabv3p_tensor[idx], map_location=torch.device(self.device))
        fcn_tensor = torch.load(self.fcn_tensor[idx], map_location=torch.device(self.device))
        pspnet_tensor = torch.load(self.pspnet_tensor[idx], map_location=torch.device(self.device))
        label_path = self.labels[idx]
        annotation = Image.open(label_path)
        if resize_flag:
            annotation = annotation.resize((512, 512),Image.NEAREST)
        annotation = np.array(annotation, dtype = np.uint8)
        annotation_tensor = torch.Tensor(annotation).long()
        return {'deeplabv3p':deeplabv3p_tensor, 'fcn':fcn_tensor, 'pspnet':pspnet_tensor}, annotation_tensor
    
    def __len__(self):
        return len(self.labels)
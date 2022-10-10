from itertools import chain
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


'''
加载ADE20K数据集
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
        ADE20KDataset: 返回为tuple, 第一个为image, 第二个为label

        return (img (Tensor.float), label (Tensor.int32))
        '''
        img_path = self.imgs[idx]
        label_path = self.labels[idx]
        img = Image.open(img_path).convert("RGB")
        label = Image.open(label_path)
        if self.transforms is not None:
            img = self.transforms(img)
            label = np.array(label, dtype = np.uint8)
            label = torch.Tensor(label).int()
        return img, label
    
    def __len__(self):
        return len(self.imgs)

    def get_img_path(self, idx):
        return str(self.imgs[idx])
    
    def get_label_path(self, idx):
        return str(self.labels[idx])

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
        deeplabv3p_tensor_path = Path(root + 'deeplabv3p_model')
        fcn_tensor_path = Path(root + 'fcn_model')
        pspnet_tensor_path = Path(root + 'pspnet_model')
        self.deeplabv3p_tensor = list(sorted(chain(deeplabv3p_tensor_path.glob('*.pt'))))
        self.fcn_tensor = list(sorted(chain(fcn_tensor_path.glob('*.pt'))))
        self.pspnet_tensor = list(sorted(chain(pspnet_tensor_path.glob('*.pt'))))
        self.labels = list(sorted(chain(self.label_root.glob('*.jpg'), self.label_root.glob('*.png'))))

    
    def __getitem__(self, idx):
        '''
        输入: idx
        返回: {'deeplabv3p': Tensor, 'fcn': Tensor, 'pspnet': Tensor}, annotation_tensor
        '''
        deeplabv3p_tensor = torch.load(self.deeplabv3p_tensor[idx], map_location=torch.device(self.device))
        fcn_tensor = torch.load(self.fcn_tensor[idx], map_location=torch.device(self.device))
        pspnet_tensor = torch.load(self.pspnet_tensor[idx], map_location=torch.device(self.device))
        label_path = self.labels[idx]
        annotation = Image.open(label_path)
        annotation = np.array(annotation, dtype = np.uint8)
        annotation_tensor = torch.Tensor(annotation).int()
        return {'deeplabv3p':deeplabv3p_tensor, 'fcn':fcn_tensor, 'pspnet':pspnet_tensor}, annotation_tensor
    
    def __len__(self):
        return len(self.tensor_list)
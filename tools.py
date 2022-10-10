from torch.utils.data import Dataset
from itertools import chain
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import mmcv
from mmcv.parallel import collate, scatter
from mmseg.datasets.pipelines import Compose
from collections import Counter

'''
多数投票融合
输入: 三个模型的推理结果, type: np.ndarray
输出：融合后的结果, type: np.ndarray
'''
def majorVote(seg_mat1, seg_mat2, seg_mat3):
    res = np.zeros(seg_mat1.shape)

    i_len = seg_mat1.shape[0]
    j_len = seg_mat1.shape[1]

    for i in range(i_len):
        for j in range(j_len):
            res_cnt = Counter([seg_mat1[i][j], seg_mat2[i][j], seg_mat3[i][j]]).most_common()
            res[i][j] = res_cnt[0][0]
    
    return res

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
        return self.imgs[idx]
    
    def get_label_path(self, idx):
        return self.labels[idx]

    def get_img_as_PIL(self, idx):
        return Image.open(self.imgs[idx]).convert("RGB")
    
    def get_label_as_PIL(self, idx):
        return Image.open(self.labels[idx])



class LoadImage:
    """A simple pipeline to load image."""

    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """

        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results

def inference_img(my_model, img_path):
    '''
    input: 
    my_model 推理使用的模型
    img_path 单张图像的path

    output:
    图像分割的logits，(1, 150, H, W)
    '''
    model = my_model
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = []
    imgs = img_path if isinstance(img_path, list) else [img_path]
    for img in imgs:
        img_data = dict(img=img)
        img_data = test_pipeline(img_data)
        data.append(img_data)
    data = collate(data, samples_per_gpu=len(imgs))
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        data['img_metas'] = [i.data[0] for i in data['img_metas']]
    img_meta = data['img_metas'][0]
    img = data['img'][0]
    output = model.inference(img, img_meta, True)
    return output


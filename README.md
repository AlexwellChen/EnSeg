# EnSeg

*Repo for SSY340 project*
![](img/Fusion%20model%20weight.png)

### 文件结构与用途

* inference_tensor：存放三个图像分割模型的推理结果，以Pytorch Tensor格式存储，大小(150, H, W)
* dataset_tools.py：与数据集加载有关的工具函数
* Ensemble_model.py：定义了分割融合模型， 以及SoftIoU Loss
* inference_demo.ipynb：mmsegmentation的教程
* inference_img_100.py：推理数据集中前100张图片
* tools.py：用于图像分割推理过程的工具函数，以及Major Vote方法的实现
* train_tools.py：用于构建融合模型的训练

### Major Vote
通过mmsegmentation的接口得到三个模型的推理结果，通过多数投票确定每一个像素的类别。

### Average
通过自定义的推理接口得到每一个像素在150个类别上的logit，平均后得到每一个像素的概率并确定最终类别

### FusionMode
通过对每一个(150, H, W)的Tensor中每一个类别赋予不同的权重，即在每一个类别中进行(H, W) * Weight。以Tensor的角度来看是(150, H, W)和(150, 1, 1)进行了广播乘法操作。我们对三个模型的结果加权求和，最终确定融合后每一个像素的类别。



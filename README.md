## Point-to-Point regression: Accurate infrared small target detection with single-point annotation
Paper: https://ieeexplore.ieee.org/document/10937752
## Contributions
1.We propose a novel P2P-HDNet that directly predicts target locations, achieving SOTA performance even with single-point annotation.

2.We devise two key components: HCEM and ATLDH. HCEM maintains high resolution throughout feature extraction, enriching deep feature maps with detailed target information. ATLDH predicts target locations by regressing a Gaussian heatmap, enhancing localization accuracy.

3.Experiments on two public datasets, NUDT-SIRST [16] and IRSTD-1k [30], demonstrate the superiority of our approach over representative SOTA IRSTD methods. Additionally, we provide single-point annotation for existing public IRSTs datasets, supporting further advancements in IRSTD.
## Structure
！[Structure](https://github.com/ActCVLab/P2P-Net/blob/main/Fig/structure.png)
![Structure](https://raw.githubusercontent.com/ActCVLab/P2P-Net/main/Fig/structure.png)
## Requirements
Recommended environment:

  Python 3.9+
  PyTorch 2.x
  CUDA 11.8+
Main dependencies:

  torch
  torchvision
  tensorboard
  numpy
  opencv-python
  pycocotools
  matplotlib 
  scipy
  pillow
  tqdm

## Datasets
**Our project has the following structure:**
  ```
  |---dataset/
  |    |--- NUDT-SIRST
  |    |    |--- test
  |    |    |    |--- annotations
  |    |    |    |      |--- annotations.json 
  |    |    |    |--- test_images
  |    |    |    |      |--- 0008.png  
  |    |    |    |      |--- 0010.png
  |    |    |    |      |--- .....
  |    |    |--- train
  |    |    |    |--- annotations
  |    |    |    |      |--- annotations.json 
  |    |    |    |--- train_images
  |    |    |    |      |--- 0001.png  
  |    |    |    |      |--- 0002.png
  |    |    |    |      |--- .....
  |    |    |--- val
  |    |    |    |--- annotations
  |    |    |    |      |--- annotations.json 
  |    |    |    |--- val_images  
  |    |    |    |      |--- 0006.png  
  |    |    |    |      |--- 0017.png
  |    |    |    |      |--- .....             
  |    |--- IRSTD-1K
  |    |    |--- test
  |    |    |    |--- annotations
  |    |    |    |      |--- annotations.json 
  |    |    |    |--- test_images
  |    |    |    |      |--- XDU0.png  
  |    |    |    |      |--- XDU10.png
  |    |    |    |      |--- .....
  |    |    |--- train
  |    |    |    |--- annotations
  |    |    |    |      |--- annotations.json 
  |    |    |    |--- train_images
  |    |    |    |      |--- XDU1.png  
  |    |    |    |      |--- XDU2.png
  |    |    |    |      |--- .....
  |    |    |--- val
  |    |    |    |--- annotations
  |    |    |    |      |--- annotations.json 
  |    |    |    |--- val_images 
  |    |    |    |      |--- XDU14.png  
  |    |    |    |      |--- XDU17.png
  |    |    |    |      |--- .....   
  |    |--- ...  
  ```
<be>
In this format:
test_images/ stores images
annotations/annotations.json is a COCO-style keypoint annotation file

#### Quantitative Results on  NUDT-SIRST, and IRSTD-1K. i.e, one weight for two Datasets.

| Model         |  pre(%) | rec(%) |  F1(%)  | 
| ------------- |:-------:|:------:|:-------:|
| NUDT-SIRST    |  99.73  |  98.42 |  94.55  |  
| IRSTD-1K      |  96.42  |  92.76 |  99.07  | 

## Training
```
python train.py
```

## Citation
If you use this code, please kindly cite our paper:
```
@ARTICLE{ni2025p2p,
  author={Ni, Rixiang and Wu, Jing and Qiu, Zhaobin and Chen, Liqiong and Luo, Changhai and Huang, Feng and Liu, Qiujiang and Wang, Binxing and Li, Yunxiang and Li, Youli},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  title={Point-to-Point regression: Accurate infrared small target detection with single-point annotation},
  year={2025},
  volume={63},
  pages={1-19},
  publisher={IEEE}
}
```
## Contact
If you have any questions, please contact:  
Author: Zhaobing Qiu  
Email: qiuzhaobing@fzu.edu.cn  
Copyright: Fuzhou University 
## License
This code is only freely available for non-commercial research use.

If you find some help for you, star is a good reward ^_^. 

# Degradation-Aware Feature Perturbation for All-in-One Image Restoration (CVPR'25)

Xiangpeng Tian, Xiangyu Liao, Xiao Liu, Meng Li, Chao Ren*

## Abstract

All-in-one image restoration aims to recover clear images from various degradation types and levels with a unified model. Nonetheless, the significant variations among degradation types present challenges for training a universal model, often resulting in task interference, where the gradient update directions of different tasks may diverge due to shared parameters. To address this issue, motivated by the routing strategy, we propose DFPIR, a novel all-in-one image restorer that introduces Degradation-aware Feature Perturbations(DFP) to adjust the feature space to align with the unified parameter space. In this paper, the feature perturbations primarily include channel-wise perturbations and attention-wise perturbations. Specifically, channel-wise perturbations are implemented by shuffling the channels in high-dimensional space guided by degradation types, while attention-wise perturbations are achieved through selective masking in the attention space. To achieve these goals, we propose a Degradation-Guided Perturbation Block (DGPB) to implement these two functions, positioned between the encoding and decoding stages of the encoder-decoder architecture. Extensive experimental results demonstrate that DFPIR achieves state-of-the-art performance on several all-in-one image restoration tasks including image denoising, image dehazing, image deraining, motion deblurring, and low-light image enhancement. 

---

## Model Architecture

![Model Architecture](./fig/shuffle-fram.jpg)  

---

## Usage

### Training

Train three types of degradations by running:

```bash
python train_3D_DFPIR.py
```
Train five types of degradations by running:
```bash
python train_5D_DFPIR.py
```
###  Testing

Test three types of degradations by running:

```bash
python test_3D_DFPIR.py
```
Test five types of degradations by running:

```bash
python test_5D_DFPIR.py
```
### Pretrained Model Weights

The pretrained weights for the three distinct degradation types under simple prompts are provided [here](https://pan.baidu.com/s/1W8mjjSB4XiL70cVK9B9Eng  )(pa3a), while detail prompts are provided [here](https://pan.baidu.com/s/1hk5JgOpl3VYEsWEecPpJkg?pwd=tjkd  )(tjkd). The pretrained weights for the five degradation types are available [here](https://pan.baidu.com/s/1LhAsRq8t4dvaD-hC6yDZrA?pwd=q0sm)(q0sm) .

We have also uploaded and shared the model weights on Google Drive. The link is: https://drive.google.com/drive/folders/15bOk4xrsK1b3nIu3-OzUj4O_ZssdFN5Y?usp=sharing

### Results
Performance results of the DFPIR framework trained under the all-in-one setting.
**Three Distinct Degradations**:

![3D](./fig/3D.jpg)  

**Five Distinct Degradations**:

![5D](./fig/5D.jpg) 

The visual results under the three degradation types are provided [here](https://pan.baidu.com/s/1xa_i7cbg5slEyLvBpC4JKg?pwd=lljd )(lljd).  The visual results under the five degradation types are provided [here](https://pan.baidu.com/s/1tfYrxfOI61om8QX9PnXLFA?pwd=tsbp)(tsbp).

### Python Runtime Environment
python: 3.11.5

pytorch: 2.1.1

numpy: 1.26.0

### Citation
If you use our work, please consider citing:
```bash
@inproceedings{tian2025degradation,
  title={Degradation-Aware Feature Perturbation for All-in-One Image Restoration},
  author={Tian, Xiangpeng and Liao, Xiangyu and Liu, Xiao and Li, Meng and Ren, Chao},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={28165--28175},
  year={2025}
}
```
## Contact

Should you have any questions, please contact tianxp@stu.scu.edu.cn

**Acknowledgment:** This code is based on the [PromptIR](https://github.com/va1shn9v/PromptIR) and [PIP]([longzilicart/pip_universal](https://github.com/longzilicart/pip_universal)) repository.

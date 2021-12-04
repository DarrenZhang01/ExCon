# ExCon: Explanation-driven Supervised Contrastive Learning

### Link to the paper: https://arxiv.org/pdf/2111.14271.pdf

### Contributors of this repo:
- Zhibo Zhang (zhibozhang@cs.toronto.edu)
- Jongseong Jang (j.jang@lgresearch.ai)
- Ruiwen Li (ruiwen.li@mail.utoronto.ca)

Copyright (c) 2021 LG AI Research and University of Toronto, all rights reserved.

If you use our code, please cite our paper:
```
@misc{zhang2021excon,
      title={ExCon: Explanation-driven Supervised Contrastive Learning for Image Classification},
      author={Zhibo Zhang and Jongseong Jang and Chiheb Trabelsi and Ruiwen Li and Scott Sanner and Yeonjeong Jeong and Dongsub Shim},
      year={2021},
      eprint={2111.14271},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

Prepare the Tiny ImageNet dataset (in the path where you want to save the dataset):
```
wget -nc https://image-net.org/data/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip
python3 ExCon/utils/val_format.py
```

### Run ExCon:
CIFAR-100 dataset
```python
python3 ExCon/main_supcon.py --epochs=200 --explainer="GradCAM" --dataset="cifar100" --batch_size=256 --method="Ex_SupCon" --learning_rate=0.5 --temp=0.1 --cosine --negative_pair=1 --validation=1 --background_anchor=0 --exp_epochs=50
```
Tiny ImageNet dataset
```python
python3 ExCon/main_supcon.py --epochs=200 --explainer="GradCAM" --dataset="ImageNet" --batch_size=128 --method="Ex_SupCon" --learning_rate=0.5 --temp=0.1 --cosine --negative_pair=1 --validation=0 --background_anchor=0 --exp_epochs=0 --data_folder=$PATH_TO_DATASET
```


## Reference Repos:

[1] https://github.com/HobbitLong/SupContrast

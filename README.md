# ExCon: Explanation-driven Supervised Contrastive Learning

### Contributors of this repo:
- Zhibo Zhang (zhibozhang@cs.toronto.edu)
- Jongseong Jang (j.jang@lgresearch.ai)
- Ruiwen Li (ruiwen.li@mail.utoronto.ca)

Copyright (c) 2021, all rights reserved.

### Run ExCon:
```python
python3 ExCon/main_supcon.py --epochs=200 --explainer="GradCAM" --dataset="cifar100" --batch_size=256 --method="Ex_SupCon" --learning_rate=0.5 --temp=0.1 --cosine --negative_pair=1 --validation=1 --background_anchor=0 --exp_epochs=0
```

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
## Reference Repos:

[1] https://github.com/HobbitLong/SupContrast

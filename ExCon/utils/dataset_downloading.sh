#!/bin/bash
# Reference: https://github.com/tjmoon0104/pytorch-tiny-imagenet/blob/master/resize.py
echo "Testing..."
# cd $SLURM_TMPDIR
wget -nc https://image-net.org/data/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip
# You may want to change the path to an absolute path to `val_format.py`.
python3 ../../val_format.py

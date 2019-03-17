# transformer-pytorch
A PyTorch implementation of transformer for text generation.

## Dependencis
- Python 3.x
- PyTorch >= 0.4
- tqdm
- numpy

## Dataset
We need to place all train/validation/test data files under the ```data``` directory,
all the files are in the same format, i.e., each sequence (sentence or document)
converted to tokenized words per line. The example data we used is
the [WMT'16 Multimodal Translation (en-de)](http://www.statmt.org/wmt16/multimodal-task.html).

## Quick Start
* Preprocess data
```
python3 preprocess.py -train_src=data/train_example.en -train_tgt=./data/train_example.de -valid_src=data/val_example.en -valid_tgt=data/val_example.de -save_data=data/en2de.pkl
```

* Training
```
python3 main_train.py -data=./data/en2de.pkl -log=./log -save_model=train -save_mode=all -proj_share_weight -label_smoothing
```

* Testing
```
python3 main_test.py -model=./log/train_xxx_xxx.ckpt -vocab=./data/en2de.pkl -src=./data/test_example.en -output_dir=./output
```

## References
[1] Vaswani et al., [Attention Is All You Need](http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf), NIPS(2017). 

[2] A PyTorch implementation [attention-is-all-you-need-pytorch](https://github.com/jadore801120/attention-is-all-you-need-pytorch).

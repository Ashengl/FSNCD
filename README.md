# FSNCD



## Installation

```shell
$ cd repository
$ pip install -r requirements.txt
```

## Config

Set paths to root and datasets in `config.py`

## Datasets

The datasets we use are:

[CUB-200](https://www.vision.caltech.edu/datasets/cub_200_2011/), [StanfordCars](https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset), [Aircraft](https://www.kaggle.com/code/metamath/fgvc-aircraft),  [Herbarium](https://www.kaggle.com/competitions/herbarium-2019-fgvc6/data),[Cifar](), [ImageNet-100](https://www.image-net.org/).

## Scripts

**Train the model**

If you wish to train SIEFormer, please run:

```
sh train.sh
```

**Eval the model**

Download the checkpoints and put them in the "checkpoints" folder and run:

```
sh test.sh
```


## Acknowledgement

Our codes are based on [Generalized Category Discovery](https://github.com/sgvaze/generalized-category-discovery) and [SimGCD](https://github.com/CVMI-Lab/SimGCD).****

# CaiT-TF (Going deeper with Image Transformers)

[![TensorFlow 2.8](https://img.shields.io/badge/TensorFlow-2.8-FF6F00?logo=tensorflow)](https://github.com/tensorflow/tensorflow/releases/tag/v2.8.0)
[![HugginFace badge](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-yellow.svg)](https://huggingface.co/spaces)
[![Models on TF-Hub](https://img.shields.io/badge/TF--Hub-Models%20on%20TF--Hub-orange)](https://tfhub.dev/sayakpaul/collections/cait/1)

This repository provides TensorFlow / Keras implementations of different CaiT
[1] variants from Touvron et al. It also provides the TensorFlow / Keras models that have been
populated with the original CaiT pre-trained params available from [2]. These
models are not blackbox SavedModels i.e., they can be fully expanded into `tf.keras.Model`
objects and one can call all the utility functions on them (example: `.summary()`).

As of today, all the TensorFlow / Keras variants of the **CaiT** models listed
[here](https://github.com/facebookresearch/deit/blob/main/README_cait.md#model-zoo) are
available in this repository. 

Refer to the ["Using the models"](https://github.com/sayakpaul/cait-tf#using-the-models)
section to get started. 

**Updates Oct 8 2022**: This project [received](https://www.kaggle.com/discussions/general/358450)
the Kaggle ML Research Spotlight Prize (September 2022)

**Updates Sept 25 2022**: [Blog post on CaiT](https://keras.io/examples/vision/cait/)

## Table of contents

* [Conversion](https://github.com/sayakpaul/cait-tf#conversion)
* [Collection of pre-trained models (converted from PyTorch to TensorFlow)](https://github.com/sayakpaul/cait-tf#models)
* [Results of the converted models](https://github.com/sayakpaul/cait-tf#results)
* [How to use the models?](https://github.com/sayakpaul/cait-tf#using-the-models)
* [References](https://github.com/sayakpaul/cait-tf#references)
* [Acknowledgements](https://github.com/sayakpaul/cait-tf#acknowledgements)

## Conversion

TensorFlow / Keras implementations are available in `cait/models.py`. Conversion
utilities are in `convert.py`.

## Models

Find the models on TF-Hub here: https://tfhub.dev/sayakpaul/collections/cait/1. You can fully inspect the
architecture of the TF-Hub models like so:

```py
import tensorflow as tf

model_gcs_path = "gs://tfhub-modules/sayakpaul/cait_xxs24_224/1/uncompressed"
model = tf.keras.models.load_model(model_gcs_path)

dummy_inputs = tf.ones((2, 224, 224, 3))
_ = model(dummy_inputs)
print(model.summary(expand_nested=True))
```

## Results

Results are on ImageNet-1k validation set (top-1 and top-5 accuracies). 

| model_name     |   top1_acc(%) |   top5_acc(%) |
|:---------------:|:--------------:|:--------------:|
| cait_s24_224   |        83.368 |        96.576 |
| cait_xxs24_224 |        78.524 |        94.212 |
| cait_xxs36_224 |        79.76  |        94.876 |
| cait_xxs36_384 |        81.976 |        96.064 |
| cait_xxs24_384 |        80.648 |        95.516 |
| cait_xs24_384  |        83.738 |        96.756 |
| cait_s24_384   |        84.944 |        97.212 |
| cait_s36_384   |        85.192 |        97.372 |
| cait_m36_384   |        85.924 |        97.598 |
| cait_m48_448   |        86.066 |        97.590 |


Results can be verified with the code in `i1k_eval`. Results are in line with [1].
[Slight differences](https://github.com/facebookresearch/deit/blob/main/README_cait.md#model-zoo) in the results stemmed
from the fact that I used a different set of augmentation transformations. Original 
transformations suggested by the authors can be found [here](https://github.com/facebookresearch/deit/blob/main/README_cait.md).


## Using the models

**Pre-trained models**:

* Off-the-shelf classification: [Colab Notebook](https://colab.research.google.com/github/sayakpaul/cait-tf/blob/main/notebooks/classification.ipynb)
* Fine-tuning: [Colab Notebook](https://colab.research.google.com/github/sayakpaul/cait-tf/blob/main/notebooks/finetune.ipynb)

These models also output attention weights from each of the Transformer blocks.
Refer to [this notebook](https://colab.research.google.com/github/sayakpaul/cait-tf/blob/main/notebooks/classification.ipynb)
for more details. Additionally, the notebook shows how to visualize the attention maps for a given image (following
figures 6 and 7 of the original paper).

| Original Image | Class Attention Maps | Class Saliency Map |
| :--: | :--: | :--: |
| ![cropped image](./assets/butterfly_cropped.png) | ![cls attn](./assets/cls_attn_heads.png) | ![saliency](./assets/cls_saliency.png) |

For the best quality, refer to the `assets` directory. You can also generate these plots using the following interactive demos on
Hugging Face Spaces:

* [Generate Class Attention plots](https://huggingface.co/spaces/probing-vits/class-attention-map)
* [Generate Class Saliency plots](https://huggingface.co/spaces/probing-vits/class-saliency)
 
**Randomly initialized models**:
 
```py
from cait.model_configs import base_config
from cait.models import CaiT
import tensorflow as tf
 
config = base_config.get_config(
    model_name="cait_xxs24_224"
)
cait_xxs24_224 = CaiT(config)

dummy_inputs = tf.ones((2, 224, 224, 3))
_ = cait_xxs24_224(dummy_inputs)
print(cait_xxs24_224.summary(expand_nested=True))
```

To initialize a network with say, 5 classes, do:

```py
config = base_config.get_config(
    model_name="cait_xxs24_224"
)
with config.unlocked():
    config.num_classes = 5
cait_xxs24_224 = CaiT(config)
```

To view different model configurations, refer to `convert_all_models.py`.

## References

[1] CaiT paper: https://arxiv.org/abs/2103.17239

[2] Official CaiT code: https://github.com/facebookresearch/deit

## Acknowledgements

* [`timm` library source code](https://github.com/rwightman/pytorch-image-models)
for the awesome codebase.
* [ML-GDE program](https://developers.google.com/programs/experts/) for
providing GCP credits that supported my experiments.

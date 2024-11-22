# My Submission to the Secret Runway Detection Challenge 🛩️

[Competition Link](https://zindi.africa/competitions/geoai-amazon-basin-secret-runway-detection-challenge)

## Overview

This is my submission for the Secret Runway Detection Challenge, part of the UN's 'AI for Good' initiative.

In this competition, we were given coordinate polygons of secret runways in the Amazon rainforest. Our task was to predict the locations of runways in areas different from the training data.

## My Approach

I started by sourcing images of size 192x192 containing existing runways from Sentinel-2 satellite imagery. I made sure to randomize the location of the runway within each image.

### First Attempt: Swin Transformer

I initially experimented with a pretrained Swin Transformer Geographical Foundation Model, as presented in [this paper](https://arxiv.org/abs/2302.04476) by Matias Mendieta et al., titled "Towards Geospatial Foundation Models via Continual Pretraining". I added a segmentation head to the model.

### Switching to a Custom CNN

Later on, I switched to a custom CNN segmentation model because I achieved better results that way. I think something went wrong with my implementation of the Swin Transformer, but I didn't have time to fix it before the deadline.

I trained the models on Google Colab using an A100 GPU.

## Inference

For inference, I ran the model on areas of interest (AOIs) that were about 10 times larger than the training images, using MONAI's sliding window technique.

You can see the outcomes of my model in `eval_model.ipynb` or `inference.ipynb`.

## Acknowledgments

As mentioned, this project builds upon the [GFM](https://github.com/mmendiet/GFM) repository developed by [mmendiet](https://github.com/mmendiet). The original GFM repository is licensed under the [Apache License 2.0](https://github.com/mmendiet/GFM/blob/main/LICENSE).

### License

The [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) governs the use, reproduction, and distribution of the GFM repository and its derivatives. A copy of the license is included in the `GFM/LICENSE` file within this project.

### Modifications

- **Integration:** The GFM repository has been integrated as a subfolder within this project to facilitate fine-tuning Swin Transformers for segmentation tasks.

---

Feel free to explore the repository and check out the code! 😊

# StegFormer: Rebuilding the Glory of the Autoencoder-Based Steganography (AAAI-2024)
**Xiao Ke, Huanqi Wu, Wenzhong Guo**

The official pytorch implementation of the paper [StegFormer: Rebuilding the Glory of the Autoencoder-Based Steganography](https://github.com/aoli-gei/StegFormer).

[[Project Page](https://aoli-gei.github.io/StegFormer.github.io/)] [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/28051)] [[Pretrain_model](https://drive.google.com/drive/folders/1L__astCCgm2GlQU-YaxZYTIzEnABdKEc?usp=sharing)]

## Abstract
Image hiding aims to conceal one or more secret images within a cover image of the same resolution. Due to strict capacity requirements, image hiding is commonly called large-capacity steganography. In this paper, we propose StegFormer, a novel autoencoder-based image-hiding model. StegFormer can conceal one or multiple secret images within a cover image of the same resolution while preserving the high visual quality of the stego image. In addition, to mitigate the limitations of current steganographic models in real-world scenarios, we propose a normalizing training strategy and a restrict loss to improve the reliability of the steganographic models under realistic conditions. Furthermore, we propose an efficient steganographic capacity expansion method to increase the capacity of steganography and enhance the efficiency of secret communication. Through this approach, we can increase the relative payload of StegFormer to 96 bits per pixel without any training strategy modifications. Experiments demonstrate that our StegFormer outperforms existing state-of-the-art (SOTA) models. In the case of single-image steganography, there is an improvement of more than 3 dB and 5 dB in PSNR for secret/recovery image pairs and cover/stego image pairs.

## News
- 2024.2.29: update README
- 2024.4.23: update pretrain model

## How to train StegFormer
- Please download the training dataset: [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
- Modify `DIV2K_path`, `path` and `model_name` in the file `config.py` 
- Training the model by `train_StegFormer_single_image.py` or `train_StegFormer_multiple_image`
    > Note that: please modify `num_secret` in `config.py` to define the number of secret images.

## How to test
- Run `test_save_single_image_hiding.py` to test StegFormer using DIV2K valid dataset and save the images in folder `image`
- Run `test_multiple_image_hiding.py` to test StegFormer in multi-image hiding
    > Note that: please modify `num_secret` in `config.py` to define the number of secret images.
- Run `test_StegFormer.py` to calculate PSNR, SSIM, MAE and RMSE

## Contact 
If you have any questions, please contact [wuhuanqi135@gmail.com](wuhuanqi135@gmail.com).

## Citation
If you find this work helps you, please cite:
```bibtex
@inproceedings{ke2024stegformer,
  title={StegFormer: Rebuilding the Glory of Autoencoder-Based Steganography},
  author={Ke, Xiao and Wu, Huanqi and Guo, Wenzhong},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={3},
  pages={2723--2731},
  year={2024}
```

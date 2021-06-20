
# Restore From Restored:<br>Video Restoration With Pseudo Clean Video
[[Paper](https://openaccess.thecvf.com/content/CVPR2021/html/Lee_Restore_From_Restored_Video_Restoration_With_Pseudo_Clean_Video_CVPR_2021_paper.html)]
[[Supplementary](https://drive.google.com/file/d/1kzXqnRBVtzfQuKlvpL1CkL0pLxAlQSYM/view?usp=sharing)]


## Abstract
In this study, we propose a self-supervised video denoising method called "restore-from-restored." This method fine-tunes a pre-trained network by using a pseudo clean video during the test phase. The pseudo clean video is obtained by applying a noisy video to the baseline network. By adopting a fully convolutional neural network (FCN) as the baseline, we can improve video denoising performance without accurate optical flow estimation and registration steps, in contrast to many conventional video restoration methods, due to the translation equivariant property of the FCN. Specifically, the proposed method can take advantage of plentiful similar patches existing across multiple consecutive frames (i.e., patch-recurrence); these patches can boost the performance of the baseline network by a large margin. We analyze the restoration performance of the fine-tuned video denoising networks with the proposed self-supervision-based learning algorithm, and demonstrate that the FCN can utilize recurring patches without requiring accurate registration among adjacent frames. In our experiments, we apply the proposed method to state-of-the-art denoisers and show that our fine-tuned networks achieve a considerable improvement in denoising performance.


## Environments
- Python 3.7
- Pytorch 1.7.1



## Test


### Examples of demo codes

Arguments
```
--video_dir: Path of test video frames
--sigma: Gaussian noise level, one of [15, 25, 40]
--online: Run RFR in online mode
--offline: Run RFR in offline mode
```

1. [DnCNN](https://github.com/SaoYan/DnCNN-PyTorch)
- Only for grayscale video frames
- We provide a single video in this repository (i.e., factory_1080p30 below)
```
python run_rfr.py --gpu 0 --net dncnn --video_dir ./testsets/derf_HD/factory_1080p30 --sigma 25 --online
```

2. [VNLnet](https://github.com/axeldavy/vnlnet) and [FastDVDnet](https://github.com/m-tassano/fastdvdnet)
- Download dataset (RGB) used for experiments in our paper ([Derf](https://drive.google.com/file/d/1dGOYfE9vZDPV4vJyenN0zHSEKUJm3xq8/view?usp=sharing) and [Davis](https://drive.google.com/file/d/16gIpZ5JX1GVaYpQGkt9GwXurvrWXVuFf/view?usp=sharing))

```
python run_rfr.py --gpu 0 --net fastdvdnet --video_dir [PATH_OF_VIDOE_FRAMES] --sigma 25 --online --offline

PYOPENCL_CTX='0' python run_rfr.py --gpu 0 --net vnlnet --video_dir [PATH_OF_VIDOE_FRAMES] --sigma 25 --online --offline
```



## Citation
```
@InProceedings{Lee_2021_CVPR,
    author    = {Lee, Seunghwan and Cho, Donghyeon and Kim, Jiwon and Kim, Tae Hyun},
    title     = {Restore From Restored: Video Restoration With Pseudo Clean Video},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {3537-3546}
}
```

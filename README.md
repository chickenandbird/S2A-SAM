<div align="center">
  <img src="resources/mmrotate-logo.png" width="450"/>
  <div>&nbsp;</div>
  <div align="center">
    <b><font size="5">OpenMMLab website</font></b>
    <sup>
      <a href="https://openmmlab.com">
        <i><font size="4">HOT</font></i>
      </a>
    </sup>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b><font size="5">OpenMMLab platform</font></b>
    <sup>
      <a href="https://platform.openmmlab.com">
        <i><font size="4">TRY IT OUT</font></i>
      </a>
    </sup>
  </div>
  <div>&nbsp;</div>

[![PyPI](https://img.shields.io/pypi/v/mmrotate)](https://pypi.org/project/mmrotate)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://mmrotate.readthedocs.io/en/latest/)
[![badge](https://github.com/open-mmlab/mmrotate/workflows/build/badge.svg)](https://github.com/open-mmlab/mmrotate/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmrotate/branch/main/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmrotate)
[![license](https://img.shields.io/github/license/open-mmlab/mmrotate.svg)](https://github.com/open-mmlab/mmrotate/blob/main/LICENSE)
[![open issues](https://isitmaintained.com/badge/open/open-mmlab/mmrotate.svg)](https://github.com/open-mmlab/mmrotate/issues)
[![issue resolution](https://isitmaintained.com/badge/resolution/open-mmlab/mmrotate.svg)](https://github.com/open-mmlab/mmrotate/issues)

[📘Documentation](https://mmrotate.readthedocs.io/en/latest/) |
[🛠️Installation](https://mmrotate.readthedocs.io/en/latest/install.html) |
[👀Model Zoo](https://mmrotate.readthedocs.io/en/latest/model_zoo.html) |
[🆕Update News](https://mmrotate.readthedocs.io/en/latest/changelog.html) |
[🚀Ongoing Projects](https://github.com/open-mmlab/mmrotate/projects) |
[🤔Reporting Issues](https://github.com/open-mmlab/mmrotate/issues/new/choose)

</div>

<!--中/英 文档切换-->

<div align="center">

English | [简体中文](README_zh-CN.md)

</div>

## Introduction

MMRotate is an open-source toolbox for rotated object detection based on PyTorch.
It is a part of the [OpenMMLab project](https://github.com/open-mmlab).

The master branch works with **PyTorch 1.6+**.

https://user-images.githubusercontent.com/10410257/154433305-416d129b-60c8-44c7-9ebb-5ba106d3e9d5.MP4

<details open>
<summary><b>Major Features</b></summary>

- **Support multiple angle representations**

  MMRotate provides three mainstream angle representations to meet different paper settings.

- **Modular Design**

  We decompose the rotated object detection framework into different components,
  which makes it much easy and flexible to build a new model by combining different modules.

- **Strong baseline and State of the art**

  The toolbox provides strong baselines and state-of-the-art methods in rotated object detection.

</details>

## What's New

### Highlight

We are excited to announce our latest work on real-time object recognition tasks, **RTMDet**, a family of fully convolutional single-stage detectors. RTMDet not only achieves the best parameter-accuracy trade-off on object detection from tiny to extra-large model sizes but also obtains new state-of-the-art performance on instance segmentation and rotated object detection tasks. Details can be found in the [technical report](https://arxiv.org/abs/2212.07784). Pre-trained models are [here](https://github.com/open-mmlab/mmrotate/tree/1.x/configs/rotated_rtmdet).

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rtmdet-an-empirical-study-of-designing-real/real-time-instance-segmentation-on-mscoco)](https://paperswithcode.com/sota/real-time-instance-segmentation-on-mscoco?p=rtmdet-an-empirical-study-of-designing-real)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rtmdet-an-empirical-study-of-designing-real/object-detection-in-aerial-images-on-dota-1)](https://paperswithcode.com/sota/object-detection-in-aerial-images-on-dota-1?p=rtmdet-an-empirical-study-of-designing-real)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rtmdet-an-empirical-study-of-designing-real/object-detection-in-aerial-images-on-hrsc2016)](https://paperswithcode.com/sota/object-detection-in-aerial-images-on-hrsc2016?p=rtmdet-an-empirical-study-of-designing-real)

| Task                     | Dataset | AP                                   | FPS(TRT FP16 BS1 3090) |
| ------------------------ | ------- | ------------------------------------ | ---------------------- |
| Object Detection         | COCO    | 52.8                                 | 322                    |
| Instance Segmentation    | COCO    | 44.6                                 | 188                    |
| Rotated Object Detection | DOTA    | 78.9(single-scale)/81.3(multi-scale) | 121                    |

<div align=center>
<img src="https://user-images.githubusercontent.com/12907710/208044554-1e8de6b5-48d8-44e4-a7b5-75076c7ebb71.png"/>
</div>

**0.3.4** was released in 01/02/2023:

- Fix compatibility with numpy, scikit-learn, and e2cnn.
- Support empty patch in Rotate Transform
- use iof for RRandomCrop validation

Please refer to [changelog.md](docs/en/changelog.md) for details and release history.

## Installation

MMRotate depends on [PyTorch](https://pytorch.org/), [MMCV](https://github.com/open-mmlab/mmcv) and [MMDetection](https://github.com/open-mmlab/mmdetection).
Below are quick steps for installation.
Please refer to [Install Guide](https://mmrotate.readthedocs.io/en/latest/install.html) for more detailed instruction.

```shell
conda create -n open-mmlab python=3.7 pytorch==1.7.0 cudatoolkit=10.1 torchvision -c pytorch -y
conda activate open-mmlab
pip install openmim
mim install mmcv-full
mim install mmdet
git clone https://github.com/open-mmlab/mmrotate.git
cd mmrotate
pip install -r requirements/build.txt
pip install -v -e .
```

## Get Started

Please see [get_started.md](docs/en/get_started.md) for the basic usage of MMRotate.
We provide [colab tutorial](demo/MMRotate_Tutorial.ipynb), and other tutorials for:

- [learn the basics](docs/en/intro.md)
- [learn the config](docs/en/tutorials/customize_config.md)
- [customize dataset](docs/en/tutorials/customize_dataset.md)
- [customize model](docs/en/tutorials/customize_models.md)
- [useful tools](docs/en/tutorials/useful_tools.md)

## Model Zoo

Results and models are available in the *README.md* of each method's config directory.
A summary can be found in the [Model Zoo](docs/en/model_zoo.md) page.

<details open>
<summary><b>Supported algorithms:</b></summary>

- [x] [Rotated RetinaNet-OBB/HBB](configs/rotated_retinanet/README.md) (ICCV'2017)
- [x] [Rotated FasterRCNN-OBB](configs/rotated_faster_rcnn/README.md) (TPAMI'2017)
- [x] [Rotated RepPoints-OBB](configs/rotated_reppoints/README.md) (ICCV'2019)
- [x] [Rotated FCOS](configs/rotated_fcos/README.md) (ICCV'2019)
- [x] [RoI Transformer](configs/roi_trans/README.md) (CVPR'2019)
- [x] [Gliding Vertex](configs/gliding_vertex/README.md) (TPAMI'2020)
- [x] [Rotated ATSS-OBB](configs/rotated_atss/README.md) (CVPR'2020)
- [x] [CSL](configs/csl/README.md) (ECCV'2020)
- [x] [R<sup>3</sup>Det](configs/r3det/README.md) (AAAI'2021)
- [x] [S<sup>2</sup>A-Net](configs/s2anet/README.md) (TGRS'2021)
- [x] [ReDet](configs/redet/README.md) (CVPR'2021)
- [x] [Beyond Bounding-Box](configs/cfa/README.md) (CVPR'2021)
- [x] [Oriented R-CNN](configs/oriented_rcnn/README.md) (ICCV'2021)
- [x] [GWD](configs/gwd/README.md) (ICML'2021)
- [x] [KLD](configs/kld/README.md) (NeurIPS'2021)
- [x] [SASM](configs/sasm_reppoints/README.md) (AAAI'2022)
- [x] [Oriented RepPoints](configs/oriented_reppoints/README.md) (CVPR'2022)
- [x] [KFIoU](configs/kfiou/README.md) (arXiv)
- [x] [G-Rep](configs/g_reppoints/README.md) (stay tuned)

</details>

## Data Preparation

Please refer to [data_preparation.md](tools/data/README.md) to prepare the data.

## FAQ

Please refer to [FAQ](docs/en/faq.md) for frequently asked questions.

## Contributing

We appreciate all contributions to improve MMRotate. Please refer to [CONTRIBUTING.md](.github/CONTRIBUTING.md) for the contributing guideline.

## Acknowledgement

MMRotate is an open source project that is contributed by researchers and engineers from various colleges and companies. We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks. We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new methods.

## Citation

If you use this toolbox or benchmark in your research, please cite this project.

```bibtex
@inproceedings{zhou2022mmrotate,
  title   = {MMRotate: A Rotated Object Detection Benchmark using PyTorch},
  author  = {Zhou, Yue and Yang, Xue and Zhang, Gefan and Wang, Jiabao and Liu, Yanyi and
             Hou, Liping and Jiang, Xue and Liu, Xingzhao and Yan, Junchi and Lyu, Chengqi and
             Zhang, Wenwei and Chen, Kai},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  year={2022}
}
```

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Projects in OpenMMLab

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MIM](https://github.com/open-mmlab/mim): MIM installs OpenMMLab packages.
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab image classification toolbox and benchmark.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab's next-generation platform for general 3D object detection.
- [MMRotate](https://github.com/open-mmlab/mmrotate): OpenMMLab rotated object detection toolbox and benchmark.
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab semantic segmentation toolbox and benchmark.
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab text detection, recognition, and understanding toolbox.
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab pose estimation toolbox and benchmark.
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d): OpenMMLab 3D human parametric model toolbox and benchmark.
- [MMSelfSup](https://github.com/open-mmlab/mmselfsup): OpenMMLab self-supervised learning toolbox and benchmark.
- [MMRazor](https://github.com/open-mmlab/mmrazor): OpenMMLab model compression toolbox and benchmark.
- [MMFewShot](https://github.com/open-mmlab/mmfewshot): OpenMMLab fewshot learning toolbox and benchmark.
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab's next-generation action understanding toolbox and benchmark.
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab video perception toolbox and benchmark.
- [MMFlow](https://github.com/open-mmlab/mmflow): OpenMMLab optical flow toolbox and benchmark.
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab image and video editing toolbox.
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMLab image and video generative models toolbox.
- [MMDeploy](https://github.com/open-mmlab/mmdeploy): OpenMMLab model deployment framework.

# S2A-SAM
建立在S2ANet之上的实例分割模型


#函数实际用法，以及进行实例分割的路径
## 工作流程：



- test.py -> 修改submission_dir='result/s2anet_dota_test_final'产生一个class 对应实例的文件，包括一个zip文件。



- 利用产生的结果文件夹调用函数DOTAresult2imgfile ->产生新的txt文件

- zw/imgfile2qiyuansubmit.py  ->用新的txt文件产生json文件，也是用于提交的文件，最新版本是imgfile2qiyuansubmit_3.py

- vis_fromjson->从json文件可视化



Mycode:

COCO_2_DOTA：将给的json train改为DOTA模式，值得注意的是，我好像漏掉了一部分数据。

dataset_seg：将训练集分割

delete：删除元数据

hbox_2_rbox：把bbox标注转为hbox标注



merge_image：

my_train:继承已经训练的模型的参数,这个函数经过实际的实验证明是没有意义的，机械地继承已训练数据集意义不大。

new_val:把val当中的200张复制到val2，用来测试的

vis:旋转框标注的可视化。



zw:

DOTAresult2imgfile:用于将test 产生的json文件转换成与图像同名txt文件      

imgfile2qiyuansubmit：用于把dota的txt文件输入sam产生启元要求的json文件。

imgfile2samvisual：zw可视化

vis_fromjson：从json文件生成可视化图片，mask标注

vis_fromjson2：连点标注可视化

img_expand：对特定种类的数据进行扩充

result2DOTAtxt：把json结果可视化

sam_test2：将所有test的txt标注输入sam可视化



## 实验数据记录:都使用大的sam模型。

1_result.json:使用水平框中心点为正点；边缘点为负点

2_result.json:在不接近边缘时用水平框为正点;在不接近边缘时旋转框边缘点为负点。

3_result.json：使用canny边缘检测筛选旋转框内部较强边缘点，边缘点的质心作为正点；

4_result.json:置信度筛选用的是0.3，使用canny边缘检测筛选旋转框内部较强边缘点，边缘点的质心作为正点；矩形端点作为负点，生成最终提交的结果。当前条件还是只使用最大连通分量的方法。优化点是canny边缘检测的质心作为正点；roundabout类中心点为负点。canny会被可视化。predictor.reset_image() 。双连通没有意义。

-->   json_img_5

5_result.json-->json_img_6

**6_result_15.json：使用DOTAv1.5作为补充数据集；储蓄罐用旋转框四条边的均值作为边长（这个似乎不靠谱，因为只有部分的目标会被排除在外）；roundabout中心取的是正点，这个比4要对，环岛指的是环岛中心的花坛。**

7_result.json:-->json_img_6

json_img_3:canny边缘检测提供中心点，但是负点还是水平框端点

json_img_4:canny边缘检测提供中心点，负点是水平框端点，但是使用了predictor.reset_image() 



经过实践，6_result_15效果比较好。canny边缘检测提供中心点，但是负点还是水平框端点。不对storage_tank做特殊处理。roundabout由于训练集数据存在问题，本来用投票的方法对外部边缘处理为负点，但实际情况是，由于环岛中心预测很难准确，这种方法是做不到的。



## 进过实践，一个比较好的学习率config：

lr_config = dict(

  policy='step',

  warmup='linear',

  warmup_iters=500,

  warmup_ratio=0.3333333333333333,

  step=[12,20])

runner = dict(type='EpochBasedRunner', max_epochs=28)

evaluation = dict(interval=2, metric='mAP')

optimizer = dict(type='AdamW', lr=1e-04, betas=(0.9, 0.999), weight_decay=0.05)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

optim_wrapper = dict(

optimizer = dict(type='AdamW', lr=1e-04, betas=(0.9, 0.999), weight_decay=0.05),

samples_per_gpu=4,workers_per_gpu=4,









## 可能的优化方法举例：

第一个思路：

所有图片everything，然后用框来判定那些是合适的目标。不过很不靠谱，因为SAM的性能限制。

第二个思路：

训两个目标检测模型，一个外界框；一个内接框，方法是取框和实例分割的loss，可以允许多个框的存在。

没有想到，两种实例的标注可能是有差的，比如DOTA数据集的环岛是包含旁边公路的，而启元实验室给的并不包含。



## 总结:

没有进行微调的SAM非常不靠谱，不能作为实例分割的核心，最好作为辅助，比如数据增强？



File "/root/autodl-tmp/S2A-SAM/tools/test.py", line 264, in main
    dataset.format_results(outputs,submission_dir='result/new_DOTA_24epoch', **kwargs)
  File "/root/autodl-tmp/S2A-SAM/mmrotate/datasets/dota.py", line 364, in format_results
    result_files = self._results2submission(id_list, dets_list,
  File "/root/autodl-tmp/S2A-SAM/mmrotate/datasets/dota.py", line 299, in _results2submission
    raise ValueError(f'The out_folder should be a non-exist path, '
ValueError: The out_folder should be a non-exist path, but result/new_DOTA_24epoch is existing

艹，这个B函数还有这种要求










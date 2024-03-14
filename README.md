# PGCL
Negative Prototypes Guided Contrastive Learning for Weakly Supervised Object Detection

## Environment setup:

* [Python 3.7](https://pytorch.org)
* [CUDA 11.0](https://developer.nvidia.com/cuda-toolkit)
* [PyTorch 1.7.1](https://pytorch.org)

## Dataset:
* [PASCAL VOC (2007, 2012)](http://host.robots.ox.ac.uk/pascal/VOC/)
* [MS-COCO (2014, 2017)](https://cocodataset.org/#download)  

```bash
mkdir -p datasets/{coco/voc}
    datasets/
    ├── voc/
    │   ├── VOC2007
    │   │   ├── Annotations/
    │   │   ├── JPEGImages/
    │   │   ├── ...
    │   ├── VOC2012/
    │   │   ├── ...
    ├── coco/
    │   ├── annotations/
    │   ├── train2014/
    │   ├── val2014/
    │   ├── train2017/
    │   ├── ...
    ├── ...
```
## Proposal:
Download .pkl file
```bash
mkdir proposal
    proposal/
    ├── SS/
    │   ├── voc
    │   │   ├── SS-voc07_trainval.pkl/
    │   │   ├── SS-voc07_test.pkl/
    │   │   ├── ...
    ├── MCG/
    │   ├── voc
    │   │   ├── ...
    │   ├── coco
    │   │   ├── MCG-coco_2014_train_boxes.pkl/
    │   │   ├── ...
    ├── ...
```
## Run:
Please run the follow code to test the model.
```bash
python -m torch.distributed.launch --nproc_per_node={NO_GPU} tools/{file}.py  
                                   --config-file "configs/{config_file}.yaml"
                                   OUTPUT_DIR {output_dir}
                                   nms {nms threshold}
                                   lmda {lambda value}
                                   iou {iou threshold}
                                   temp {temperature}
```
Some parts of the code are borrowed from <a href="https://github.com/NVlabs/wetectron">wetectron</a>

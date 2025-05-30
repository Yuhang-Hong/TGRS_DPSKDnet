# Cross-Scene Hyperspectral Image Classification Network with Dynamic Perturbation and Self-Knowledge Distillation

The Pytorch Implementation of “Cross-Scene Hyperspectral Image Classification Network with Dynamic Perturbation and Self-Knowledge Distillation”.

## Requirements

1. We build the project with python=3.8.

   ```shell
   conda create -n DPSKDnet python=3.8
   ```

2. Clone the repo:

   ```shell
   git clone https://github.com/Yuhang-Hong/TGRS_DPSKDnet.git
   ```

3. Activate the environment:

   ```shell
   conda activate DPSKDnet
   ```

4. Install the requirements:

   ```shell
   cd TGRS_DPSKDnet
   pip install -r requirements.txt
   ```

## Dataset

```python
dataset
├── Houston
│   ├── Houston13.mat
│   ├── Houston13_7gt.mat
│   ├── Houston18.mat
│   ├── Houston18_7gt.mat
├── HyRANK
│   ├── Dioni.mat
│   ├── Dioni_gt.mat
│   ├── Dioni_gt_out68.mat
│   ├── Loukia.mat
│   ├── Loukia_gt.mat
│   ├── Loukia_gt_out68.mat
└── Pavia
    ├── paviaC.mat
    ├── paviaC_7gt.mat
    ├── paviaU.mat
    └── paviaU_7gt.mat
```

## Usage

1. You can download [Houston ＆HyRANK ＆Pavia](https://github.com/YuxiangZhang-BIT/Data-CSHSI) dataset here. 

2. You can change the --data_path , the source_name and the target_name in "run.sh" to run different datasets.

3. For Houston dataset:

   ```shell
   python train.py --data_path dataset/Houston/ --source_name Houston13 --target_name Houston18 --re_ratio 5 --training_sample_ratio 0.8 --d_se 64  --batch_size 256 --lambda_1 1.0 --lambda_2 0.1 --seed 233
   ```

4. For HyRANK dataset:

   ```shell
   python train.py --data_path dataset/HyRANK/ --source_name Dioni --target_name Loukia --re_ratio 1 --training_sample_ratio 0.5 --d_se 64  --batch_size 256 --lambda_1 1.0 --lambda_2 0.1 --seed 233
   ```

5. For Pavia dataset:

   ```shell
   python train.py --data_path dataset/Pavia/ --source_name paviaU --target_name paviaC --re_ratio 1 --training_sample_ratio 0.5 --d_se 64  --batch_size 256 --lambda_1 1.0 --lambda_2 0.1 --seed 233
   ```

## Acknowledgement

- We thank [YuxiangZhang-BIT](https://github.com/YuxiangZhang-BIT) for elegant and efficient code base.

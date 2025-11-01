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
## Result
CLASSIFICATION ACCURACY (%) OF DIFFERENT METHODS ON THE LOUKIA DATASET.

|class|SDEnet|LLURnet|LDGnet|FDGnet|DTAM|EHSnet|S2ECnet|ACB|DSU|FACT|SagNets|XDED|DPSKDnet|
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
|1|24.66|30.97|42.43|29.81|31.07|13.01|**48.84**|18.05|21.65|32.56|20.58|20.39|31.07|
|2|12.22|2.96|5.55|20.37|30.00|16.67|**62.59**|24.07|3.70|18.61|15.19|10.74|7.04|
|3|46.62|51.27|45.87|49.20|55.35|39.06|54.79|51.92|52.63|31.24|**71.45**|50.66|43.57|
|4|14.18|7.60|22.79|11.90|16.46|21.26|27.34|21.52|21.52|0.00|**29.37**|5.82|17.22|
|5|1.21|0.27|0.00|0.38|1.37|7.30|0.14|1.66|0.11|**32.63**|0.02|1.37|0.96|
|6|32.37|33.27|**42.32**|32.94|39.72|32.75|28.24|23.60|27.77|3.92|14.98|16.68|32.32|
|7|76.37|73.95|77.39|76.19|75.35|78.50|76.44|79.91|79.45|74.92|76.58|**83.64**|80.25|
|8|**71.91**|68.70|48.14|70.28|58.24|54.83|61.51|62.56|65.66|64.13|66.39|54.06|71.50|
|9|47.42|26.01|**73.83**|56.64|50.53|59.55|70.38|22.56|23.16|45.01|20.60|19.50|33.88|
|10|0.00|**27.82**|0.80|0.04|3.36|20.53|0.00|0.00|0.31|0.44|0.00|10.60|0.00|
|11|**100.0**|99.99|**100.0**|**100.0**|**100.0**|**100.0**|**100.0**|**100.0**|**100.0**|**100.0**|**100.0**|**100.0**|**100.0**|
|12|**100.0**|**100.0**|**100.0**|**100.0**|98.34|**100.0**|**100.0**|**100.0**|99.95|**100.0**|**100.0**|**100.0**|**100.0**|
|OA|62.09±0.82|61.20±1.35|58.64±1.44|62.20±0.61|59.79±0.52|60.38±1.46|61.52±0.62|59.84±1.36|60.53±1.36|61.48±0.67|60.09±0.84|58.89+-1.85|**62.57±0.17**|
|KC|52.72±0.86|51.89±1.48|49.38±1.52|53.01±0.70|50.55±0.71|50.96±1.91|52.82±0.92|49.46±1.66|50.65±1.45|52.17±1.27|50.21±1.04|48.38+-2.12|**53.07±0.32**|




CLASSIFICATION ACCURACY (%) OF DIFFERENT METHODS ON THE HOUSTON2018 DATASET.

|class|SDEnet|LLURnet|LDGnet|FDGnet|DTAM  |EHSnet|S2ECnet|ACB|DSU|FACT  |SagNets|XDED|DPSKDnet|
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
|1|42.45|43.40|50.66|51.19|24.05|48.44|**77.00**|40.10|30.86|21.02|20.31|32.64|35.79|
|2|82.19|80.34|80.43|82.23|**90.65**|78.97|63.37|79.80|49.32|65.38|37.70|70.25|84.10|
|3|**64.64**|59.14|54.76|63.00|50.87|60.01|45.70|58.17|38.46|43.40|38.19|53.59|53.77|
|4|**100.0**|**100.0**|89.09|**100.0**|94.54|85.45|**100.0**|88.18|92.73|78.82|80.91|92.73|**100.0**|
|5|70.46|65.19|69.08|67.65|**84.92**|66.64|74.58|71.52|76.36|66.05|64.26|64.24|54.35|
|6|88.79|56.84|90.79|83.27|85.89|85.24|82.49|89.64|92.02|94.65|**97.09**|92.76|90.19|
|7|41.83|62.56|31.93|53.93|51.16|55.69|**66.35**|42.73|33.28|12.04|1.26|23.56|59.66|
|OA|78.29±0.75|76.34±1.19|77.72±1.65|76.23±1.88|78.68±0.74|77.02±2.17|75.96±1.22|78.44±1.73|75.15±2.40|74.66±1.09|71.84±2.13|75.98±1.19|**79.10±0.57**|
|KC|62.65±1.45|60.77±2.07|60.72±3.67|61.13±1.77|**64.54±0.93**|61.96±2.51|60.81±1.87|62.79±2.58|54.68±4.26|51.34±2.18|41.05±6.34|55.44±3.32|63.96±0.77|




CLASSIFICATION ACCURACY (%) OF DIFFERENT METHODS ON THE PAVIA CENTER DATASET.

|class|SDEnet|LLURnet|LDGnet|FDGnet|DTAM|EHSnet|S2ECnet|ACB|DSU|FACT  |SagNets|XDED|DPSKDnet|
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
|1|85.67|85.93|**94.88**|84.63|93.43|92.70|91.47|94.16|71.74|89.37|93.74|92.51|90.42|
|2|83.79|81.36|82.48|80.61|82.26|83.75|**85.56**|80.50|78.72|84.40|80.63|84.70|84.45|
|3|69.22|67.35|48.07|68.62|43.61|51.36|45.59|61.75|17.12|**73.88**|35.91|29.81|61.81|
|4|77.93|81.34|72.95|83.38|79.87|74.01|75.42|69.20|55.56|16.79|21.74|67.07|**85.78**|
|5|86.03|90.83|88.13|89.00|93.05|85.99|**95.44**|93.45|79.50|85.43|88.32|91.18|85.30|
|6|78.48|75.60|66.16|74.62|69.40|72.89|**78.53**|69.32|62.13|76.21|67.71|71.91|74.28|
|7|76.28|78.70|72.27|81.68|77.03|80.60|72.01|79.84|**86.84**|78.40|74.75|77.89|78.42|
|OA|80.56±1.33|81.06±1.16|78.18±1.60|81.40±0.64|80.24±1.57|80.21±2.58|79.98±1.01|79.72±1.88|68.99±3.66|70.55±1.38|67.77±2.33|77.52±1.57|**82.56±0.43**|
|KC|76.67±1.61|77.26±1.35|73.74±1.89|77.70±0.73|76.20±1.86|76.15±3.13|75.90±1.55|75.66±2.25|62.58±4.42|64.99±1.49|61.17±2.82|72.82±1.99|**78.97±0.51**|

## Usage

1. You can download [Houston ＆HyRANK ＆Pavia](https://github.com/YuxiangZhang-BIT/Data-CSHSI) dataset here. 

2. You can change the --data_path , the source_name and the target_name in "run.sh" to run different datasets.

3. For three datasets :

   ```shell
   bash run.sh 
   ```

4. For a single dataset, you can run it from the command line.

   For Houston dataset:

   ```shell
   python train.py --data_path dataset/Houston/ --source_name Houston13 --target_name Houston18 --re_ratio 5 --training_sample_ratio 0.8 --d_se 64  --batch_size 256 --lambda_1 1.0 --lambda_2 0.1 --seed 233
   ```

   For HyRANK dataset:

   ```shell
   python train.py --data_path dataset/HyRANK/ --source_name Dioni --target_name Loukia --re_ratio 1 --training_sample_ratio 0.5 --d_se 64  --batch_size 256 --lambda_1 1.0 --lambda_2 0.1 --seed 233
   ```

   For Pavia dataset:

   ```shell
   python train.py --data_path dataset/Pavia/ --source_name paviaU --target_name paviaC --re_ratio 1 --training_sample_ratio 0.5 --d_se 64  --batch_size 256 --lambda_1 1.0 --lambda_2 0.1 --seed 233
   ```

## Acknowledgement

- We thank [YuxiangZhang-BIT](https://github.com/YuxiangZhang-BIT) for elegant and efficient code base.

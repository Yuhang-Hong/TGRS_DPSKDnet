python train.py --data_path dataset/HyRANK/ --source_name Dioni --target_name Loukia --re_ratio 1 --training_sample_ratio 0.5 --d_se 64  --batch_size 256 --lambda_1 1.0 --lambda_2 0.1 --seed 233;
python train.py --data_path dataset/Houston/ --source_name Houston13 --target_name Houston18 --re_ratio 5 --training_sample_ratio 0.8 --d_se 64  --batch_size 256 --lambda_1 1.0 --lambda_2 0.1 --seed 233;
python train.py --data_path dataset/Pavia/ --source_name paviaU --target_name paviaC --re_ratio 1 --training_sample_ratio 0.5 --d_se 64  --batch_size 256 --lambda_1 1.0 --lambda_2 0.1 --seed 233;

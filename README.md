# This is the source codes of Hashing for Localization (HfL): A Baseline for Efficiently Searching A Large-Scale Scene for Specific Objects.
## Environment
NVIDIA GPU + CUDA and corresponidng Pytorch framework (v0.4.1)<br>
Python 3.6
## Dataset
The dataset is aviliable though the following link: [SpaceNet_data](https://pan.baidu.com/s/1BIhuKppEJLQ6g3-Z4SOz6w). The password is `3ipr`. The dataset contains three subset: database set, train set and test set. Additionally，we release the 24 large-scale images as follows: [SpaceNet-24](https://pan.baidu.com/s/1RxaBdAA8K6tuAbuFUxQrLw). The password is ly92.
## Train
```
python train.py --data_name SpaceNet --hash_bit 64 --gpus 0,1 --model_type resnet18 --lambda1 0  --lambda2 0.05  --R 50
```
## Test for hash retrieval performence
You should set test.py
```
database_list = 'data/' + args.data_name + '/train.txt'
```
and then 
```
python test.py --data_name SpaceNet --gpus 0,1  --R 50  --model_name 'name' 
```
## Test for localization performence
You should set test.py
```
database_list = 'data/' + args.data_name + '/database.txt'
```
and then
```
python test.py --data_name SpaceNet --gpus 0,1  --R 50  --model_name 'name' 
```

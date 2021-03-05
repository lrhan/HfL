# This is the source codes of Hashing for Localization (HfL): A Baseline for Efficiently Searching A Large-Scale Scene for Specific Objects.
## Environment
NVIDIA GPU + CUDA and corresponidng Pytorch framework (v0.4.1)<br>
Python 3.6
## Dataset
The dataset is aviliable though the following link: [https://pan.baidu.com/s/1BIhuKppEJLQ6g3-Z4SOz6w] (SpaceNet_data). The password is '3ipr'. The dataset contains three subset: database set, train set and test set. 
## Train
```
python train.py --data_name SpaceNet --hash_bit 64 --gpus 0,1 --model_type resnet18 --lambda1 0  --lambda2 0.05  --R 50
```

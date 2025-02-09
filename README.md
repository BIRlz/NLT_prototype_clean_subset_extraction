# Prototype-oriented Clean Subset Extraction for Noisy Long-tailed Classification

## 1. Prepare for the environment
```
pip install -r requirements.txt
```

## 2. Prepare for the data and pre-trained weights
Please download CIFAR-10 and CIFAR-100, and put them into the folder *data/*.

For pre-trained weights, we put it in [Google Driver](https://drive.google.com/drive/folders/1bQ-OcgNlCzeqp4m2DjoC4zSKVLVYFF9y?usp=sharing).

## 3. How to run on CIFAR-10/100.

For CIFAR-10/100 with different imbalance factor and noise ratio:
```
bash train.sh
```
More detailed settings and how to run ablation studies can also be found in train.sh

## 4. How to run on WebVision-50.
```
python wenbvsion_main.py
```

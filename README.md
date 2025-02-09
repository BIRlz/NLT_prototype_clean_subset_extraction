# Prototype-oriented Clean Subset Extraction for Noisy Long-tailed Classification

## 1. Prepare for the environment
```
pip install -r requirements.txt
```

## 2. Prepare for the data
Please download CIFAR-10 and CIFAR-100, and put them into the folder *data/*.

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

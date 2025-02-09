# Prototype-oriented Clean Subset Extraction for Noisy Long-tailed Classification

## 1. Prepare for the environment
```
pip install -r requirements.txt
```

## 2. Prepare for the data
Please download CIFAR-10 and CIFAR-100, and put them into the folder *data/*.

## 3. How to run

For CIFAR-10 with imbalance factor=100 and different noise ratios based on Warming-Up,
```
bash train.sh
```

# ETANA / F-ETANA
## Introduction

ETANA is an dynamic instance–wise joint feature selection and classification algorithm. F–ETANA is a fast implementation of ETANA

## Prerequisites

python 2.7 or higher, and the following libraries

```
numpy
sklearn
scipy
```

## Files

```
Joint_FS_and_C.py: All the necessary functions for ETANA and F-ETANA
datasets: Sample MLL dataset spitted into training and validation sets
```

## How to use

```
Step 1. Load the dataset:
    Xtrain: Train data 
    Ytrain: Train labels
    Xtest:  Test data
    Ytest: Test labels

Step 2. Define configuration parameters:
    feat_cost: feature evaluation cost
    bins: number of bins considered when quantizing the feature space
    neta: parameter used to quantize the probability simplex
    SPSA_params: parameter required for SPSA stochastic gradient algorithm 
    exp_feat_para: parameter required to compute expected number of features for classification

Step 3. Initiate an instance of ETANA or F-ETANA using "config" file, and call run function

Step 4. Print classification summary report

Optional -- To compute expected number of features for classification

Step 5. Initiate an instance of Exp_Features using "config" file, and call run function

Step 6. Print expected number of features to achieve defined "alpha" error probability using defined "distribution"
```


## Example

```
See example.py
```
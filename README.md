# ETANA

We present a Dynamic Instance-wise Feature Selection and Classification algorithm. To reduce the computational complexity of ETANA, while maintaining its classification performance, we further present a fast implementation of ETANA, named F-ETANA.

## Prerequisites

python 2.7 or above and the following libraries

```
numpy
sklearn
scipy
```

## Files

```
Joint_FS_and_C.py: Include all the necessary functions of ETANA and F-ETANA
datasets: include a sample dataset (i.e., MLL dataset splitted into training and validation sets) 
```

## How to use

```
Step 1. Load dataset:
    Xtrain: Train data 
    Ytrain: Train labels
    Xtest:  Test data
    Ytest: Test labels

Step 2. Define configuration parameters:
    feat_cost: feature evaluation cost
    bins: number of bins conisdered when quantizing the feature space
    neta: parameter used to quantize the probability simplex
    SPSA_params: parameter required for SPSA stochastic gradient algorithm 
    exp_feat_para: parameter required to compute expected number of features for classification

Step 3. Initiate an instance of ETANA (or F_ETANA) using config file and call run function

Step 4. Print classification summary report

Optional -- To compute expected number of features for classification
Step 5. Initiate an instance of Exp_Features using config file and call run function

Step 6. Print expected number of features to achieve defined "alpha" error probability using defined "distribution"
```


## Example

```
See example.py
```

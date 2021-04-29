from Joint_FS_and_C import ETANA, F_ETANA, Exp_Features
import numpy as np
from scipy import io

'''
Step 1. Load dataset:
    Xtrain: Train data 
    Ytrain: Train labels
    Xtest: test data
    Ytest: test labels
'''
mat = io.loadmat('Datasets/Gene/MLL_data_label_all.mat')
   
Xtrain = np.array(mat['train_data'],dtype = None)
Ytrain = mat['train_label'].astype(int)[:,0]
Xtest = np.array(mat['test_data'],dtype =None)
Ytest = mat['test_label'].astype(int)[:,0]

dataset = {'Xtrain': Xtrain, 'Ytrain': Ytrain, 'Xtest':Xtest, 'Ytest':Ytest}


'''
Step 2. Define configuration parameters:
    feat_cost: feature evaluation cost
    bins: number of bins considered when quantizing the feature space
    neta: parameter used to quantize the probability simplex
    SPSA_params: parameter required for SPSA stochastic gradient algorithm 
    exp_feat_para: parameter required to compute expected number of features for classification
'''
config = {'feat_cost': 0.01, 'bins':3, 'neta': 10,
         'SPSA_params': {'mu': 2, 'epsilon':0.1667, 'zeta': 0.5, 
                         'kappa': 0.602, 'nu': 0.2, 't_max': 100000,'rho':1e-5},
         'exp_feat_para':{'alpha':0, 'distribution': 'best'}}

'''
Step 3. Initiate an instance of ETANA (or F_ETANA) using "config" file and call run function
'''
clf = ETANA(config)
clf.run(dataset)

'''
Step 4. Print classification summary report
'''
print("Classification Report: "+str(clf.summary)+'\n')

'''
Optional -- To compute expected number of features for classification
Step 5. Initiate an instance of Exp_Features using "config" file and call run function
'''
obj = Exp_Features(config)
obj.run(dataset)

'''
Step 6. Print expected number of features to achieve "alpha" error probability using defined "distribution"
'''
print("Expected Number of Features: "+str(obj.summary))
import numpy as np
from sklearn.metrics import accuracy_score
from scipy.linalg import toeplitz
import time, math, itertools
from iteround import saferound

class ETANA(object):
    """
    ETANA Class Object
    Initializing Attributes:
        feat_cost: feature evaluation cost
        bins: number of bins considered when quantizing the feature space
        neta: parameter used to quantize the probability simplex
    Functions: 
        run: main function of ETANA object which perform model training and testing. input must be a dataset
        fit: train the model
        predict: perform instance-wise dynamic joint feature selection and classification on test dataset
        preprocess: preprocessing step; mainly generates feature distributions and class prior distribution
    """   
    def __init__(self, config):
        self.feat_cost = config['feat_cost']
        self.bins      = config['bins']
        self.neta      = config['neta']
        
    def run(self, data):        
        # preprocessing
        self.preprocess(data['Xtrain'], data['Ytrain'])             
        # training
        train_time = self.fit()       
        # joint feature selection and classification
        fs_and_c_time = self.predict(data['Xtest'])        
        # classification report
        self.summary = {'accuracy': accuracy_score(data['Ytest'], self.predictions), 'avg_feat': np.mean(self.n_feat),
                       'std_feat': np.std(self.n_feat),'training_time': train_time, 'testing_time': fs_and_c_time}
        
    def fit(self):  # ETANA: Training      
        self.A =  [dict() for _ in range(self.K)]
        J = self.g_w.copy()  
        start = time.time() # start time
        for i in range(self.K-1,-1,-1):            
            f_order = self.ordering[i]
            J_next = J.copy()
            
            for key in self.W:
                pi = np.array(key)/self.neta                                
                sigma = self.feat_cost
                for j in range(self.bins): 
                    
                    delta = self.feat_prob[f_order,:,j]
                    a = np.dot(pi, delta)
                    next_pi = np.multiply(pi,delta)/a           
                               
                    next_pi_key = np.array(saferound(next_pi,1))*self.neta 
                    next_pi_key = tuple(next_pi_key.astype(int))
                    sigma += a*J_next[next_pi_key]
                self.A[i][key] = sigma
                J[key] = min(self.g_w[key], sigma)

        train_time = time.time() - start # training time
        return train_time

    
    def predict(self, Xtest): # ETANA: Joint Feature Selection and Classification      
        self.predictions = []  
        self.n_feat = []          
        start = time.time() # start time
        for z in range(np.size(Xtest,axis=0)):    
            obs = Xtest[z,:]  # test instance
            pi = np.ones(self.L)/self.L # initial belief
            for k in range(self.K):
                
                pi_key = np.array(saferound(pi,1))*self.neta  
                pi_key = tuple(pi_key.astype(int))           
                if self.g_w[pi_key] <= self.A[k][pi_key]:
                    break             
                f_order = self.ordering[k]  # features  
                f = obs[f_order]  # observing a feature assignment
                f_index = find_range(f,self.edges[f_order][1:])   # index after discretization the feature  
                
                delta = self.feat_prob[f_order,:,f_index]
                pi = np.multiply(pi,delta)/np.dot(pi, delta)           

            D_opt = np.argmin(np.dot(self.MC,pi))            
            self.n_feat.append(k) # number of features used    
            self.predictions.append(self.C[D_opt])  # predictions

        fs_and_c_time =  time.time()- start  # joint feature selection and classification time  
        return fs_and_c_time
    
    def preprocess(self, Xtrain, Ytrain):         
        self.C = list(set(Ytrain))       # classes
        self.L = len(self.C)             # number of classes
        self.K = np.size(Xtrain,axis=1)  # number of features
        self.W = quantize_simplex(self.L, self.neta) # uniformly quantizing the probability simplex
        self.MC = cost_matrix(self.L) # misclassification cost matrix
        self.get_stop_cost() # optimum cost of stopping on each quantized point of the probability simplex
        self.prior = compute_prior(Ytrain, self.L, self.C) # priori probabilities   
        # feature distributions and corresponding bin edges
        self.feat_prob, self.edges = compute_feat_prob(Xtrain, Ytrain, self.C, self.K, self.L, self.bins) 
        self.ordering = get_feat_ord(self.feat_prob, self.K) # feature ordering
    
    def get_stop_cost(self):
        self.g_w = dict()
        for key in self.W:
            pi = np.array(key)/self.neta
            self.g_w[key] = np.min(np.dot(self.MC,pi))
            
class F_ETANA(object):
    """
    F_ETANA Class Object
    Initializing Attributes:
        feat_cost: feature evaluation cost
        bins: number of bins considered when quantizing the feature space
        SPSA_params: parameter required for stochastic gradient algorithm 
    Functions: 
        run: main function of F_ETANA object which perform model training and testing. input must be a dataset
        fit: train the model
        predict: perform instance-wise dynamic joint feature selection and classification on test dataset
        func_J: calculate cost function value for gradient estimate using SPSA algorithm
        preprocess: preprocessing step; mainly generates feature distributions and class prior distribution 
    """   
    def __init__(self, config):
        self.feat_cost = config['feat_cost']
        self.bins      = config['bins']
        self.SPSA_params = config['SPSA_params']
          
    def run(self, data):
        # preprocessing
        self.preprocess(data['Xtrain'], data['Ytrain'])             
        # training
        train_time = self.fit(data['Xtrain'], data['Ytrain'], self.SPSA_params)        
        # joint feature selection and classification
        fs_and_c_time = self.predict(data['Xtest'])       
        # classification report
        self.summary = {'accuracy': accuracy_score(data['Ytest'], self.predictions), 'avg_feat': np.mean(self.n_feat),
                       'std_feat': np.std(self.n_feat),'training_time': train_time, 'testing_time': fs_and_c_time}
        
    def fit(self, Xtrain, Ytrain, params): # training F-ETANA       
        self.thresh = np.ones((self.L, self.K, self.L))
        start = time.time() # starting time
        for ci in range(self.L):  
            new_train = Xtrain[Ytrain==self.C[ci]]        
            for t in range(params['t_max']):   # SPSA algorithm 
                ind = np.random.randint(0,np.size(new_train,axis=0))
                obs = new_train[ind,:]   #present instance

                rand_dir = np.random.choice([-1,1],(self.K,self.L))
                a_n = params['epsilon']/((t+1+params['zeta'])**params['kappa'])
                b_n = params['mu']/((t+1)**params['nu'])

                x_plus = self.thresh[ci] + b_n*rand_dir
                J_plus  = self.func_J(x_plus, obs, ci)
                x_minus = self.thresh[ci] - b_n*rand_dir
                J_minus = self.func_J(x_minus, obs, ci)

                grad = (J_plus-J_minus)*rand_dir/(2*b_n)
                self.thresh[ci] -= a_n*grad
                if np.linalg.norm(grad) < params['rho']:
                        break

        train_time = time.time() - start # training time
        return train_time

    def predict(self, Xtest):      
        self.predictions = [] 
        self.n_feat      = []  
        start = time.time()
        for z in range(np.size(Xtest,axis=0)):    
            obs = Xtest[z,:]  # test instance
            pin = np.ones(self.L)/self.L
            STOP = 0
            D = np.random.randint(self.L)
            for k in range(self.K):                               
                cost = []
                for i in range(self.L):
                    cost.append(np.dot(self.thresh[i][k],pin)) 
                    if any(c < 0 for c in cost):
                        D = np.argmin(cost, axis=0)
                        STOP =1
                        break
                if STOP ==1:
                    break                        
                f_order = self.ordering[k]    
                f = obs[f_order]  #Extracting number of words           
                f_index = find_range(f, self.edges[f_order][1:])
                pin = np.divide(np.multiply(pin, self.feat_prob[f_order,:,f_index]),
                                np.dot(pin, self.feat_prob[f_order,:,f_index])) 

            self.predictions.append(self.C[D]) 
            self.n_feat.append(k+1)

        fs_and_c_time =  time.time() - start  # joint feature selection and classification time  
        return fs_and_c_time
    
    def func_J(self, temp_thresh, obs, ci):    
        J = 0
        pin = np.ones(self.L)/self.L
        for k in range(self.K):          
            f_order = self.ordering[k]            
            if np.dot(temp_thresh[k],pin) <0:
                break
            J += self.feat_cost           
            f = obs[f_order]  # observing the feature assignment 
            f_index = find_range(f, self.edges[f_order][1:])  # index after discretization the feature       
            pin = np.divide(np.multiply(pin, self.feat_prob[f_order,:,f_index]),
                            np.dot(pin, self.feat_prob[f_order,:,f_index])) # belief update
        return J + np.dot(self.MC[ci],pin)
    
    def preprocess(self, Xtrain, Ytrain):        
        self.C = list(set(Ytrain))       # classes
        self.L = len(self.C)             # number of classes
        self.K = np.size(Xtrain,axis=1)  # number of features
        self.MC = cost_matrix(self.L) # misclassification cost matrix       
        self.prior = compute_prior(Ytrain, self.L, self.C) # priori probabilities         
        # feature distributions and corresponding bin edges
        self.feat_prob, self.edges = compute_feat_prob(Xtrain, Ytrain, self.C, self.K, self.L, self.bins)    
        self.ordering = get_feat_ord(self.feat_prob, self.K) # feature ordering
        

class Exp_Features(object):
    """
    Exp_Features Class Object: To compute Expected Number of Features for Classification
    Initializing Attributes:
        feat_cost: feature evaluation cost
        bins: number of bins considered when quantizing the feature space
        alpha: error probability bound
        distribution: type of the identical feature distribution. Set to either "best", or "worst" or "avg"
    Functions: 
        run: main function of Exp_Feature object which computes expected number of features for classification
        one_vs_rest: compute one vs rest feature distributions and class prior distribution
        class_exp_feat: compute expected number of features per class variable assignment
    """ 
    
    def __init__(self, config):
        self.feat_cost = config['feat_cost']
        self.bins      = config['bins']
        self.alpha     = config['exp_feat_para']['alpha']
        self.distribution = config['exp_feat_para']['distribution']
        
    def run(self, data):
        self.C = list(set(data['Ytrain']))
        Exp_feat = 0        
        for target in self.C:
            feat_prob, prior = self.one_vs_rest(data['Xtrain'], data['Ytrain'], target)        
            class_feat = self.class_exp_feat(feat_prob, prior)
            Exp_feat += class_feat*prior[0]            
        # Expected number of features
        self.summary = {'exp_features': Exp_feat}
        
    def one_vs_rest(self, Xtrain, Ytrain, target):  # compute one vs rest feature distributions and class prior
        L = len(self.C)
        K = np.size(Xtrain,axis=1)      
        # feature distributions and corresponding bin edges
        feat_prob, edges = compute_feat_prob(Xtrain, Ytrain, self.C, K, L, self.bins) 
        ordering = get_feat_ord(feat_prob, K) # feature ordering
        prior = one_vs_rest_prior(Ytrain, target) # one vs rest prior
        feat_prob_new = one_vs_rest_feat_prob(Xtrain, Ytrain, target, K, self.bins) # one vs rest feature distributions  
        return feat_prob_new[ordering,:], prior

    def class_exp_feat(self, feat_prob, prior):   # expected number of features for a specific target    
        num = (1-2*self.alpha)*math.log((1-self.feat_cost)/self.feat_cost) + math.log(prior[1]/prior[0])
        if self.distribution =='avg':
            kl = 0
            # average KL distance
            for i in range(len(feat_prob)):
                theta = feat_prob[i][0]
                theta_bar = feat_prob[i][1]
                # kl distance
                for v,theta_v in enumerate(theta):
                    kl += theta_v*math.log(theta_v/theta_bar[v])
            den = kl/len(feat_prob)

        elif self.distribution =='best':
            kl = 0
            theta = feat_prob[0][0]
            theta_bar = feat_prob[0][1]
            #kl distance
            for v,theta_v in enumerate(theta):
                kl += theta_v*math.log(theta_v/theta_bar[v])
            den = kl

        elif self.distribution =='worst':
            kl = 0
            theta = feat_prob[-1][0]
            theta_bar = feat_prob[-1][1]
            #kl distance
            for v,theta_v in enumerate(theta):
                kl += theta_v*math.log(theta_v/theta_bar[v])
            den = kl
        else:
            raise ValueError('Check the distribution parameter (avg/best/worst)')
        return num/den

def quantize_simplex(L, neta):
    # uniformly quantizing the probability simplex
    w = np.arange(neta+1, dtype=int)    
    W = set([seq for seq in itertools.product(w, repeat=L) if sum(seq)==neta])
    return W

def cost_matrix(L):
    # misclassification cost matrix
    arr = np.ones(L)
    arr[0] = 0
    MC = toeplitz(arr)   
    return MC

def compute_prior(Ytrain, L, C):
    # compute priori probabilities   
    prior = np.zeros(L)
    for j in range(L):
        prior[j] = sum(Ytrain == C[j] )/len(Ytrain)         
    return prior

def compute_feat_prob(Xtrain, Ytrain, C, K, L, bins):
    # compute feature distributions
    edges = np.zeros((K, bins+1))
    feat_prob = np.zeros((K, L, bins))    
    for i in range(K):
        # discretization feature space
        min_r = np.floor(Xtrain[:,i].min())
        max_r = np.ceil(Xtrain[:,i].max())
        edges[i] = np.linspace(min_r, max_r, num = bins+1)
    
        for j in range(L):
            # feature distributions conditioned on class C_j
            cpd = np.histogram(Xtrain[:,i][Ytrain == C[j]], bins=edges[i])[0]
            feat_prob[i,j,:] = (cpd+1)/(sum(cpd) + bins)            
    return feat_prob, edges

def get_feat_ord(feat_prob, K):
    # get feature ordering based on feature errors 
    PE_var = np.zeros(K) 
    # feature ordering             
    for mi in range(K):
        # error vector for features
        PE_var[mi] = sum(np.var(feat_prob[mi,:,:],axis =0))
    ordering = sorted(range(len(PE_var)), key=lambda k:PE_var[k],reverse=True)     
    return ordering
    
def find_range(val,edge):
    # find bin number of the given feature value
    for i,f in enumerate(edge):
        if val < f:
            return i
    return len(edge)-1

def one_vs_rest_feat_prob(Xtrain, Ytrain, target, K, bins):
    # compute one vs rest probability distributions 
    feat_prob_new = np.zeros((K,2,bins))
    for i in range(K):
        min_r = np.floor(Xtrain[:,i].min())
        max_r = np.ceil(Xtrain[:,i].max())
        edges = np.linspace(min_r, max_r, num=bins+1)
        #CPD for target class
        target_feat = Xtrain[:,i][Ytrain == target]   
        CPD = np.histogram(target_feat, bins = edges)[0]
        feat_prob_new[i,0,:] = (CPD+1)/(sum(CPD) + bins)     
        #CPD for rest classes
        target_feat = Xtrain[:,i][Ytrain != target]   
        CPD = np.histogram(target_feat, bins=edges)[0]
        feat_prob_new[i,1,:] = (CPD+1)/(sum(CPD) + bins)          
    return feat_prob_new

def one_vs_rest_prior(Ytrain, target):
    # compute one vs rest prior for a given target
    P = np.zeros(2)   
    P[0] = sum(Ytrain == target )/len(Ytrain)
    P[1] = 1-P[0]    
    return P
# thats all folks!
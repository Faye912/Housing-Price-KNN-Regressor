#%%
import os
import math
import numpy as np
import pandas as pd
#pip install rfit
import rfit 
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

import warnings
warnings.filterwarnings("ignore")
print("\nReady to continue.")

#%%[markdown]
# 1. Data Preprocessing

#%%
df = pd.read_csv(f'train.csv', header=0)

#%%
numerical_var = ['LotArea', 'SalePrice','LotFrontage', 
                 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
                 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', 
                 '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 
                 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 
                 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
                 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars',
                 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 
                 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 
                 'PoolArea', 'MiscVal']  
df[numerical_var] = StandardScaler().fit_transform(df[numerical_var])

#%%

# Ordinal encoding
ordinal_var = ['OverallQual', 'OverallCond', 'ExterQual', 'ExterCond',
               'BsmtQual', 'BsmtCond', 'BsmtFinType1', 'BsmtFinType2',
               'HeatingQC', 'KitchenQual', 'Functional', 'FireplaceQu', 
               'GarageQual', 'GarageCond', 'GarageCond', 'PoolQC', 'Fence']  

df['OverallQual'] = df['OverallQual'].map({1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:8, 10:9}) 
df['OverallCond'] = df['OverallCond'].map({1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:8, 10:9})
df['BsmtFinType1'] = df['BsmtFinType1'].map({'NA': 0, 'Unf': 0, 'LwQ': 1, 'Rec': 2, 'BLQ': 3, 'ALQ': 4, 'GLQ': 5})
df['BsmtFinType2'] = df['BsmtFinType2'].map({'NA': 0, 'Unf': 0, 'LwQ': 1, 'Rec': 2, 'BLQ': 3, 'ALQ': 4, 'GLQ': 5})
df['Functional'] = df['Functional'].map({'Sal':0, 'Sev':1, 'Maj2':2, 'Maj1':3,
                                         'Mod':4, 'Min2':5, 'Min1':6, 'Typ':7})
df['Fence'] = df['Fence'].map({'NA': 0, 'MnWw': 0, 'GdWo': 1, 'MnPrv': 2, 'GdPrv': 3})

# loop for mapping quality levels
quality_mapping = {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}
for col in ['ExterQual', 'ExterCond', 'HeatingQC', 'KitchenQual']:
    df[col] = df[col].map(quality_mapping)

# loop for variables with "NA" and quality levels
na_quality_mapping = {'NA': 0, 'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}
for col in ['BsmtQual', 'BsmtCond', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']:
    df[col] = df[col].map(na_quality_mapping)



#%%
# Drop purely categorical features
categorical_var = ['MSSubClass', 'MSZoning', 'Street',
                   'Alley', 'LotShape', 'LandContour', 
                   'Utilities', 'LotConfig', 'LandSlope',
                   'Neighborhood', 'Condition1', 'Condition2', 
                   'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl',
                   'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation',
                   'Heating', 'CentralAir', 'Electrical', 'GarageType',
                   'GarageFinish', 'PavedDrive', 'MiscFeature', 'SaleType', 
                   'MoSold', 'YrSold', 'SaleCondition', 'BsmtExposure']  
df = df.drop(columns=categorical_var)

#%%

# Separate features and target
data_x = df.drop(columns=['SalePrice'])
data_y = df['SalePrice']

print("\nReady to continue.")


#%%[markdown]
# 2. Modify the sknn class to perform K-NN regression.

#%%

#%%[markdown]
# 3. Modify the sknn class as you see fit to improve the algorithm performance, logic, or presentations.
# 4. Find optimized scaling factors for the features for the best model score.
# 5. Modify the sknn class to save some results (such as scores, scaling factors, gradients, etc, at various points, like every 100 epoch).
# 6. Compare the results of the optimized scaling factors to Feature Importance from other models, such as Tree regressor for example.




#%%

class sknn:
    '''
    Scaling k-NN model
    v2
    Using gradient to find max
    '''
    import os
    import math
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neighbors import KNeighborsRegressor
    # from sklearn.linear_model import LogisticRegression
    # from sklearn.svm import SVC, LinearSVC
    # from sklearn.tree import DecisionTreeClassifier

    # contructor and properties
    def __init__(self, 
                 data_x, 
                 data_y, 
                 resFilePfx='results', 
                 classifier=True, 
                 k=7, 
                 kmax=33, 
                 zscale=True, 
                 caleExpos_init = (), 
                 scales_init = (), 
                 ttsplit=0.5, 
                 max_iter = 100, 
                 seed=1, 
                 scoredigits = 6, 
                 learning_rate_init = 0.1, 
                 atol = 1e-8 ) :
        """
        Scaling kNN model, using scaling parameter for each feature to infer feature importance and other info about the manifold of the feature space.

        Args:
            data_x (numpy ndarray or pandas DataFrame): x-data
            data_y (numpy ndarray or pandas Series or DataFrame): y-data
            resFilePfx (str, optional): result file prefix. Defaults to 'scores'.
            classifier (bool, optional): classifier vs regressor. Defaults to True.
            k (int, optional): k-value for k-N. Defaults to 7.
            kmax (int, optional): max k-value. Defaults to 33.
            zscale (bool, optional): start with standardized z-score. Defaults to True.
            probeExpos (tuple, optional): Tuple of the exponents for scaling factors. Defaults to ().
            scaleExpos (tuple, optional): Tuple of the scaling factors. Defaults to ().
            ttsplit (float, optional): train-test-split ratio. Defaults to 0.5.
            max_iter (int, optional): maximum iteration. Defaults to 100.
            seed (int, optional): seed value. Defaults to 1.
            scoredigits (int, optional): number of digitis to show/compare in results. Defaults to 6.
            learning_rate_init (float, optional): learning rate, (0,1]. Defaults to 0.01.
            tol (_type_, optional): tolerance. Defaults to 1e-4.
        """
        self.__classifierTF = classifier  # will extend to regression later
        self.k = k
        self.__kmax = kmax
        self.__iter = 0 # the number of trials/iterations
        self.max_iter = max_iter
        self.__seed = seed
        self.__scoredigits = scoredigits
        # self.__resFilePfx = resFilePfx
        self.__learning_rate_init = abs(learning_rate_init)
        self.learning_rate = abs(learning_rate_init)
        self.__atol = atol
        
        # prep data
        self.data_x = data_x
        self.data_xz = data_x # if not to be z-scaled, same as original
        self.zscaleTF = zscale
        # transform z-score 
        if (self.zscaleTF): self.zXform() # will (re-)set self.data_xz
        self.data_y = data_y
        # train-test split
        self.__ttsplit = ttsplit if (ttsplit >=0 and ttsplit <= 1) else 0.5 # train-test split ratio
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.__xdim = 0  # dimension of feature space
        self.traintestsplit() # will set X_train, X_test, y_train, y_test, __xdim
        # set x data column names
        # self.__Xcolnames = (); self.__setXcolnames()
        self.__vector0 = np.zeros(self.__xdim)
        # self.__vector1 = np.ones(self.__xdim)
        
        # set exponents and scaling factors 
        self.__scaleExpos = [] # tuple or list. length set by number of features. Because of invariance under universal scaling (by all features with same factor), we can restrict total sum of exponents to zero.
        # self.__scaleExpos_init = [] # tuple or list. length set by number of features
        self.__scaleFactors = None # numpy array. always calculate from self.__setExpos2Scales
        self.__setExpos2Scales([]) # will set the initial self.scaleExpos and self.__scaleFactors
        # self.__gradients = [] # partial y/partial exponents (instead of partial scaling factors)
        
        # set sklearn knnmodel objects, train, and get benchmark scores on test data
        self.__knnmodels = [np.nan, np.nan] # matching index value as k value
        for i in range(2,self.__kmax +1): 
            if (self.__classifierTF): 
                self.__knnmodels.append( KNeighborsClassifier(n_neighbors=i, weights='uniform').fit(self.X_train, self.y_train ) )
            else: 
                self.__knnmodels.append( KNeighborsRegressor(n_neighbors=i, weights='uniform').fit(self.X_train, self.y_train ) ) # TODO
        self.benchmarkScores = [np.nan, np.nan] +  [ round(x.score(self.X_test, self.y_test ), self.__scoredigits) for x in self.__knnmodels[2:] ]
        print(f'These are the basic k-NN scores for different k-values: {repr(self.benchmarkScores)}, where no individual feature scaling is performed.') 
        
        # set pandas df to save some results
        # self.__resultsDF = None
        
    # END constructor
    
    def zXform(self):
        '''
        standardize all the features (if zscale=True). Should standardize/scale before train-test split
        :return: None
        '''
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        self.data_xz = scaler.fit_transform(self.data_x)  # data_x can be ndarray or pandas df, data_xz will be ndarray
        return
    
    def traintestsplit(self):
        '''
        train-test split, 50-50 as default
        :return: None
        '''
        # train-test split
        from sklearn.model_selection import train_test_split
        # data_y can be pd series here, or 
        dy = self.data_y.values if (isinstance(self.data_y, pd.core.series.Series) or isinstance(self.data_y, pd.core.frame.DataFrame)) else self.data_y # if (isinstance(data_y, np.ndarray)) # the default
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data_xz, dy, test_size=self.__ttsplit, random_state = self.__seed)
        # these four sets should be all numpy ndarrays.

        nrows_Xtest, self.__xdim = self.X_test.shape  # total rows and columns in X_test. # not needed for nrows
        # notice that 
        # self.__xdim == self.X_test.shape[1]   # True
        # self.__xdim is self.X_test.shape[1]   # True
        # nrows_Xtest == self.X_test.shape[0]   # True
        # nrows_Xtest is self.X_test.shape[0]   # False
        return

    # def __setProbeExpos(self, expos):
    #     '''
    #     set Probing Exponents, a tuple
    #     param expos: list/tuple of floats
    #     :return: None
    #     '''
    #     # Can add more checks to ensure expos is numeric list/tuple
    #     self.__probeExpos = expos if (len(expos)>2) else (-6, -3, -1, -0.5, 0, 0.5, 1, 1.5, 2, 4, 6) # tuple, exp(tuple) gives the scaling factors.
    #     self.__probeFactors = tuple( [ math.exp(i) for i in self.__probeExpos ] )
    #     return

    def __setExpos2Scales(self, expos=[]):
        """
        set Scaling Exponents, a tuple or list
        Should make sure expos is centered (using __shiftCenter)

        Args:
            expos (list, optional): _description_. Defaults to [], should match number of features in data_x
        """

        # Can add more checks to ensure expos is numeric list/tuple
        if (len(expos) != self.__xdim):
            self.__scaleExpos = np.zeros(self.__xdim) # tuple, exp(tuple) gives the scaling factors.
            if self.__xdim >1: 
                self.__scaleExpos[0] = 1
                self.__scaleExpos[1] = -1
        else:
            self.__scaleExpos =  expos
        self.__scaleFactors = np.array( [ math.exp(i) for i in self.__scaleExpos ] ) # numpy array
        return
    
    def __shiftCenter(self, expos = []):
        """
        Enforce sum of exponents or any vectors like gradient = 0 (for xdim > 1)

        Args:
            expos (np array, optional): array of scaling exponents. Defaults to [].
        """
        return expos.copy() - expos.sum()/len(expos) if len(expos) > 1 else expos.copy()
        
    
    def __evalGradients(self, learning_rate=0, use = 'test'):
        """
        evaluate Gradients/partial derivatives with respect to exponential factors (not scaling factor)
        Args:
            learning_rate (float, optional): learning_rate. Defaults to 0.
            use (str, optional): use 'test' (default) or 'train' dataset to score. 
        """
        # set learning_rate
        grad = np.array( [ self.__eval1Gradient(i, learning_rate, use=use) for i in range(self.__xdim) ] )
        # normalize grad here?
        # self.__gradients = grad.copy()
        # return
        return grad # gradient as numpy array
    
    def __eval1Gradient(self, i, learning_rate=0, use='test'):
        """
        evaluate a single Gradient/partial derivative with respect to the exponential factor (not scaling factor)

        Args:
            i (int): the column/feature index.
            learning_rate (float, optional): learning_rate. Defaults to 0.
            use (str, optional): use 'test' (default) or 'train' dataset to score. 
        """
        thescale = self.__scaleExpos[i]
        thestep = max(learning_rate, self.learning_rate, abs(thescale)*self.learning_rate ) # modify step value appropriately if needed.
        # maxexpo = thescale + thestep/2
        # minexpo = thescale - thestep/2
        maxexpos = self.__scaleExpos.copy()
        maxexpos[i] += thestep/2
        minexpos = self.__scaleExpos.copy()
        minexpos[i] -= thestep/2
        slope = ( self.scorethis(scaleExpos=maxexpos, use=use) - self.scorethis(scaleExpos=minexpos, use=use) ) / thestep
        return slope
    
    def __setNewExposFromGrad(self, grad=() ):
        """
        setting new scaling exponents, from the gradient info
        steps: 
        1. center grad (will take care of both grad = 0 and grad = (1,1,...,1) cases)
        2. normalize grad (with learning rate as well)
        3. add to original expos

        Args:
            grad (tuple, optional): the gradient calculated. Defaults to empty tuple ().
        """
        grad = self.__shiftCenter(grad)
        if np.allclose(grad, self.__vector0, atol=self.__atol): 
            print(f"Gradient is zero or trivial: {grad}, \n__scaleExpos= {self.__scaleExpos}, \n__scaleFactors= {self.__scaleFactors}, \nmodel score-train is {self.scorethis(use='train')}, \nscore-test is {self.scorethis(use='test')}\n")
            return False
        norm = np.sqrt( np.dot(grad,grad) )
        deltaexpos = grad / norm * self.learning_rate
        self.__scaleExpos += deltaexpos
        self.__setExpos2Scales(self.__scaleExpos)
        return True
    
    def optimize(self, scaleExpos_init = (), maxiter = 0, learning_rate=0):
        """
        Optimizing scaling exponents and scaling factors

        Args:
            scaleExpos_init (np array, optional): initial search vector. Defaults to empty.
            maxiter (int, optional): max iteration. Defaults to 1e5.
            learning_rate (float, optional): learning_rate. Defaults to 0 or self.learning_rate
        """
        maxi = max( self.max_iter, maxiter, 1000)
        skip_n = 10 # rule of thumb math.floor(1/learning_rate)
        expos = scaleExpos_init 
        if (len(scaleExpos_init) == self.__xdim): self.__scaleExpos = scaleExpos_init # assumes the new input is the desired region.
        print(f"Begin: \n__scaleExpos= {self.__scaleExpos}, \n__scaleFactors= {self.__scaleFactors}, \nmodel score-train is {self.scorethis(use='train')}, \nscore-test is {self.scorethis(use='test')}, \nmaxi= {maxi}, k={self.k}, learning_rate={self.learning_rate}\n")
        for i in range(maxi):
            grad = self.__evalGradients(learning_rate, use='train')
            # Cases
            # 1. grad = 0, stop (built into __setNewExposFromGrad)
            # 2. grad parallel to (1,1,1,...,1) direction, stop.
            # 3. maxiter reached, stop. (end of loop)
            # 4. ?? dy < tol, stop??            # 
            result = self.__setNewExposFromGrad(grad)
            if (i<10 or i%skip_n==0 ): print(f"i: {i}, |grad|^2={np.dot(grad,grad)}, \ngrad= {grad}, \n__scaleExpos= {self.__scaleExpos}, \n__scaleFactors= {self.__scaleFactors}, \nmodel score-train is {self.scorethis(use='train')}, \nscore-test is {self.scorethis(use='test')}\n")
            if not result: break
            
        if i==maxi-1: print(f"max iter reached. Current |grad|^2={np.dot(grad,grad)}, \ngrad= {grad}, \n__scaleExpos= {self.__scaleExpos}, \n__scaleFactors= {self.__scaleFactors}, \nmodel score-train is {self.scorethis(use='train')}, \nscore-test is {self.scorethis(use='test')}\n")
            
    
    def scorethis(self, scaleExpos = [], scaleFactors = [], use = 'test'):
        if len(scaleExpos)==self.__xdim :
            self.__setExpos2Scales( self.__shiftCenter(scaleExpos) )
        # elif len(scaleFactors)==self.__xdim:
        #     self.__scaleFactors = np.array(scaleFactors)
        #     self.__scaleExpos = [ round(math.log(x), 2 ) for x in scaleFactors ]
        else:
            # self.__setExpos2Scales(np.zeros(self.__xdim))
            if (len(scaleExpos)>0 or len(scaleFactors)>0) : print('Scale factors set to default values of unit (all ones). If this is not anticipated, please check your input, making sure the length of the list matches the number of features in the dataset.')
        
        sfactors = self.__scaleFactors.copy() # always start from the pre-set factors, whatever it might be
        self.__knnmodels[self.k].fit(sfactors*self.X_train, self.y_train)
        # For optimizing/tuning the scaling factors, use the train set to tune. 
        newscore = self.__knnmodels[self.k].score(sfactors*self.X_train, self.y_train) if use=='train' else self.__knnmodels[self.k].score(sfactors*self.X_test, self.y_test)
        return newscore

###### END class sknn

#%%
# test code
# diabetes = sknn(data_x=data_x, data_y=data_y)
housing_price = sknn(data_x=data_x, data_y=data_y, learning_rate_init=0.01)
housing_price.optimize()


#%%

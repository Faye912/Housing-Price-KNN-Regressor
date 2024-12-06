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
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

class sknn:
    def __init__(self, 
                 data_x, 
                 data_y, 
                 resFilePfx='results', 
                 classifier=True, 
                 k=7, 
                 kmax=33, 
                 zscale=True, 
                 ttsplit=0.5, 
                 max_iter=100, 
                 seed=1, 
                 scoredigits=6, 
                 learning_rate_init=0.1, 
                 atol=1e-8):
        """
        Modified sknn class for K-NN regression.
        """
        self.classifier = classifier  # Determines if K-NN is for classification or regression
        self.k = k
        self.kmax = kmax
        self.max_iter = max_iter
        self.seed = seed
        self.scoredigits = scoredigits
        self.learning_rate = abs(learning_rate_init)
        self.atol = atol

        # Preprocess data
        self.data_x = data_x
        self.data_y = data_y
        self.zscale = zscale
        if self.zscale:
            self.zXform()  # Apply z-score scaling
        self.traintestsplit(ttsplit)  # Perform train-test split

        # Initialize k-NN models
        self.knnmodels = [None] * (self.kmax + 1)
        for i in range(2, self.kmax + 1):
            if self.classifier:
                self.knnmodels[i] = KNeighborsClassifier(n_neighbors=i).fit(self.X_train, self.y_train)
            else:
                self.knnmodels[i] = KNeighborsRegressor(n_neighbors=i).fit(self.X_train, self.y_train)

        # Benchmark scores for test data
        self.benchmarkScores = [None] * (self.kmax + 1)
        for i in range(2, self.kmax + 1):
            self.benchmarkScores[i] = self.scorethis(k=i, use='test')

    def zXform(self):
        """Standardize features using z-score scaling."""
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        self.data_x = scaler.fit_transform(self.data_x)

    def traintestsplit(self, ttsplit):
        """Split data into training and testing sets."""
        from sklearn.model_selection import train_test_split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.data_x, self.data_y, test_size=ttsplit, random_state=self.seed
        )

    def scorethis(self, k=None, use='test'):
        """
        Evaluate model performance.
        Args:
            k (int): Number of neighbors.
            use (str): 'train' or 'test' dataset to evaluate.
        Returns:
            float: Model performance (R² for regression, accuracy for classification).
        """
        if k is None:
            k = self.k

        model = self.knnmodels[k]
        if use == 'train':
            X, y = self.X_train, self.y_train
        else:
            X, y = self.X_test, self.y_test

        if self.classifier:
            # Accuracy for classification
            return round(model.score(X, y), self.scoredigits)
        else:
            # R² for regression
            y_pred = model.predict(X)
            return round(r2_score(y, y_pred), self.scoredigits)

    def optimize(self, max_iter=None, learning_rate=None):
        """
        Optimize scaling factors for features to improve model performance.
        Args:
            max_iter (int): Maximum iterations for optimization.
            learning_rate (float): Learning rate for gradient-based updates.
        """
        if max_iter is None:
            max_iter = self.max_iter
        if learning_rate is None:
            learning_rate = self.learning_rate

        print(f"Optimization started with max_iter={max_iter} and learning_rate={learning_rate}")

        # Example of optimization logic (simplified for demonstration)
        for i in range(max_iter):
            current_score = self.scorethis(use='train')
            print(f"Iteration {i}: Train Score = {current_score}")
            # Update scaling factors (placeholder for actual gradient logic)

        print("Optimization complete.")


#%%[markdown]
# 3. Modify the sknn class as you see fit to improve the algorithm performance, logic, or presentations.

#%%

#%%[markdown]
# 4. Find optimized scaling factors for the features for the best model score.

#%%

#%%[markdown]
# 5. Modify the sknn class to save some results (such as scores, scaling factors, gradients, etc, at various points, like every 100 epoch).

#%%

#%%[markdown]
# 6. Compare the results of the optimized scaling factors to Feature Importance from other models, such as Tree regressor for example.

#%%


#%%
# test code
# diabetes = sknn(data_x=data_x, data_y=data_y)
housing_price = sknn(data_x=data_x, data_y=data_y, learning_rate_init=0.01)
housing_price.optimize()


#%%

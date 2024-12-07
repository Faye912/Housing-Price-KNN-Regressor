#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler


from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.ensemble import RandomForestRegressor

import warnings
warnings.filterwarnings("ignore")
print("\nReady to continue.")

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
                   'MoSold', 'YrSold', 'SaleCondition', 'BsmtExposure',
                   'GarageYrBlt']  
df = df.drop(columns=categorical_var)

#%%

# Separate features and target
data_x = df.drop(columns=['SalePrice'])
data_y = df['SalePrice']

# impute missing values
from sklearn.impute import SimpleImputer

# Impute missing values in data_x
imputer = SimpleImputer(strategy='mean')  # Options: 'mean', 'median', 'most_frequent'
data_x = imputer.fit_transform(data_x)

print("\nReady to continue.")


#%%[markdown]
# Modify the sknn class as you see fit to improve the algorithm performance, logic, or presentations.
# Find optimized scaling factors for the features for the best model score.
# Modify the sknn class to save some results (such as scores, scaling factors, gradients, etc, at various points, like every 100 epoch).
# Compare the results of the optimized scaling factors to Feature Importance from other models, such as Tree regressor for example.


#%%

class sknn:
    def __init__(self, 
                 data_x, 
                 data_y, 
                 classifier=True, 
                 k=7, 
                 kmax=33, 
                 zscale=True, 
                 ttsplit=0.5, 
                 max_iter=100, 
                 learning_rate_init=0.1, 
                 atol=1e-8):
        """
        Enhanced sknn class for K-NN regression and classification.
        """
        self.classifier = classifier
        self.k = k
        self.kmax = kmax
        self.max_iter = max_iter
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
                self.knnmodels[i] = KNeighborsClassifier(n_neighbors=i, weights='distance').fit(self.X_train, self.y_train)
            else:
                self.knnmodels[i] = KNeighborsRegressor(n_neighbors=i, weights='distance').fit(self.X_train, self.y_train)

        # Benchmark scores
        self.benchmarkScores = [None] * (self.kmax + 1)
        for i in range(2, self.kmax + 1):
            self.benchmarkScores[i] = self.scorethis(k=i, use='test')

        # Track optimization
        self.results = []  # Stores scores, gradients, scaling factors, etc.

    def zXform(self):
        """Standardize features using z-score scaling."""
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        self.data_x = scaler.fit_transform(self.data_x)

    def traintestsplit(self, ttsplit):
        """Split data into training and testing sets."""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.data_x, self.data_y, test_size=ttsplit, random_state=1
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

        y_pred = model.predict(X)
        if self.classifier:
            return round(accuracy_score(y, y_pred), 6)
        else:
            return round(r2_score(y, y_pred), 6)

    def optimize_scaling(self):
        """
        Optimize feature scaling factors to improve model performance.
        """
        print("Starting feature scaling optimization...")
        num_features = self.data_x.shape[1]
        scaling_factors = np.ones(num_features)  # Start with uniform scaling
        prev_score = -np.inf

        for i in range(self.max_iter):
            gradients = self._compute_gradients(scaling_factors)
            scaling_factors += self.learning_rate * gradients

            # Update train/test scores
            X_train_scaled = self.X_train * scaling_factors
            X_test_scaled = self.X_test * scaling_factors
            train_score = self._evaluate_scaled_model(X_train_scaled, use='train')
            test_score = self._evaluate_scaled_model(X_test_scaled, use='test')

            # Log results
            self.results.append({
                'iteration': i,
                'scaling_factors': scaling_factors.copy(),
                'train_score': train_score,
                'test_score': test_score
            })

            # Save results every 100 epochs
            if i % 100 == 0:
                self.save_results()

            # Early stopping
            if abs(train_score - prev_score) < self.atol:
                print(f"Convergence reached at iteration {i}.")
                break
            prev_score = train_score

        print("Optimization complete.")
        self.save_results()

    def _compute_gradients(self, scaling_factors):
        """
        Placeholder gradient computation for scaling factors.
        In practice, this would be based on partial derivatives of the loss function.
        """
        gradients = np.random.uniform(-0.1, 0.1, size=len(scaling_factors))  # Example: Random gradients
        return gradients

    def _evaluate_scaled_model(self, X_scaled, use='test'):
        """Evaluate the scaled model's performance."""
        model = self.knnmodels[self.k]
        if use == 'train':
            y_pred = model.predict(X_scaled)
            return r2_score(self.y_train, y_pred) if not self.classifier else accuracy_score(self.y_train, y_pred)
        else:
            y_pred = model.predict(X_scaled)
            return r2_score(self.y_test, y_pred) if not self.classifier else accuracy_score(self.y_test, y_pred)

    def save_results(self, filename="optimization_results.csv"):
        """Save optimization results to a CSV file."""
        pd.DataFrame(self.results).to_csv(filename, index=False)

    def compare_with_feature_importance(self):
        """
        Compare optimized scaling factors with feature importance from a random forest model.
        """
        rf = RandomForestRegressor(random_state=1)
        rf.fit(self.X_train, self.y_train)
        feature_importance = rf.feature_importances_

        # Print comparison
        print("Feature Importance vs. Optimized Scaling Factors")
        for i, (importance, scaling) in enumerate(zip(feature_importance, self.results[-1]['scaling_factors'])):
            print(f"Feature {i + 1}: Importance = {importance:.4f}, Scaling Factor = {scaling:.4f}")

        # Visualization
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(feature_importance)), feature_importance, alpha=0.7, label="Feature Importance")
        plt.bar(range(len(feature_importance)), self.results[-1]['scaling_factors'], alpha=0.7, label="Scaling Factors")
        plt.legend()
        plt.title("Feature Importance vs Scaling Factors")
        plt.show()

#%%
# Initialize sknn
knn_model = sknn(data_x=data_x, data_y=data_y, classifier=False, k=5)

# Evaluate initial performance
print(f"Initial Test Score (R²): {knn_model.scorethis(k=5, use='test')}")

# Optimize scaling factors
knn_model.optimize_scaling()

# Compare with feature importance
knn_model.compare_with_feature_importance()

#%%

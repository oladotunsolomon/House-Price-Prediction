# House Price Prediction
The AmesHousing project focuses on the residential real estate market in **Ames, Iowa,** using publicly available data that includes detailed information on over 2,900 home sales. The goal is to develop a predictive system that estimates house prices based on features such as location, structure, and quality, while also identifying market patterns through clustering techniques. As a data analyst, my role is to apply machine learning methods to uncover pricing drivers, segment the housing market, and provide actionable insights. The findings can help real estate professionals, buyers, and investors make smarter pricing, renovation, and investment decisions based on data-driven evidence.

## Project Overview
This project uses the **AmesHousing** dataset to build predictive models that estimate house prices and segment the housing market. It integrates both **supervised learning** (Linear Regression, Random Forest, XGBoost) and **unsupervised learning** (PCA, KMeans) to support data-driven real estate decisions.

## Dataset
- Source: Ames Housing Dataset - Kaggle
- Records: 2,930 residential property sales
- Features: 82 variables covering detailed housing attributes

## Tools and Technologies
- Language: Python
- Libraries: pandas, numpy, scikit-learn, XGBoost, matplotlib, seaborn
- Environment: Jupyter Notebook / Google Colab

## Methodology
#### Data Preprocessing
- Removed columns with >40% missing values
- Imputed numerical and categorical values
- Log-transformed skewed features (e.g., SalePrice, GrLivArea)

#### Feature Engineering
- Created HouseAge, GarageAge, TotalSF, TotalBathrooms, etc.
- Encoded categorical variables using ordinal and one-hot encoding

#### Exploratory Data Analysis (EDA)
-  Correlation heatmaps, boxplots, and scatterplots
-  Neighborhood and condition analysis to understand price trends

#### Unsupervised Learning (Market Segmentation)
-  Dimensionality reduction using PCA
-  KMeans clustering (4 segments: Luxury Builds, Modern Mid-Size, Historic Starters, Older Family Homes)

#### Supervised Learning (Price Prediction)
-  Models: Linear Regression, Random Forest, XGBoost
-  Evaluation metrics: RMSE and R² Score

| Model            | RMSE       | R² Score |
|------------------|------------|------------|
| Linear Regression | 30,401     | 0.8847     |
| Random Forest     | 25,496     | 0.9189     |
| XGBoost           | 23,574     | 0.9307     |

## Key Insights
-  XGBoost performed best with lowest RMSE and highest R²
-  Feature importance showed OverallQual, TotalSF, and KitchenQual as top predictors
-  Cluster label (from KMeans) emerged as a significant feature in price prediction

```python
# House Price Prediction and Regional Market Segmentation

# Import necessary libraries
from google.colab import drive
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from xgboost import XGBRegressor, plot_importance
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Mount Google Drive
drive.mount('/content/drive')

# Load dataset
file_path = '/content/drive/MyDrive/AmesHousing.csv'
df = pd.read_csv(file_path)

# Display dataset info
print("Dataset shape:", df.shape)
print(df.head())

# Check for missing values
missing_percentage = df.isnull().mean() * 100
missing_percentage = missing_percentage[missing_percentage > 0].sort_values(ascending=False)

# Plot missing values
plt.figure(figsize=(12, 6))
missing_percentage.plot(kind='bar', color='salmon', edgecolor='black')
plt.title('Percentage of Missing Values by Column')
plt.ylabel('Missing Percentage (%)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Feature engineering - age features
df['HouseAge'] = df['Yr Sold'] - df['Year Built']
df['YearsSinceRemodel'] = df['Yr Sold'] - df['Year Remod/Add']
df['GarageAge'] = df['Yr Sold'] - df['Garage Yr Blt']

# Drop columns with >40% missing values
threshold = 0.4
missing_ratio = df.isnull().mean()
columns_to_drop = missing_ratio[missing_ratio > threshold].index
df_cleaned = df.drop(columns=columns_to_drop)

# Drop irrelevant columns
df_cleaned = df_cleaned.drop(columns=['Order', 'PID'])

# Fill missing values
num_cols = df_cleaned.select_dtypes(include=['float64', 'int64']).columns
df_cleaned[num_cols] = df_cleaned[num_cols].fillna(df_cleaned[num_cols].median())
cat_cols = df_cleaned.select_dtypes(include=['object']).columns
df_cleaned[cat_cols] = df_cleaned[cat_cols].fillna(df_cleaned[cat_cols].mode().iloc[0])

# Ordinal encoding
ordinal_mappings = {
    'Exter Qual': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'Exter Cond': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'Bsmt Qual': ['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'Bsmt Cond': ['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'Heating QC': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'Kitchen Qual': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'Fireplace Qu': ['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'Garage Qual': ['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'Garage Cond': ['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'Pool QC': ['NA', 'Fa', 'TA', 'Gd', 'Ex'],
    'Bsmt Exposure': ['NA', 'No', 'Mn', 'Av', 'Gd'],
    'BsmtFin Type 1': ['NA', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
    'BsmtFin Type 2': ['NA', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
    'Functional': ['Sal', 'Sev', 'Maj2', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ'],
    'Fence': ['NA', 'MnWw', 'GdWo', 'MnPrv', 'GdPrv']
}
ordinal_cols = [col for col in ordinal_mappings if col in df_cleaned.columns]
encoder = OrdinalEncoder(categories=[ordinal_mappings[col] for col in ordinal_cols])
df_cleaned[ordinal_cols] = encoder.fit_transform(df_cleaned[ordinal_cols])

# One-hot encoding
nominal_cols = [col for col in cat_cols if col not in ordinal_cols]
df_encoded = pd.get_dummies(df_cleaned, columns=nominal_cols, drop_first=True)

# Log transformation
df_encoded['SalePrice_Log'] = np.log1p(df_cleaned['SalePrice'])
df_encoded['GrLivArea_Log'] = np.log1p(df_cleaned['Gr Liv Area'])

# Additional feature engineering
df_encoded['TotalBathrooms'] = (
    df_cleaned['Full Bath'] + df_cleaned['Half Bath'] * 0.5 +
    df_cleaned['Bsmt Full Bath'] + df_cleaned['Bsmt Half Bath'] * 0.5
)
df_encoded['TotalSF'] = (
    df_cleaned['1st Flr SF'] + df_cleaned['2nd Flr SF'] + df_cleaned['Total Bsmt SF']
)
df_encoded['HouseAge'] = df_cleaned['Yr Sold'] - df_cleaned['Year Built']

# PCA for clustering
features_for_clustering = df_encoded.drop(columns=['SalePrice_Log'], errors='ignore')
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features_for_clustering)
pca = PCA(n_components=2)
df_pca = pd.DataFrame(pca.fit_transform(scaled_features), columns=['PC1', 'PC2'])

# KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=42)
df_pca['Cluster'] = kmeans.fit_predict(df_pca)
df_encoded['Cluster'] = df_pca['Cluster']

# Supervised learning setup
X = df_encoded.drop(columns=['SalePrice'], errors='ignore')
y = df_encoded['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr = r2_score(y_test, y_pred_lr)

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

# XGBoost
xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
r2_xgb = r2_score(y_test, y_pred_xgb)

# Comparison table
model_scores = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest', 'XGBoost'],
    'RMSE': [rmse_lr, rmse_rf, rmse_xgb],
    'R2 Score': [r2_lr, r2_rf, r2_xgb]
})

# Feature importance
importances = rf.feature_importances_
feature_names = X_train.columns
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

importance_dict = xgb.get_booster().get_score(importance_type='gain')
xgb_importance_df = pd.DataFrame({
    'Feature': list(importance_dict.keys()),
    'Gain': list(importance_dict.values())
}).sort_values(by='Gain', ascending=False)

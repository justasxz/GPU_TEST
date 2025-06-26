import pandas as pd
import numpy as np
from scipy.stats import skew
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# 1. Load raw data
# ----------------
df_raw = pd.read_csv("Data/house_train.csv")
df_test_raw = pd.read_csv("Data/house_test.csv")

# 2. Compute imputation statistics from train raw
# -----------------------------------------------
# comment: calculate once on raw train to use for both train & test
lotfrontage_median = df_raw['LotFrontage'].median()
masvnrtype_mode      = df_raw['MasVnrType'].mode()[0]
masvnrarea_median    = df_raw['MasVnrArea'].median()
electrical_mode      = df_raw['Electrical'].mode()[0]
garageyrbult_median  = df_raw['GarageYrBlt'].median()

# 3. Copy train for processing
# ----------------------------
df = df_raw.copy()


# 5. Drop uninformative columns (train only)
# -------------------------------------------
df = df.drop_duplicates().drop(columns=['PoolQC', 'MiscFeature'])

# 6. Impute missing values in train using train stats
# ---------------------------------------------------
df['LotFrontage'] = df['LotFrontage'].fillna(lotfrontage_median)
df['MasVnrType']  = df['MasVnrType'].fillna(masvnrtype_mode)
df['MasVnrArea']  = df['MasVnrArea'].fillna(masvnrarea_median)
df['Electrical']  = df['Electrical'].fillna(electrical_mode)
df['GarageYrBlt'] = df['GarageYrBlt'].fillna(garageyrbult_median)

bsmt_cols   = ['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2']
garage_cols = ['GarageType','GarageFinish','GarageQual','GarageCond']
for col in bsmt_cols + garage_cols:
    df[col] = df[col].fillna("None")  # comment: treat missing basement/garage as None
for col in ['FireplaceQu','Alley','Fence']:
    df[col] = df[col].fillna("None")

# 7. Convert MSSubClass to categorical in train
# ---------------------------------------------
df['MSSubClass'] = df['MSSubClass'].astype(str)  # comment: as category for dummies

# 8. One-hot encode train categories
# ----------------------------------
df = pd.get_dummies(df, drop_first=True) 

# 9. Handle outliers and transform LotArea in train
# ------------------------------------------------
Q1  = df['LotArea'].quantile(0.25)
Q3  = df['LotArea'].quantile(0.75)
IQR = Q3 - Q1
# comment: remove extreme LotArea outliers
df = df[(df['LotArea'] >= Q1 - 1.5*IQR) & (df['LotArea'] <= Q3 + 1.5*IQR)]
# comment: log-transform skewed LotArea
df['LotArea'] = np.log1p(df['LotArea'])

# 10. Separate features and target for train
# ------------------------------------------
y_raw  = df['SalePrice']
X_raw  = df.drop(columns=['SalePrice']).copy()

# 11. Standardize numeric features on train & save scaler
# -------------------------------------------------------
num_cols = X_raw.select_dtypes(include=['int64','float64']).columns
scaler   = StandardScaler()
X_raw[num_cols] = scaler.fit_transform(X_raw[num_cols])

# 12. (Optional) Save preprocessed train
# --------------------------------------
X_raw['SalePrice'] = y_raw
X_raw.to_csv('train_preprocessed.csv', index=False)
print("Išsaugota paruošta treniravimui")

# 13. Split into train/validation
# -------------------------------
y_log = np.log1p(y_raw)
X     = X_raw.drop(columns=['SalePrice'])
X_train, X_val, y_train_log, y_val_log = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)

# 14. Train model
# ---------------
model = DecisionTreeRegressor(max_depth=5, random_state=42)
model.fit(X_train, y_train_log)

# 15. Evaluate on validation set
# ------------------------------
y_pred_log_val = model.predict(X_val)
rmse_log       = np.sqrt(mean_squared_error(y_val_log, y_pred_log_val))
y_val_real     = np.expm1(y_val_log)
y_pred_real    = np.expm1(y_pred_log_val)
rmse_eur       = np.sqrt(mean_squared_error(y_val_real, y_pred_real))
print(f"RMSE log erdvėje: {rmse_log:.4f}")
print(f"RMSE tikrose kainose: ±€{rmse_eur:,.0f}")

# 16. PREPARE TEST DATA (preserve all rows)
# ------------------------------------------
df_test = df_test_raw.copy()
# comment: drop same columns
if set(['PoolQC','MiscFeature']).issubset(df_test.columns):
    df_test = df_test.drop(columns=['PoolQC','MiscFeature'])

# comment: impute test using train statistics
df_test['LotFrontage'] = df_test['LotFrontage'].fillna(lotfrontage_median)
df_test['MasVnrType']  = df_test['MasVnrType'].fillna(masvnrtype_mode)
df_test['MasVnrArea']  = df_test['MasVnrArea'].fillna(masvnrarea_median)
df_test['Electrical']  = df_test['Electrical'].fillna(electrical_mode)
df_test['GarageYrBlt'] = df_test['GarageYrBlt'].fillna(garageyrbult_median)
for col in bsmt_cols + garage_cols:
    df_test[col] = df_test[col].fillna("None")
for col in ['FireplaceQu','Alley','Fence']:
    df_test[col] = df_test[col].fillna("None")

# comment: MSSubClass & one-hot encode test
df_test['MSSubClass'] = df_test['MSSubClass'].astype(str)
df_test = pd.get_dummies(df_test, drop_first=True)

# 17. Align test to train features
# --------------------------------
X_test = df_test.reindex(columns=X_raw.drop(columns=['SalePrice']).columns, fill_value=0)

# 18. Standardize numeric features on test
# ----------------------------------------
X_test[num_cols] = scaler.transform(X_test[num_cols])

# 19. Predict and create submission
# ---------------------------------
df_submission = pd.DataFrame({
    'Id': df_test_raw['Id'],  # comment: use raw test IDs
    'SalePrice': np.expm1(model.predict(X_test))  # comment: inverse log
})

df_submission.to_csv('submission.csv', index=False)
print("Submission file created: submission.csv")

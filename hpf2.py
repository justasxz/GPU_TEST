from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.stats import skew

# įkrauname pradinius duomenis
df = pd.read_csv("Data/house_train.csv")
df_test = pd.read_csv("Data/house_test.csv")


# surandame skaitinius stulpelius ir jų šleivumą
numeric_cols  = df.select_dtypes(include=['int64', 'float64']).columns
skewed_feats  = df[numeric_cols].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewed_to_fix = skewed_feats[skewed_feats > 0.75]
print("Stulpeliai su dideliu šleivumu:")
print(skewed_to_fix)

# pašaliname pasikartojančias eilutes ir tuščius stulpelius
df = df.drop_duplicates().drop(columns=['PoolQC', 'MiscFeature'])
df_test = df_test.drop(columns=['PoolQC', 'MiscFeature'])

# trūkstamų reikšmių užpildymas skaitiniams stulpeliams
df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())
df['MasVnrType']  = df['MasVnrType'].fillna(df['MasVnrType'].mode()[0])
df['MasVnrArea']  = df['MasVnrArea'].fillna(df['MasVnrArea'].median())
df['Electrical']  = df['Electrical'].fillna(df['Electrical'].mode()[0])
df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['GarageYrBlt'].median())

df_test['LotFrontage'].fillna(df['LotFrontage'].median(), inplace=True)
df_test['MasVnrType'].fillna(df['MasVnrType'].mode()[0], inplace=True)
df_test['MasVnrArea'].fillna(df['MasVnrArea'].median(), inplace=True)
df_test['Electrical'].fillna(df['Electrical'].mode()[0], inplace=True)
df_test['GarageYrBlt'].fillna(df['GarageYrBlt'].median(), inplace=True)




# trūkstamų reikšmių užpildymas kategoriniams stulpeliams
bsmt_cols   = ['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2']
garage_cols = ['GarageType','GarageFinish','GarageQual','GarageCond']
for col in bsmt_cols + garage_cols:
    df[col] = df[col].fillna("None")
    df_test[col] = df_test[col].fillna("None")
for col in ['FireplaceQu','Alley','Fence']:
    df[col] = df[col].fillna("None")
    df_test[col] = df_test[col].fillna("None")



# MSSubClass kaip kategorija
df['MSSubClass'] = df['MSSubClass'].astype(str)
df_test['MSSubClass'] = df_test['MSSubClass'].astype(str)

from sklearn.preprocessing import OneHotEncoder

# vienkartinis kodavimas kategorijoms
encoder = OneHotEncoder(
    drop='first',  # pašaliname pirmą kategoriją, kad išvengtume multikolineariškumo
    handle_unknown='ignore'  # ignoruojame nežinomas kategorijas testavimo rinkinyje
)
# koduojame kategorinius stulpelius bet df turi islikti dataframe
df = encoder.fit_transform(df)
df_test = encoder.transform(df_test)

# LotArea pasiskirstymo vaizdavimas prieš log transformaciją
# plt.hist(df["LotArea"], bins=50)
# plt.title("LotArea prieš log transformaciją")
# plt.show()

# ekstremalių reikšmių šalinimas pagal IQR
Q1 = df["LotArea"].quantile(0.25)
Q3 = df["LotArea"].quantile(0.75)
IQR = Q3 - Q1
df = df[(df["LotArea"] >= Q1 - 1.5 * IQR) & (df["LotArea"] <= Q3 + 1.5 * IQR)]

# log transformacija LotArea
df['LotArea'] = np.log1p(df['LotArea'])
df_test['LotArea'] = np.log1p(df_test['LotArea'])

# atskiriame požymius ir tikslinį kintamąjį
y_raw = df["SalePrice"]
X_raw = df.drop(columns=["SalePrice"])

scaler = StandardScaler()

# standartizuojame skaitinius požymius
num_cols        = X_raw.select_dtypes(include=['int64','float64']).columns
X_raw[num_cols] = scaler.fit_transform(X_raw[num_cols])

# užtikriname, kad testavimo duomenų rinkinys turi tuos pačius požymius
# X_raw, X_test = X_raw.align(df_test, join='left', axis=1, fill_value=0)

# standartizuojame skaitinius požymius testavimo duomenų rinkinyje
df_test[num_cols] = scaler.transform(df_test[num_cols])

# saugome paruoštą rinkinį
X_raw["SalePrice"] = y_raw
X_raw.to_csv("train_preprocessed.csv", index=False)
print("Išsaugota paruošta treniravimui")

# atskiriame log kainas ir padalijame mokymui bei testavimui
X = X_raw.drop(columns=["SalePrice"])
y_log = np.log1p(y_raw)
X_train, X_test, y_train_log, y_test_log = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)

# modelio kūrimas ir treniravimas su ankstyvu sustabdymu
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(
    max_depth=5,
    random_state=42
)

model.fit(
    X_train, y_train_log,
)

# log prognozių generavimas
y_pred_log = model.predict(X_test)
print(df_test.shape)
# let's allign the test set to the training set so we don't have any issues with missing columns from get_dummies let's make the same columns without reindexing

# y_pred_test_log = model.predict(df_test)

# RMSE log erdvėje
val_rmse_log = np.sqrt(mean_squared_error(y_test_log, y_pred_log))

# grąžiname tikras kainas ir apskaičiuojame RMSE eurais
y_test_real  = np.expm1(y_test_log)
y_pred_real  = np.expm1(y_pred_log)

# y_test_pred_real = np.expm1(y_pred_test_log)
# print(y_test_pred_real.shape)

val_rmse_eur = np.sqrt(mean_squared_error(y_test_real, y_pred_real))

print(f"RMSE log erdvėje: {val_rmse_log:.4f}")
print(f"RMSE tikrose kainose: ±€{val_rmse_eur:,.0f}")

# print(y_test_real)

# # LotArea paskirstymo vaizdavimas po log transformacijos
# plt.hist(np.log1p(df["LotArea"]), bins=50)
# plt.title("LotArea po log transformacijos")
# plt.show()

# patikriname trūkstamas ir neigiamos kainos atvejus
# print(df["SalePrice"].isnull().sum())
# print((df["SalePrice"] <= 0).sum())
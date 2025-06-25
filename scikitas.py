from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
iris = load_iris()
# X, y = iris.data, iris.target

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# labai svarbu susipažinti su duomenimis
# print(df.head())
# print(df.describe())
# print(df.info())

# Plot some data

# print(sns.load_dataset("titanic"))

# sns.pairplot(df, hue='target', markers=["o", "s", "D"])
# plt.show()

# DUomenys sutvarkyti # ir paruošti mokymui

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test  = train_test_split(df.drop(columns='target'), df['target'], test_size=0.20, random_state=42, stratify=df['target'])


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) # fit - pritaiko duomenis, suranda vidurkius ir standartinius nuokrypius
X_test = scaler.transform(X_test)

# import normalization
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)


# print(X_train[:5]) # Pirmi 5 duomenys po transformacijos

# # strify=df['target'] - tai svarbu, kad duomenys būtų subalansuoti
# print(X_train.shape)
# print(X_test.shape)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train) # Treniravimas (training)
# Model evaluation
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, classification_report, recall_score
y_pred = model.predict(X_test) # Predicting(Inference) Spejimas

accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.2f}")
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)
precision = precision_score(y_test, y_pred, average='weighted')
print(f"Precision: {precision:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
recall = recall_score(y_test, y_pred, average='weighted')
print(f"Recall: {recall:.2f}")

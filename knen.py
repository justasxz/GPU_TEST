# pasiimsime regresijos duomenis is sklearn arba sns ir pameginsime atlikti regresija su knn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, r2_score
# Load student performance dataset
df = pd.read_csv('students.csv')
print(df.describe())
target_column = 'GPA' 
df.drop(columns=['GradeClass'], inplace=True)  # Remove unnecessary columns
# Duomenu paruosimas, bet duomenys jau yra paruošti
# # Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=[target_column]), df[target_column], test_size=0.2, random_state=42)
# # Standardize the features
columns_to_scale = ['Age','StudyTimeWeekly', 'Absences', 'ParentalEducation','Ethnicity','ParentalSupport']
scaler = StandardScaler()
X_train[columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])
X_test[columns_to_scale] = scaler.transform(X_test[columns_to_scale])
# Create a KNN regressor
# # Train the model
# let's pick the best k value with cross-validation
from sklearn.model_selection import cross_val_score
# k_range = range(1, 20)
# mae_scores = []
# for k in k_range:
#     knn = KNeighborsRegressor(n_neighbors=k)
#     scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
#     mae_scores.append(-scores.mean())

# # let's plot the results to find the best k
# import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 6))
# plt.plot(k_range, mae_scores, marker='o')
# plt.title('KNN Regression: Mean Absolute Error vs. Number of Neighbors')
# plt.xlabel('Number of Neighbors (k)')
# plt.ylabel('Mean Absolute Error')
# plt.grid()
# plt.show()
# let's pick best k from our test
# best_k = mae_scores.index(min(mae_scores)) + 1  # +1 because index starts at 0
# print(f"Best k value: {best_k}")

# knn = KNeighborsRegressor(n_neighbors=best_k)  # Use the best k value found
# knn.fit(X_train, y_train)

# Let's use SVR
from sklearn.svm import SVR
svr = SVR(kernel='linear')  # Using SVR as a regression model
# let's pick hyperparameters for SVR with cross-validation

c_range = [0.1, 1, 10, 100]
epsilon_range = [0.01, 0.1, 0.5, 1]
best_score = float('inf')
best_params = {}
for c in c_range:
    for epsilon in epsilon_range:
        svr = SVR(kernel='linear', C=c, epsilon=epsilon)
        scores = cross_val_score(svr, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
        mean_score = -scores.mean()  # Convert to positive MAE
        if mean_score < best_score:
            best_score = mean_score
            best_params = {'C': c, 'epsilon': epsilon}

print(f"Best parameters found: {best_params} with Mean Absolute Error: {best_score:.2f}")
svr = SVR(kernel='linear', C=best_params['C'], epsilon=best_params['epsilon'])  # Use the best parameters found
svr.fit(X_train, y_train)  # Train the model


# Make predictions
y_pred = svr.predict(X_test)
# # Evaluate the model
mse = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mse:.2f}")

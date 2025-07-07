# import os
# import seaborn as sns
# import matplotlib.pyplot as plt
# from datetime import datetime
# import optuna
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, InputLayer
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import (
#     EarlyStopping,
#     ReduceLROnPlateau,
#     ModelCheckpoint,
#     TensorBoard,
# )

# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, confusion_matrix

# # ─── Data Prep ────────────────────────────────────────────────────────────────────
# titanic = sns.load_dataset('titanic')
# X = titanic.drop(
#     ['survived', 'deck', 'embark_town', 'alive', 'who', 'embarked'],
#     axis=1
# )
# X = X.select_dtypes(include=['int64', 'float64']).fillna(0)
# y = titanic['survived']

# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# def objective(trial):
#     units = trial.suggest_int('units', 32, 128, step=16) # 32, 48, 64, 80, 96, 112, 128
#     learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
#     optimizers = trial.suggest_categorical(
#         'optimizer', ['adam', 'sgd', 'rmsprop']
#     )
#     batch_size = trial.suggest_int('batch_size', 8,128, step=16)
#     layer_amount = trial.suggest_int('layer_amount', 1, 7, step=1)  
#     activation = trial.suggest_categorical(
#         'activation', ['relu', 'tanh', 'sigmoid']
#     )
#     # Build model
#     model = Sequential()
#     model.add(InputLayer(input_shape=(X_train.shape[1],)))

#     for _ in range(layer_amount):
#         model.add(Dense(units, activation=activation))

#     model.add(Dense(1, activation='sigmoid'))
    
#     if optimizers == 'adam':
#         optimizer = Adam(learning_rate=learning_rate)
#     elif optimizers == 'sgd':
#         from tensorflow.keras.optimizers import SGD
#         optimizer = SGD(learning_rate=learning_rate)
#     elif optimizers == 'rmsprop':
#         from tensorflow.keras.optimizers import RMSprop
#         optimizer = RMSprop(learning_rate=learning_rate)
    
#     model.compile(
#         optimizer=optimizer,
#         loss='binary_crossentropy',
#         metrics=['accuracy']
#     )
    
#     history = model.fit(
#         X_train, y_train,
#         epochs=10,
#         batch_size=batch_size,
#         validation_split=0.2,
#         verbose=0
#     )
    
#     val_accuracy = history.history['val_accuracy']
#     return val_accuracy[-1]  # Return the last validation accuracy

# # ─── Optuna ──────────────────────────────────────────────────────────────────────
# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=30, n_jobs=-1)

# print("Best trial:")
# trial = study.best_trial
# print(f"  Value: {trial.value}")
# print("  Params:")
# for key, value in trial.params.items():
#     print(f"    {key}: {value}")
# # ─── Model ───────────────────────────────────────────────────────────────────────
# # # Uncomment the following lines to build and train the model using the best parameters found by Optuna
# best_params = study.best_params
# units = best_params['units']
# learning_rate = best_params['learning_rate']
# optimizer_name = best_params['optimizer']
# batch_size = best_params['batch_size']
# layer_amount = best_params['layer_amount']
# activation = best_params['activation']  # Not used in the final model, but can be added if needed

# # ─── Build Model ─────────────────────────────────────────────────────────────────
# model = Sequential()
# model.add(InputLayer(input_shape=(X_train.shape[1],)))

# for _ in range(layer_amount):
#     model.add(Dense(units, activation=activation))

# model.add(Dense(1, activation='sigmoid'))

# if optimizer_name == 'adam':
#     optimizer = Adam(learning_rate=learning_rate)
# elif optimizer_name == 'sgd':
#     from tensorflow.keras.optimizers import SGD
#     optimizer = SGD(learning_rate=learning_rate)
# elif optimizer_name == 'rmsprop':
#     from tensorflow.keras.optimizers import RMSprop
#     optimizer = RMSprop(learning_rate=learning_rate)
    
# model.compile(
#     optimizer=optimizer,
#     loss='binary_crossentropy',
#     metrics=['accuracy']
# )
# model.summary()

# # ─── Callbacks ───────────────────────────────────────────────────────────────────
# timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
# log_dir = os.path.join("logs", timestamp)

# callbacks = [
#     EarlyStopping(
#         monitor='val_loss', patience=7, restore_best_weights=True
#     ),
#     ReduceLROnPlateau(
#         monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5
#     ),
#     ModelCheckpoint(
#         filepath='best_model.keras',
#         monitor='val_loss',
#         save_best_only=True
#     ),
#     TensorBoard(
#         log_dir=log_dir,
#         histogram_freq=1,
#         write_graph=True,
#         write_images=True
#     ),
# ]

# # ─── Train ───────────────────────────────────────────────────────────────────────
# history = model.fit(
#     X_train,
#     y_train,
#     epochs=50,
#     batch_size=batch_size,
#     validation_split=0.2,
#     callbacks=callbacks,
# )

# # ─── Plot ────────────────────────────────────────────────────────────────────────
# plt.plot(history.history['accuracy'], label='train_acc')
# plt.plot(history.history['val_accuracy'], label='val_acc')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()

# # ─── Evaluate ────────────────────────────────────────────────────────────────────
# preds = (model.predict(X_test) > 0.5).astype(int)
# print(classification_report(y_test, preds))
# print(confusion_matrix(y_test, preds))

# # ─── Launch TensorBoard ─────────────────────────────────────────────────────────
# # In Jupyter:
# #   %load_ext tensorboard
# #   %tensorboard --logdir logs
# #
# # In a terminal (if running as a script):
# #   tensorboard --logdir logs --host=localhost --port=6006


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from sklearn.datasets import load_diabetes

# load a simple regression dataset
data = load_diabetes()
X, y = data.data, data.target
# make a dataframe
# import pandas as pd
# df = pd.DataFrame(X, columns=data.feature_names)
# df['target'] = y
# # print first 5 rows
# print(df.head())

# build model
model = Sequential()
model.add(Input(shape=(X.shape[1],)))         # 10 inputs
model.add(Dense(8, activation='relu'))        # hidden layer
model.add(Dense(1, activation='linear'))      # output layer

# compile & train
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=100, verbose=1)

# quick check
print(model.predict(X[:5]))
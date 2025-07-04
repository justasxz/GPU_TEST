import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, InputLayer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
 
# iris = sns.load_dataset('iris')
# X = iris.drop('species', axis=1)
# y = iris['species'].map({'setosa': 0, 'versicolor': 1, 'virginica': 2})
titanic = sns.load_dataset('titanic')
X = titanic.drop(['survived', 'deck', 'embark_town', 'alive', 'who', 'embarked'], axis=1)
X = X.select_dtypes(include=['int64', 'float64']).fillna(0)
y = titanic['survived']
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
 
from sklearn.model_selection import train_test_split
X, X_test, y, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# model = Sequential()
# model.add(InputLayer(shape=(X.shape[1],)))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# # idedame output layer with 3 neurons for the 3 species
# model.add(Dense(1, activation='sigmoid'))  
 
# early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
# rlopl = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001)
# model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)

# model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
# load_model
model = load_model('best_model.keras')
model.summary()
# history = model.fit(X, y, epochs=50, batch_size=16, validation_split=0.2, callbacks=[early_stopping,rlopl, model_checkpoint])
# plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label='val_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()
# # print last val accuracy
# print(f"Last validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
# predictions
predictions = model.predict(X_test)
# convert predictions to binary
predictions = (predictions > 0.5).astype(int)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
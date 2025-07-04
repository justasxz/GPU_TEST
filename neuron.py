import os
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
    TensorBoard,
)

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ─── Data Prep ────────────────────────────────────────────────────────────────────
titanic = sns.load_dataset('titanic')
X = titanic.drop(
    ['survived', 'deck', 'embark_town', 'alive', 'who', 'embarked'],
    axis=1
)
X = X.select_dtypes(include=['int64', 'float64']).fillna(0)
y = titanic['survived']

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ─── Build Model ─────────────────────────────────────────────────────────────────
model = Sequential([
    InputLayer(input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid'),
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
model.summary()

# ─── Callbacks ───────────────────────────────────────────────────────────────────
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join("logs", timestamp)

callbacks = [
    EarlyStopping(
        monitor='val_loss', patience=7, restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5
    ),
    ModelCheckpoint(
        filepath='best_model.keras',
        monitor='val_loss',
        save_best_only=True
    ),
    TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=True
    ),
]

# ─── Train ───────────────────────────────────────────────────────────────────────
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.2,
    callbacks=callbacks,
)

# ─── Plot ────────────────────────────────────────────────────────────────────────
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# ─── Evaluate ────────────────────────────────────────────────────────────────────
preds = (model.predict(X_test) > 0.5).astype(int)
print(classification_report(y_test, preds))
print(confusion_matrix(y_test, preds))

# ─── Launch TensorBoard ─────────────────────────────────────────────────────────
# In Jupyter:
#   %load_ext tensorboard
#   %tensorboard --logdir logs
#
# In a terminal (if running as a script):
#   tensorboard --logdir logs --host=localhost --port=6006

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import tensorflow as tf


# def mish(x):
#     return x * tf.math.tanh(tf.math.softplus(x))

# Charger les données
data = pd.read_csv('./Pima_Indians_Diabetes.csv')

# Pré-traitement des données
train_data = data.drop('Outcome', axis=1)
train_data = train_data.replace(0, np.nan)
eval_data = data['Outcome']
column_means = train_data.mean(skipna=True)
train_data = train_data.fillna(column_means)

X_train, X_test, y_train, y_test = train_test_split(train_data, eval_data, test_size=0.2, random_state=42)

# Normalisation des données
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Construire le modèle de réseau de neurones
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=8))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# model.add(Dense(128, activation=mish))
# model.add(BatchNormalization())
# model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))

# Compiler le modèle
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Entraîner le modèle avec early stopping pour éviter le surapprentissage
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1, validation_split=0.1, callbacks=[early_stopping])

# Évaluer les performances du modèle
_, train_accuracy = model.evaluate(X_train, y_train)
_, test_accuracy = model.evaluate(X_test, y_test)

print(f'Train accuracy: {train_accuracy * 100:.2f}%')
print(f'Test accuracy: {test_accuracy * 100:.2f}%')

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Abrir el fichero de imagenes en modo binario
f = open('./train-images-idx3-ubyte/train-images.idx3-ubyte', 'rb')
f_labels = open('./train-labels-idx1-ubyte/train-labels.idx1-ubyte', 'rb')

# Descartar cabecera (16 bytes)
f.read(16)

# El resto de datos son las imagenes.
# La intensidad en escala de grises para cada pixel se guarda en un byte
# Las imagenes son de 28 * 28 por lo que ocupan 784 bytes adyacentes
images = []
'''while True:
    # Leemos la siguiente imagen
    image = f.read(784)
    if len(image) != 784: # Si no queda imagen parar
        break
    else: # Conversion de binario a entero[0,1]
        images.append([x/255 for x in image])

f_labels.read(8) # Descartar cabecera (8 bytes)

labels = []
while True:
    label = f_labels.read(1)
    if len(label) != 1:
        break
    else:
        labels.append(int.from_bytes(label, byteorder='big'))'''

(images, labels), (imagesT, labelsT) = keras.datasets.mnist.load_data()

# Normalizar a [0,1]
images = images.astype("float32") / 255.0
imagesT = imagesT.astype("float32") / 255.0

# Reshape para añadir el canal de color (Train Y Test)
images = np.expand_dims(images, -1)
imagesT = np.expand_dims(imagesT, -1)

# Aplanar imágenes (28x28 a 784)

print(f'Numero de labels: {len(labels)}')

# Conversion a array de NumPy
images = np.array(images)

# crear modelo
def create_mlp():
    model = models.Sequential([
        layers.RandomRotation(0.05, input_shape=(28,28,1)),
        layers.RandomZoom(0.05),
        layers.RandomTranslation(height_factor=0.05, width_factor=0.05),

        # --- Bloque 1: Aprendizaje de bordes y formas simples ---
        # Doble convolución para capturar más detalles antes de comprimir
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)), 
        layers.Dropout(0.20), # Dropout ligero al principio

        # --- Bloque 2: Aprendizaje de formas complejas ---
        # Aumentamos filtros a 64
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3), # Dropout medio

        # --- Bloque 3: Detalles finos (Opcional, pero ayuda al 99.5%) ---
        layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Dropout(0.4), # Dropout más alto en capas profundas
        
        layers.Flatten(),

        # --- Clasificador ---
        layers.Dense(128, activation='relu', kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Dropout(0.5), # Dropout fuerte antes de la decisión final

        layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001), # Algoritmo de optimización
        loss='sparse_categorical_crossentropy', # Función de pérdida
        metrics=[ # Métricas durante entrenamiento,
                  # aquí solo he puesto una pero siempre debéis usar varias
                  # y utilizar la matriz de confusión para ver problemas específicos de clases
                  # en problemas de clasificación
            'accuracy'
        ]
    )
    return model

#Hacer split de entrenamiento y validación
im_train_split, im_val, lab_train_split, lab_val = train_test_split(
    images, labels,
    test_size=0.2,
    random_state=42,
    stratify=labels  # mantiene proporción de clases
)

early_stopper = EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=5, min_lr=1e-6)

# Configurando epocas
model = create_mlp()
history = model.fit(
    im_train_split, lab_train_split, # Partición de entrenamiento
    validation_data=(im_val, lab_val), # Partición de validación
    epochs=100, # Número de épocas de entrenamiento (veces que se pasa por el conjunto de datos entero)
    batch_size=128, # Número de muestras que se procesan a la vez antes de hacer el paso hacia atrás
    verbose=2,
    callbacks=[early_stopper, reduce_lr]
)

val_metrics = model.evaluate(im_val, lab_val, verbose=0)
metric_names = model.metrics_names
results = dict(zip(metric_names, val_metrics))

print("\n🔹 Métricas finales de validación:")
for k, v in results.items():
    print(f"{k:>10s}: {v:.4f}")

# Matriz de confusión y reporte de clasificación
y_val_pred = np.argmax(model.predict(im_val), axis=1)

print("\n Classification Report (Validación):")
print(classification_report(lab_val, y_val_pred, digits=4))

cm = confusion_matrix(lab_val, y_val_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Matriz de confusión - Validación")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.show()

plt.figure(figsize=(14,6))

plt.subplot(1,3,1)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Evolución de la pérdida')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.legend()


plt.tight_layout()
plt.show()

print(model.summary())

#model.save_weights('pesosMdl_final.weights.h5')

val_metrics = model.evaluate(imagesT, labelsT, verbose=0)
metric_names = model.metrics_names
results = dict(zip(metric_names, val_metrics))

print("\n🔹 Métricas finales de validación:")
for k, v in results.items():
    print(f"{k:>10s}: {v:.4f}")

# Matriz de confusión y reporte de clasificación
y_test_pred = np.argmax(model.predict(imagesT), axis=1)

print("\n Classification Report (Validación):")
print(classification_report(labelsT, y_test_pred, digits=4))

cm = confusion_matrix(labelsT, y_test_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Matriz de confusión - Test")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.show()

print(''.join(list(map(str,np.argmax(model.predict(imagesT),axis=1)))))
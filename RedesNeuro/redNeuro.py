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

# Aplanar imágenes (28x28 a 784)
images = images.reshape(-1, 784)

print(f'Numero de labels: {len(labels)}')

# Conversion a array de NumPy
images = np.array(images)

# crear modelo
def create_mlp():
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(784,)), # Primera Capa oculta
        layers.Dropout(0.2),

        layers.Dense(256, activation='relu'), # Segunda Capa oculta
        layers.Dropout(0.3),

        layers.Dense(128, activation='relu'), # Tercera Capa oculta
        layers.Dropout(0.2),

        layers.Dense(256, activation='relu'), # Cuarta Capa oculta
        layers.Dropout(0.25),

        layers.Dense(128, activation='relu'), # Quinta Capa oculta
        layers.Dropout(0.15),

        layers.Dense(10, activation='softmax') # Capa de Salida
    ])
    
    model.compile(
        optimizer=optimizers.AdamW(learning_rate=0.001), # Algoritmo de optimización
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

early_stopper = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=3, min_lr=0.0001)

# Configurando epocas
model = create_mlp()
history = model.fit(
    im_train_split, lab_train_split, # Partición de entrenamiento
    validation_data=(im_val, lab_val), # Partición de validación
    epochs=40, # Número de épocas de entrenamiento (veces que se pasa por el conjunto de datos entero)
    batch_size=128, # Número de muestras que se procesan a la vez antes de hacer el paso hacia atrás
    verbose=2,
    callbacks=[early_stopper,reduce_lr]
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

#Enseña la imagen

"""
# Muestra por pantalla el primer numero para comprobar
plt.imshow(images[1].reshape(28,28), cmap='gray')

plt.show()
"""
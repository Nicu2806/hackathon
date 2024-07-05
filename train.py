import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Setările de antrenament
img_width, img_height = 224, 224
train_data_dir = '/home/drumea/Desktop/Hackathon/train/dataset'  # Actualizează cu calea către directorul tău de date
batch_size = 16
epochs = 5
fine_tune_epochs = 10
total_epochs = epochs + fine_tune_epochs
model_save_path = "/home/drumea/Desktop/Hackathon/train"

# Inițializarea generatorului de date cu preprocesare și împărțire pentru validare
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2)  # 20% din date vor fi folosite pentru validare

# Generator pentru datele de antrenament
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training')  # Set de antrenament

# Generator pentru datele de validare
validation_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation')  # Set de validare

# Încărcarea MobileNetV2 ca bază de model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Construirea modelului personalizat
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Antrenarea doar a straturilor adăugate; straturile din MobileNetV2 sunt înghețate
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks pentru oprirea antrenării dacă val_loss nu se îmbunătățește
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=1, mode='min', min_lr=0.00001)

# Antrenarea modelului (etapa inițială)
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    callbacks=[early_stopping, reduce_lr])

# Fine-tuning
# "Dezghețăm" straturile superioare din MobileNetV2
for layer in base_model.layers[-250:]:  # Ajustează acest număr în funcție de necesitățile tale
    layer.trainable = True

# Recompilăm modelul (este necesar după modificarea setării trainable ale straturilor)
model.compile(optimizer=Adam(learning_rate=0.00001),  # Folosim un learning rate mai mic pentru fine-tuning
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Continuăm antrenarea modelului (etapa de fine-tuning)
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=total_epochs,
    initial_epoch=epochs,  # Continuăm de unde am rămas
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    callbacks=[early_stopping, reduce_lr])

# Salvăm modelul în calea specificată
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

model.save(os.path.join(model_save_path, 'diases.keras'))
print(f"Modelul a fost salvat la: {os.path.join(model_save_path, 'diases.keras')}")

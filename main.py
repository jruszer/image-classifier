import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix)


image_dir = 'cat_dog'
labels_path = 'cat_dog.csv'

df = pd.read_csv(labels_path)
df['filepath'] = df['image'].apply(lambda x: os.path.join(image_dir, x))
df['labels'] = df['labels'].astype(str)

IMG_SHAPE = (224, 224) 
BATCH = 32 
EPOCHS = 3 


def build_model():
    base = MobileNetV2(weights='imagenet',
                       include_top=False,
                       input_shape=(*IMG_SHAPE, 3))
    base.trainable = False

    x = base.output
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    out = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base.input, outputs=out)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


testowy = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]

for i, test_size in enumerate(testowy, start=1):
    print(f"\n\n-------------- {i}. ZBIOR TESTOWY = {test_size:.1f} ---------------\n")

    train_df, val_df = train_test_split(df, test_size=test_size, stratify=df['labels'], random_state=42)

    train_gen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    ).flow_from_dataframe(
        dataframe=train_df,
        x_col='filepath',
        y_col='labels',
        target_size=IMG_SHAPE,
        batch_size=BATCH,
        class_mode='binary',
        seed=42
    )

    val_gen = ImageDataGenerator(rescale=1. / 255).flow_from_dataframe(
        dataframe=val_df,
        x_col='filepath',
        y_col='labels',
        target_size=IMG_SHAPE,
        batch_size=BATCH,
        class_mode='binary',
        seed=42,
        shuffle=False
    )

    tf.keras.backend.clear_session()  
    model = build_model()
    model.fit(
        train_gen,
        epochs=EPOCHS,
        steps_per_epoch=train_gen.samples // BATCH,
        validation_data=val_gen,
        validation_steps=val_gen.samples // BATCH,
        verbose=0  # tutaj jak dla 1 to zmienić na 1 bo widać pasek postepu
    )

    val_gen.reset() 
    preds = model.predict(val_gen, verbose=0).flatten() 
    y_pred = (preds > 0.5).astype(int) 
    y_true = val_gen.classes 
    
    prec = precision_score(y_true, y_pred, average=None)
    rec = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)

    acc = accuracy_score(y_true, y_pred)
    prec_avg = precision_score(y_true, y_pred)
    rec_avg = recall_score(y_true, y_pred)
    f1_avg = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print(f"Dokładność  (Accuracy): {acc:.4f}")
    print(f"Precyzja    (avg)     : {prec_avg:.4f}")
    print(f"Czułość     (avg)     : {rec_avg:.4f}")
    print(f"F1-score    (avg)     : {f1_avg:.4f}\n")

    print("Precyzja / Czułość / F1 na klasy:")
    print(f"  Kot – precision: {prec[0]:.4f} | recall: {rec[0]:.4f} | F1: {f1[0]:.4f}")
    print(f"  Pies – precision: {prec[1]:.4f} | recall: {rec[1]:.4f} | F1: {f1[1]:.4f}\n")

    print("Macierz Pomyłek (Confusion Matrix):")
    print("            Przewidziane: Kot   Przewidziane: Pies")
    print(f"Prawdziwe: Kot     {cm[0][0]:>10}                 {cm[0][1]:>10}")
    print(f"Prawdziwe: Pies    {cm[1][0]:>10}                 {cm[1][1]:>10}")

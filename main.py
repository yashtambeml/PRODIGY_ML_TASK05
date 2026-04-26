# ==========================================
# 🍽️ INDIAN FOOD CLASSIFIER (IMPROVED UI)
# ==========================================

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog
import threading

# ==========================================
# 📂 PATHS
# ==========================================
DATASET_PATH = "Indian Food Images"
MODEL_PATH = "indian_food_model.keras"

# ==========================================
# 🧠 CLASSES
# ==========================================
classes = sorted(os.listdir(DATASET_PATH))
np.save("classes.npy", classes)

# ==========================================
# 🧠 DATA GENERATOR
# ==========================================
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# ==========================================
# 🧠 MODEL
# ==========================================
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(len(classes), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ==========================================
# 🚀 TRAIN / LOAD MODEL
# ==========================================
if not os.path.exists(MODEL_PATH):
    print("🚀 Training model...")
    model.fit(train_data, validation_data=val_data, epochs=10)
    model.save(MODEL_PATH)
else:
    print("✅ Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)

classes = np.load("classes.npy", allow_pickle=True)

# ==========================================
# 🎯 PREDICTION (SINGLE OUTPUT ONLY)
# ==========================================
def predict_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((224,224))

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)

    return classes[class_index]

# ==========================================
# 🖥️ IMPROVED GUI APP (ONLY UI UPGRADE)
# ==========================================
def upload_image():
    file_path = filedialog.askopenfilename()

    if not file_path:
        return

    img = Image.open(file_path)
    img = img.resize((250, 250))
    img = ImageTk.PhotoImage(img)

    panel.config(image=img)
    panel.image = img

    result = predict_image(file_path)

    # 🎯 Better styled output
    label.config(
        text=f"🍽️ Predicted Food:\n{result}",
        bg="#ffffff",
        fg="#2e7d32",
        font=("Arial", 16, "bold"),
        padx=10,
        pady=10,
        relief="solid",
        bd=2
    )


# ==========================================
# 🪟 UI DESIGN (IMPROVED LOOK)
# ==========================================
root = tk.Tk()
root.title("🍛 Indian Food Classifier")
root.geometry("420x600")
root.configure(bg="#f0f0f0")

# Title
title = tk.Label(
    root,
    text="Indian Food Classifier 🍽️",
    font=("Arial", 20, "bold"),
    bg="#f0f0f0",
    fg="#333"
)
title.pack(pady=15)

# Upload Button
btn = tk.Button(
    root,
    text="📤 Upload Image",
    command=upload_image,
    font=("Arial", 12, "bold"),
    bg="#4CAF50",
    fg="white",
    padx=10,
    pady=5
)
btn.pack(pady=10)

# Image Panel (with better spacing)
panel = tk.Label(
    root,
    bg="#f0f0f0",
    bd=2,
    relief="groove"
)
panel.pack(pady=15)

# Result Label (modern output box)
label = tk.Label(
    root,
    text="Prediction will appear here",
    font=("Arial", 14),
    bg="#ffffff",
    fg="#555",
    padx=10,
    pady=10,
    relief="solid",
    bd=1
)
label.pack(pady=20, fill="x", padx=20)

root.mainloop()
# ==========================================
# 🪟 UI DESIGN
# ==========================================
root = tk.Tk()
root.title("🍛 Indian Food Classifier")
root.geometry("420x600")
root.configure(bg="#f5f5f5")

title = tk.Label(root, text="Indian Food Classifier 🍽️", font=("Arial", 18, "bold"), bg="#f5f5f5")
title.pack(pady=10)

btn = tk.Button(root, text="📤 Upload Image", command=upload_image, font=("Arial", 12), bg="#4CAF50", fg="white")
btn.pack(pady=10)

panel = tk.Label(root, bg="#f5f5f5")
panel.pack(pady=10)

label = tk.Label(root, text="Prediction will appear here", font=("Arial", 12), bg="#f5f5f5", justify="left")
label.pack(pady=20)

root.mainloop()

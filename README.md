# 🍛 Indian Food Classifier

A desktop application that classifies Indian food images using MobileNetV2 and a simple Tkinter GUI.

## 🚀 Features
- Upload an image and get instant prediction  
- Clean and user-friendly interface  
- Uses pre-trained MobileNetV2 (Transfer Learning)  
- Automatic model saving & loading  

## 🛠️ Tech Stack
- Python  
- TensorFlow / Keras  
- Tkinter (GUI)  
- PIL  
- NumPy  

## 📂 Project Structure
Indian-Food-Classifier/
│── main.py
│── indian_food_model.keras
│── classes.npy
│── Indian Food Images/

## ⚙️ Installation
pip install tensorflow pillow numpy

## ▶️ Run
python main.py

## 📷 How It Works
1. Loads dataset from folder  
2. Trains model if not already saved  
3. Upload image via GUI  
4. Predicts food category  

## ⚠️ Notes
- Dataset must be in folders (one folder per class)  
- First run may take time due to training  

## 🔮 Future Improvements
- Add webcam detection  
- Improve accuracy  
- Convert to web app  

## 👨‍💻 Author
Yash Tambe 🚀

# Deep Fake Detection Using Images

## 📌 Overview
Deep Fake Detection is a machine learning project that identifies fake (AI-generated) and real images using deep learning techniques. The model is trained on a dataset of real and fake images to classify them accurately.

## 🏗 Features
- Uses **Convolutional Neural Networks (CNNs)** for image classification.
- Implements **TensorFlow/Keras** for deep learning.
- Supports **real-time image detection**.
- Provides **visualizations** for model accuracy and loss.

## 📂 Dataset
The dataset consists of:
- **Real images**: Authentic human faces.
- **Fake images**: AI-generated deepfake faces.

Dataset Directory Structure:
```
D:/real-vs-fake/
│── train/
│   ├── real/
│   ├── fake/
│── valid/
│   ├── real/
│   ├── fake/
│── test/
│   ├── real/
│   ├── fake/
```

## 🔧 Installation
1. Clone the repository:
   ```sh
  
   cd deepfake-detection
   ```
2. Create and activate a virtual environment:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   venv\Scripts\activate  # On Windows
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## 🚀 Training the Model
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# Data Augmentation
image_gen = ImageDataGenerator(rescale=1./255)
train_flow = image_gen.flow_from_directory('D:/real-vs-fake/train/', target_size=(224, 224), batch_size=64, class_mode='binary')
valid_flow = image_gen.flow_from_directory('D:/real-vs-fake/valid/', target_size=(224, 224), batch_size=64, class_mode='binary')

# Model Architecture
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile Model
opt = Adam(learning_rate=0.0001)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

# Train Model
model.fit(train_flow, epochs=10, validation_data=valid_flow)
```

## 🧪 Testing the Model
```python
test_flow = image_gen.flow_from_directory('D:/real-vs-fake/test/', target_size=(224, 224), batch_size=1, shuffle=False, class_mode='binary')
model.evaluate(test_flow)
```

## 📊 Results
- Performance metrics visualized using `matplotlib`.

## 📜 License
This project is open-source and available under the **MIT License**.

## 🤝 Contributing
Feel free to **fork** this repository, create a new branch, and submit a **pull request**!

## 📬 Contact
For questions or suggestions, reach out via [your.email@example.com](mailto:your.email@example.com) or open an issue on GitHub.

---
🚀 **Happy Coding!**


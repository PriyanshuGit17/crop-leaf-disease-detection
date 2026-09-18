# 🌱 Crop Leaf Disease Detection

A **CNN-based crop leaf disease detection web application** built with **Python, TensorFlow/Keras, Flask, and HTML/CSS/JavaScript**.

The application allows users to upload a leaf image and receive a predicted disease class, prediction probability, infection-level indicator, treatment suggestion, and top predictions through a simple web interface.

## 🚀 Features

* 🌿 Leaf disease classification using a **Convolutional Neural Network (CNN)**
* 🖼️ Upload and preview leaf images
* 🤖 TensorFlow/Keras-based image classification
* 🔌 Flask REST API for predictions
* 📊 Prediction probability and top-3 predictions
* 🩺 Rule-based infection-level indicator
* 💡 Disease-specific treatment suggestions
* 💚 Rule-based healing score indicator
* 🌐 Simple HTML/CSS/JavaScript frontend
* 🔄 Image preprocessing and normalization
* 📁 Saved trained model and class-index mapping

## 🛠️ Technologies Used

* **Python**
* **TensorFlow / Keras**
* **Flask**
* **Flask-CORS**
* **NumPy**
* **Pillow (PIL)**
* **HTML5**
* **CSS3**
* **JavaScript**
* **CNN (Convolutional Neural Network)**

## 🧠 Machine Learning Model

The project uses a CNN for multiclass image classification.

### Model Architecture

```text
Input Image
128 × 128 × 3
      ↓
Conv2D (32 filters)
      ↓
MaxPooling2D
      ↓
Conv2D (64 filters)
      ↓
MaxPooling2D
      ↓
Conv2D (128 filters)
      ↓
MaxPooling2D
      ↓
Flatten
      ↓
Dense (128 neurons)
      ↓
Dropout (0.4)
      ↓
Softmax Output
```

### Training Configuration

| Parameter         | Value                        |
| ----------------- | ---------------------------- |
| Image Size        | 128 × 128                    |
| Batch Size        | 32                           |
| Epochs            | 10                           |
| Optimizer         | Adam                         |
| Loss Function     | Categorical Crossentropy     |
| Output            | Multiclass Softmax           |
| Validation Split  | 20%                          |
| Data Augmentation | Shear, Zoom, Horizontal Flip |
| Pixel Scaling     | 1/255                        |

## 📂 Project Structure

```text
crop-leaf-disease-detection/
│
├── leaf dataset/
│   └── train/
│       ├── Class 1/
│       ├── Class 2/
│       └── ...
│
├── app.py                    # Flask API
├── load.py                   # Model loading and prediction logic
├── train_model.py            # CNN training script
├── index.html                # Web interface
├── leaf_disease_model.h5     # Trained CNN model
├── class_indices.json        # Class-to-index mapping
├── .gitignore
├── .gitattributes
└── README.md
```

## ⚙️ How It Works

```text
User uploads leaf image
          ↓
     HTML Frontend
          ↓
      Flask API
          ↓
 Image preprocessing
  (128 × 128 + normalization)
          ↓
      CNN Model
          ↓
    Class prediction
          ↓
 Prediction confidence
          ↓
Disease + infection level
+ suggestions + top predictions
          ↓
      Web Interface
```

## 💻 Installation

### 1. Clone the repository

```bash
git clone https://github.com/PriyanshuGit17/crop-leaf-disease-detection.git
cd crop-leaf-disease-detection
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install tensorflow flask flask-cors pillow numpy
```

### 4. Verify the required files

Make sure these files are present:

```text
leaf_disease_model.h5
class_indices.json
app.py
load.py
index.html
```

## ▶️ Running the Application

### Start the Flask backend

Open a terminal in the project folder and run:

```bash
python app.py
```

The Flask server will start at:

```text
http://127.0.0.1:5000
```

### Open the frontend

Open `index.html` in your web browser.

Then:

1. Click **Upload Image**
2. Select a crop leaf image
3. Click **Predict**
4. View the prediction results

## 🔌 API

The application provides a prediction endpoint:

```text
POST /predict
```

### Request

Send a leaf image using the `file` form field.

Example:

```text
POST http://127.0.0.1:5000/predict
```

### Response

The API returns information such as:

```json
{
  "top_class": "Apple___Black_rot",
  "probability": "85.42%",
  "infection_level": "🔴 Severe Infection",
  "suggestion": "Use copper fungicide and prune infected black spots.",
  "advice": "Use copper fungicide and prune infected black spots. Prune infected parts and apply copper fungicide.",
  "healing_score": 20,
  "all_results": []
}
```

## 📊 Prediction Output

The application displays:

* **Predicted disease/class**
* **Prediction probability**
* **Infection-level indicator**
* **Treatment suggestion**
* **Additional advice**
* **Healing score indicator**
* **Top 3 predictions**

### Infection-Level Logic

The current implementation uses prediction probability thresholds:

| Prediction Probability | Indicator             |
| ---------------------- | --------------------- |
| `< 30%`                | 🟢 Very Low / Healthy |
| `30% – <60%`           | 🟡 Early Stage        |
| `60% – <80%`           | 🟠 Moderate Infection |
| `≥ 80%`                | 🔴 Severe Infection   |

> **Note:** The infection level and healing score are rule-based indicators created for this project. They are derived from model prediction confidence and should not be considered a scientifically validated measure of disease severity or plant recovery.

## 🗃️ Dataset

The training script expects the dataset to follow a directory-based class structure:

```text
leaf_dataset/
└── train/
    ├── Class_Name_1/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    │
    ├── Class_Name_2/
    │   ├── image1.jpg
    │   └── ...
    │
    └── ...
```

Keras automatically assigns class labels based on the directory structure.

The trained class mapping is stored in:

```text
class_indices.json
```

## 🔬 Model Training

To train the model again:

```bash
python train_model.py
```

The script:

1. Loads images from the dataset.
2. Resizes images to 128 × 128 pixels.
3. Applies image augmentation.
4. Creates training and validation sets.
5. Builds the CNN architecture.
6. Trains the model for 10 epochs.
7. Saves the trained model.
8. Saves the class-index mapping.

Generated files:

```text
leaf_disease_model.h5
class_indices.json
```

## 📌 Current Limitations

* The application currently runs locally through Flask.
* The frontend is configured to communicate with `127.0.0.1:5000`.
* Model performance depends on the quality and variety of the training dataset.
* The current infection-level system is based on prediction probability rather than a dedicated disease-severity model.
* The healing score is a rule-based project indicator and is not a scientifically validated recovery metric.
* Disease-treatment suggestions are currently defined using a predefined dictionary in `load.py`.

## 🔮 Future Improvements

* 📈 Add model accuracy and loss visualization
* 🧪 Improve dataset quality and class balance
* 🧠 Experiment with transfer learning models such as MobileNet, EfficientNet, or ResNet
* 🎯 Add a dedicated disease-severity classification system
* 📱 Create a responsive mobile-friendly interface
* ☁️ Deploy the Flask application online
* 🔐 Add API validation and error handling
* 📷 Support camera-based leaf image capture
* 📊 Add prediction history
* 🌐 Add support for more crops and diseases

## 🎯 Project Purpose

This project demonstrates the practical use of **machine learning, computer vision, deep learning, REST APIs, and web development** to build an end-to-end image classification application.

It covers the complete workflow:

```text
Dataset
   ↓
Data Preprocessing
   ↓
CNN Training
   ↓
Model Saving
   ↓
Flask API
   ↓
Web Frontend
   ↓
Prediction
```

## 👨‍💻 Author

### Priyanshu Singh

**BCA Graduate | Frontend Developer | Python | Computer Graphics | AI/ML**

* 💻 GitHub: [PriyanshuGit17](https://github.com/PriyanshuGit17)
* 💼 LinkedIn: [Priyanshu Singh](https://www.linkedin.com/in/priyanshu-singh-8644702a9/)
* 📧 Email: [priyanshusingh52509@gmail.com](mailto:priyanshusingh52509@gmail.com)

---

⭐ If you find this project useful, consider giving the repository a star.

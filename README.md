# 🧠 Brain Tumor MRI Image Classification

This project focuses on classifying brain MRI images into four categories — **glioma**, **meningioma**, **pituitary tumor**, and **no tumor** — using a custom-built Convolutional Neural Network (CNN). The trained model is deployed as a real-time web app using **Streamlit**, allowing users to upload images and receive predictions instantly.

---

## 📁 Dataset

The dataset is structured into training, validation, and test sets:

Brain_Tumor_MRI_Dataset/
├── train/
│ ├── glioma/
│ ├── meningioma/
│ ├── no_tumor/
│ └── pituitary/
├── valid/
├── test/
└── Models/


Each subfolder contains labeled MRI scans. A total of over **2,400+** images were used across all sets.

---

## 🚀 Project Pipeline

### 1. **Data Preprocessing**
- Resized all images to **224×224**
- Normalized pixel values
- Created data generators using `ImageDataGenerator`

### 2. **Exploratory Data Analysis**
- Inspected class distributions
- Sample visualizations
- Checked dimensions, formats, and color channels

### 3. **Model Building**
- Built a custom CNN with:
  - 3 convolutional layers
  - Batch normalization & max-pooling
  - Global average pooling
  - Dropout regularization
- Compiled with `categorical_crossentropy` and Adam optimizer

### 4. **Training & Evaluation**
- Achieved **Test Accuracy:** `87.8%`
- Visualized accuracy/loss curves
- Evaluated performance with classification report and confusion matrix

### 5. **Transfer Learning (Explored but Underperformed)**
- Tried EfficientNetB0 and fine-tuning
- Achieved sub-optimal accuracy (~32%)

### 6. **Deployment with Streamlit**
- Web interface for MRI image upload
- Live tumor type prediction
- Displays model confidence scores

---

## 🖥️ Web Application Features

- 📤 Upload any brain MRI image (JPG/PNG)
- 📈 View model prediction and probability distribution
- 💡 Intuitive interface for fast, real-time testing

> ✅ Hosted on: [Streamlit Cloud](https://streamlit.io/cloud)  
> (Replace this with your live link once deployed)

---

## 📦 Folder Structure

BrainTumorClassifierApp/
├── app.py
├── custom_cnn_best.h5
├── requirements.txt
├── runtime.txt
└── .streamlit/
└── config.toml


---

## ⚙️ Setup Instructions

1. **Clone the repo**
```bash
git clone https://github.com/your-username/brain-tumor-classifier.git
cd brain-tumor-classifier
```
2. **Install dependencies**
   pip install -r requirements.txt

3. **Run locally**
   streamlit run app.py
   
4. **Requirements**
* Python 3.10

* TensorFlow 2.14.0

* Streamlit 1.35.0

* Pillow, NumPy

 Author
Siddhartha Ram Konasani
mail: siddharthakonasani.77@gmail.com
Brain Tumor MRI Classifier Project (2025)


---

### 📌 To Use:
- Save this as `README.md` in your GitHub repo root.
- Replace the placeholder deployment URL with your Streamlit app link once it's live.
- You can also attach screenshots or badges (optional).

Let me know if you'd like help converting this to a **PDF report**, **submission ZIP**, or adding **demo screenshots!**

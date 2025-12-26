# 🌸 Flower Detection & Classification

A deep learning-powered web application that classifies flower images using a Convolutional Neural Network (CNN). Upload an image and instantly identify whether it's a daisy, sunflower, tulip, dandelion, or rose with confidence scores.

![Python](https://img.shields.io/badge/Python-3.13-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🌟 Features

- **Real-time Classification**: Upload flower images and get instant predictions
- **5 Flower Types**: Classifies daisy, sunflower, tulip, dandelion, and rose
- **Confidence Scores**: View probability distribution across all classes
- **Modern UI**: Clean, responsive web interface with dark mode design
- **Fast API Backend**: Built with FastAPI for high performance
- **Deep Learning Model**: Custom CNN architecture trained for flower recognition

## 🚀 Demo

Upload any flower image and the model will:
1. Process the image
2. Classify it into one of 5 categories
3. Display confidence percentages for each class

## 🛠️ Tech Stack

### Backend
- **FastAPI** - Modern, fast web framework
- **TensorFlow/Keras** - Deep learning framework
- **Pillow** - Image processing
- **Uvicorn** - ASGI server
- **NumPy** - Numerical computations

### Frontend
- **HTML5** - Structure
- **CSS3** - Modern styling with gradients and animations
- **Vanilla JavaScript** - Interactive functionality

### Model
- **Architecture**: Custom CNN with 4 convolutional layers
- **Input Size**: 64x64 RGB images
- **Layers**: Conv2D, BatchNormalization, MaxPooling, Dropout, Dense
- **Output**: 5-class softmax classification

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/flower_detection_classification.git
cd flower_detection_classification-main
```

2. **Install backend dependencies**
```bash
cd backend
pip install -r requirements.txt
```

3. **Verify model file exists**
Ensure `models/my_model_exp2.h5` is present in the backend directory.

## 🎯 Usage

### Running Locally

1. **Start the Backend Server**
```bash
cd backend
python app.py
```
Backend will run on `http://localhost:8000`

2. **Start the Frontend Server**
```bash
cd frontend
python -m http.server 5500
```
Frontend will run on `http://localhost:5500`

3. **Open your browser**
Navigate to `http://localhost:5500` and start classifying flowers!

### API Endpoints

- **GET** `/health` - Health check endpoint
- **POST** `/predict` - Upload image for classification

Example API usage:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@flower.jpg"
```

Response:
```json
{
  "prediction": "rose",
  "confidence": 0.9523,
  "probabilities": {
    "daisy": 0.0123,
    "sunflower": 0.0089,
    "tulip": 0.0156,
    "dandelion": 0.0109,
    "rose": 0.9523
  }
}
```

## 🧠 Model Architecture

```
Conv2D (32 filters, 5x5) → BatchNorm → MaxPool2D
    ↓
Conv2D (64 filters, 3x3) → BatchNorm → MaxPool2D
    ↓
Conv2D (96 filters, 3x3) → BatchNorm → MaxPool2D
    ↓
Conv2D (96 filters, 3x3) → BatchNorm → MaxPool2D
    ↓
Flatten → Dense(512) → BatchNorm → Dropout(0.5)
    ↓
Dense(5, softmax)
```

**Input Shape**: (64, 64, 3)  
**Output**: 5 classes with probability distribution

## 📊 Supported Flower Classes

1. 🌼 **Daisy**
2. 🌻 **Sunflower**
3. 🌷 **Tulip**
4. 🌼 **Dandelion**
5. 🌹 **Rose**

## 🎨 Frontend Features

- Drag-and-drop image upload
- Image preview before prediction
- Real-time confidence visualization
- Responsive design for all devices
- Modern dark-mode UI with gradients
- Smooth animations and transitions

## 📁 Project Structure

```
flower_detection_classification-main/
├── backend/
│   ├── app.py                 # FastAPI application
│   ├── requirements.txt       # Python dependencies
│   └── models/
│       └── my_model_exp2.h5   # Trained CNN model
├── frontend/
│   └── index.html             # Web interface
├── Copy_of_EXP_FLOWER_CNN_F1.ipynb  # Model training notebook
└── README.md
```

## 🔧 Configuration

### Environment Variables
- `MODEL_PATH` - Path to the model file (default: `models/my_model_exp2.h5`)

### CORS Settings
The backend is configured to accept requests from `http://localhost:5500`. Update in `app.py` if needed:
```python
allow_origins=["http://localhost:5500"]
```

## 🚀 Deployment

### Option 1: Hugging Face Spaces (Recommended for ML)
Free hosting specifically for ML models with unlimited usage.

### Option 2: Render.com
Free tier with auto-deployment from GitHub.

### Option 3: Docker
Build and deploy as a container:
```bash
docker build -t flower-classifier .
docker run -p 8000:8000 flower-classifier
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

Your Name - [@mukesh-843](https://github.com/mukesh-843)

## 🙏 Acknowledgments

- TensorFlow and Keras teams for the amazing framework
- FastAPI for the excellent web framework
- Flower dataset contributors

## 📧 Contact

For questions or feedback, please open an issue or reach out via email.

---

⭐ **Star this repository if you found it helpful!** ⭐

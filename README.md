# Plant Disease Classification API 🌱

A FastAPI-based REST API for classifying plant diseases using a ResNet-9 model trained on the PlantVillage dataset.

## Features

- **Plant Disease Classification**: Classify 38 different plant diseases and healthy plant conditions
- **Multiple Predictions**: Get top 3 predictions with confidence scores
- **Easy to Use**: Simple REST API with image upload
- **Fast Inference**: Optimized ResNet-9 model for quick predictions
- **Comprehensive**: Covers diseases in crops like tomato, potato, corn, apple, grape, and more

## Supported Plant Classes

The model can classify 38 different conditions including:
- **Tomato**: Late blight, Early blight, Leaf mold, Bacterial spot, etc.
- **Apple**: Cedar apple rust, Apple scab, Black rot
- **Potato**: Late blight, Early blight
- **Corn**: Northern leaf blight, Common rust, Cercospora leaf spot
- **Grape**: Black rot, Leaf blight, Esca
- **And many more healthy and diseased plant conditions**

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure Model File Exists**:
   - Make sure `plant-disease-model.pth` is in the same directory as `app.py`

3. **Run the API**:
   ```bash
   python app.py
   ```
   
   Or using uvicorn directly:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```

## API Endpoints

### 1. Root Endpoint
- **GET** `/`
- Returns API information and available endpoints

### 2. Health Check
- **GET** `/health`
- Check if the API and model are loaded properly

### 3. Get Classes
- **GET** `/classes`
- Returns list of all 38 supported plant disease classes

### 4. Predict Disease
- **POST** `/predict`
- Upload an image to get disease prediction
- **Request**: Multipart form data with image file
- **Response**: JSON with prediction, confidence, and top 3 results

## Usage Examples

### Using curl
```bash
# Health check
curl http://localhost:8000/health

# Get all supported classes
curl http://localhost:8000/classes

# Predict disease from image
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@path/to/your/plant_image.jpg"
```

### Using Python requests
```python
import requests

# Predict disease
with open('plant_image.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/predict', files=files)
    result = response.json()
    print(f"Predicted disease: {result['prediction']['class']}")
    print(f"Confidence: {result['prediction']['confidence']:.2%}")
```

## Model Information

- **Architecture**: ResNet-9 (Custom lightweight ResNet)
- **Dataset**: PlantVillage Dataset (87K+ images)
- **Input Size**: 256x256 RGB images
- **Classes**: 38 plant disease/healthy conditions
- **Accuracy**: ~99% on validation set

## File Structure

```
Backend/
├── app.py                          # FastAPI application
├── plant-disease-model.pth         # Trained model weights
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── plant-disease-classification-resnet-99-2.ipynb  # Training notebook
```

## Response Format

### Successful Prediction
```json
{
  "prediction": {
    "class": "Tomato___Late_blight",
    "class_index": 0,
    "confidence": 0.9876
  },
  "top_predictions": [
    {
      "class": "Tomato___Late_blight",
      "confidence": 0.9876
    },
    {
      "class": "Tomato___Early_blight",
      "confidence": 0.0098
    },
    {
      "class": "Tomato___Septoria_leaf_spot",
      "confidence": 0.0026
    }
  ],
  "filename": "tomato_leaf.jpg"
}
```

## Error Handling

The API includes comprehensive error handling:
- **400 Bad Request**: Invalid image format or file type
- **503 Service Unavailable**: Model not loaded
- **500 Internal Server Error**: Prediction processing errors

## Development

The API is built using:
- **FastAPI**: Modern, fast web framework for building APIs
- **PyTorch**: Deep learning framework for model inference
- **Pillow**: Image processing library
- **Uvicorn**: ASGI server for production deployment

## License

This project is based on the PlantVillage dataset and is intended for educational and research purposes.

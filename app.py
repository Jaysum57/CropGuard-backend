import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from typing import Dict, Any

app = FastAPI(
    title="Plant Disease Classification API",
    description="API for classifying plant diseases using a ResNet-9 model trained on the PlantVillage dataset",
    version="1.0.0"
)

# Plant disease class names from the training notebook (one per line for readability)
CLASS_NAMES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)__Powdery_mildew',
    'Cherry(including_sour)__healthy',
    'Corn(maize)__Cercospora_leaf_spot Gray_leaf_spot',
    'Corn(maize)_Common_rust',
    'Corn(maize)__Northern_Leaf_Blight',
    'Corn(maize)healthy',
    'Grape___Black_rot',
    'Grape___Esca(Black_Measles)',
    'Grape___Leaf_blight(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# ResNet9 and helper block definitions from the notebook
def ConvBlock(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
             nn.BatchNorm2d(out_channels),
             nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss
        
    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = (out.argmax(dim=1) == labels).float().mean()
        return {"val_loss": loss.detach(), "val_accuracy": acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x["val_loss"] for x in outputs]
        batch_accuracy = [x["val_accuracy"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        epoch_accuracy = torch.stack(batch_accuracy).mean()
        return {"val_loss": epoch_loss, "val_accuracy": epoch_accuracy}
        
    def epoch_end(self, epoch, result):
        print(f"Epoch [{epoch}], train_loss: {result['train_loss']:.4f}, val_loss: {result['val_loss']:.4f}, val_acc: {result['val_accuracy']:.4f}")

class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
        self.conv3 = ConvBlock(128, 256, pool=True)
        self.conv4 = ConvBlock(256, 512, pool=True)
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        self.classifier = nn.Sequential(
            nn.MaxPool2d(4),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

# Load the trained model
model = None
load_error = None

def load_model():
    global model, load_error
    try:
        # Method 1: Add all required safe globals for PyTorch modules
        import torch.nn.modules.container
        import torch.nn.modules.conv
        import torch.nn.modules.batchnorm
        import torch.nn.modules.activation
        import torch.nn.modules.pooling
        import torch.nn.modules.flatten
        import torch.nn.modules.linear
        
        safe_globals = [
            ResNet9,
            ConvBlock,
            ImageClassificationBase,
            torch.nn.modules.container.Sequential,
            torch.nn.modules.conv.Conv2d,
            torch.nn.modules.batchnorm.BatchNorm2d,
            torch.nn.modules.activation.ReLU,
            torch.nn.modules.pooling.MaxPool2d,
            torch.nn.modules.flatten.Flatten,
            torch.nn.modules.linear.Linear,
        ]
        
        torch.serialization.add_safe_globals(safe_globals)
        model = torch.load('plant-disease-model.pth', map_location=torch.device('cpu'), weights_only=True)
        model.eval()
        print("Model loaded successfully with safe globals!")
        return
    except Exception as e:
        print(f"Safe globals method failed: {str(e)}")
    
    try:
        # Method 2: Use the safe_globals context manager
        import torch.nn.modules.container
        import torch.nn.modules.conv
        import torch.nn.modules.batchnorm
        import torch.nn.modules.activation
        import torch.nn.modules.pooling
        import torch.nn.modules.flatten
        import torch.nn.modules.linear
        
        safe_globals = [
            ResNet9,
            ConvBlock, 
            ImageClassificationBase,
            torch.nn.modules.container.Sequential,
            torch.nn.modules.conv.Conv2d,
            torch.nn.modules.batchnorm.BatchNorm2d,
            torch.nn.modules.activation.ReLU,
            torch.nn.modules.pooling.MaxPool2d,
            torch.nn.modules.flatten.Flatten,
            torch.nn.modules.linear.Linear,
        ]
        
        with torch.serialization.safe_globals(safe_globals):
            model = torch.load('plant-disease-model.pth', map_location=torch.device('cpu'), weights_only=True)
            model.eval()
            print("Model loaded successfully with safe globals context manager!")
            return
    except Exception as e:
        print(f"Safe globals context manager method failed: {str(e)}")
    
    try:
        # Method 3: Load state dict into new model (cleanest approach)
        print("Attempting to load as state dict...")
        model = ResNet9(3, len(CLASS_NAMES))
        
        # Try to load just the state dict
        checkpoint = torch.load('plant-disease-model.pth', map_location=torch.device('cpu'), weights_only=False)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
        else:
            raise RuntimeError("Checkpoint format not recognized for loading state dict.")
        
        model.eval()
        print("Model loaded successfully as state dict!")
        return
    except Exception as e:
        print(f"State dict method failed: {str(e)}")
    
    try:
        # Method 4: Fallback to weights_only=False (less secure but works)
        print("Using fallback method with weights_only=False...")
        model = torch.load('plant-disease-model.pth', map_location=torch.device('cpu'), weights_only=False)
        model.eval()
        print("Model loaded successfully with weights_only=False (fallback method)!")
        return
    except Exception as e:
        print(f"Fallback method failed: {str(e)}")

    # If all methods fail
    load_error = "Failed to load model with all attempted methods. Please check the model file."
    print(load_error)

# Load the model on startup
load_model()

def transform_image(image_bytes):
    """
    Transform the uploaded image to tensor format expected by the model.
    Based on the notebook, the model expects 256x256 images with ToTensor transform.
    """
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to 256x256 as per training
        transforms.ToTensor()           # Convert to tensor and normalize to [0,1]
    ])
    
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img_tensor = transform(image)
        if isinstance(img_tensor, torch.Tensor):
            return img_tensor.unsqueeze(0)  # Add batch dimension
        else:
            raise ValueError("Transform did not return a tensor")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

def predict_image(img_tensor, model):
    """
    Predict image class using the trained model.
    Based on the predict_image function from the notebook.
    """
    with torch.no_grad():
        # Get predictions from model
        outputs = model(img_tensor)
        _, predicted = outputs.max(1)
        class_idx = predicted.item()
        
        if class_idx < len(CLASS_NAMES):
            class_name = CLASS_NAMES[class_idx]
        else:
            class_name = f"Unknown_Class_{class_idx}"
            
        return class_name, class_idx, outputs

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Plant Disease Classification API",
        "model": "ResNet-9",
        "dataset": "PlantVillage",
        "classes": len(CLASS_NAMES),
        "endpoints": {
            "/predict": "POST - Upload image for disease prediction",
            "/classes": "GET - List all supported plant disease classes",
            "/health": "GET - Check API health status"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "error": load_error if load_error else None
    }

@app.get("/classes")
async def get_classes():
    """Get list of all supported plant disease classes"""
    return {
        "classes": CLASS_NAMES,
        "total_classes": len(CLASS_NAMES)
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict plant disease from uploaded image
    
    Args:
        file: Image file (JPG, PNG, etc.)
        
    Returns:
        Dictionary containing prediction results
    """
    if model is None:
        raise HTTPException(status_code=503, detail=f"Model not available: {load_error}")
    
    # Validate file type
    if file.content_type and not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and transform image
        image_bytes = await file.read()
        img_tensor = transform_image(image_bytes)
        
        # Make prediction
        class_name, class_idx, outputs = predict_image(img_tensor, model)
        
        # Get confidence scores
        probabilities = F.softmax(outputs[0], dim=0)
        confidence = float(probabilities[class_idx])
        
        # Get top 3 predictions
        top3_prob, top3_indices = torch.topk(probabilities, 3)
        top3_predictions = [
            {
                "class": CLASS_NAMES[int(idx.item())],
                "confidence": float(prob)
            }
            for prob, idx in zip(top3_prob, top3_indices)
        ]
        
        return {
            "prediction": {
                "class": class_name,
                "class_index": class_idx,
                "confidence": confidence
            },
            "top_predictions": top3_predictions,
            "filename": file.filename
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8002)

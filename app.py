# app.py
import os
import numpy as np
from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import pretrainedmodels
import torchvision.models as models

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def get_xception_model():
    # Load pretrained Xception
    model = pretrainedmodels.__dict__['xception'](pretrained='imagenet')
    # Modify for binary classification
    model.last_linear = nn.Linear(model.last_linear.in_features, 1)
    return model

def get_resnet_model():
    # Load pretrained ResNet
    model = models.resnet50(pretrained=False)
    # Modify for binary classification
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model

# Load the pre-trained models
try:
    # Initialize model architectures
    xception_model = get_xception_model()
    resnet_model = get_resnet_model()
    
    # Load the saved weights
    xception_model.load_state_dict(torch.load('models/xception_model_best.pth', map_location=device))
    resnet_model.load_state_dict(torch.load('models/resnet_model_best.pth', map_location=device))
    
    # Move models to device
    xception_model = xception_model.to(device)
    resnet_model = resnet_model.to(device)
    
    # Set models to evaluation mode
    xception_model.eval()
    resnet_model.eval()
    
    models_loaded = True
    print("Models loaded successfully")
except Exception as e:
    print(f"Error loading models: {e}")
    models_loaded = False

# Define transformations for the input images - using the standard pretrainedmodels preprocessing
xception_transform = transforms.Compose([
    transforms.Resize((299, 299)),     # Resize to match input size
    transforms.ToTensor(),             # Convert PIL to tensor (C, H, W), scaled to [0, 1]
    transforms.Normalize(              # Normalize using ImageNet stats
        mean=[0.485, 0.456, 0.406],    
        std=[0.229, 0.224, 0.225]
    )
])

resnet_transform = transforms.Compose([
    transforms.Resize((299, 299)),     # Resize to match input size
    transforms.ToTensor(),             # Convert PIL to tensor (C, H, W), scaled to [0, 1]
    transforms.Normalize(              # Normalize using ImageNet stats
        mean=[0.485, 0.456, 0.406],    
        std=[0.229, 0.224, 0.225]
    )
])

def predict_image(img_path):
    if not models_loaded:
        return {"error": "Models not loaded properly"}
    
    try:
        img = Image.open(img_path).convert('RGB')
        
        # Prepare inputs for both models
        xception_input = xception_transform(img).unsqueeze(0).to(device)
        resnet_input = resnet_transform(img).unsqueeze(0).to(device)
        
        # Get predictions
        with torch.no_grad():
            xception_output = xception_model(xception_input)
            resnet_output = resnet_model(resnet_input)
            
            # Convert to probabilities using sigmoid
            xception_prob = torch.sigmoid(xception_output).item()
            resnet_prob = torch.sigmoid(resnet_output).item()
            
            # IMPORTANT: Since 1 = fake, 0 = real in your labeling convention
            # Higher probability now means more likely to be fake
            xception_fake_prob = xception_prob
            resnet_fake_prob = resnet_prob
            
            # Calculate real probabilities (inverse of fake)
            xception_real_prob = 1 - xception_fake_prob
            resnet_real_prob = 1 - resnet_fake_prob
            
            # Combine predictions (simple average)
            avg_fake_prob = (xception_fake_prob + resnet_fake_prob) / 2
            
            # Prediction based on the average probability
            prediction = "Fake" if avg_fake_prob > 0.5 else "Real"
            
            result = {
                "xception_confidence": float(xception_prob),  # Raw model output
                "resnet_confidence": float(resnet_prob),      # Raw model output
                "xception_fake_prob": float(xception_fake_prob),
                "resnet_fake_prob": float(resnet_fake_prob),
                "combined_fake_prob": float(avg_fake_prob),
                "prediction": prediction,
                "fake_probability": float(avg_fake_prob)
            }
            
            return result
    except Exception as e:
        print(f"Error during prediction: {e}")
        return {"error": str(e)}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload_and_predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        # Run prediction
        result = predict_image(file_path)
        
        # Add the image path for display
        result['image_path'] = f"/static/uploads/{file.filename}"
        
        return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

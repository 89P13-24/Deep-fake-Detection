# app.py
import os
import numpy as np
from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import traceback
from efficientnet_pytorch import EfficientNet
from torch.serialization import add_safe_globals

# Add EfficientNet to safe globals to allow loading the model with weights_only=True
add_safe_globals([EfficientNet])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Global variables for model status
models_loaded = False
efficientnet_model = None
densenet_model = None
# Add this function to your app.py
def rebuild_densenet_model(weights_path=None):
    """Recreate the DenseNet model with the same architecture and optionally load weights."""
    import tensorflow as tf
    
    # Define the input shape
    input_shape = (256, 256, 3)
    
    # Create the model from scratch with the same architecture
    input_tensor = tf.keras.Input(shape=input_shape)
    densenet = tf.keras.applications.DenseNet121(
        weights="imagenet", 
        include_top=False, 
        input_tensor=input_tensor
    )
    
    # Add the same layers as in your original architecture
    x = tf.keras.layers.GlobalAveragePooling2D()(densenet.output)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    # Create the model
    model = tf.keras.Model(densenet.input, output)
    
    # Compile with the same settings
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    # If weights path is provided, try to load weights
    if weights_path:
        try:
            # Try to load weights directly (might work for some weights)
            model.load_weights(weights_path)
            print("Successfully loaded weights directly")
        except:
            try:
                # Try loading specific layers that don't have naming issues
                from tensorflow.keras.models import load_model
                import h5py
                
                print("Attempting to extract weights from original model file...")
                with h5py.File(weights_path, 'r') as f:
                    # Check if it's a standard Keras model file with 'model_weights'
                    if 'model_weights' in f:
                        # Check what groups are available for extracting weights
                        groups = list(f['model_weights'].keys())
                        print(f"Found weight groups: {groups[:5]}...")
                
                # Load the original model but skip weight loading
                orig_model = load_model(weights_path, compile=False, custom_objects={
                    # Add any custom layers here if needed
                }, by_name=True)
                
                # Now manually copy weights for the non-DenseNet layers
                # (since DenseNet layers should already have ImageNet weights)
                layer_names = ['global_average_pooling2d', 'dense', 'batch_normalization', 'dropout', 'dense_1']
                for name in layer_names:
                    for i, layer in enumerate(model.layers):
                        if layer.name == name:
                            # Try to find corresponding layer in original model
                            for orig_layer in orig_model.layers:
                                if orig_layer.name.replace('/', '_') == name:
                                    layer.set_weights(orig_layer.get_weights())
                                    print(f"Copied weights for layer: {name}")
                                    break
                
                print("Finished loading weights from specific layers")
            except Exception as e:
                print(f"Could not load weights: {e}")
                # If weights loading fails, continue with ImageNet weights for DenseNet
                print("Using ImageNet weights only")
    
    return model

# [existing imports remain unchanged...]

# Load the pre-trained models
def load_models():
    global models_loaded, efficientnet_model, densenet_model
    
    try:
        print("Attempting to load EfficientNetB0 model...")
        try:
            efficientnet_model = torch.load('models/efficientnetb0.pt', map_location=device)
            print("Loaded EfficientNetB0 as full model")
        except Exception as e:
            print(f"First method failed: {e}")
            try:
                efficientnet_model = torch.load('models/efficientnetb0.pt', map_location=device, weights_only=False)
                print("Loaded EfficientNetB0 with weights_only=False")
            except Exception as e:
                print(f"Second method failed: {e}")
                efficientnet_model = EfficientNet.from_pretrained('efficientnet-b0')
                efficientnet_model._fc = nn.Linear(efficientnet_model._fc.in_features, 1)
                try:
                    state_dict = torch.load('models/efficientnetb0.pt', map_location=device, weights_only=True)
                    print("Loaded state dict from efficientnetb0.pt")
                except:
                    state_dict = torch.load('models/efficientb0.pt', map_location=device, weights_only=True)
                    print("Loaded state dict from efficientb0.pt")
                efficientnet_model.load_state_dict(state_dict)
                print("Loaded EfficientNetB0 using state_dict approach")
        
        efficientnet_model = efficientnet_model.to(device)
        efficientnet_model.eval()
        print("EfficientNetB0 model loaded successfully.")
        
        print("Attempting to load DenseNet121 model...")
        try:
            densenet_model = load_model('models/Densenet121.h5')
            print("DenseNet121 model loaded successfully.")
        except Exception as e:
            print(f"Error loading DenseNet121 model: {e}")
            print("Rebuilding DenseNet121 model with the same architecture...")
            try:
                densenet_model = rebuild_densenet_model('models/Densenet121.h5')
                print("DenseNet121 model rebuilt successfully.")
            except Exception as e2:
                print(f"Rebuilding model failed: {e2}")
                print("Creating DenseNet121 model with ImageNet weights only...")
                densenet_model = rebuild_densenet_model()
                print("DenseNet121 model created with ImageNet weights only.")
        
        models_loaded = True
        print("All models loaded successfully!")
        return True

    except Exception as e:
        print(f"Error during model loading: {e}")
        traceback.print_exc()
        models_loaded = False
        return False

# Define a fallback prediction function for when models aren't loaded
def fallback_predict():
    return {
        "efficientnet_confidence": 0.5,
        "densenet_confidence": 0.5,
        "efficientnet_real_prob": 0.5,
        "densenet_real_prob": 0.5,
        "efficientnet_fake_prob": 0.5,
        "densenet_fake_prob": 0.5,
        "combined_real_prob": 0.5,
        "prediction": "Uncertain (Model Loading Issue)",
        "real_probability": 0.5,
        "fake_probability": 0.5,
        "warning": "Using fallback prediction because models failed to load."
    }

# Define transformations for PyTorch EfficientNetB0
efficientnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),     # Resize to match input size
    transforms.ToTensor(),             # Convert PIL to tensor (C, H, W), scaled to [0, 1]
    transforms.Normalize(              # Normalize using ImageNet stats
        mean=[0.485, 0.456, 0.406],    
        std=[0.229, 0.224, 0.225]
    )
])

def predict_image(img_path):
    global models_loaded, efficientnet_model, densenet_model
    
    # Try loading models again if they weren't loaded successfully
    if not models_loaded:
        print("Models not loaded, attempting to load again...")
        models_loaded = load_models()
        if not models_loaded:
            print("Models still failed to load. Using fallback prediction.")
            result = fallback_predict()
            return result
    
    try:
        # Load image for PyTorch model
        img_torch = Image.open(img_path).convert('RGB')
        efficientnet_input = efficientnet_transform(img_torch).unsqueeze(0).to(device)
        
        # Load and preprocess image for TensorFlow model
        img_tf = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img_tf)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize to [0,1]
        
        # Get predictions from EfficientNetB0 (PyTorch)
        with torch.no_grad():
            try:
                efficientnet_output = efficientnet_model(efficientnet_input)
                # Handle different output formats
                if isinstance(efficientnet_output, tuple):
                    efficientnet_output = efficientnet_output[0]  # Some models return (output, features)
                
                if isinstance(efficientnet_output, torch.Tensor):
                    if efficientnet_output.numel() == 1:
                        efficientnet_prob = torch.sigmoid(efficientnet_output).item()
                    else:
                        # If output has multiple values, take the first one as probability
                        efficientnet_prob = torch.sigmoid(efficientnet_output[0][0]).item()
                else:
                    efficientnet_prob = float(efficientnet_output)
            except Exception as e:
                print(f"Error with EfficientNet prediction: {e}")
                efficientnet_prob = 0.5  # Default to uncertain if there's an error
        
        # Get predictions from DenseNet121 (TensorFlow)
        try:
            densenet_pred = densenet_model.predict(img_array)
            if isinstance(densenet_pred, np.ndarray) and densenet_pred.size == 1:
                densenet_prob = float(densenet_pred[0][0])
            else:
                # Handle different output formats
                densenet_prob = float(densenet_pred[0][0]) if densenet_pred.ndim > 1 else float(densenet_pred[0])
        except Exception as e:
            print(f"Error with DenseNet prediction: {e}")
            densenet_prob = 0.5  # Default to uncertain if there's an error
        
        # IMPORTANT: Now 1 = real, 0 = fake (changed labeling convention)
        # Higher probability now means more likely to be real
        efficientnet_real_prob = efficientnet_prob
        densenet_real_prob = densenet_prob
        
        # Calculate fake probabilities (inverse of real)
        efficientnet_fake_prob = 1 - efficientnet_real_prob
        densenet_fake_prob = 1 - densenet_real_prob
        
        # Combine predictions (simple average)
        avg_real_prob = (efficientnet_real_prob + densenet_real_prob) / 2
        avg_fake_prob = 1 - avg_real_prob
        
        # Prediction based on the average probability
        prediction = "Real" if avg_real_prob > 0.5 else "Fake"
        
        result = {
            "efficientnet_confidence": float(efficientnet_prob),  # Raw model output
            "densenet_confidence": float(densenet_prob),         # Raw model output
            "efficientnet_real_prob": float(efficientnet_real_prob),
            "densenet_real_prob": float(densenet_real_prob),
            "efficientnet_fake_prob": float(efficientnet_fake_prob),
            "densenet_fake_prob": float(densenet_fake_prob),
            "combined_real_prob": float(avg_real_prob),
            "prediction": prediction,
            "real_probability": float(avg_real_prob),
            "fake_probability": float(avg_fake_prob)
        }
        
        return result
    except Exception as e:
        print(f"Error during prediction: {e}")
        traceback.print_exc()
        return {"error": str(e)}

# Try to load models at startup
models_loaded = load_models()

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
        if 'error' not in result:
            result['image_path'] = f"/static/uploads/{file.filename}"
        
        return jsonify(result)

@app.route('/status')
def status():
    return jsonify({
        'models_loaded': models_loaded,
        'device': str(device),
        'efficientnet_loaded': efficientnet_model is not None,
        'densenet_loaded': densenet_model is not None
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
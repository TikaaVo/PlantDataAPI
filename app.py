import io
import traceback
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import torch
import clip
import tensorflow as tf

# --- Configuration ---
class Config:
    """Groups all required configuration variables in one place."""
    PLANT_MODEL_PATH = 'models/BestModel.keras'

    PLANT_CLASSES = ['tomato', 'basil', 'mint', 'lettuce', 'rosemary', 'strawberry']
    IMG_SIZE = (384, 384)
    
    # Health analysis prompt
    HEALTH_PROMPTS = [
        "a photo of a healthy {plant} plant with vibrant green leaves",
        "a photo of a sick {plant} plant with yellow spots or discoloration",
        "a photo of a dehydrated {plant} plant with wilted or drooping leaves",
        "a photo of a dead {plant} plant with brown, dry, or crispy leaves"
    ]
    HEALTH_LABELS = ["Healthy", "Diseased", "Dehydrated", "Dead"]

# --- Application Setup ---
app = Flask(__name__)

# --- Model Loading ---
def load_models():
    """Loads and initializes all required machine learning models."""
    # Load plant identification model
    print("1. Loading plant identification model...")
    plant_model = tf.keras.models.load_model(Config.PLANT_MODEL_PATH)
    
    # Load model for health analysis
    print("2. Loading CLIP model for health analysis...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    
    print("\nAll models loaded successfully.")
    return plant_model, clip_model, clip_preprocess, device

# Load models
plant_model, clip_model, clip_preprocess, device = load_models()

# --- Core ML Functions ---
def identify_plant(image):
    """Plant identification"""
    img = image.resize(Config.IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    preds = plant_model.predict(img_array, verbose=0)
    best_idx = np.argmax(preds[0])
    
    plant_name = Config.PLANT_CLASSES[best_idx]
    confidence = float(np.max(preds))
    
    return plant_name, confidence

def assess_health(plant_name, image):
    """Plant health"""
    prompts = [p.format(plant=plant_name) for p in Config.HEALTH_PROMPTS]
    image_input = clip_preprocess(image).unsqueeze(0).to(device)
    text_tokens = clip.tokenize(prompts).to(device)
    
    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
        text_features = clip_model.encode_text(text_tokens)
        
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        probs = similarity.cpu().numpy()[0]

    status = Config.HEALTH_LABELS[np.argmax(probs)]
    confidence = float(np.max(probs))
    probabilities = {label: float(p) for label, p in zip(Config.HEALTH_LABELS, probs)}
    
    return status, confidence, probabilities

# --- API Endpoint ---
@app.route('/analyze', methods=['POST'])
def analyze_plant_image():
    """Image analysis."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    try:
        file = request.files['image']
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        
        plant_name, plant_conf = identify_plant(image.copy())
        health_status, health_conf, health_probs = assess_health(plant_name, image)
        
        return jsonify({
            'plant_species': plant_name,
            'identification_confidence': f"{plant_conf:.2%}",
            'health_status': health_status,
            'health_confidence': f"{health_conf:.2%}",
            'health_breakdown': health_probs
        })
        
    except Exception as e:
        print("An error occurred:", str(e))
        traceback.print_exc()
        return jsonify({'error': 'An internal server error occurred.'}), 500

# --- Main Execution ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
import os
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing import image
import numpy as np
import json
import tensorflow as tf
from flask_cors import CORS
import logging
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Path to your model
model_path = '/Users/shweta/CDD/Crop Disease Recognition/extra/crop_disease_model_new.tflite'  

# Initialize interpreter as None
interpreter = None
class_labels = {}

# Try to load model and class indices
try:
    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    
    # Load TensorFlow Lite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    logger.info("TFLite model loaded successfully")
    
    # Load class indices
    class_indices_file = 'class_indices.json'
    if not os.path.exists(class_indices_file):
        raise FileNotFoundError(f"Class indices file not found at: {class_indices_file}")
    
    with open(class_indices_file, 'r') as f:
        class_indices = json.load(f)
        # Reverse class indices to get labels
        class_labels = {v: k for k, v in class_indices.items()}
        logger.info(f"Loaded {len(class_labels)} class labels")
        
except Exception as e:
    logger.error(f"Initialization error: {e}")

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Set upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define confidence thresholds
CONFIDENCE_THRESHOLD = 0.7
HIGH_CONFIDENCE_THRESHOLD = 0.98  # For healthy plants

# Mapping to readable disease names
READABLE_NAMES = {
    "Pepper,_bell___Bacterial_spot": "Pepper Bell Bacterial Spot",
    "Pepper,_bell___healthy": "Healthy Pepper Bell",
    "Potato___Early_blight": "Potato Early Blight",
    "Potato___Late_blight": "Potato Late Blight",
    "Potato___healthy": "Healthy Potato",
    "Tomato___Bacterial_spot": "Tomato Bacterial Spot",
    "Tomato___Early_blight": "Tomato Early Blight",
    "Tomato___Late_blight": "Tomato Late Blight",
    "Tomato___Leaf_Mold": "Tomato Leaf Mold",
    "Tomato___Septoria_leaf_spot": "Tomato Septoria Leaf Spot",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Tomato Spider Mites",
    "Tomato___Target_Spot": "Tomato Target Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Tomato Yellow Leaf Curl Virus",
    "Tomato___Tomato_mosaic_virus": "Tomato Mosaic Virus",
    "Tomato___healthy": "Healthy Tomato",
    "Apple___Apple_scab": "Apple Scab",
    "Apple___Black_rot": "Apple Black Rot",
    "Apple___Cedar_apple_rust": "Apple Cedar Rust",
    "Apple___healthy": "Healthy Apple",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Corn Gray Leaf Spot",
    "Corn_(maize)___Common_rust_": "Corn Common Rust",
    "Corn_(maize)___Northern_Leaf_Blight": "Corn Northern Leaf Blight",
    "Corn_(maize)___healthy": "Healthy Corn",
    "Grape___Black_rot": "Grape Black Rot",
    "Grape___Esca_(Black_Measles)": "Grape Black Measles",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Grape Leaf Blight",
    "Grape___healthy": "Healthy Grape",
    "Rice___Brown_Spot": "Rice Brown Spot",
    "Rice___Healthy": "Healthy Rice",
    "Rice___Leaf_Blast": "Rice Leaf Blast",
    "Rice___Neck_Blast": "Rice Neck Blast",
    "Strawberry___Leaf_scorch": "Strawberry Leaf Scorch",
    "Strawberry___healthy": "Healthy Strawberry",
    "Wheat___Brown_Rust": "Wheat Brown Rust",
    "Wheat___Healthy": "Healthy Wheat",
    "Wheat___Yellow_Rust": "Wheat Yellow Rust"
}

def is_plant_leaf(image_path):
    """Check if the image appears to be a plant leaf using color analysis"""
    try:
        # Load image with OpenCV
        img = cv2.imread(image_path)
        if img is None:
            return False
            
        # Convert to HSV color space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Define green color range (plant leaves)
        lower_green = np.array([30, 40, 40])
        upper_green = np.array([90, 255, 255])
        
        # Create mask for green areas
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Calculate percentage of green pixels
        green_percentage = np.count_nonzero(mask) / (img.shape[0] * img.shape[1])
        
        # Also check image entropy (texture complexity)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        entropy = np.log(np.var(gray) + 1e-10)

        logger.info(f"Green percentage: {green_percentage:.2f}, Entropy: {entropy:.2f}")
        
        # Consider it a plant leaf if there's sufficient green OR sufficient texture
        return green_percentage > 0.15 or entropy > 4.0

    except Exception as e:
        logger.error(f"Error in plant detection: {e}")
        return False

@app.route('/')
def index():
    status = "Model loaded successfully" if interpreter else "Model NOT loaded"
    return jsonify({
        "message": "Crop Disease Detection API (TFLite)",
        "status": status,
        "endpoints": {
            "/predict": "POST method - Upload an image to get predictions"
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    if interpreter is None:
        return jsonify({
            'error': 'Model not loaded. Service unavailable.',
            'details': 'The disease detection model failed to load during startup.'
        }), 503

    if 'image' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not file.filename.lower().endswith(('png', 'jpg', 'jpeg')):
        return jsonify({'error': 'Invalid file type. Only PNG, JPG, or JPEG are allowed.'}), 400

    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # First: Check if this looks like a plant leaf
        if not is_plant_leaf(filepath):
            return jsonify({
                'error': 'The uploaded image does not appear to be a plant leaf. Please upload a clear image of a plant leaf.',
                'details': 'Detection algorithm found insufficient plant characteristics'
            }), 400

        # Preprocess the image
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # Get input and output tensors
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], img_array.astype(np.float32))

        # Run the interpreter
        interpreter.invoke()

        # Get the prediction
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_class = np.argmax(output_data, axis=1)
        confidence = np.max(output_data)
        
        # Enhanced confidence check
        original_label = class_labels.get(predicted_class[0], 'Unknown Class')
        
        # Get readable name
        readable_label = READABLE_NAMES.get(original_label, original_label)
        
        # For healthy plants, require higher confidence
        is_healthy = "healthy" in original_label.lower()
        confidence_threshold = HIGH_CONFIDENCE_THRESHOLD if is_healthy else CONFIDENCE_THRESHOLD
        
        if confidence < confidence_threshold:
            return jsonify({
                'error': f'Low confidence prediction ({confidence:.2f} < {confidence_threshold:.2f}). Please upload a clearer image of a plant leaf.',
                'predicted': readable_label,
                'confidence': float(confidence),
                'threshold': confidence_threshold
            }), 400

        # Return prediction with symptoms
        symptoms_data = {
            "Pepper,_bell___Bacterial_spot": [
                "Dark, water-soaked spots on leaves.",
                "Yellowing of leaf margins.",
                "Small, dark, sunken lesions on fruits.",
                "Defoliation and reduced yield.",
                "Wilting and stunted growth."
            ],
            "Pepper,_bell___healthy": [
                "No visible symptoms.",
                "Bright green leaves.",
                "Healthy fruit development.",
                "Sturdy stems.",
                "Normal growth patterns."
            ],
            "Potato___Early_blight": [
                "Dark brown spots on older leaves.",
                "Yellowing of leaf margins.",
                "Defoliation.",
                "Reduced tuber quality.",
                "Stunted plant growth."
            ],
            "Potato___Late_blight": [
                "Water-soaked spots on leaves.",
                "White mold on the undersides of leaves.",
                "Rapid wilting of plants.",
                "Dark, greasy spots on tubers.",
                "Rotting of tubers in storage."
            ],
            "Potato___healthy": [
                "No visible symptoms.",
                "Healthy, green leaves.",
                "Robust tuber development.",
                "Strong stems.",
                "Normal growth patterns."
            ],
            "Tomato___Bacterial_spot": [
                "Dark, water-soaked lesions on leaves.",
                "Yellowing of leaf edges.",
                "Small, dark spots on fruits.",
                "Leaf curling and wilting.",
                "Reduced fruit quality."
            ],
            "Tomato___Early_blight": [
                "Dark, concentric rings on leaves.",
                "Yellowing and dropping of lower leaves.",
                "Reduced fruit yield.",
                "Dark lesions on stems.",
                "Stunted plant growth."
            ],
            "Tomato___Late_blight": [
                "Large, irregularly shaped water-soaked spots.",
                "White mold on the underside of leaves.",
                "Brown lesions on stems.",
                "Rapid wilting of plants.",
                "Tubers rot in the ground."
            ],
            "Tomato___Leaf_Mold": [
                "Yellowing of leaves.",
                "Fuzzy greenish-gray mold on the underside.",
                "Leaf curling.",
                "Defoliation.",
                "Reduced fruit quality."
            ],
            "Tomato___Septoria_leaf_spot": [
                "Small, round spots with dark borders.",
                "Yellowing leaves.",
                "Defoliation.",
                "Reduced yield.",
                "Dark, sunken spots on stems."
            ],
            "Tomato___Spider_mites Two-spotted_spider_mite": [
                "Fine webbing on leaves.",
                "Yellowing and stippling of leaves.",
                "Leaf drop.",
                "Stunted growth.",
                "Brown, crispy leaves."
            ],
            "Tomato___Target_Spot": [
                "Dark, concentric ring spots on leaves.",
                "Leaf drop.",
                "Reduced yield.",
                "Spots may appear on fruits.",
                "Stunted growth."
            ],
            "Tomato___Tomato_Yellow_Leaf_Curl_Virus": [
                "Yellowing of leaves.",
                "Curling and distortion of leaves.",
                "Stunted growth.",
                "Reduced fruit set.",
                "Plant wilting."
            ],
            "Tomato___Tomato_mosaic_virus": [
                "Mosaic patterns on leaves.",
                "Stunted growth.",
                "Leaf curling.",
                "Deformed fruits.",
                "Reduced yield."
            ],
            "Tomato___healthy": [
                "No visible symptoms.",
                "Healthy green leaves.",
                "Robust fruit development.",
                "Strong stems.",
                "Normal growth patterns."
            ],
            "Apple___Apple_scab": [
                "Olive green spots on leaves.",
                "Scabby lesions on fruit and leaves.",
                "Premature leaf drop.",
                "Reduced fruit yield.",
                "Deformed or cracked fruits."
            ],
            "Apple___Black_rot": [
                "Dark, sunken lesions on fruit.",
                "Leaf yellowing and premature drop.",
                "Black fungal growth on fruit surface.",
                "V-shaped lesions on leaves.",
                "Weakened branches."
            ],
            "Apple___Cedar_apple_rust": [
                "Bright orange spots on leaves.",
                "Yellowing and premature defoliation.",
                "Galls on branches.",
                "Reduced fruit size.",
                "Lower tree vigor."
            ],
            "Apple___healthy": [
                "No visible symptoms.",
                "Bright green leaves.",
                "Healthy fruit growth.",
                "Strong branches.",
                "Normal yield."
            ],
            "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": [
                "Gray, rectangular lesions on leaves.",
                "Yellowing leaf margins.",
                "Reduced photosynthesis.",
                "Lower kernel production.",
                "Weak stalks."
            ],
            "Corn_(maize)___Common_rust_": [
                "Reddish-brown pustules on leaves.",
                "Yellowing of leaf tissues.",
                "Reduced plant growth.",
                "Decreased yield.",
                "Wilting in severe cases."
            ],
            "Corn_(maize)___Northern_Leaf_Blight": [
                "Long, elliptical lesions on leaves.",
                "Grayish-brown necrotic patches.",
                "Premature leaf death.",
                "Lower yield.",
                "Weakened stalks."
            ],
            "Corn_(maize)___healthy": [
                "No visible symptoms.",
                "Lush green leaves.",
                "Strong stalks.",
                "Healthy cob formation.",
                "Normal growth pattern."
            ],
            "Grape___Black_rot": [
                "Small, brown circular lesions on leaves.",
                "Black fungal fruiting bodies.",
                "Withered fruit clusters.",
                "Brown, shriveled grapes.",
                "Reduced vine vigor."
            ],
            "Grape___Esca_(Black_Measles)": [
                "Interveinal chlorosis on leaves.",
                "Dark brown streaks on stems.",
                "Sunken lesions on fruit.",
                "Premature defoliation.",
                "Reduced grape quality."
            ],
            "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": [
                "Small, reddish-brown spots on leaves.",
                "Leaf curling and drying.",
                "Stunted growth.",
                "Reduced fruit quality.",
                "Weakened vines."
            ],
            "Grape___healthy": [
                "No visible symptoms.",
                "Bright green leaves.",
                "Strong vine growth.",
                "Plump, healthy grapes.",
                "Normal fruit set."
            ],
            "Rice___Brown_Spot": [
                "Small, brown lesions on leaves.",
                "Reduced plant growth.",
                "Yellowing leaves.",
                "Poor grain formation.",
                "Weakened stems."
            ],
            "Rice___Healthy": [
                "No visible symptoms.",
                "Vibrant green leaves.",
                "Healthy grain development.",
                "Strong stems.",
                "Normal plant growth."
            ],
            "Rice___Leaf_Blast": [
                "Diamond-shaped lesions on leaves.",
                "Yellow halos around lesions.",
                "Leaf tip blight.",
                "Reduced grain yield.",
                "Weak, lodging plants."
            ],
            "Rice___Neck_Blast": [
                "Dark brown lesions on necks of rice panicles.",
                "Incomplete grain filling.",
                "Grain shattering.",
                "Lodging of plants.",
                "Reduced yield."
            ],
            "Strawberry___Leaf_scorch": [
                "Purple to brown spots on leaves.",
                "Leaf curling and browning.",
                "Reduced fruit production.",
                "Stunted growth.",
                "Premature leaf drop."
            ],
            "Strawberry___healthy": [
                "No visible symptoms.",
                "Bright green leaves.",
                "Healthy flower and fruit formation.",
                "Strong plant growth.",
                "Normal yield."
            ],
            "Wheat___Brown_Rust": [
                "Reddish-brown pustules on leaves.",
                "Yellowing and drying of leaves.",
                "Reduced grain yield.",
                "Weak, thin stems.",
                "Delayed maturity."
            ],
            "Wheat___Healthy": [
                "No visible symptoms.",
                "Lush green leaves.",
                "Strong stems.",
                "Well-formed grains.",
                "Normal growth."
            ],
            "Wheat___Yellow_Rust": [
                "Bright yellow pustules on leaves.",
                "Stunted growth.",
                "Leaf curling and drying.",
                "Reduced grain yield.",
                "Weakened plant structure."
            ]
        }

        symptoms = symptoms_data.get(original_label, ["No symptoms available"])

        return jsonify({
            'prediction': readable_label,
            'symptoms': symptoms,
            'confidence': float(confidence),
            'is_healthy': is_healthy
        })

    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return jsonify({'error': 'Failed to process image'}), 500

if __name__ == '__main__':
    logger.info("Starting TFLite server...")
    port = int(os.environ.get("PORT", 8080))  
    app.run(host="0.0.0.0", port=port)

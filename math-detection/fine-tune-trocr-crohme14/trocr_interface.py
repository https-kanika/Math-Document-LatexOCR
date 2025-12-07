import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import io
import base64

# Import your inference class
from TrOCRinfer import LaTeXOCRInference

app = Flask(__name__, template_folder='templates')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize model once at startup
print("Loading model...")
ocr_model = LaTeXOCRInference()
print("Model loaded successfully!")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: PNG, JPG, JPEG, GIF, BMP'}), 400
    
    try:
        # Read image
        image = Image.open(file.stream).convert('RGB')
        
        # Run inference
        latex_output, token_ids = ocr_model.predict(image, return_raw=True)
        
        # Convert image to base64 for display
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'latex': latex_output,
            'token_ids': token_ids[:50],  # First 50 tokens
            'image': f'data:image/png;base64,{img_str}'
        })
    
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
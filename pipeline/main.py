from flask import Flask, request, jsonify, send_file, render_template
from werkzeug.utils import secure_filename
import os
import cv2
from ultralytics import YOLO
import uuid
import json
import torch
from utils import (
    apply_binarization, 
    transcribe_text_with_trocr, 
    load_latex_model, 
    transcribe_latex_with_wap,
    read_metadata_and_generate_tex  # ADD THIS
)

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
MODEL_PATH = r"segmentation_model\document_detection_model.pt"
LATEX_CHECKPOINT = r"math_model\checkpoint_best.pth"  
WORD2IDX_PATH = r"vocab\word2idx.pkl"  
IDX2WORD_PATH = r"vocab\idx2word.pkl"  

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load YOLO model
model = YOLO(MODEL_PATH)

# Load LaTeX transcription model (WAP)
print("Loading LaTeX transcription model...")
latex_encoder, latex_decoder, word2idx, idx2word, latex_device = load_latex_model(
    checkpoint_path=LATEX_CHECKPOINT,
    word2idx_path=WORD2IDX_PATH,
    idx2word_path=IDX2WORD_PATH,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)
print("LaTeX model loaded successfully!")

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('home.html')


@app.route('/segment', methods=['POST'])
def segment_document():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        # Check if LaTeX generation is requested
        generate_latex = request.form.get('generate_latex', 'true').lower() == 'true'
        
        file = request.files['image']
        request_id = str(uuid.uuid4())
        request_folder = os.path.join(OUTPUT_FOLDER, request_id)
        os.makedirs(request_folder, exist_ok=True)
        
        # Save input image
        input_path = os.path.join(UPLOAD_FOLDER, f"{request_id}_{file.filename}")
        file.save(input_path)
        
        # Binarize the input image using Otsu method
        binarized_path = os.path.join(request_folder, 'binarized_input.jpg')
        binarized_img = apply_binarization(input_path, save_path=binarized_path, method='otsu')
        
        # Read original image for dimensions
        original_image = cv2.imread(input_path)
        img_height, img_width = original_image.shape[:2]
        
        # Run YOLO on the BINARIZED image
        results = model.predict(source=binarized_path, imgsz=1024)[0]
        
        # Save annotated image with bounding boxes
        annotated_image = results.plot()
        annotated_path = os.path.join(request_folder, 'predicted_image.jpg')
        cv2.imwrite(annotated_path, annotated_image)
        
        # Process detections
        segments = []
        if results.boxes is not None:
            for idx, box in enumerate(results.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                
                # Crop and save segment FROM BINARIZED IMAGE
                cropped = binarized_img[y1:y2, x1:x2]
                segment_filename = f"segment_{idx}.jpg"
                segment_path = os.path.join(request_folder, segment_filename)
                cv2.imwrite(segment_path, cropped)
                
                # TRANSCRIPTION LOGIC
                transcription = None
                
                if class_name == 'text':
                    # Use TrOCR for text
                    print("Transcribing text segment with TrOCR...")
                    transcription = transcribe_text_with_trocr(segment_path)
                    
                elif class_name in ['equation', 'symbol']:
                    # Use WAP model for equations and symbols
                    print("Transcribing LaTeX segment with WAP model...")
                    transcription = transcribe_latex_with_wap(
                        image_path=segment_path,
                        encoder=latex_encoder,
                        decoder=latex_decoder,
                        word2idx=word2idx,
                        idx2word=idx2word,
                        device=latex_device,
                        beam_width=10,
                        max_len=150
                    )
                
                # Store location info for stitching
                segments.append({
                    'id': idx,
                    'filename': segment_filename,
                    'bbox': {
                        'x1': x1, 'y1': y1,
                        'x2': x2, 'y2': y2,
                        'width': x2 - x1,
                        'height': y2 - y1
                    },
                    'class': class_name,
                    'confidence': float(box.conf[0]),
                    'transcription': transcription
                })
        
        # Prepare metadata
        metadata = {
            'request_id': request_id,
            'original_image': {
                'filename': file.filename,
                'width': img_width,
                'height': img_height
            },
            'preprocessing': {
                'binarization': 'otsu',
                'binarized_image': 'binarized_input.jpg'
            },
            'coordinate_type': 'absolute_pixels',
            'total_segments': len(segments),
            'segments': segments,
            'annotated_image': 'predicted_image.jpg'
        }
        
        # Save metadata to JSON file
        metadata_path = os.path.join(request_folder, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Generate LaTeX document if requested
        latex_file_path = None
        if generate_latex:
            try:
                output_tex_path = os.path.join(request_folder, 'document.tex')
                latex_file_path = read_metadata_and_generate_tex(metadata_path, output_tex_path)
                print(f"LaTeX document generated: {latex_file_path}")
            except Exception as latex_error:
                print(f"Error generating LaTeX: {str(latex_error)}")
                latex_file_path = f"Error: {str(latex_error)}"
        
        response = {
            'message': 'Segmentation and transcription completed successfully!',
            'request_id': request_id,
            'image_size': {'width': img_width, 'height': img_height},
            'total_segments': len(segments),
            'output_folder': request_folder,
            'files': {
                'metadata': metadata_path,
                'annotated_image': annotated_path,
                'binarized_image': binarized_path,
                'latex_document': latex_file_path
            }
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/download/<request_id>/<filename>', methods=['GET'])
def download_file(request_id, filename):
    """
    Download any file from a request folder (metadata.json, document.tex, images, etc.)
    """
    try:
        file_path = os.path.join(OUTPUT_FOLDER, request_id, filename)
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        return send_file(file_path, as_attachment=True)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/latex/<request_id>', methods=['GET'])
def get_latex_document(request_id):
    """
    Get the LaTeX document content as text
    """
    try:
        latex_path = os.path.join(OUTPUT_FOLDER, request_id, 'document.tex')
        
        if not os.path.exists(latex_path):
            return jsonify({'error': 'LaTeX document not found'}), 404
        
        with open(latex_path, 'r', encoding='utf-8') as f:
            latex_content = f.read()
        
        return jsonify({
            'request_id': request_id,
            'latex_content': latex_content,
            'file_path': latex_path
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
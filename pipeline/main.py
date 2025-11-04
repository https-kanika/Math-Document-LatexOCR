from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
from ultralytics import YOLO
import uuid
import json

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
MODEL_PATH = r"C:\Users\kani1\Desktop\Errorcode500_results\working\document_detection_model.pt"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load model
model = YOLO(MODEL_PATH)

@app.route('/segment', methods=['POST'])
def segment_document():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        request_id = str(uuid.uuid4())
        request_folder = os.path.join(OUTPUT_FOLDER, request_id)
        os.makedirs(request_folder, exist_ok=True)
        
        # Save input image
        input_path = os.path.join(UPLOAD_FOLDER, f"{request_id}_{file.filename}")
        file.save(input_path)
        
        # Read image and run YOLO
        image = cv2.imread(input_path)
        img_height, img_width = image.shape[:2]
        results = model.predict(source=input_path, imgsz=1024)[0]
        
        # Save annotated image with bounding boxes
        annotated_image = results.plot()  # This draws boxes, labels, and confidence scores
        annotated_path = os.path.join(request_folder, 'predicted_image.jpg')
        cv2.imwrite(annotated_path, annotated_image)
        
        # Process detections
        segments = []
        if results.boxes is not None:
            for idx, box in enumerate(results.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                # Crop and save segment
                cropped = image[y1:y2, x1:x2]
                segment_filename = f"segment_{idx}.jpg"
                segment_path = os.path.join(request_folder, segment_filename)
                cv2.imwrite(segment_path, cropped)
                
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
                    'class': model.names[int(box.cls[0])],
                    'confidence': float(box.conf[0])
                })
        
        # Prepare metadata
        metadata = {
            'request_id': request_id,
            'original_image': {
                'filename': file.filename,
                'width': img_width,
                'height': img_height
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
        
        # return jsonify({
        #     'request_id': request_id,
        #     'image_size': {'width': img_width, 'height': img_height},
        #     'segments': segments,
        #     'output_folder': request_folder,
        #     'metadata_file': metadata_path,
        #     'annotated_image': annotated_path
        # }), 200
        return jsonify("Great success"), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
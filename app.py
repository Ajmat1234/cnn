from flask import Flask, render_template, request, jsonify
from paddleocr import PaddleOCR
import io
from PIL import Image
import os
from tempfile import NamedTemporaryFile
import logging

# Set up detailed logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = Flask(__name__)

# Global OCR instance
ocr = None

@app.before_first_request
def initialize_ocr():
    global ocr
    try:
        logger.info("Starting PaddleOCR initialization...")
        ocr = PaddleOCR(use_angle_cls=True, lang='hi', use_gpu=False, show_log=False)
        logger.info("PaddleOCR initialized successfully.")
    except Exception as init_err:
        logger.error(f"PaddleOCR initialization failed: {str(init_err)}", exc_info=True)
        ocr = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global ocr
    if ocr is None:
        logger.error("OCR not initialized - returning error")
        return jsonify({'error': 'OCR engine not ready. Please wait a moment and try again.'}), 503
    
    if 'file' not in request.files:
        logger.warning("No file in request")
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        logger.warning("Empty filename")
        return jsonify({'error': 'No file selected'}), 400
    
    temp_path = None
    try:
        logger.info(f"Starting prediction for file: {file.filename} (size: {len(file.read())} bytes)")
        file.seek(0)  # Reset file pointer after reading size
        
        # Load image
        logger.info("Loading image from bytes...")
        image_bytes = file.read()
        if not image_bytes:
            logger.error("Empty image bytes")
            return jsonify({'error': 'Invalid empty image'}), 400
        
        image = Image.open(io.BytesIO(image_bytes))
        logger.info(f"Image loaded: {image.size}, mode: {image.mode}")
        
        # Save to temp file
        logger.info("Creating temporary file for OCR...")
        with NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            image.save(temp_file.name, 'JPEG', quality=95)
            temp_path = temp_file.name
            logger.info(f"Temp file saved: {temp_path}")
        
        # Run OCR
        logger.info("Executing OCR on image...")
        result = ocr.ocr(temp_path, cls=True)
        logger.info(f"OCR raw result received: {len(result[0]) if result and result[0] else 0} detections")
        
        # Parse results
        recognized_text = []
        confidences = []
        if result and result[0]:
            for idx, line in enumerate(result[0]):
                logger.info(f"Line {idx}: {line}")
                if line and len(line) > 1 and line[1]:
                    text = line[1][0] if isinstance(line[1], (list, tuple)) else str(line[1])
                    conf = line[1][1] if isinstance(line[1], (list, tuple)) and len(line[1]) > 1 else 0.0
                    if text and text.strip():
                        clean_text = text.strip()
                        recognized_text.append(clean_text)
                        confidences.append(float(conf))
                        logger.info(f"Extracted: '{clean_text}' (conf: {conf:.2f})")
        else:
            logger.warning("No OCR results detected")
        
        full_text = ' '.join(recognized_text)
        avg_conf = sum(confidences) / max(1, len(confidences)) if confidences else 0.0
        
        logger.info(f"Final output - Text: '{full_text}' | Avg Conf: {avg_conf:.2f} | Lines: {len(recognized_text)}")
        
        # Cleanup
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
            logger.info("Temp file deleted")
        
        return jsonify({
            'recognized_text': full_text,
            'confidence': f'{avg_conf:.2f}',
            'num_lines': len(recognized_text)
        })
        
    except Exception as e:
        logger.error(f"Detailed predict error: {str(e)}", exc_info=True)
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
            logger.info("Temp file deleted on error")
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

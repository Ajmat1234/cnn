from flask import Flask, render_template, request, jsonify
import easyocr
import io
from PIL import Image
import os
from tempfile import NamedTemporaryFile
import logging
import threading  # For lazy init lock

# Minimal logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = Flask(__name__)

# Global EasyOCR reader with lazy init
reader = None
reader_lock = threading.Lock()

def init_reader():
    global reader
    with reader_lock:
        if reader is None:
            try:
                logger.info("Lazy-loading EasyOCR for Hindi handwriting...")
                reader = easyocr.Reader(['hi', 'en'], gpu=False)  # Hindi + English fallback
                logger.info("EasyOCR loaded successfully.")
            except Exception as init_err:
                logger.error(f"EasyOCR init failed: {str(init_err)}", exc_info=True)
                reader = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Lazy init on first call
    init_reader()
    if reader is None:
        return jsonify({'error': 'OCR setup failed (memory?). Try smaller image.'}), 503

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    temp_path = None
    try:
        logger.info(f"Processing handwriting: {file.filename}")

        # Read & resize image (mem saver)
        image_bytes = file.read()
        if not image_bytes:
            return jsonify({'error': 'Empty image'}), 400
        image = Image.open(io.BytesIO(image_bytes))

        # Resize to max 800px
        max_dim = 800
        if image.width > max_dim or image.height > max_dim:
            ratio = min(max_dim / image.width, max_dim / image.height)
            new_size = (int(image.width * ratio), int(image.height * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            logger.info(f"Resized to: {new_size}")

        logger.info(f"Image ready: {image.size}")

        # Save temp (EasyOCR accepts path or numpy, but path safer)
        with NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            image.save(temp_file.name, 'JPEG', quality=85)
            temp_path = temp_file.name

        # Run EasyOCR
        logger.info("Running handwriting OCR...")
        result = reader.readtext(temp_path, detail=1)  # detail=1 for text + conf
        logger.info(f"Detections: {len(result)} lines")

        # Parse (EasyOCR format: [(bbox, (text, conf)), ...])
        recognized_text = []
        confidences = []
        for (bbox, (text, conf)) in result:
            if text.strip():
                clean_text = text.strip()
                recognized_text.append(clean_text)
                confidences.append(float(conf))
                logger.info(f"Line: '{clean_text}' (conf: {conf:.2f})")

        full_text = ' '.join(recognized_text)
        avg_conf = sum(confidences) / max(1, len(confidences)) if confidences else 0.0

        logger.info(f"Handwriting result: '{full_text[:50]}...' (avg conf: {avg_conf:.2f})")

        # Cleanup
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

        return jsonify({
            'recognized_text': full_text,
            'confidence': f'{avg_conf:.2f}',
            'num_lines': len(recognized_text)
        })

    except MemoryError:
        logger.error("Out of memory")
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
        return jsonify({'error': 'Low RAM - use smaller image.'}), 503
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
        return jsonify({'error': f'Failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

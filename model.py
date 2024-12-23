import logging
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
import time
import queue
import threading
import tensorflow as tf
import numpy as np
from ultralytics import YOLO
import requests
import cv2
import tempfile

app = Flask(__name__)
CORS(app)

# Queue to store video data and metadata for processing
data_queue = queue.Queue()

# Paths for ESRGAN and YOLO models
ESRGAN_MODEL_PATH = './esrgan_model'
YOLO_MODEL_PATH = 'best.pt'

# Load models
yolo_model = YOLO(YOLO_MODEL_PATH)
esrgan_model = tf.saved_model.load(ESRGAN_MODEL_PATH)

# Configure logging
logging.basicConfig(
    filename='pipeline.log',
    level=logging.DEBUG, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

@app.route('/test', methods=['GET'])
def test():
    return 'deployed'

@app.route('/upload_blob', methods=['POST'])
def upload_blob():
    """
    Endpoint to handle blob file upload and save each frame with a unique filename in the queue.
    """
    logging.debug("Received POST request on /upload_blob")

    if 'video' not in request.files:
        logging.error("No video file in the request")
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files['video']
    latitude = request.form.get('latitude')
    longitude = request.form.get('longitude')

    if not latitude or not longitude:
        logging.error("Missing latitude or longitude in the request")
        return jsonify({"error": "Missing latitude or longitude"}), 400

    timestamp = int(time.time() * 1000)
    filename = secure_filename(f"{timestamp}_{video_file.filename}")

    video_data = video_file.read()

    metadata = {
        "latitude": latitude,
        "longitude": longitude,
        "filename": filename,
        "timestamp": timestamp
    }

    data_queue.put({"video": video_data, "metadata": metadata})
    logging.info(f"Data added to queue: filename={filename}, metadata={metadata}")

    return jsonify({"message": "Video and metadata added to queue successfully"}), 200

def preprocess_image(image):
    """
    Preprocess an image for ESRGAN super-resolution.
    """
    hr_image = tf.convert_to_tensor(image, dtype=tf.float32)
    hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
    hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
    return tf.expand_dims(hr_image, 0)

def process_videos_from_queue():
    """
    Continuously process videos from the queue.
    """
    while True:
        if data_queue.empty():
            continue

        data_item = data_queue.get()
        video_data = data_item['video']
        metadata = data_item['metadata']
        latitude = metadata['latitude']
        longitude = metadata['longitude']
        filename = metadata['filename']

        try:
            with tempfile.NamedTemporaryFile(delete=True, suffix=".webm") as temp_video:
                temp_video.write(video_data)
                temp_video.flush() 

                cap = cv2.VideoCapture(temp_video.name)

                frame_counter = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_counter += 1

                    if frame_counter % 6 == 2 or frame_counter % 6 == 4 or frame_counter % 6 == 0:
                        logging.debug(f"Skipping frame {frame_counter} in video {filename}.")
                        continue

                    try:
                        lr_image = preprocess_image(frame)
                        sr_image = esrgan_model(lr_image)
                        sr_image = tf.squeeze(sr_image).numpy().astype(np.uint8)
                        logging.debug(f"Frame enhanced from video {filename}.")
                    except Exception as e:
                        logging.error(f"Error enhancing frame from video {filename}: {str(e)}")
                        continue

                    try:
                        results = yolo_model.predict(sr_image)
                        pothole_detected = False
                        for r in results:
                            if len(r.boxes) > 0:
                                pothole_detected = True
                                break

                        if pothole_detected:
                            logging.info(f"Pothole detected in frame {frame_counter} of video {filename}. Sending GPS data...")
                            data = {
                                "latitude": latitude,
                                "longitude": longitude,
                                "val":4
                            }

                        else:
                            data = {
                                "latitude": latitude,
                                "longitude": longitude,
                                "val":0
                            }
                            logging.info(f"No pothole detected in {frame_counter}.")

                        try:
                            url = "https://potholebackend.onrender.com/api/location/"
                            response = requests.post(url, json=data)
                            if response.status_code == 200:
                                logging.info(f"GPS data sent successfully for frame {frame_counter} of video {filename}.")
                            else:
                                logging.error(f"Failed to send GPS data for frame {frame_counter} of video {filename}. Response: {response.text}")

                        except Exception as e:
                            logging.error(f"Error sending GPS data for frame {frame_counter} of video {filename}: {str(e)}")

                    except Exception as e:
                        logging.error(f"Error during YOLO detection on video {filename}: {str(e)}")

                cap.release()
                logging.info(f"Finished processing video {filename}.")
        except Exception as e:
            logging.error(f"Error processing video {filename}: {str(e)}")

def start_flask():
    """
    Start the Flask server.
    """
    app.run(host="0.0.0.0", port=5050)

def start_processing():
    """
    Start the video processing thread.
    """
    processing_thread = threading.Thread(target=process_videos_from_queue, daemon=True)
    processing_thread.start()

if __name__ == '__main__':
    threading.Thread(target=start_flask, daemon=True).start()
    start_processing()
    while True:
        time.sleep(1)
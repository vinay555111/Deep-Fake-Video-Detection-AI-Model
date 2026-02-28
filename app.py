from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory,jsonify
import os
import tensorflow as tf
import keras
import cv2
import numpy as np

IMG_SIZE = 224
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048


model = keras.models.load_model('./deepfake_video_model.h5')


def build_feature_extractor():
    feature_extractor= keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
)
    preprocess_input = keras.applications.inception_v3.preprocess_input
    
    inputs=keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)
    
    outputs = feature_extractor (preprocessed)
    return keras. Model (inputs, outputs, name="feature_extractor")
feature_extractor= build_feature_extractor()

def load_video(path, max_frames=0, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture (path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square (frame)
            frame = cv2.resize(frame, resize) 
            frame = frame [:, :, [2, 1, 0]]
            frames.append(frame)
            
            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)

def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame [start_y: start_y + min_dim, start_x : start_x + min_dim]

def prepare_single_video(frames): 
    frames = frames [None, ...]
    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
    frame_features = np.zeros (shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")
    for i, batch in enumerate (frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            frame_features [i, j, :] = feature_extractor.predict(batch [None, j,:])
        frame_mask[i, :length] = 1 #1 = not masked, 0 masked

    return frame_features, frame_mask


# Load the pre-trained model

app = Flask(__name__)

# Configuration for upload folder and content length
app.config['UPLOAD_FOLDER'] = 'uploads/videos'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # Max upload size: 100MB
app.secret_key = 'supersecretkey'

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file extensions for uploads
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

# Helper function to check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home route to render the index page
@app.route('/')
def home():
    return render_template('index.html')

# Endpoint to predict if the video is deepfake or not
@app.route('/predict', methods=['POST'])
def predict():
    # Check if a video file is present in the request
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    video = request.files['video']

    # Check if the uploaded file is allowed
    if video and allowed_file(video.filename):
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video.filename)
        video.save(video_path)  # Save the video file

        # Process the video and extract features
        frames = load_video(video_path)
        frame_features, frame_mask = prepare_single_video(frames)
        
        # Make prediction using the pre-trained model
        prediction = model.predict([frame_features, frame_mask])[0]
        result = 'FAKE' if prediction >= 0.51 else 'REAL'
        confidence = float(prediction)  # Convert prediction to Python float

        # Clean up by removing the uploaded video
        os.remove(video_path)

        # Return the result and confidence as a JSON response
        return jsonify({'result': result, 'confidence': confidence})
    
    # Handle invalid file uploads
    return jsonify({'error': 'Invalid file type. Allowed types are mp4, avi, mov, mkv'}), 400

# Main entry point to run the Flask app
if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    # Run the app in debug mode
    app.run(debug=True)

from flask import Flask, request, jsonify
from flask_cors import CORS
import face_recognition
import cv2
import os

app = Flask(__name__)
CORS(app)  
reference_images_dir = "images/"

known_face_encodings = []
known_face_names = []

for image_name in os.listdir(reference_images_dir):
    image_path = os.path.join(reference_images_dir, image_name)
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb_image)
    if len(encodings) > 0:
        known_face_encodings.append(encodings[0])
        known_face_names.append(image_name)

@app.route('/identify', methods=['POST'])
def identify():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    image = face_recognition.load_image_file(file)
    encodings = face_recognition.face_encodings(image)

    if len(encodings) == 0:
        return jsonify({"error": "No faces found in the image"}), 400

    image_to_compare_encoding = encodings[0]
    best_match_index = None
    best_match_distance = float('inf')
    for i, known_face_encoding in enumerate(known_face_encodings):
        distance = face_recognition.face_distance([known_face_encoding], image_to_compare_encoding)[0]
        if distance < best_match_distance:
            best_match_distance = distance
            best_match_index = i

    if best_match_index is not None:
        result = {
            "name": known_face_names[best_match_index],
            "distance": best_match_distance
        }
        return jsonify(result), 200
    else:
        return jsonify({"error": "No match found"}), 400

if __name__ == '__main__':
    app.run(debug=True)

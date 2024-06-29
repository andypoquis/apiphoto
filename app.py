from flask import Flask, request, jsonify
import cv2
import face_recognition
import numpy as np
from minio import Minio
from minio.error import S3Error
from PIL import Image
import io

app = Flask(__name__)

minio_client = Minio(
    "bucket-production-e5ac.up.railway.app:443",
    access_key="DbzPDe4KUIr9zJcV62FJ",
    secret_key="PeGICqOrGUr9YeeYE5z0QA1qXAYhQMn5AMFW7Pfd",
    secure=True
)

bucket_name = "imagesusers"

def convert_image_to_rgb(image_file):
    image = Image.open(image_file).convert('RGB')
    return np.array(image)

def get_image_from_minio(email):
    try:
        response = minio_client.get_object(bucket_name, f"{email}.jpg")
        image_data = response.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        return np.array(image)
    except S3Error as e:
        print(f"Error fetching image for {email}: {e}")
        return None

@app.route('/recognize', methods=['POST'])
def recognize():
    file = request.files['file']
    email = request.form['email']
    if not file or not email:
        return jsonify({"error": "No file or email provided"}), 400

    try:
        # Convert the uploaded image to RGB format
        uploaded_image = convert_image_to_rgb(file)

        # Get the stored image from Minio
        stored_image = get_image_from_minio(email)
        if stored_image is None:
            return jsonify({"error": f"No image found for {email}"}), 404

        # Get face encodings for both images
        uploaded_face_locations = face_recognition.face_locations(uploaded_image)
        uploaded_face_encodings = face_recognition.face_encodings(uploaded_image, uploaded_face_locations)

        stored_face_locations = face_recognition.face_locations(stored_image)
        stored_face_encodings = face_recognition.face_encodings(stored_image, stored_face_locations)

        if not uploaded_face_encodings or not stored_face_encodings:
            return jsonify({"error": "No face found in one of the images"}), 400

        response = []

        for uploaded_face_encoding in uploaded_face_encodings:
            matches = face_recognition.compare_faces(stored_face_encodings, uploaded_face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(stored_face_encodings, uploaded_face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = email
                response.append({"status": "success", "email": email})
            else:
                response.append({"status": "error", "email": email})

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

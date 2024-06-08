import os
from flask import Flask, render_template, flash, redirect, url_for, session, logging, request, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename

import re
import re
import time
import warnings
warnings.filterwarnings('ignore')

ALLOWED_EXTENSIONS = ['csv', 'xlsx', 'tsv', 'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov']

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)
app.secret_key = "m4xpl0it"

# Define the directory to store uploaded datasets
UPLOAD_FOLDER = 'uploaded_datasets'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


class user(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80))
    email = db.Column(db.String(120))
    password = db.Column(db.String(80))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/user")
def index_auth():
    return render_template("index_auth.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        uname = request.form["uname"]
        passw = request.form["passw"]

        login = user.query.filter_by(username=uname, password=passw).first()
        if login is not None:
            return redirect(url_for("index_auth"))
    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        uname = request.form['uname']
        mail = request.form['mail']
        passw = request.form['passw']

        register = user(username=uname, email=mail, password=passw)
        db.session.add(register)
        db.session.commit()

        return redirect(url_for("login"))
    return render_template("register.html")




@app.route('/pred_page')
def pred_page():
    pred = session.get('pred_label', None)
    f_name = session.get('filename', None)
    return render_template('pred.html', pred=pred, f_name=f_name)


@app.route("/upload", methods=['POST', 'GET'])
def upload():

    print("Uplaoding...")
    try:
        if request.method == 'POST':
            f = request.files['bt_image']
            filename = str(f.filename)

            if filename != '':
                ext = filename.split(".")

                if ext[1] in ALLOWED_EXTENSIONS:
                    filename = secure_filename(f.filename)
                    # f.save('test_images/'+filename)
                    f.save('static/test_images/test.jpg')

                    pred = predict_deepfake('static/test_images/test.jpg',model)

                    # if result[0][0] >0.45:
                    #     pred = "is"
                    #     prob=str(result[0][0]*100)[:4]
                    # else:
                    #     pred = "is not"
                    #     prob=str(100-(result[0][0]*100))[:4]
                   
                    return render_template('pred.html', prob="", pred=pred, f_name='../../test_images/'+filename)

                    # return redirect(url_for('pred_page'))

    except Exception as e:
        print("Exception\n")
        print(e, '\n')

    return render_template("upload.html")




from keras.preprocessing import image

import cv2
import numpy as np
import skimage
from keras.models import model_from_json
from skimage import transform

categories=['AI Generated Fake Image','Real Image']


def load_model(weight,config):

    # Load the model architecture from JSON file
    json_file = open(config, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # Load the model weights from h5 file
    loaded_model.load_weights(weight)

    return loaded_model



def predict_deepfake(file_path,model):
  img = cv2.imread(file_path,cv2.IMREAD_GRAYSCALE)
  img = transform.resize(img, (224, 224, 3))
  # img = image.load_img(photo, target_size=(64, 64))
  x = image.img_to_array(img)

  x = np.expand_dims(x, axis=0)

  preds = model.predict(x,batch_size=None, verbose=1)
  labels = np.argmax(preds, axis=-1)
  print("\nPREDICTION : "+categories[labels[0]])
  probs = np.exp(preds) / np.sum(np.exp(preds), axis=1, keepdims=True)
  all_percentages =   probs * 100
  perc  = max(all_percentages[0])

  return str(perc)[0:5] +" % probability of " + categories[labels[0]]


model = load_model(r"models/model_weight.h5",r"models/model.json")





from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from keras.preprocessing import image
from skimage import transform
from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from keras.preprocessing import image
from skimage import transform

FAKE_IMAGES_FOLDER = 'static/fake_images'

app.config['FAKE_IMAGES_FOLDER'] = FAKE_IMAGES_FOLDER
os.makedirs(FAKE_IMAGES_FOLDER, exist_ok=True)

 
@app.route('/fake_images/<path:filename>')
def serve_fake_image(filename):
    return send_from_directory(app.config['FAKE_IMAGES_FOLDER'], filename)


def predict_deepfake_frame(frame, model):
    # Preprocess the frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = transform.resize(frame, (224, 224, 3))
    x = image.img_to_array(frame)
    x = np.expand_dims(x, axis=0)

    # Predict the frame
    preds = model.predict(x, batch_size=None, verbose=0)
    labels = np.argmax(preds, axis=-1)

    return 'AI Generated Fake Image' if labels[0] == 0 else 'Real Image'


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/upload_video")
def upload_video():
    return render_template("upload_video.html")





import os
import cv2
from mtcnn import MTCNN

@app.route("/detect_deepfake", methods=['POST'])
def detect_deepfake():
    print("Started")
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400

    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No selected video file'}), 400

    if video_file and allowed_file(video_file.filename):
        # Save the uploaded video temporarily
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(video_file.filename))
        video_file.save(video_path)

        # Load the deepfake detection model
        model = load_model("models/model_weight.h5", "models/model.json")

        # Initialize the MTCNN face detector
        detector = MTCNN()

        # Remove all existing images from the fake images folder
        for filename in os.listdir(app.config['FAKE_IMAGES_FOLDER']):
            file_path = os.path.join(app.config['FAKE_IMAGES_FOLDER'], filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)

        # Open the video file
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(total_frames)

        # Calculate frame skipping based on the total number of frames
        skip_frames = 0
        if total_frames >= 100 and total_frames < 200:
            skip_frames = 1
        elif total_frames >= 200 and total_frames < 300:
            skip_frames = 3
        elif total_frames >= 300 and total_frames < 400:
            skip_frames = 4
        elif total_frames >= 400 and total_frames < 500:
            skip_frames = 5
        elif total_frames >= 500 and total_frames < 600:
            skip_frames = 6
        elif total_frames >= 600 and total_frames < 700:
            skip_frames = 7
        elif total_frames >= 700 and total_frames < 800:
            skip_frames = 8
        elif total_frames >= 800 and total_frames < 900:
            skip_frames = 9
        elif total_frames >= 900 and total_frames < 1000:
            skip_frames = 10
        elif total_frames >= 1000 and total_frames < 1100:
            skip_frames = 11
        elif total_frames >= 1100 and total_frames < 1200:
            skip_frames = 12
        elif total_frames >= 1200 and total_frames < 1300:
            skip_frames = 13
        elif total_frames >= 1300 and total_frames < 1400:
            skip_frames = 14

        frame_count = 0
        fake_frame_count = 0
        real_frame_count = 0
        fake_image_paths = []

        # Create an OpenCV window
        cv2.namedWindow('Video with Fake Frames', cv2.WINDOW_NORMAL)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            if frame_count % skip_frames != 0:
                continue

            # Detect faces in the frame
            faces = detector.detect_faces(frame)

            for face in faces:
                # Extract face coordinates
                x, y, width, height = face['box']
                x2, y2 = x + width, y + height

                # Extract face region
                face_img = frame[y:y2, x:x2]

                # Perform deepfake detection on the face
                pred = predict_deepfake_frame(face_img, model)

                # Save fake frames as images
                if pred == 'AI Generated Fake Image':
                    fake_frame_count += 1
                    fake_image_path = os.path.join(app.config['FAKE_IMAGES_FOLDER'], f"fake_{fake_frame_count}.jpg")
                    fake_image_paths.append(fake_image_path)
                    # Draw bounding box around face
                    cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 2)


                    # Draw text "Fake Face" above the bounding box
                    cv2.putText(frame, 'Fake Face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                    cv2.imwrite(fake_image_path, frame)

                else:
                    real_frame_count += 1
                    cv2.rectangle(frame, (x, y), (x2, y2), (0,255,0), 2)

            # Show the frame in the OpenCV window
            cv2.imshow('Video with Fake Frames', frame)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break


        # Calculate average prediction or count of fake and real frames
        total_frames = fake_frame_count + real_frame_count
        if total_frames > 0:
            fake_percentage = (fake_frame_count / total_frames) * 100
            real_percentage = (real_frame_count / total_frames) * 100
        else:
            fake_percentage = 0
            real_percentage = 0

        return jsonify({
            'fake_frame_count': fake_frame_count,
            'real_frame_count': real_frame_count,
            'fake_percentage': fake_percentage,
            'real_percentage': real_percentage,
            'fake_image_paths': fake_image_paths
        })

    return jsonify({'error': 'Invalid file format'}), 400

if __name__ == "__main__":
    db.create_all()
    app.run(debug=False, port=3000)

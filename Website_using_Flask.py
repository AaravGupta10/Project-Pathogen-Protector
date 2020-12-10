from keras.models import load_model
import cv2
import numpy as np
import time
from flask import Flask, render_template, Response


app = Flask(__name__)

model = load_model('model-017.model')

face_clsfr = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# source=cv2.VideoCapture(0)

labels_dict = {0:'MASK',1:'NO MASK'}
color_dict = {0:(0,255,0), 1:(0,0,255)}

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')




def gen():
    """Video streaming generator function."""
    source = cv2.VideoCapture(0)

    # Read until video is completed
    while (source.isOpened()):
        # Capture frame-by-frame
        ret, img = source.read()
        if ret == True:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_clsfr.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face_img = gray[y:y + w, x:x + w]
                resized = cv2.resize(face_img, (100, 100))
                normalized = resized / 255.0
                reshaped = np.reshape(normalized, (1, 100, 100, 1))
                result = model.predict(reshaped)

                label = np.argmax(result, axis=1)[0]

                cv2.rectangle(img, (x, y), (x + w, y + h), color_dict[label], 2)
                cv2.rectangle(img, (x, y - 40), (x + w, y), color_dict[label], -1)
                cv2.putText(img, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            img = cv2.resize(img, (0, 0), fx=1, fy=1)
            frame = cv2.imencode('.jpg', img)[1].tobytes()
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.1)
        else:
            break


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run()





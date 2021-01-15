from keras.models import load_model
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import csv
import time
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders


model = load_model('model-017.model')

face_clsfr=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')



labels_dict={0:'MASK',1:'NO MASK'}
color_dict={0:(0,255,0),1:(0,0,255)}
section = "9D"
current_time = datetime.now()
date = f'{current_time.day}-{current_time.month}-{current_time.year}'
filepath = 'Mask Defaulters Lists'
filename = f'Mask Defaulters_{section}_{date}.csv'
filename_path = f'{filepath}/{filename}'
source = cv2.VideoCapture(1)

path = 'recog_images'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

fromaddr = "noreply.pathogenprotector@gmail.com"
toaddr = "anujaarav06@gmail.com"

msg = MIMEMultipart()

msg['From'] = fromaddr
msg['To'] = toaddr
msg['Subject'] = "Mask Defaulters List"
body = f"PFA the list for mask defaulters for {section} on {date}."


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
        return encodeList


def markAttendance(name):
    with open(filename_path, 'w') as f:
        with open(filename_path, 'r+') as f:
            _writer = csv.writer(f)
            _writer.writerow(['Mask_Defaulter_Name', 'Time'])
            myDataList = f.readlines()
            nameList = []
            for line in myDataList:
                entry = line.split(',')
                nameList.append(entry[0])
            if name not in nameList:
                now = datetime.now()
                dtString = now.strftime('%H:%M:%S')
                f.writelines(f'\n{name},{dtString}')


encodeListKnown = findEncodings(images)


while True:


        ret, img = source.read()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_clsfr.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = gray[y:y + w, x:x + w]
            resized = cv2.resize(face_img, (100, 100))
            normalized = resized / 255.0
            reshaped = np.reshape(normalized, (1, 100, 100, 1))
            result = model.predict(reshaped)

            label = np.argmax(result, axis=1)[0]
            if label==0:
                cv2.rectangle(img, (x, y), (x + w, y + h), color_dict[label], 2)
                cv2.rectangle(img, (x, y - 40), (x + w, y), color_dict[label], -1)
                cv2.putText(img, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            elif label==1:
                imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
                imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

                facesCurFrame = face_recognition.face_locations(imgS)
                encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

                for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                    matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                    faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                    matchIndex = np.argmin(faceDis)

                    if matches[matchIndex]:
                        name = classNames[matchIndex].upper()
                        y1, x2, y2, x1 = faceLoc
                        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)
                        cv2.putText(img, f'NO MASK: {name}', (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
                    markAttendance(name)

        cv2.imshow("Pathogen Protector", img)

        if cv2.waitKey(1) == 27:
            msg.attach(MIMEText(body, 'plain'))
            attachment = open(filename_path, "rb")
            p = MIMEBase('application', 'octet-stream')
            p.set_payload((attachment).read())
            encoders.encode_base64(p)
            p.add_header('Content-Disposition', "attachment; filename= %s" % filename)
            msg.attach(p)
            s = smtplib.SMTP('smtp.gmail.com', 587)
            s.starttls()
            s.login(fromaddr, "pcubed_csv")
            text = msg.as_string()
            s.sendmail(fromaddr, toaddr, text)
            s.quit()
            print('Email sent')






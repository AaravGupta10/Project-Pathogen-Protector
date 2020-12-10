import cv2
import numpy as np
from pyzbar.pyzbar import decode
from datetime import *

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    success, img = cap.read()

    for barcode in decode(img):
        # print(barcode.data)
        myData = barcode.data.decode('utf-8')
        print(myData)

        def MarkAttendance():
            with open('temp.csv', 'r+') as file:

                Status = 'Outside'

                Date = date.today()
                Date = Date.strftime('%d/%m/%Y')

                Time = datetime.now()
                Time = Time.strftime('%H:%M:%S')

                file.writelines(f'\n{myData},{Date},{Time},{Status}')

        pts = np.array([barcode.polygon], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True, (255, 0, 255), 5)
        pts2 = barcode.rect
        cv2.putText(img, myData, (pts2[0], pts2[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)

        MarkAttendance()

        cv2.imshow('Result', img)
        cv2.waitKey(1)

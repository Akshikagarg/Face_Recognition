import numpy as np
import cv2
from recognize_faces import recognize
from encode_faces import encode
import os

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            print("Path exists")
        else:
            print("Exception")

c = cv2.VideoCapture(0)

while(True):
    ret, frame = c.read()

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('p'):
        print(recognize(frame))
    if cv2.waitKey(1) & 0xFF == ord('r'):
        name = input("Enter the name - ")
        path = 'dataset/' + str(name)
        mkdir_p(path)
        cv2.imwrite("dataset/" + str(name) + "/" + str(name) + ".jpg", frame)
        encode()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

c.release()
cv2.destroyAllWindows()

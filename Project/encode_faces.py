# import the necessary packages
from imutils import paths
import face_recognition
import pickle
import cv2
import os

def encode():
	
	print("[INFO] Quantifying Faces...")
	Paths = list(paths.list_images("dataset"))

	knownEncodings = []
	knownNames = []

	for (i, Path) in enumerate(Paths):
		
		print("[INFO] processing image {}/{}".format(i + 1,
			len(Paths)))
		name = Path.split(os.path.sep)[-2]

		image = cv2.imread(Path)
		rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
 
		boxes = face_recognition.face_locations(rgb,
			model="hog")

		encodings = face_recognition.face_encodings(rgb, boxes)

		for encoding in encodings:
			knownEncodings.append(encoding)
			knownNames.append(name)

	print("[INFO] serializing encodings...")
	data = {"encodings": knownEncodings, "names": knownNames}
	f = open("encodings.pickle", "wb")
	f.write(pickle.dumps(data))
	f.close()

import face_recognition
import pickle
import cv2

def recognize(image):
	# load the known faces and embeddings
	print("[INFO] loading encodings...")
	data = pickle.loads(open("encodings.pickle", "rb").read())

	# load the input image and convert it from BGR to RGB
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# detect the (x, y)-coordinates of the bounding boxes then compute the facial embeddings
	print("[INFO] recognizing faces...")
	boxes = face_recognition.face_locations(rgb,
		model="hog")
	encodings = face_recognition.face_encodings(rgb, boxes)

	names = []

	# loop over the facial embeddings
	for encoding in encodings:

		matches = face_recognition.compare_faces(data["encodings"],
			encoding)
		name = "Unknown"

		# check to see if we have found a match
		if True in matches:
			
			matchedIdxs = [i for (i, b) in enumerate(matches) if b]
			counts = {}

			# loop over the matched indexes and maintain a count for each recognized face
			for i in matchedIdxs:
				name = data["names"][i]
				counts[name] = counts.get(name, 0) + 1

			name = max(counts, key=counts.get)
		
		
		names.append(name)

	# loop over the recognized faces
	# for ((top, right, bottom, left), name) in zip(boxes, names):
	# draw the predicted face name on the image
	# cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
	# y = top - 15 if top - 15 > 15 else top + 15
	# cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
	# 0.75, (0, 255, 0), 2)

	return names
	

from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os

#creating argument parsing
ap = argparse.ArgumentParser()
ap.add_argument("-i","--dataset",required=True,help="Path of the dataset.(Compulsory)")
ap.add_argument("-e","--encodings",required=True,help="path to serialized db")
ap.add_argument("-d","--detection-method",type=str,default="cnn",help="hog or cnn")
args = vars(ap.parse_args())

#getting all image paths
image_path = list(paths.list_images(args["dataset"]))

knownEncodings =[]
knownNames = []

for (i,imagePath) in enumerate(image_path):
    print("**Image {0} of {1}**".format(i+1,len(image_path)))
    name = imagePath.split(os.path.sep)[-2]
    print(name)
    image = cv2.imread(imagePath)
    image  = cv2.resize(image,(640,480))
    rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(image,model=args["detection_method"])
    encodings = face_recognition.face_encodings(image,boxes)

    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(name)


print("**Serializing Encoding...**")
data = {"encodings":knownEncodings,"names":knownNames}
f = open(args["encodings"],"wb")
f.write(pickle.dumps(data))
f.close()
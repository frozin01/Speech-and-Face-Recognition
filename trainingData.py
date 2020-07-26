import cv2
import os
import numpy as np
from PIL import Image

faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
path='datacollected'

#function baca label
def getImageLabel(path):
    imagePaths = [os.path.join(path, file) for file in os.listdir(path)] #buat list dari path yang berisi gambar wajah
    faces = []
    IDs = []

    #baca dalam folder
    for imagePath in imagePaths:
        imgPIL = Image.open(imagePath).convert('L') #convert ke gray
        imgNum = np.array(imgPIL, 'uint8') #convert PIL image to numpy array
        ID = int(os.path.split(imagePath)[-1].split(".")[1])
        face=faceDetector.detectMultiScale(imgNum) #detect wajah di imgNum

        #tambah label ke wajah
        for (x,y,w,h) in face:
            faces.append(imgNum[y:y+h,x:x+w])
            IDs.append(ID)

    return IDs, faces

#training data
print("Mesin sedang melakukan training data wajah. Please wait...")
Ids, faces = getImageLabel(path)
faceRecognizer.train(faces, np.array(Ids))

#tulis file hasil training
faceRecognizer.write('trainingfiles' + '/training.xml')
print("Training telah selesai")

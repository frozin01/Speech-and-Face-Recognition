import cv2

#setting camera
camera = cv2.VideoCapture(0)
camera.set(3, 640) #height camera
camera.set(4,480) #widht camera

#detector
faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#input nama data
faceID = input("Masukkan User ID : ")
print("Collecting data ....")
ambilData = 1

#looping camera on
while True:
    retValue, frame = camera.read()
    frame_abu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #convert to gray
    faces = faceDetector.detectMultiScale(frame_abu, 1.3, 5) #deteksi wajah dari frame_abu
    cv2.imshow('Collecting Data', frame)

    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2) #rectangle untuk wajah
        namaFile = 'User.'+faceID+'.'+str(ambilData)+'.jpg'
        cv2.imwrite('datacollected/'+namaFile, frame) #simpan data frame
        ambilData +=1
        cv2.waitKey(100) #delay 100ms

    if ambilData>50:
        break

print("Pengambilan data selesai")
camera.release()
cv2.destroyAllWindows()
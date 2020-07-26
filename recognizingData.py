import cv2
import datetime
import speech_recognition as sr
import webbrowser
import playsound
import os
import random
from gtts import gTTS

#for speech recognizer
speech = sr.Recognizer()

#function for camera
def camera():
    # setting camera
    camera = cv2.VideoCapture(0)
    camera.set(3, 640)  # height camera
    camera.set(4, 480)  # widht camera

    faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
    faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    faceRecognizer.read("trainingfiles" + '/training.xml')
    font = cv2.FONT_HERSHEY_SIMPLEX

    names = ['Tidak Diketahui', 'Rozin', 'Jokowi']  # data nama

    minWidth = 0.1 * camera.get(3)
    minHeight = 0.1 * camera.get(4)

    while True:
        retValue, frame = camera.read()
        frame = cv2.flip(frame, 1)
        abu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceDetector.detectMultiScale(abu, 1.2, 5, minSize=(round(minWidth), round(minHeight)))
        date = str(datetime.datetime.now())

        for (x, y, w, h) in faces:
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            id, confidence = faceRecognizer.predict(abu[y:y + h, x:x + w])
            if confidence <= 75:
                nameID = names[id]
                confidenceTxt = "{0}%".format(round(120 - confidence))
                cv2.putText(frame, str(confidenceTxt), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)  # persennya
            else:
                nameID = names[0]
            cv2.putText(frame, str(nameID), (x + 5, y - 5), font, 1, (255, 255, 255), 2)  # namanya

        cv2.putText(frame, date, (10, 50), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)  # date and time
        cv2.imshow('Recognize Wajah', frame)

        # close camera
        k = cv2.waitKey(1) & 0xFF
        if k == 27 or k == ord('q'): #press q or esc
            break

    camera.release()
    cv2.destroyAllWindows()

#function for record audio
def record_audio():
    with sr.Microphone() as source:
        audio = speech.listen(source)
        voice_data=''
        try:
            voice_data = speech.recognize_google(audio)
        except sr.UnknownValueError:
            None
        except sr.RequestError:
            None
        return voice_data

#function for respond audio
def respond_audio(voice_data):
    if "what is your name" in voice_data:
        speak("my name is Rozin's Assistant")
    if "help please" in voice_data:
        speak("What do you want to search for?")
        search = record_audio()
        url="https://google.com/search?q="+search
        webbrowser.get().open(url)
        speak('here is what i found for '+search)
    if "open please" in voice_data:
        speak("open the camera")
        camera()
    if "thank you" in voice_data:
        speak("your welcome sir")
    if "stop" in voice_data:
        speak('thank you sir, have a nice day. Good bye')
        exit()

#function for speak audio
def speak(audio_string):
    tts=gTTS(text=audio_string,lang='en')
    speech=random.randint(1,20000000)
    audio_file='audio-'+str(speech)+'.mp3'
    tts.save(audio_file)
    playsound.playsound(audio_file)
    os.remove(audio_file)

#inisiasi
speak("Welcome back sir!")
loop=True

#main code
while loop:
    voice_data=record_audio()
    respond_audio(voice_data)
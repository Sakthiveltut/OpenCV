import cv2

trainedDataset = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#video
video = cv2.VideoCapture("E:\BACKUP\SD Card\Videos\Jerald.mp4")

#Webcam
# video = cv2.VideoCapture(2)

while True:
    success, frame = video.read()
    if success==True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = trainedDataset.detectMultiScale(gray)
        print(faces)

        for x,y,w,h in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255),2)

        cv2.imshow("video", frame)
        key = cv2.waitKey(1) 
        if key == 81 or key == 113:
            break
    else:
        print("Video completed or Frame Nil")
        break


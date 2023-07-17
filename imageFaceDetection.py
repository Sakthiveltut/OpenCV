import cv2

#Trained Dataset
trainedDataset = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#Read Images
img = cv2.imread("E:\BACKUP\SD Card\Photos\Enjoy.jpg")

#convert into grayscale
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces = trainedDataset.detectMultiScale(img)
print(faces)

for x,y,w,h in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255),2)

cv2.imshow("Original",img)
# cv2.imshow("Gray",gray)
cv2.waitKey()
import cv2
import numpy

face_path=r"C:\ProgramData\Anaconda3\pkgs\opencv3-3.1.0-py35_0\Library\etc\haarcascades\haarcascade_frontalface_default.xml"
faceCascade=cv2.CascadeClassifier(face_path)

#cv2.namedWindow('image2')
#cv2.waitKey(0)
img=cv2.imread(r'C:\Users\admin\Downloads\ae.jpg')
faces=faceCascade.detectMultiScale(img, scaleFactor=1.05, minNeighbors=5, minSize=(15, 15), flags=cv2.CASCADE_SCALE_IMAGE)
#img2=cv2.imread(r'C:\Users\admin\Downloads\image.png')
#cv2.rectangle(img, (10, img.shape[0]-20), (110, img.shape[0]), (0,0,0), -1)
#cv2.putText(img, )
#print(type(img))
#print(img.shape)
#cv2.rectangle(img, (500, 20), (580, 100), (0, 0, 255), 0)
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x,y), (x+w, y+h), (128,255,0), 0)

cv2.namedWindow("Facedetect")
cv2.imshow("Facedetect", img)

#cv2.imshow('image1',img)
#cv2.imshow('image2',img2)

cv2.waitKey(0)
cv2.imwrite(r'C:\Users\admin\desktop\img.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
cv2.destroyAllWindows()

#update by 20181031
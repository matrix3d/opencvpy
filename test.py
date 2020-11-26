import cv2
import numpy
kernel=numpy.ones((5,5),numpy.uint8)
img=cv2.imread("Body_tex_003.jpg");
imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow("",imgGray)
imgBlur=cv2.GaussianBlur(img,(7,7),0)
cv2.imshow("",imgBlur)
imgCanny=cv2.Canny(img,100,100)
cv2.imshow("",imgCanny)
imgDialation=cv2.dilate(imgCanny,kernel)
cv2.imshow("",imgDialation)

#cv2.imshow("output",img)
cv2.waitKey(0)

cap=cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

#while True:
#    success,img=cap.read()
#    cv2.imshow("vidow",img)
#    if cv2.waitKey(1)&0xff==ord("q"):
#        break
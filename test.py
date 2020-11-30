import cv2
import numpy
kernel=numpy.ones((5,5),numpy.uint8)
img=cv2.imread("shapes.png");
imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgBlur=cv2.GaussianBlur(imgGray,(7,7),1)
imgCanny=cv2.Canny(img,100,100)#描边
imgDialation=cv2.dilate(imgCanny,kernel,iterations=1)#膨胀
imgEroded=cv2.erode(imgDialation,kernel,iterations=2)#qinzhu
imgResize=cv2.resize(img,(1300,300))
imgCropped=img[0:200,200:400]#裁剪 miny:maxy,minx:maxx

cv2.imshow("",imgGray)
cv2.imshow("",imgBlur)
cv2.imshow("",imgCanny)
cv2.imshow("",imgDialation)
cv2.imshow("",imgEroded)
cv2.imshow("",imgResize)
cv2.imshow("",imgCropped)

img=numpy.zeros((200,300,3),numpy.uint8)
img[0:100,1:100]=255,0,0
cv2.line(img,(0,0),(img.shape[1],img.shape[0]),(0,255,0),3)
cv2.rectangle(img,(0,0),(100,200),(255,0,255),2)
cv2.circle(img,(150,150),20,(255,255,0),5)
cv2.putText(img,"11",(200,200),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),3)
cv2.imshow("",img)

width,height=200,200
#1   2
#3   4

pts1=numpy.float32([[0,0],[200,0],[0,300],[100,300]])
pts2=numpy.float32([[0,0],[width,0],[0,height],[width,height]])
matr=cv2.getPerspectiveTransform(pts1,pts2)
imgOutput=cv2.warpPerspective(img,matr,(width,height))

cv2.imshow("",imgOutput)


img=cv2.imread("Body_tex_003.jpg");
#色调（H），饱和度（S），明度（V
imgHSV=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
mask=cv2.inRange(imgHSV,numpy.array([100,0,0]),numpy.array([255,255,255]));

imgResult=cv2.bitwise_and(img,img,mask=mask)
def getContours(img):
    contours,hierarchy=cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area=cv2.contourArea(cnt)
        if area>500:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 2)
            peri=cv2.arcLength(cnt,True)#周长
            approx=cv2.approxPolyDP(cnt,0.02*peri,True)
            objCor=len(approx)
            x,y,w,h=cv2.boundingRect(approx)
            cv2.rectangle(imgContour,(x,y),(x+w,y+h),(0,255,0),2)
            if objCor==3:ObjType="Tri"
            else:ObjType=""


img=cv2.imread("shapes.png");
imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgBlur=cv2.GaussianBlur(imgGray,(7,7),1)
imgCanny=cv2.Canny(imgBlur,50,50)
imgContour=img.copy()
getContours(imgCanny)

#人脸识别
faceCascade=cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
eyeCascade=cv2.CascadeClassifier("haarcascades\haarcascade_eye.xml")
img=cv2.imread("pic.jpg")
imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces=faceCascade.detectMultiScale(imgGray,1.1,1)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),3)
    faceImgGray=imgGray[y:y+h,x:x+w]
    #cv2.imshow(str(x),faceImgGray)
    eyes=eyeCascade.detectMultiScale(faceImgGray,1.1,1)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(img,(x+ex,y+ey),(x+ex+ew,y+ey+eh),(255,255,0),1)

cv2.imshow("",imgHSV)
cv2.imshow("",mask)
cv2.imshow("",imgResult)
cv2.imshow("",imgCanny)
cv2.imshow("",imgContour)
cv2.imshow("",img)
#cv2.waitKey(0)
#exit()
cap=cv2.VideoCapture("2.mp4")
cap.set(3,640)
cap.set(4,480)

# 人脸识别
faceCascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
eyeCascade = cv2.CascadeClassifier("haarcascades\haarcascade_eye.xml")
while True:
    success,img=cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray, 1.1, 20)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 3)
        faceImgGray = imgGray[y:y + h, x:x + w]
        # cv2.imshow(str(x),faceImgGray)
        eyes = eyeCascade.detectMultiScale(faceImgGray, 1.1, 20)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(img, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (255, 255, 0), 1)
    cv2.imshow("vidow",img)
    if cv2.waitKey(int(1000/60)):
        if cv2.getWindowProperty("vidow",cv2.WND_PROP_VISIBLE)<=0:
            cv2.destroyAllWindows()
            break


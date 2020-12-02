import cv2
import numpy as np
import dlib


detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
def colorImg(img,mask,color):
    imgColor = np.zeros_like(img)
    imgColor[:] = color
    imgColor = cv2.bitwise_and(mask, imgColor)
    imgColor = cv2.GaussianBlur(imgColor, (7, 7), 3)
    # imgColorLips=cv2.bitwise_or(imgOriginal,imgColorLips)
    img = cv2.addWeighted(img, 1, imgColor, 1, 0)
    return img


cap=cv2.VideoCapture("2.mp4")
cap.set(3,640)
cap.set(4,480)

# 人脸识别
while True:
    success,img=cap.read()
    img=cv2.resize(img,None,None,0.5,0.5)

    maskLisp = np.zeros_like(img)
    maskEye = np.zeros_like(img)

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(imgGray)
    for face in faces:
        myPoints = []
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()
        # imgOriginal=cv2.rectangle(imgOriginal,(x1,y1),(x2,y2),(255,0,255))
        landmarks = predictor(imgGray, face)
        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            myPoints.append([x, y])
            # cv2.circle(imgOriginal,(x,y),2,(244,0,0),cv2.FILLED)
            # cv2.putText(imgOriginal,str(n),(x,y-10),cv2.FONT_HERSHEY_COMPLEX,0.1,(0,0,255),1)
        lipsPoints = myPoints[48:59]  # 嘴外圈
        lipsPoints = np.array(lipsPoints)
        maskLisp = cv2.fillPoly(maskLisp, [lipsPoints], (255, 255, 255))

        lipsPoints = myPoints[60:67]  # 挖去嘴内圈
        lipsPoints = np.array(lipsPoints)
        maskLisp = cv2.fillPoly(maskLisp, [lipsPoints], (0, 0, 0))

        lipsPoints = myPoints[36:41]  # 眼睛1
        lipsPoints = np.array(lipsPoints)
        maskEye = cv2.fillPoly(maskEye, [lipsPoints], (255, 255, 255))

        lipsPoints = myPoints[42:47]  # 眼睛2
        lipsPoints = np.array(lipsPoints)
        maskEye = cv2.fillPoly(maskEye, [lipsPoints], (255, 255, 255))

    img = colorImg(img, maskEye, (0, 0, 255))
    img = colorImg(img, maskLisp, (255, 100, 0))
    cv2.imshow("", img)


    if cv2.waitKey(int(1000/60)):
        if cv2.getWindowProperty("",cv2.WND_PROP_VISIBLE)<=0:
            cv2.destroyAllWindows()
            break
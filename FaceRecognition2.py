import face_recognition
import cv2

cap=cv2.VideoCapture("2.mp4")
cap.set(3,640)
cap.set(4,480)

all_face_encodings=[]
# 人脸识别
while True:
    success,img=cap.read()
    img = cv2.resize(img, None, None, 0.5, 0.5)
    face_encodings = face_recognition.face_encodings(img)
    face_locations = face_recognition.face_locations(img)
    for j in range(len(face_encodings)):
        face_encoding = face_encodings[j]
        best_location=face_locations[j]
        #results= face_recognition.compare_faces(all_face_encodings, face_encoding,0.5)
        distances= face_recognition.face_distance(all_face_encodings,face_encoding)
        bestValue,bestIndex=100,-1
        for i in range(len(distances)):
            v=distances[i]
            if v<bestValue:
                bestValue,bestIndex=v,i
        if bestValue >0.5:
            bestIndex=len(all_face_encodings)
            all_face_encodings.append(face_encoding)
        print("index"+str(bestIndex))
        cv2.rectangle(img,(best_location[3],best_location[0]),(best_location[1],best_location[2]),(255,0,255),2)
        cv2.putText(img,str(bestIndex),(best_location[3],best_location[0]),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),2)

    cv2.imshow("", img)
    if cv2.waitKey(int(1000/60)):
        if cv2.getWindowProperty("",cv2.WND_PROP_VISIBLE)<=0:
            cv2.destroyAllWindows()
            break
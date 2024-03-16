import cv2

cap=cv2.VideoCapture("D:\opencv_udemy/09_car_detection\car.mp4")
car_cascade=cv2.CascadeClassifier("D:\opencv_udemy/09_car_detection\car.xml")

while True:
    ret,frame=cap.read()
    frame=cv2.resize(frame,(640,360))

    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cars=car_cascade.detectMultiScale(gray,1.2,2)

    for (x,y,w,h) in cars:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)

    cv2.imshow("Video",frame)

    if cv2.waitKey(1)==27:
        break

cap.release()
cv2.destroyAllWindows()
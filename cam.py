import cv2
import numpy as np
cap = cv2.VideoCapture(0)

while True:
  ret,frame = cap.read()
  if ret==True:
    framegray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    _,umbral = cv2.threshold(framegray, 90, 255, cv2.THRESH_BINARY)
    umbral2 = cv2.bitwise_not(umbral)
    contornos,_ = cv2.findContours(umbral2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(umbral2, contornos, -1, (0,255,0), 3)
    for c in contornos:
      area = cv2.contourArea(c)
      if area > 30:
        M = cv2.moments(c)
        if (M["m00"]==0): M["m00"]=1
        x = int(M["m10"]/M["m00"])
        y = int(M['m01']/M['m00'])
        cv2.circle(frame, (x,y), 7, (0,255,0), -1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, '{},{}'.format(x,y),(x+10,y), font, 0.75,(0,255,0),1,cv2.LINE_AA)
        nuevoContorno = cv2.convexHull(c)
        cv2.drawContours(frame, [nuevoContorno], 0, (255,0,0), 3)

    cv2.imshow('umbral2',umbral2)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('s'):
  break
cap.release()
cv2.destroyAllWindows()

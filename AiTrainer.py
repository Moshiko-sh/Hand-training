import cv2
import numpy as np
import time
import PoseModule as pm

cap = cv2.VideoCapture(0)
pTime = 0
detector = pm.PoseDetection()
count= 0
dir = 0
count2= 0
dir2 = 0
while True:
    success, img = cap.read()
    img = detector.findPose(img,draw= False)
    lmList = detector.findPosion(img, draw=False)

    if len(lmList) != 0:

        #Right arm
        Rangel=detector.findAngel(img,12, 14, 16)
        Rangel = 360 - Rangel
        Rper = np.interp(Rangel, (200, 310), (0, 100))
        Rbar = np.interp(Rangel, (200,310), (400, 150))

        # Left arm
        Langel=detector.findAngel(img,11,13,15)
        per = np.interp(Langel,(200,310),(0,100))
        bar = np.interp(Langel,(200,310),(400,150))
        #print(angel,per)

        #Check for the dunmbbel cycle
        if per == 100:
            if dir == 0:  # up
                count += 0.5
                dir = 1

        if per == 0:
            if dir == 1:   # Down
                count += 0.5
                dir =0
        print (count)

        cv2.putText(img,str(int(count2)),(50,110),cv2.FONT_HERSHEY_TRIPLEX,2,(255,0,255),2)

        cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
        cv2.rectangle(img, (50, int(Rbar)), (85, 400), (255, 0, 0), cv2.FILLED)
        cv2.putText(img, f'{int(Rper)}%', (40, 450), cv2.FONT_HERSHEY_COMPLEX,
                    1, (0, 255, 0), 3)

        # Check for the dunmbbel cycle
        if Rper == 100:
            if dir2 == 0:  # up
                count2 += 0.5
                dir2 = 1

        if Rper == 0:
            if dir2 == 1:  # Down
                count2 += 0.5
                dir2 = 0
        print(count2)

        cv2.putText(img, str(int(count)), (550, 110), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 0, 255), 2)

        cv2.rectangle(img, (550, 150), (585, 400), (255, 0, 0), 3)
        cv2.rectangle(img, (550, int(bar)), (585, 400), (255, 0, 0), cv2.FILLED)
        cv2.putText(img, f'{int(per)}%', (550, 450), cv2.FONT_HERSHEY_COMPLEX,
                    1, (0, 255, 0), 3)



    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)

    cv2.imshow("Hand training", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
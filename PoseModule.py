import cv2
import mediapipe as mp
import time
import math

class PoseDetection:
    def __init__(self, mode=False, model_complexity=1, smooth=True, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.model_complexity = model_complexity
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.mode,
                                     model_complexity=self.model_complexity,
                                     smooth_landmarks=self.smooth,
                                     min_detection_confidence=self.detectionCon,
                                     min_tracking_confidence=self.trackCon)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosion(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 4, (0, 0, 255), cv2.FILLED)
        return self.lmList

    def findAngel(self ,img,p1,p2,p3,draw=True):
        #get line marks

        x1,y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        #Calculate the angel
        angel = math.degrees(math.atan2(y3-y2,x3-x2)-
                          math.atan2(y1-y2,x1-x2))
        if angel <0:
            angel  += 360
        #print (angel)


         # Draw
        if draw:
            cv2.line(img,(x1, y1),(x2, y2),(0, 255, 0),3)
            cv2.line(img, (x3, y3), (x2, y2), (0, 255, 0), 3)
            cv2.circle(img, (x1, y1), 4, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 10, (255, 0, 255),2)
            cv2.circle(img, (x2, y2), 4, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 0, 255), 2)
            cv2.circle(img, (x3, y3), 4, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 10, (255, 0, 255), 2)

            cv2.putText(img, str(int(angel)), (x2-20, y2+50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)

            return angel




def main():
    cap = cv2.VideoCapture(0)

    pTime = 0
    detector = PoseDetection()
    while True:
        success, img = cap.read()
        if not success:
            break

        img = detector.findPose(img)
        lmList = detector.findPosion(img)
        print(lmList)  # Corrected the print function call
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

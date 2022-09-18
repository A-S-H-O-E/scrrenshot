import ssl
import numpy as np
import mediapipe as mp 
import cv2
import pyautogui 
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands()
camera = cv2.VideoCapture(0)
ft = [8,12,16,20]
tt = 4
def ts(img,handslandmarks):
    if(handslandmarks):
        for h in handslandmarks:
            lmlist = []
            for lm in h.landmark:
                lmlist.append(lm)
            fingerfoldstatus = []
            for id in ft:
                if(lmlist[id].x<lmlist[id-3].x):
                    fingerfoldstatus.append(True)
                else:
                    fingerfoldstatus.append(False)
            if all(fingerfoldstatus)==True:
                ss = pyautogui.screenshot()
                ss = cv2.cvtColor(np.array(ss),cv2.COLOR_RGB2BGR)
                cv2.imwrite('ss.png',ss)
            mp_drawing.draw_landmarks(img, h, mp_hands.HAND_CONNECTIONS, mp_drawing.DrawingSpec((0,0,255),2,2), mp_drawing.DrawingSpec((0,255,0),4,2))
while True:
    ret,img = camera.read()
    height,width,channels = img.shape
    results = hands.process(img)
    handslandmarks = results.multi_hand_landmarks
    ts(img,handslandmarks)
    cv2.imshow('screenshot',img)
    cv2.waitKey(1)
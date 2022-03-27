from turtle import left
import cv2 as cv # Let's count the cars
import time

from matplotlib.pyplot import contour      # Let's count only after a few time

# Vamos cortar o vídeo da rodovia em duas regiões
# Região da direita e região da esquerda

cap = cv.VideoCapture("parking.mp4")

# Object detection drom stable camera 
object_detector = cv.createBackgroundSubtractorMOG2(history=100, varThreshold=40) # Extraindo só os objetos em movimento na camera estável


while True:
    ret, frame = cap.read()

    # Extraindo as regiões  

    height, width, _ = frame.shape

    leftRegion = frame[0:360,0:320]
    rightRegion = frame[150:360, 320:640]

    mask = object_detector.apply(rightRegion)
    _, mask = cv.threshold(mask, 254, 255, cv.THRESH_BINARY) # Vamos deixar somente os pixel totalmente branco e totalmente preto
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        
        area = cv.contourArea(cnt)

        if area > 100:
            #cv.drawContours(rightRegion, [cnt], -1, (0, 255, 0), 2)
            x, y, h, w = cv.boundingRect(cnt)
            cv.rectangle(rightRegion, (x, y), (x+w, y+h), (0, 255, 0), 3)


    cv.imshow("Frame", rightRegion)
    cv.imshow("Mask", mask)

    key = cv.waitKey(30)

    if key == 27: # Para podermos fechar com uma tecla, possivelmente 27 é o S
        break

cap.release()
cv.destroyAllWindows()
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np
import time


IMAGE_SIZE = (1280, 720)
GRAD_THRESHOLD = 50
BG_THRESHOLD = 50
MOTION_THRESHOLD = 20
WIRE_BLACK = (10, 10, 10)
ERODE_SIZE = 11
DILATE_SIZE = 11
FIELD_DILATE_SIZE = 51
APPROX_EPS = 20
TIME_SCALE = 100.0

ROBOT_MIN = 50000
MOVING_MIN = 50000
PARK_MIN = 50000
NO_PARK_MIN = 30000



WAITING = "WAITING"
PARKED = "PARKED"
STUMBLED = "STUMBLED"


def Sobel_1_color(gray):
    ddepth = cv.CV_16S
    mix = 0.5

    grad_x = cv.Sobel(gray, ddepth = ddepth, dx = 1, dy = 0)
    grad_y = cv.Sobel(gray, ddepth = ddepth, dx = 0, dy = 1)
    grad_x_abs = cv.convertScaleAbs(grad_x)
    grad_y_abs = cv.convertScaleAbs(grad_y)

    grad = cv.addWeighted(grad_x_abs, mix, grad_y_abs, mix, 0)

    return grad


def findFinishBox(frame):
    W = frame.shape[1]
    H = frame.shape[0]
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    edges = Sobel_1_color(gray)
    edges = cv.inRange(edges, 50, 255)
    mask = cv.copyMakeBorder(edges, 1, 1, 1, 1, cv.BORDER_REPLICATE)
    cv.floodFill(edges, mask, (W // 2, H // 2), 255)

    kernel = np.ones((ERODE_SIZE, ERODE_SIZE), np.uint8)
    cv.erode(edges, kernel, edges)
    kernel = np.ones((DILATE_SIZE, DILATE_SIZE), np.uint8)
    cv.dilate(edges, kernel, edges)

    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    areas = np.vectorize(cv.contourArea, signature='(m,1,2)->()')(contours)
    boxContour = contours[np.argmax(areas)]
    #box = cv.approxPolyDP(boxContour, APPROX_EPS, True)
    box = boxContour

    kernel = np.ones((FIELD_DILATE_SIZE, FIELD_DILATE_SIZE), np.uint8)
    field = cv.dilate(edges, kernel)

    return box, edges, field

def findMotion(frame, background, threshold):
    diff = cv.absdiff(frame, background)
    diff = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
    mask = cv.inRange(diff, threshold, 255)
    return mask

def processMasks(bgdiffmask, motionmask, boxmask, fieldmask, wiremask):
    robotmask = cv.bitwise_and(bgdiffmask, fieldmask)
    movemask = cv.bitwise_and(motionmask, fieldmask)
    parkmask = cv.bitwise_and(robotmask, boxmask)
    noparkmask = cv.bitwise_and(cv.bitwise_and(robotmask, wiremask), cv.bitwise_xor(boxmask, fieldmask))

    displaymask = cv.merge((movemask, parkmask, noparkmask))
    #blackmask = cv.inRange(displaymask, (0,0,0), (1,1,1))
    #cv.copyTo(np.ones_like(displaymask) * 255, blackmask, displaymask)

    return np.sum(robotmask), np.sum(movemask), np.sum(parkmask), np.sum(noparkmask), displaymask

def fancyDisplayImage(frame, displaymask, state=None, time=None):
    displaymask = cv.resize(displaymask, IMAGE_SIZE)
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
    cv.polylines(frame, [finishBox], True, (0, 255, 0), 1)
    frame = cv.add(frame, displaymask)

    if state is not None:
        if state == PARKED:
            color = (0,255,0)
        elif state == STUMBLED:
            color = (0,0,255)
        else:
            color = (255,0,0)
        text = str(int(time * TIME_SCALE) / TIME_SCALE) + "s"

        cv.putText(frame, text, (100, 100), cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    return frame

def processSums(robotsum, movesum, parksum, noparksum, state):
    if movesum > MOVING_MIN:
        return WAITING
    elif robotsum > ROBOT_MIN:
        if parksum > PARK_MIN and noparksum < NO_PARK_MIN:
            return PARKED
        else:
            return STUMBLED
    else:
        return WAITING
        
    


cap = cv.VideoCapture("./videos/2019-04-17-234435.webm")

ret, background = cap.read()
background = cv.resize(background, IMAGE_SIZE)
prevframe = background.copy()
finishBox, boxmask, fieldmask = findFinishBox(background)

state = "WAITING"
stateTime = time.time()

while (cap.isOpened()):
    
    ret, frame = cap.read()
    if ret == False:
        break
    frame = cv.resize(frame, IMAGE_SIZE)

    start_time = time.time()

    motionmask = findMotion(frame, prevframe, MOTION_THRESHOLD)
    bgdiffmask = findMotion(frame, background, BG_THRESHOLD)
    wiremask = cv.inRange(frame, WIRE_BLACK, (255, 255, 255))
    robotsum, movesum, parksum, noparksum, displaymask = processMasks(bgdiffmask, motionmask, boxmask, fieldmask, wiremask)

    cv.copyTo(frame, None, prevframe)

    if state == WAITING:
        cv.imshow("Image", fancyDisplayImage(frame, displaymask, state, time.time() - stateTime))
    else:
        cv.imshow("Image", fancyDisplayImage(frame, displaymask))
    if cv.waitKey(1) & 0xFF == ord('q'):
        break


    newState = processSums(robotsum, movesum, parksum, noparksum, state)
    if (state != newState):
        state = newState
        newTime = time.time()
        print(state, newTime, newTime - stateTime, robotsum, movesum, parksum, noparksum)

        if state != WAITING:
            cv.imshow("Image", fancyDisplayImage(frame, displaymask, state, newTime - stateTime))
            cv.waitKey()
        stateTime = newTime


    #print(str(int((time.time() - start_time) * 1000)) + " ms")


cap.release()


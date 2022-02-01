import cv2
import threading
from cv2 import aruco
import numpy as np
import math
import pandas as pd
import socket
import pyrealsense2 as rs
import numpy as np
from collections import deque
import time
from datetime import datetime
import csv




# UDP_IP = "192.168.139.174"
# UDP_PORT = 5065
# sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
last = []
direction = ""
global_centroids = {1: (0, 0), 2: (0, 0), 3: (0, 0), 4: (0, 0), 5: (0, 0), 6: (0, 0), 7: (0, 0), 8: (0, 0),
                         9: (0, 0)}
angle1 = 90
angle3 = 90
overall_update = 0

filename = "thread_test.csv"
fields = ["overall_update","time", "forward angle", "sideways angle", "direction string", "updated by", "count for thread"]


with open(filename, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)



class camThread(threading.Thread):
    def __init__(self, previewName, camID):
        threading.Thread.__init__(self)
        self.previewName = previewName
        self.camID = camID
        self.lock = threading.Lock()
    def run(self):
        print("Starting ", self.previewName)
        camPreview(self, self.previewName, self.camID, self.lock)



def camPreview(threadname, previewName, camID, lock):
    global direction
    global global_centroids
    global angle1
    global angle3
    global overall_update
    global filename
    lock.acquire()
    cv2.namedWindow(previewName)
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(camID)
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 60)
    pipeline.start(config)
    count = 1
    prev_centroids = {1: (0, 0), 2: (0, 0), 3: (0, 0), 4: (0, 0), 5: (0, 0), 6: (0, 0), 7: (0, 0), 8: (0, 0), 9: (0, 0)}
    c = np.array(np.zeros([9, 4, 2]))
    ids = [[1], [2], [3], [4], [5], [6], [7], [8], [9]]
    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            # depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            image = np.asanyarray(color_frame.get_data())
            #image = cv2.resize(color_image, (1200, 800))
            arucodict = aruco.Dictionary_get(aruco.DICT_6X6_50)
            arucoparams = aruco.DetectorParameters_create()
            (corners, id, rejected) = aruco.detectMarkers(image, arucodict, parameters=arucoparams)
            # print(id,corners)
            if id is not None:
                for (a, b) in zip(corners, id):
                    if b in range(0, 10):
                        c[b - 1] = a
            centroids = {1: (0, 0), 2: (0, 0), 3: (0, 0), 4: (0, 0), 5: (0, 0), 6: (0, 0), 7: (0, 0), 8: (0, 0),
                         9: (0, 0)}
            if len(corners) > 0:
                id = id.flatten()
                for (markerCorner, markerID) in zip(c, ids):
                    corners = markerCorner.reshape((4, 2))
                    (topLeft, topRight, bottomRight, bottomLeft) = corners
                    topRight = (int(topRight[0]), int(topRight[1]))
                    bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                    bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                    topLeft = (int(topLeft[0]), int(topLeft[1]))
                    # cv2.line(image, topLeft, topRight, (255, 0, 0), 5)
                    # cv2.line(image, topRight, bottomRight, (255, 0, 0), 5)
                    # cv2.line(image, bottomRight, bottomLeft, (255, 0, 0), 5)
                    # cv2.line(image, bottomLeft, topLeft, (255, 0, 0), 5)
                    cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                    cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                    # cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
                    if (cX == 0 and cY == 0 and count > 1):
                        centroids[markerID[0]] = prev_centroids[markerID[0]]
                    else:
                        centroids[markerID[0]] = (cX, cY)
                        prev_centroids[markerID[0]] = centroids[markerID[0]]
                global_centroids = centroids
                color = (0, 0, 255)
                if (len(centroids) > 1):
                    forwards = ""
                    sideways = ""
                    if previewName == "Camera 2":
                        output = cv2.line(image, global_centroids[4], global_centroids[8], color, 2)
                        # output = cv2.line(image, centroids[2], centroids[4], color, 2)
                        # output = cv2.line(image, centroids[4], centroids[6], color, 2)
                        output = cv2.line(image, global_centroids[4], global_centroids[8], (0, 255, 0), 2)
                    else:
                        output = cv2.line(image, global_centroids[1], global_centroids[2], color, 2)
                        # output = cv2.line(image, centroids[2], centroids[4], color, 2)
                        # output = cv2.line(image, centroids[2], centroids[6], color, 2)
                        output = cv2.line(image, global_centroids[1], global_centroids[2], (0, 255, 0), 2)

                    try:
                        angle1 = round(angle2([global_centroids[4], global_centroids[8]]), 2)
                        print("forward angle={} updated by {} at {}".format(angle1, threadname, datetime.now().time()))
                    except:
                        angle1 = 90

                    if angle1 in range(50,80):
                        forwards = "F1"
                    elif angle1 < 50:
                        forwards = "F2"
                    elif angle1 > 100:
                        forwards = "Re"

                    try:
                        angle3 = round(angle2([global_centroids[1], global_centroids[2]]), 2)
                        print("sideways angle={} updated by {} at {}".format(angle3, threadname, datetime.now().time()))
                    except:
                        angle3 = 90


                    if angle3 in range(70,80):
                        sideways = "R1"
                    elif angle3 < 70:
                        sideways = "R2"
                    elif angle3 in range(100,110):
                        sideways = "L1"
                    elif angle3 >= 110:
                        sideways = "L2"

                    if forwards != "" and sideways != "":
                        direction = "{}-{}".format(forwards, sideways)
                    else:
                        if forwards != "":
                            direction = forwards
                        else:
                            direction = sideways
                    message=direction+",{},{},{}".format(datetime.now().time(), threadname,count)
                    print(message)
                    overall_update +=1
                    rowtowrite = [overall_update, "'"+str(datetime.now().time()), angle1, angle3, direction, threadname, count]
                    with open(filename, 'a') as csvfile:
                        csvwriter = csv.writer(csvfile)
                        csvwriter.writerow(rowtowrite)
                    count += 1
                    # sock.sendto((message).encode(), (UDP_IP, UDP_PORT))
                    # cv2.putText(output, str(angle1), (0, 230), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255))
                    x= 960*(int(previewName[-1])-1)
                    cv2.moveWindow(previewName, x, 0)
                    cv2.imshow(previewName, output)
                    cv2.waitKey(1)
            cv2.imshow(previewName, image)
            cv2.waitKey(1)
    finally:
        # Stop streaming
        pipeline.stop()
    cv2.destroyWindow(previewName)
    lock.release()



def angle2(c):
    c1 = c[0]
    c2 = c[1]
    (x1,y1)=(c1[0],c1[1])
    (x2,y2)=(c2[0],c2[1])
    if x2 - x1 == 0:
        angle1 = 90
    else:
        slope1 = (-y2 + y1) / (x2 - x1)
        angle1 = math.atan(slope1) * 180 / 3.14
        if x2 > x1:
            angle1 = 180 + angle1
    return angle1
# Create two threads as follows


vid1 = "108322073120"
vid2 = "108322073108"
lock = threading.Lock()
thread1 = camThread("Camera 1", vid1)
thread2 = camThread("Camera 2", vid2)
thread1.start()
thread2.start()
thread1.join()
thread2.join()
import threading
import numpy as np
import math
import pandas as pd
import socket
import numpy as np
from collections import deque
import time
from datetime import datetime
import csv
import random



last = []
direction = ""
angle1 = 90
angle3 = 90
overall_update = 0

filename = "other_test.csv"
fields = ['Entry','Time', 'Thread', 'angle1 modified', 'angle2 modified']


with open(filename, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)



class camThread(threading.Thread):
    def __init__(self, previewName, camID):
        threading.Thread.__init__(self)
        self.previewName = previewName
        self.camID = camID
        self.lock = threading.Lock()
        # self.writer = csv.writer(filename)
    def run(self):
        print("Starting ", self.previewName)
        camPreview(self, self.previewName, self.camID, self.lock)



def camPreview(threadname, previewName, camID, lock):
    global direction
    global angle1
    global angle3
    global overall_update
    global filename
    lock.acquire()
    count = 1
    try:
        while True:

            now = "'"+str(datetime.now().time())
            angle1 = angle1 + random.randint(0, 20)
            angle3 = angle3 + random.randint(0, 20)
            overall_update += 1
            message = "{},{},{},{},{}".format(overall_update, datetime.now().time(), threadname, angle1, angle3)
            rowtowrite = [overall_update, now, threadname, angle1, angle3]
            print(message)
            with open(filename, 'a') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(rowtowrite)
            count += 1
    finally:
        print("done")
    lock.release()

lock = threading.Lock()
thread1 = camThread("Camera 1", "vid1")
thread2 = camThread("Camera 2", "vid2")
thread3 = camThread("Camera 3", "vid3")
thread4 = camThread("Camera 4", "vid4")
thread1.start()
thread2.start()
thread3.start()
thread4.start()
thread1.join()
thread2.join()
thread3.join()
thread4.join()
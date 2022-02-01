from threading import Thread
import time
import socket

import ctypes  # An included library with Python install.
cur_time = time.time()

print("starting communication...")
HOST = "0.0.0.0"
MESSAGE = " 1 "

PORT1 = 9999
UDP_IP = "192.168.1.132"
# PORT1, UDP_IP= 8000, "172.1.14.35" //P1
# PORT1, UDP_IP= 7000, "172.1.14.36" //P12
# PORT1, UDP_IP= 6000, "172.1.14.37" //P13


s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind((HOST, PORT1))

# d = '1'
# s.sendto(d.encode(), (UDP_IP, PORT1))

count =0
rate = 0
while 1:
        if (count % 30) == 0:
                cur_time = time.time()
        if (count % 30) == 29:
                rate = 30 / (time.time() - cur_time)
        data = s.recv(1024)
        data = data.decode()
        count +=1
        print(rate, count)


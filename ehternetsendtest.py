import socket

UDP_IP = "172.1.14.37"
UDP_PORT = 6000
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
i=1
while 1 :
    sock.sendto(('{}'.format(i)).encode(), (UDP_IP, UDP_PORT))
    print(i)
    i+=1

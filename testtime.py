import time

before = time.time()
for i in range(0,20000000):
    continue
delay = time.time() - before
print(delay)
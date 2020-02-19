import time
from pynput.mouse import Controller

while (True):
    print("Current position: " + str(Controller().position))
    time.sleep(1)

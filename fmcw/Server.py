import socket
import threading
import numpy as np
from scipy.io import wavfile
from Analyze import *
import matplotlib.pyplot as plt

BANDWIDTH = 32768


def handleConnection(con):
    distances = []
    with open('raw.wav', 'wb') as f:
        while True:
            data = con.recv(BANDWIDTH)
            if not data:
                break
            f.write(data)
            """
            sig = np.frombuffer(data, dtype=np.short)
            d = ComputeDistance(sig, 0)
            print(d)
            if d is not None and len(d) > 0:
                # print(d)
                distances.append(d[0])
            """
    process()
    addHead()

def process():
    data = open('raw.wav', 'rb').read()
    data = np.frombuffer(data, dtype=np.short)
    ComputeDistance(data, 0)

def addHead():
    data = open('raw.wav', 'rb').read()
    wavfile.write('recv.wav', 48000, np.frombuffer(data, dtype=np.short))

def main():
    listen = socket.socket()
    listen.bind(('0.0.0.0', 23333))
    listen.listen()
    print('listening')
    while True:
        try:
            con, addr = listen.accept()
            print(addr)
            threading.Thread(target=handleConnection, args=(con,), daemon=True).start()
        except KeyboardInterrupt:
            break


if __name__ == '__main__':
    #main()
    process()
    #addHead()

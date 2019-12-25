import socket
import threading
import numpy as np
from scipy.io import wavfile
from Analyze import *
from Generate import *
import matplotlib.pyplot as plt

def handleConnection(con):
    total = 0
    with open('raw.wav', 'wb') as f:
        while True:
            data = con.recv(32768)
            if not data:
                break
            f.write(data)
            sig = np.frombuffer(data, dtype=np.short)
            total += sig.shape[0]
            print(sig.shape)
    wavfile.write('recv.wav', 48000, np.frombuffer(open('raw.wav', 'rb').read(), dtype=np.short))
    process(np.frombuffer(open('raw.wav', 'rb').read(), dtype=np.short))

def process(data):
    AnalyzeAudio(data)


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
    main()
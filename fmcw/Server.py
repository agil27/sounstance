import socket
import threading
import numpy as np
from scipy.io import wavfile
from Analyze import *
import matplotlib.pyplot as plt
from tkinter import Tk, Label, StringVar
from tkinter.font import Font as TkFont

BANDWIDTH = 32768

top = Tk()
ft = TkFont(size=40)
distanceVar = StringVar()

distanceLabel = Label(top, height=9, width=16, font=ft, textvariable=distanceVar)
distanceLabel.pack()

distanceVar.set(0)


def handleConnection(con):
    global distanceVar
    with open('raw.wav', 'wb') as f:
        while True:
            data = con.recv(BANDWIDTH)
            if not data:
                break
            f.write(data)
    process()


def process():
    data = open('raw.wav', 'rb').read()
    data = np.frombuffer(data, dtype=np.short)
    PlotDistance(data, 0)


def addHead():
    data = open('raw.wav', 'rb').read()
    wavfile.write('recv.wav', 48000, np.frombuffer(data, dtype=np.short))


def test():
    data = np.frombuffer(open('raw.wav', 'rb').read(), dtype=np.short)
    last = 0
    ds = []
    start = 0
    for i in range(len(data) // 7680):
        cur = data[i * 7680:(i + 1) * 7680]
        if i < 5:
            continue
        if i == 5:
            start = ComputeStartPosition(cur)
        else:
            start = start + 4
        d = ComputeDistance(cur, 0, start)
        #d = d + last
        #last = d
        ds.append(d.mean())
    ds = np.array(ds)
    plt.plot(ds)
    plt.show()


def listen():
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


def main():
    global top
    threading.Thread(target=listen, daemon=True).start()
    top.mainloop()


if __name__ == '__main__':
    # main()
    test()
    #process()

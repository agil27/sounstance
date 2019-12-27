import socket
import threading
import numpy as np
from scipy.io import wavfile
from Analyze import *
from Generate import *
import matplotlib.pyplot as plt
from tkinter import Tk, Label, StringVar
from tkinter.font import Font as TkFont

top = Tk()
ft = TkFont(size=40)
distanceVar = StringVar()

distanceLabel = Label(top, height=9, width=16, font=ft, textvariable=distanceVar)
distanceLabel.pack()

distanceVar.set(0)


def handleConnection(con):
    total = 0
    with open('raw.wav', 'wb') as f:
        while True:
            data = con.recv(32768)
            if not data:
                break
            f.write(data)
            sig = np.frombuffer(data, dtype=np.short)
            d = AnalyzeAudio(sig)
            print(d[0].mean())
            distanceVar.set("%.3f" % d[0].mean())
            total += sig.shape[0]
            # print(sig.shape)
    wavfile.write('recv.wav', 48000, np.frombuffer(open('raw.wav', 'rb').read(), dtype=np.short))
    process(np.frombuffer(open('raw.wav', 'rb').read(), dtype=np.short))


def process(data):
    PlotAudio(data)

def test():
    data = np.frombuffer(open('raw.wav', 'rb').read(), dtype=np.short)
    last = 0
    ds = []
    for i in range(len(data) // 7680):
        cur = data[i * 7680:(i + 1) * 7680]
        d = AnalyzeAudio(cur, i * 7680 / SAMPLE_FREQUENCY)
        d = d - d[0][0] + last
        last = d[0][-1]
        ds.append(d[0].mean())
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
    #main()
    test()

import socket
import threading
from Analyze import *
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
        start = 0
        index = -1
        lastLength = 0
        lastChirpNum = 0
        while True:
            data = con.recv(BANDWIDTH)
            if not data:
                break
            sig = np.frombuffer(data, dtype=np.short)
            index = index + 1
            if index < 5:
                continue
            if index == 5:
                start = ComputeStartPosition(sig)
                lastLength = sig.shape[0]
                lastChirpNum = math.ceil(lastLength / NUMBER_OF_SAMPLES / 2)
            else:
                start = start - lastLength + lastChirpNum * 2 * NUMBER_OF_SAMPLES
                lastLength = sig.shape[0]
                lastChirpNum = math.ceil(lastLength / NUMBER_OF_SAMPLES / 2)
            d = ComputeDistance(sig, 0, start)
            print(d)
            if d is not None and len(d) > 0 and d.mean() != 0:
                distanceVar.set('%.3f' % d.mean())
            f.write(data)
    process()


def process():
    data = open('raw.wav', 'rb').read()
    data = np.frombuffer(data, dtype=np.short)
    PlotDistance(data, 0)


def addHead():
    data = open('raw.wav', 'rb').read()
    wavfile.write('recv.wav', 48000, np.frombuffer(data, dtype=np.short))


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
    main()
    #test()
    #process()

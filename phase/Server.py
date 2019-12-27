import socket
import threading
from Analyze import *
from Generate import *
from tkinter import Tk, Label, StringVar
from tkinter.font import Font as TkFont

top = Tk()
ft = TkFont(size=40)
distanceVar = StringVar()
distanceLabel = Label(top, height=9, width=16, font=ft, textvariable=distanceVar)
distanceLabel.pack()
distanceVar.set(0)


def handleConnection(con):
    with open('raw.wav', 'wb') as f:
        index = 0
        last = 0
        while True:
            data = con.recv(32768)
            if not data:
                break
            f.write(data)
            sig = np.frombuffer(data, dtype=np.short)
            # 对每个片段，都会加上总的相位偏移
            d = AnalyzeAudio(sig, index * 7680 / SAMPLE_FREQUENCY)
            d = d - d[0][0] + last
            last = d[0][-1]
            distanceVar.set("%.3f" % d[0].mean())
            index = index + 1
    wavfile.write('recv.wav', 48000, np.frombuffer(open('raw.wav', 'rb').read(), dtype=np.short))
    process()


def process():
    PlotAudio(np.frombuffer(open('raw.wav', 'rb').read(), dtype=np.short))


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

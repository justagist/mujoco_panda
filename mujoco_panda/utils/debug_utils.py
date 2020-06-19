
import threading
from .keyboard_poll import KBHit
from tkinter import *
class ParallelPythonCmd(object):

    def __init__(self, callable_func, kbhit=False):

        if kbhit:
            target = self._run
        else:
            target = self._run_tk
        self._th = threading.Thread(target=target)
        self._callable = callable_func
        self.start_thread()

    def start_thread(self):
        self._th.start()

    def _run(self):
        kb = KBHit()
        print ("Running KBHit debugger...")
        string = ''
        while True:
            c = kb.getch()
            if ord(c) == 27:
                string = ''
            elif ord(c) == 10:
                self._callable(string)
                string = ''
            elif ord(c) == 127:
                if string != '':
                    string = string[:-1]
            else:
                string += c
    
    def _run_tk(self):

        self._root = Tk()
        self._root.title("Python Cmd")
        self._e = Entry(self._root)
        self._e.pack()
        self._text = Text(self._root)
        self._text.insert(INSERT, "Hello.....")
        self._text.pack()
        self._e.focus_set()
        self._root.bind('<Return>', self._printtext)
        self._root.mainloop()

    def _printtext(self, event=None):
        string = self._e.get()
        # print(string)
        # # exec(string)
        # try:
        a = self._callable(string)
        if a is not None:
            self._text.delete("%d.%d" % (0, 0), END)
            self._text.insert(INSERT, str(a))
                
        # except Exception as e:
        #     print ("Exception:", e)
        self._e.delete(0, 'end')

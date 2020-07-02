try:
    from tkinter import *
    _NO_TK = False
except ImportError:
    print("tkinter not found. Cannot run with ParallelPythonCmd with tkinter.")
    _NO_TK = True

import threading
from .keyboard_poll import KBHit


class ParallelPythonCmd(object):

    def __init__(self, callable_func, kbhit=False):
        """
        Provides a debugger thread when using mujoco visualiser rendering.
        Debug commands can be sent in this separate thread
        to control or retrieve info from the simulated robot.

        :param callable_func: a handle to a function (should be defined somewhere with
            access to the mujoco robot) which executes the provided command. 
            See `exec_func` below.
        :type callable_func: callable
        :param kbhit: a handle to a function, defaults to False
        :type kbhit: bool, optional
        """

        if kbhit or _NO_TK:
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
        print("Running KBHit debugger...")
        string = ''
        while True:
            c = kb.getch()
            if ord(c) == 27:
                string = ''
            elif ord(c) == 10:
                ret = self._callable(string)
                if ret is not None:
                    print("Cmd: {}\nOutput: {}".format(string, ret))
                string = ''
            elif ord(c) == 127:
                if string != '':
                    string = string[:-1]
            else:
                string += c

    def _run_tk(self):
        print("Running tk debugger...")
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
        a = self._callable(string)
        if a is not None:
            self._text.delete("%d.%d" % (0, 0), END)
            self._text.insert(INSERT, str(a))
        self._e.delete(0, 'end')

# demo exec function handle


def exec_func(cmd):
    """
    Sample handle that can be used with :py:class`ParallelPythonCmd`.
    Copy this to somewhere which has access to all required objects 
    (eg: PandaArm object), and pass the handle to this function as 
    argument to :py:class`ParallelPythonCmd` instance.

    Requirement for function handle:
        - take an argument `cmd`
        - perform eval or exec on `cmd`
        - optionally return a string    
    """

    if cmd == '':
        return None
    print(cmd)
    try:
        if "=" in cmd:
            exec(cmd)
            a = cmd
        else:
            a = eval(cmd)
            print(a)
    except Exception as e:
        a = "Exception: {}: {}".format(e.what(),e)
    if a is not None:
        return str(a)

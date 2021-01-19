#!/usr/bin/env python
# coding: utf-8
import socket
import time
from time import sleep 

from os import system, name 
from IPython.display import clear_output

def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

# define our clear function 
def clear(): 
    if not isnotebook():
        # for windows 
        if name == 'nt': 
            system('cls') 

        # for mac and linux(here, os.name is 'posix') 
        else: 
            system('clear')
    else:
        clear_output(wait=True)
    return

print(isnotebook())
sock = socket.socket()
sock.connect(('localhost', 2002))
for i in range(0,100000):
    print('->')
    #sock.send(b'hello, world!')
    data = sock.recv(1024)
    clear()
    print(data)
    
print("100 sent")
sock.close()

print(data)
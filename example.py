import random
import socket
import time

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


port = 9090
message_max_length = 1024
connectionsCount = 0
screenFlipper = False
msg = '!{0}"{1}"{2}"{3}"{4}"{5}'

def CreateRandomMessage():
    global screenFlipper
    global msg
    msg_copy = msg
    errorFound = random.randint(0,1)
    isWorking = random.randint(0,1)
    flipping = (0,1)[screenFlipper]
    screenFlipper = not screenFlipper
    possibility=str(random.randint(0,100)).zfill(3)
    size = str(random.randint(0,999)).zfill(3)
    msg_copy = msg_copy.format(errorFound,isWorking,flipping,errorFound,possibility,size)
    return msg_copy.encode()

def SocketCycle(conn):
    global message_max_length
    while True:
        data = conn.recv(message_max_length)
        if not data:
            break
        conn.send(CreateRandomMessage())

def SocketHandle(sock):
    print("Server started")
    global port
    sock.bind(('', port))
    sock.listen(1)
    conn, addr = sock.accept()
    print('connected:', addr)
    SocketCycle(conn)

    
def Main():
    global connectionsCount
    while True:
        sock = socket.socket()
        SocketHandle(sock)
        sock.close()
        clear()
        print("Server stopped due to disconnect, restarting")
        print("Restart count = {0}".format(connectionsCount))
        connectionsCount = connectionsCount + 1
    
    
Main()
   

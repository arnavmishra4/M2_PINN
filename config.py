import os
import socket

# Set data type
DTYPE = 'float32'

hostname = socket.gethostname().lower()

if 'gru' in hostname:
    DATADIR = '/mnt/data/rzhang/pinndata'
elif 'hpc3' in hostname:
    DATADIR = '~/pinndata'
elif 'poison' in hostname:
    DATADIR = '/home/ziruz16/models'
else:
    DATADIR = './'
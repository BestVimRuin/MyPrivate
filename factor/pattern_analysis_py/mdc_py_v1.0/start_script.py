#!/usr/bin/python
#-*-coding:utf-8-*-

import zmq
import os
context = zmq.Context()
socket = context.socket(zmq.PULL)
socket.bind('tcp://*:5000')
while True:
	data = socket.recv()
	print(data)
	os.system("python read_data.py")

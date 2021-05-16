'''
Author: your name
Date: 2021-01-12 19:30:18
LastEditTime: 2021-05-16 17:42:40
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /XAVIER/src/tcpClient.py
'''
from socket import *

HOST = '192.168.137.20'
PORT = 9000
BUFSIZ = 1024
ADDRESS = (HOST, PORT)

tcpClientSocket = socket(AF_INET, SOCK_STREAM)
tcpClientSocket.connect(ADDRESS)

while True:
    data = input('>')
    if not data:
        break

    # 发送数据
    tcpClientSocket.send(data.encode('utf-8'))
    # 接收数据
    data, ADDR = tcpClientSocket.recvfrom(BUFSIZ)
    if not data:
        break
    print("服务器端响应：", data.decode('utf-8'))

print("链接已断开！")
tcpClientSocket.close()
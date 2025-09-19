import socket

def sendit(msg):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.sendto(msg, ("<FILL SOME IP HERE>", 1234))

for i in range(256):
    sendit(b"a"*300)

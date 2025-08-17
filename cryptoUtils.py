from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
import base64

def pad(data):
    return data + b"\0" * (AES.block_size - len(data) % AES.block_size)

def encryptData(data, key):
    data = pad(data.encode("utf-8"))
    iv = get_random_bytes(AES.block_size)
    cipher = AES.new(key,AES.MODE_CBC, iv)
    encrypted = cipher.encrypt(data)
    return base64.b64encode(iv + encrypted).decode("utf-8")

def decryptData(data,key):
    data = base64.b64decode(data)
    iv = data[:AES.block_size]
    cipher = AES.new(key, AES.MODE_CBC, iv)
    decrypted = cipher.decrypt(data[AES.block_size:]).rstrip(b"\0")
    return decrypted.decode("utf-8")


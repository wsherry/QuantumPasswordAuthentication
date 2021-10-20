import psutil
import time
from ClientSSL import ClientSSL
from ServerSSL import ServerSSL

"""SSL connection between different entities example"""

print('before example',psutil.cpu_percent())
print('before ex',psutil.virtual_memory())

start = time.time()
# server setup
server = ServerSSL()
pk = server.get_public_key()

auth_server = ServerSSL()
auth_pk = auth_server.get_public_key()
# client generates an AES key and sends it to the server
user = ClientSSL()
ciphertext = user.generate_AES_key(pk)

# server decrypts the encrypted AES key to establish secure connection with client
server.set_decrypted_AES_key(ciphertext)

# client sends their password with symmetric encryption to the server
password = b'000110011100010001110010101000111011010101100000100011011000011000101100110101011011'
password = 'pass'
ciphertext, nonce, tag = user.encrypt(password)

# server decrypts password
server.decrypt(ciphertext, nonce, tag)

end = time.time()
print(end-start, ' seconds')
print('after',psutil.cpu_percent())
print('after',psutil.virtual_memory())
from Crypto.Cipher import AES

class EntitySSL:
    def __init__(self):
        self.AES_key = ""

    def encrypt(self, data):
        """Encrypts data using AES-256"""
        cipher = AES.new(self.AES_key, AES.MODE_EAX)
        nonce = cipher.nonce
        if type(data) == str:
            data = data.encode("ascii")
        ciphertext, tag = cipher.encrypt_and_digest(data)
        return ciphertext, nonce, tag

    def decrypt(self, ciphertext, nonce, tag):
        """Decrypts data using AES-256"""
        cipher = AES.new(self.AES_key, AES.MODE_EAX, nonce=nonce)
        plaintext = cipher.decrypt(ciphertext)
        # verify authenticity of ciphertext
        try:
            cipher.verify(tag)
            print("The message is authentic:", plaintext)
        except ValueError:
            print("Key incorrect or message corrupted")

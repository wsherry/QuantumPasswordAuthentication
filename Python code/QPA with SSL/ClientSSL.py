from EntitySSL import EntitySSL
import oqs

class ClientSSL(EntitySSL):
    def __init__(self, kemalg="FrodoKEM-1344-AES"):
        self.client = oqs.KeyEncapsulation(kemalg)

    def generate_AES_key(self, public_key):
        """Generates an encrypted AES key using a public key for establishing secure communication with another party"""
        ciphertext, shared_secret = self.client.encap_secret(public_key)
        self.AES_key = shared_secret
        return ciphertext

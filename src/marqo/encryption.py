from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding, hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
from marqo.tensor_search.utils import read_env_vars_and_defaults
from marqo.tensor_search.enums import EnvVars
from marqo.tensor_search.constants import MARQO_ENCRYPTION_SALT
import os, base64


def derive_key_from_password(password, salt = MARQO_ENCRYPTION_SALT):
    """
    This function derives a key from a password and salt using PBKDF2HMAC.
    The salt is predefined in the constants.py file, as `b"marqo-encryption-salt"`.
    The password is read from the environment variable `MARQO_AES_ENCRYPTION_KEY`.
    Users can provide this environment variable to achieve encryption at rest.
    Args:
        password: a plain string password to derive the key for AES encryption
        salt: a pre-defined salt to generate the key, the value is `b"marqo-encryption-salt"`.
    Returns:
        A key generated from the password and salt for AES encryption
    """
    backend = default_backend()
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=backend
    )
    return kdf.derive(password.encode())


password = read_env_vars_and_defaults(EnvVars.MARQO_AES_ENCRYPTION_KEY)
key = derive_key_from_password(password)


def encrypt_string(plain_text):
    """
    This function encrypts a plain text string using AES encryption.
    Args:
        plain_text: a string to be encrypted
    Returns:
        An encrypted string
    """
    iv = os.urandom(16)
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
    encryptor = cipher.encryptor()

    padder = padding.PKCS7(128).padder()
    padded_data = padder.update(plain_text.encode()) + padder.finalize()
    cipher_text = encryptor.update(padded_data) + encryptor.finalize()

    encrypted_message = base64.b64encode(iv + cipher_text).decode('utf-8')
    return encrypted_message


def decrypt_string(encrypted_message):
    """
    This function decrypts an encrypted string using AES decryption.
    Args:
        encrypted_message: a string to be decrypted
    Returns:
        A decrypted string
    """
    data = base64.b64decode(encrypted_message.encode('utf-8'))
    iv = data[:16]
    cipher_text = data[16:]

    cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
    decryptor = cipher.decryptor()
    padded_data = decryptor.update(cipher_text) + decryptor.finalize()

    unpadder = padding.PKCS7(128).unpadder()
    plain_text = (unpadder.update(padded_data) + unpadder.finalize()).decode('utf-8')
    return plain_text

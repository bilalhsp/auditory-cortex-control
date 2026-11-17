"""Single-operator HTTP authentication. Deploy behind HTTPS."""

import getpass
import hashlib
import secrets


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 600000)
    return "pbkdf2_sha256$600000${}${}".format(salt.hex(), digest.hex())


def password_matches(password: str, encoded: str) -> bool:
    algorithm, iterations, salt, expected = encoded.split("$")
    if algorithm != "pbkdf2_sha256":
        raise ValueError("Unsupported password hash")
    actual = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), bytes.fromhex(salt), int(iterations)
    )
    return secrets.compare_digest(actual, bytes.fromhex(expected))


if __name__ == "__main__":
    password = getpass.getpass("Operator password: ")
    if len(password) < 12:
        raise SystemExit("Use at least 12 characters.")
    print(hash_password(password))

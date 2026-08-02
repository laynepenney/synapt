"""Small age X25519 boundary for the encrypted-sync spike.

Cryptography is delegated to pyrage, the Python bindings for rage.  The
relay never imports this module and never receives the identity.
"""

from __future__ import annotations


def _pyrage():
    try:
        import pyrage
        from pyrage import x25519
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise RuntimeError(
            "pyrage is required for the encrypted-sync spike; install pyrage==1.3.0"
        ) from exc
    return pyrage, x25519


def generate_team_identity() -> tuple[str, str]:
    """Return a new ``(identity, recipient)`` age X25519 pair."""
    _pyrage_module, x25519 = _pyrage()
    identity = x25519.Identity.generate()
    return str(identity), str(identity.to_public())


def recipient_from_identity(identity_text: str) -> str:
    """Derive the public recipient from a private team identity."""
    _pyrage_module, x25519 = _pyrage()
    try:
        identity = x25519.Identity.from_str(identity_text.strip())
    except Exception as exc:
        raise ValueError("invalid age team identity") from exc
    return str(identity.to_public())


def encrypt_archive(plaintext: bytes, recipient_text: str) -> bytes:
    """Encrypt one portable recall archive for the team recipient."""
    pyrage, x25519 = _pyrage()
    try:
        recipient = x25519.Recipient.from_str(recipient_text.strip())
        return pyrage.encrypt(plaintext, [recipient])
    except Exception as exc:
        raise ValueError("age archive encryption failed") from exc


def decrypt_archive(ciphertext: bytes, identity_text: str) -> bytes:
    """Decrypt one portable recall archive with the team identity."""
    pyrage, x25519 = _pyrage()
    try:
        identity = x25519.Identity.from_str(identity_text.strip())
        return pyrage.decrypt(ciphertext, [identity])
    except Exception as exc:
        raise ValueError("age archive decryption failed") from exc

import secrets
# token URL-safe ~ menghasilkan string panjang variatif, gunakan token_bytes untuk ukuran presisi
key = secrets.token_urlsafe(48)  # cukup panjang dan URL-safe
# atau presisi 32 bytes (256 bit) lalu encode base64-url tanpa padding:
import base64
raw = secrets.token_bytes(32)
key = base64.urlsafe_b64encode(raw).rstrip(b'=').decode()
print(key)

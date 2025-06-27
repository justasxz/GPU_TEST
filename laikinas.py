import time
import json
import base64
import hmac
import hashlib

# Your secret key and identifiers
secret_key = "KbPeShVmYq3t6w9z$C&F)J@NcQfTjWnZr4u7x!A%D*G-KaPdSgVkXp2s5v8y/B?E"
issuer = "airtel_africa"
subject = "airtel_africa"
airtel_system_transaction_id = "12345"

# Helper for base64url encoding without padding
def b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")

# Build JWT header
header = {
    "alg": "HS512",
    "typ": "JWT"
}

# Build JWT payload
now = int(time.time())
payload = {
    "jti": f"jwt-id-{now}",
    "iat": now,
    "sub": subject,
    "iss": issuer,
    "payload": {
        "txnId": airtel_system_transaction_id
    },
    "exp": now + 28800  # expires in 8 hours
}

# Encode header and payload
header_b64 = b64url(json.dumps(header, separators=(",", ":")).encode())
payload_b64 = b64url(json.dumps(payload, separators=(",", ":")).encode())
signing_input = f"{header_b64}.{payload_b64}".encode()

# Sign with HMAC-SHA512
signature = hmac.new(secret_key.encode(), signing_input, hashlib.sha512).digest()
signature_b64 = b64url(signature)

# Construct the full token
token = f"{header_b64}.{payload_b64}.{signature_b64}"

# Output the token
print(token)

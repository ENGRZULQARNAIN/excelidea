from datetime import datetime, timedelta, timezone
from typing import Any
import base64
import hashlib
import hmac
import json
import os
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from app.core.db import get_db
from app.models import Users
from app.core.config import settings
import logging

# Password hashing using hashlib instead of passlib
def get_password_hash(password: str) -> str:
    # Generate a random salt
    salt = os.urandom(32)
    # Use PBKDF2 with SHA256 for password hashing
    key = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt,
        100000  # Number of iterations
    )
    # Store salt and key together
    return base64.b64encode(salt + key).decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        # Decode the stored hash
        decoded = base64.b64decode(hashed_password.encode('utf-8'))
        # Extract salt (first 32 bytes) and stored key
        salt, stored_key = decoded[:32], decoded[32:]
        # Compute hash with the same salt
        computed_key = hashlib.pbkdf2_hmac(
            'sha256',
            plain_password.encode('utf-8'),
            salt,
            100000  # Same number of iterations
        )
        # Compare in constant time to prevent timing attacks
        return hmac.compare_digest(stored_key, computed_key)
    except Exception:
        return False

# Custom token implementation instead of JWT
def create_access_token(subject: str | Any, expires_delta: timedelta) -> str:
    expire = datetime.now(timezone.utc) + expires_delta
    to_encode = {
        "exp": int(expire.timestamp()),
        "sub": str(subject),
        "iat": int(datetime.now(timezone.utc).timestamp())
    }
    # Convert payload to JSON
    payload = json.dumps(to_encode).encode('utf-8')
    # Create signature using HMAC
    signature = hmac.new(
        settings.SECRET_KEY.encode('utf-8'),
        payload,
        hashlib.sha256
    ).digest()
    # Encode payload and signature
    token_parts = [
        base64.urlsafe_b64encode(payload).decode('utf-8'),
        base64.urlsafe_b64encode(signature).decode('utf-8')
    ]
    return ".".join(token_parts)

def decode_token(token: str) -> dict:
    try:
        # Split token into parts
        parts = token.split(".")
        if len(parts) != 2:
            raise ValueError("Invalid token format")
        
        payload_b64, signature_b64 = parts
        
        # Decode payload
        payload = base64.urlsafe_b64decode(payload_b64.encode('utf-8'))
        
        # Verify signature
        expected_signature = hmac.new(
            settings.SECRET_KEY.encode('utf-8'),
            payload,
            hashlib.sha256
        ).digest()
        
        actual_signature = base64.urlsafe_b64decode(signature_b64.encode('utf-8'))
        
        if not hmac.compare_digest(expected_signature, actual_signature):
            raise ValueError("Invalid signature")
        
        # Parse payload
        data = json.loads(payload.decode('utf-8'))
        
        # Check expiration
        if data["exp"] < datetime.now(timezone.utc).timestamp():
            raise ValueError("Token expired")
            
        return data
    except Exception as e:
        raise ValueError(f"Invalid token: {str(e)}")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/v1/login", auto_error=False)

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> Users:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    logging.info(f"Token: {token}")
    if not token:
        raise credentials_exception
        
    try:
        payload = decode_token(token)
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except ValueError:
        raise credentials_exception

    user = db.query(Users).filter(Users.id == user_id).first()
    if user is None:
        raise credentials_exception
    return user

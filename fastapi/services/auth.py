import os
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from pydantic import BaseModel

# อ่านจาก env แต่มี fallback แบบ hardcode สำหรับ dev (ควรเปลี่ยนในโปรดักชัน)
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-me")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

# สำหรับอนาคตถ้าจะต่อ Keycloak ให้ใช้ค่า issuer / audience ตรงนี้
KEYCLOAK_ISSUER = os.getenv("KEYCLOAK_ISSUER")
KEYCLOAK_AUDIENCE = os.getenv("KEYCLOAK_AUDIENCE")

FAKE_USERS_DEMO = os.getenv("FAKE_USERS_DEMO")
FAKE_PASS_DEMO = os.getenv("FAKE_PASS_DEMO")

FAKE_USERS_ADMIN = os.getenv("FAKE_USERS_ADMIN")
FAKE_PASS_ADMIN = os.getenv("FAKE_PASS_ADMIN")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


class TokenData(BaseModel):
    sub: str
    roles: List[str] = []


class User(BaseModel):
    username: str
    roles: List[str] = []


# In-memory user สำหรับเริ่มต้น (ปรับเป็น DB/Keycloak ภายหลัง)
FAKE_USERS = {
    "demo": {"username": FAKE_USERS_DEMO, "password": FAKE_PASS_DEMO, "roles": ["user"]},
    "admin": {"username": FAKE_USERS_ADMIN, "password": FAKE_PASS_ADMIN, "roles": ["admin"]},
}


def authenticate_user(username: str, password: str) -> Optional[User]:
    user = FAKE_USERS.get(username)
    if not user or user["password"] != password:
        return None
    return User(username=user["username"], roles=user.get("roles", []))


def create_access_token(data: dict, expires_minutes: int = ACCESS_TOKEN_EXPIRE_MINUTES) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=expires_minutes)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def _decode_local_jwt(token: str) -> TokenData:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: Optional[str] = payload.get("sub")
        roles = payload.get("roles", [])
        if username is None:
            raise ValueError("sub missing")
        return TokenData(sub=username, roles=roles)
    except (JWTError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


def _decode_with_keycloak(token: str) -> TokenData:
    # โครงสำหรับอนาคต: ดึง JWKS จาก Keycloak เพื่อตรวจลายเซ็น + issuer/audience
    # ปัจจุบันยังไม่ implement ให้ใช้งาน local JWT ไปก่อน
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Keycloak validation not implemented yet",
    )


def verify_token(token: str = Depends(oauth2_scheme)) -> TokenData:
    if KEYCLOAK_ISSUER:
        return _decode_with_keycloak(token)
    return _decode_local_jwt(token)


def require_roles(required: List[str]):
    def checker(user: TokenData = Depends(verify_token)):
        if required and not any(role in user.roles for role in required):
            raise HTTPException(status_code=403, detail="Forbidden")
        return user

    return checker

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from fastapi.security import OAuth2PasswordRequestForm

from services.auth import (
    User,
    authenticate_user,
    create_access_token,
    verify_token,
    require_roles,
)

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


@router.post("/login", response_model=TokenResponse)
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    รองรับ Swagger Authorize (OAuth2 password flow) ที่ส่ง form-urlencoded (username, password)
    """
    user: User | None = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token({"sub": user.username, "roles": user.roles})
    return TokenResponse(access_token=token)


@router.get("/me")
def me(user=Depends(verify_token)):
    return {"username": user.sub, "roles": user.roles}


@router.get("/admin-probe")
def admin_only(user=Depends(require_roles(["admin"]))):
    return {"ok": True, "user": user.sub, "roles": user.roles}

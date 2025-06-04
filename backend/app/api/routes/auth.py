from fastapi import APIRouter, Depends, HTTPException, status, Response, Cookie
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import Optional, List
import bcrypt
import secrets
from pydantic import BaseModel, EmailStr
from ...db.database import get_db
from ...db.models import User, Session as DbSession, TrainedModel

router = APIRouter()

class UserCreate(BaseModel):
    name: str
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: int
    name: str
    email: str

class TrainedModelResponse(BaseModel):
    id: int
    name: str
    model_type: str
    dataset_name: str
    metrics: dict
    hyperparameters: dict
    created_at: datetime

def create_session(db: Session, user_id: int) -> str:
    # Delete any existing sessions for the user
    db.query(DbSession).filter(DbSession.user_id == user_id).delete()
    
    # Create new session
    session_token = secrets.token_urlsafe(32)
    expires_at = datetime.utcnow() + timedelta(days=7)
    
    db_session = DbSession(
        user_id=user_id,
        session_token=session_token,
        expires_at=expires_at
    )
    db.add(db_session)
    db.commit()
    
    return session_token

async def get_current_user(
    session_token: Optional[str] = Cookie(None, alias="session_token"),
    db: Session = Depends(get_db)
) -> Optional[User]:
    if not session_token:
        return None
        
    db_session = db.query(DbSession).filter(
        DbSession.session_token == session_token,
        DbSession.expires_at > datetime.utcnow()
    ).first()
    
    if not db_session:
        return None
        
    return db.query(User).filter(User.id == db_session.user_id).first()

@router.post("/signup")
def signup(user: UserCreate, db: Session = Depends(get_db)):
    # Check if email already exists
    if db.query(User).filter(User.email == user.email).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Hash password
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(user.password.encode(), salt)
    
    # Create user
    db_user = User(
        email=user.email,
        password_hash=hashed_password.decode(),
        name=user.name
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    return {"message": "User created successfully"}

@router.post("/signin")
def signin(user: UserLogin, response: Response, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    
    if not bcrypt.checkpw(user.password.encode(), db_user.password_hash.encode()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    
    session_token = create_session(db, db_user.id)
    response.set_cookie(
        key="session_token",
        value=session_token,
        httponly=True,
        secure=False,  # Set to False for local development
        samesite="lax",
        max_age=7 * 24 * 60 * 60,  # 7 days
        domain="localhost"  # Explicitly set domain for local development
    )
    
    return {
        "message": "Signed in successfully",
        "user": UserResponse(
            id=db_user.id,
            name=db_user.name,
            email=db_user.email
        )
    }

@router.post("/signout")
def signout(
    response: Response,
    session_token: Optional[str] = Cookie(None, alias="session_token"),
    db: Session = Depends(get_db)
):
    if session_token:
        db.query(DbSession).filter(DbSession.session_token == session_token).delete()
        db.commit()
    
    response.delete_cookie(
        key="session_token",
        domain="localhost",
        path="/"
    )
    return {"message": "Signed out successfully"}

@router.get("/me", response_model=dict)
async def get_current_user_info(
    current_user: Optional[User] = Depends(get_current_user)
):
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )
    
    return {
        "user": UserResponse(
            id=current_user.id,
            name=current_user.name,
            email=current_user.email
        )
    }

@router.get("/my-models", response_model=List[TrainedModelResponse])
async def get_user_models(
    current_user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )
    
    models = db.query(TrainedModel).filter(
        TrainedModel.user_id == current_user.id
    ).order_by(TrainedModel.created_at.desc()).all()
    
    return models

@router.get("/my-models/{model_id}", response_model=TrainedModelResponse)
async def get_model_details(
    model_id: int,
    current_user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )
    
    model = db.query(TrainedModel).filter(
        TrainedModel.id == model_id,
        TrainedModel.user_id == current_user.id
    ).first()
    
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found"
        )
    
    return model 
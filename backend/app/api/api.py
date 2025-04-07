from fastapi import APIRouter

from app.api.routes import auth
from app.api.routes import generate_project_route

api_router = APIRouter()
api_router.include_router(auth.router, tags=["Auth"])
api_router.include_router(generate_project_route.router, tags=["Project Proposal"])

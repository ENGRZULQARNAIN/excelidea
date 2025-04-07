from fastapi import FastAPI
from fastapi.routing import APIRoute
from starlette.middleware.cors import CORSMiddleware

from app.api.api import api_router
from app.core.config import settings
from app.core.db import engine
from app.models import Base

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="EXCELIDEA",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
)


# Set all CORS enabled origins
app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
@app.get("/")
async def root():
    return {"message": "Excel IDEA v1.0.0"}

app.include_router(api_router, prefix=settings.API_V1_STR)

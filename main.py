# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from routers.analyze import router as analyze_router

# app = FastAPI(title="Code Review Assistant")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:5173"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# app.include_router(analyze_router)

# @app.get("/")
# def root():
#     return {"message": "Code Review Assistant API is running"}

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers.analyze import router as analyze_router
import os

app = FastAPI(
    title="Code Review Assistant API",
    description="AI-powered code analysis using ML + graph theory",
    version="1.0.0"
)

# Production CORS - allow your frontend domain
allowed_origins = [
    "http://localhost:5173",           # local dev
    "https://*.vercel.app",            # vercel preview/prod
    os.getenv("FRONTEND_URL", ""),     # custom domain
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # We'll restrict this after deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze_router)

@app.get("/")
def root():
    return {
        "message": "Code Review Assistant API",
        "version": "1.0.0",
        "status": "healthy"
    }

@app.get("/health")
def health():
    """Health check endpoint for deployment monitoring"""
    return {"status": "ok"}
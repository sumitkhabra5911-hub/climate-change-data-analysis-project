"""
FastAPI Enterprise Application Factory & Middleware Pipeline.
"""
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from src.config import settings
from src.api.routes import api_router
from src.api.service import model_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure artifacts are loaded
    print(f"[*] Starting {settings.PROJECT_NAME} v{settings.VERSION}")
    model_service.load_artifacts()
    print(f"[+] Loaded {len(model_service.models)} models and {len(model_service.df)} records.")
    yield
    print("[-] Shutting down service.")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description="Production REST API for Climate Change Risk Assessment, Anomaly Forecasting, and Country Intelligence.",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount API Router
    app.include_router(api_router, prefix=settings.API_V1_STR)

    # Mount Frontend Assets
    frontend_dir = os.path.join(settings.BASE_DIR, "frontend")
    if os.path.exists(frontend_dir):
        assets_dir = os.path.join(frontend_dir, "assets")
        if os.path.exists(assets_dir):
            app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

        @app.get("/", include_in_schema=False)
        def serve_index():
            return FileResponse(os.path.join(frontend_dir, "index.html"))

    return app

app = create_app()

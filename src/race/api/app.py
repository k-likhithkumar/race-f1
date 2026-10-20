import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from src.race.api.routes import router as api_router
from src.race.api.websocket import router as ws_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="RACE F1 Replay Engine API",
        description="Historical F1 race telemetry replay and state synchronization API.",
        version="0.3.2",
    )

    # Enable CORS for local development (Vite frontend)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router)
    app.include_router(ws_router)

    # Mount built UI dist directory if present
    dist_dir = Path(__file__).resolve().parent.parent.parent.parent / "ui" / "dist"
    if dist_dir.exists():
        assets_dir = dist_dir / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

        @app.get("/", include_in_schema=False)
        def serve_index():
            return FileResponse(dist_dir / "index.html")

    return app


app = create_app()

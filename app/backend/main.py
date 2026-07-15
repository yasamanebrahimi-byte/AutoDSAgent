"""FastAPI entrypoint for the AutoDS Agent backend."""

from fastapi import FastAPI

from app.backend.routes import cleaning, profile, runs, upload


app = FastAPI(
    title="AutoDS Agent Backend",
    description="Backend foundation for autonomous tabular data analysis.",
    version="0.1.0",
)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return a simple service health check."""

    return {"status": "ok", "service": "autods-agent-backend"}


app.include_router(upload.router)
app.include_router(runs.router)
app.include_router(profile.router)
app.include_router(cleaning.router)

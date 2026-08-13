"""FastAPI entrypoint for the AutoDS Agent backend."""

from fastapi import FastAPI

from app.backend.routes import config, cleaning, eda, modeling, profile, reports, runs, upload, workflow
from app.tools.app_logging import configure_logging


configure_logging()


app = FastAPI(
    title="AutoDS Agent Backend",
    description="Backend foundation for deterministic tabular data analysis workflows.",
    version="0.1.0",
)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return a simple service health check."""

    return {"status": "ok", "service": "autods-agent-backend"}


app.include_router(upload.router)
app.include_router(config.router)
app.include_router(runs.router)
app.include_router(profile.router)
app.include_router(cleaning.router)
app.include_router(eda.router)
app.include_router(modeling.router)
app.include_router(reports.router)
app.include_router(workflow.router)

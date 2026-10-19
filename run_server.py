"""
CLI Entry Point to launch the production FastAPI server.
"""
import os
import uvicorn
from src.config import settings

if __name__ == "__main__":
    port = int(os.getenv("PORT", settings.PORT))
    host = os.getenv("HOST", settings.HOST)
    print(f"Starting {settings.PROJECT_NAME} on http://{host}:{port}")
    print(f"Interactive Swagger Docs: http://{host}:{port}/docs")
    print(f"Dashboard UI: http://{host}:{port}/")
    uvicorn.run(
        "src.api.app:app",
        host=host,
        port=port,
        reload=False,
        workers=1
    )

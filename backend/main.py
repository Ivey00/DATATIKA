import uvicorn
import os
from dotenv import load_dotenv
from app.main import app  # Import the FastAPI app instance

# Load environment variables from .env file if it exists
load_dotenv()

if __name__ == "__main__":
    # Get configuration from environment variables or use defaults
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "True").lower() in ("true", "1", "t")
    
    print(f"Starting server on {host}:{port}")
    print(f"Reload mode: {'enabled' if reload else 'disabled'}")
    print("Documentation available at:")
    print(f"  - Swagger UI: http://localhost:{port}/docs")
    print(f"  - ReDoc: http://localhost:{port}/redoc")
    
    # Run the application with uvicorn
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )
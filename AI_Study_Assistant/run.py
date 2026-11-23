import sys
from pathlib import Path
import uvicorn

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

if __name__ == "__main__":
    print("=" * 60)
    print("STARTING AI STUDY ASSISTANT SERVER")
    print("=" * 60)
    print("Dashboard UI:  http://127.0.0.1:8000/dashboard")
    print("API Docs:      http://127.0.0.1:8000/docs")
    print("Health Status: http://127.0.0.1:8000/health")
    print("=" * 60)
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)

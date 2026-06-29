import uvicorn
import sys
import os

if __name__ == "__main__":
    # Ensure backend directory is in the path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    print("Starting BioCoworker FastAPI Backend Server...")
    print("Host: http://127.0.0.1:8989")
    print("Interactive API Docs: http://127.0.0.1:8989/docs")
    
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8989, reload=True)

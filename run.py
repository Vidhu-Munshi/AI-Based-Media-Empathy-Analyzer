"""
Single entry point to launch FastAPI server and open browser automatically.
Run with: python run.py
"""

import sys
import threading
import time
import webbrowser

def open_browser():
    time.sleep(1.5)
    print("\n[INFO] Opening your browser at http://localhost:8000 ...")
    webbrowser.open("http://localhost:8000")

if __name__ == "__main__":
    threading.Thread(target=open_browser, daemon=True).start()
    
    import uvicorn
    print("=" * 65)
    print("      Starting AI Screenplay & Censor Rating Analyzer")
    print("      URL: http://localhost:8000")
    print("      API Docs: http://localhost:8000/docs")
    print("=" * 65)
    
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)

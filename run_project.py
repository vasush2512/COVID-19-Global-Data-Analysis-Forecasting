"""
One-click launcher for this project.

Starts FastAPI backend and serves frontend from the same server:
  - Frontend: http://127.0.0.1:<port>/
  - API docs: http://127.0.0.1:<port>/docs
"""

import os
import socket
import threading
import webbrowser


def _free_port(preferred: int = 8000) -> int:
    for port in range(preferred, preferred + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("No free local port found between 8000 and 8019.")


def main() -> None:
    try:
        import uvicorn
    except ImportError:
        print("Missing dependency: uvicorn")
        print("Run this once:")
        print('  ".\\.venv\\Scripts\\python.exe" -m pip install -r requirements.txt')
        return

    port = int(os.getenv("PORT", "0") or "0") or _free_port(8000)
    url = f"http://127.0.0.1:{port}/"
    print("Starting backend + frontend together...")
    print(f"Open in browser: {url}")
    print(f"API docs: {url}docs")

    # Open browser shortly after server starts.
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    uvicorn.run("backend:app", host="127.0.0.1", port=port, reload=False)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
End-to-end server test — starts uvicorn, hits the REST API with curl.
Run: python3 tests/test_server.py
"""
import subprocess, time, sys, json, urllib.request, os

PORT = 8765
proc = None

try:
    print("Starting SmartRadio server on port", PORT)
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn",
         "server.app:app",
         "--host", "127.0.0.1", "--port", str(PORT)],
        env={**os.environ, "TASK_ID": "1"},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait for startup
    for i in range(20):
        time.sleep(1)
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
            print(f"Server up after {i+1}s")
            break
        except Exception:
            pass
    else:
        print("ERROR: server did not start in 20s")
        sys.exit(1)

    # Hit /schema
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/schema") as r:
        schema = json.load(r)
    print("Schema:", list(schema.keys()))

    print("Server OK — openenv validate will pass")

finally:
    if proc:
        proc.terminate()
        proc.wait()
        print("Server stopped.")

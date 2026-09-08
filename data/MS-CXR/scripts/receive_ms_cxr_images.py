from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse
import json

ROOT = Path("/Users/yanjiangtao/Desktop/ms-cxr/ms-cxr-making-the-most-of-text-semantics-to-improve-biomedical-vision-language-processing-1.1.0")
IMAGE_ROOT = ROOT / "mimic-cxr-jpg"
LOG_PATH = ROOT / "image_receive_log.jsonl"


class Handler(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "https://physionet.org")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS, GET")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Image-Path")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/missing":
            missing_path = ROOT / "missing_image_paths.json"
            body = missing_path.read_bytes() if missing_path.exists() else b"[]"
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if parsed.path != "/status":
            self.send_response(404)
            self._cors()
            self.end_headers()
            return
        count = sum(1 for _ in IMAGE_ROOT.glob("files/**/*.jpg")) if IMAGE_ROOT.exists() else 0
        body = json.dumps({"received": count}, ensure_ascii=False).encode()
        self.send_response(200)
        self._cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        path = self.headers.get("X-Image-Path", "")
        path = unquote(path).lstrip("/")
        if not path.startswith("files/") or ".." in Path(path).parts:
            self.send_response(400)
            self._cors()
            self.end_headers()
            return

        length = int(self.headers.get("Content-Length", "0"))
        data = self.rfile.read(length)
        out = IMAGE_ROOT / path
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)

        with LOG_PATH.open("a") as log:
            log.write(json.dumps({"path": path, "bytes": len(data)}) + "\n")

        body = b"ok"
        self.send_response(200)
        self._cors()
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        return


if __name__ == "__main__":
    IMAGE_ROOT.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", 8765), Handler)
    print(f"listening on http://127.0.0.1:8765, writing to {IMAGE_ROOT}", flush=True)
    server.serve_forever()

"""Local configuration editor. Run: python -m gui_periphery.editor.app"""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import threading
from urllib.parse import urlsplit
import webbrowser

from config.expressions import preview_expression

try:
    from config.yaml_parser import ConfigError, dump_yaml, load_yaml, new_config, new_stage, save_yaml
except ModuleNotFoundError as exc:
    if exc.name != "yaml":
        raise
    raise SystemExit("Install the editor dependency: python -m pip install -r gui_periphery/requirements.txt") from exc


ROOT = Path(__file__).resolve().parent


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        if str(args[1]) not in ("200", "304"):
            super().log_message(format, *args)

    def reply(self, value, status=200):
        payload = json.dumps(value, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def local_request(self):
        port = self.server.server_port
        hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        origin = self.headers.get("Origin")
        return self.headers.get("Host") in hosts and (origin is None or origin in {"http://" + host for host in hosts})

    def do_GET(self):
        if not self.local_request():
            return self.reply({"error": "This editor accepts local requests only."}, 403)
        route = urlsplit(self.path).path
        if route == "/api/defaults":
            return self.reply({"config": new_config(), "stage": new_stage(),
                               "save_path": str(Path.home() / "Downloads" / "experiment.yaml")})
        if route == "/api/example":
            return self.reply({"config": load_yaml((ROOT.parents[1] / "config" / "examples" / "kick_and_hold.yaml").read_text())})
        files = {"/": "index.html", "/app.js": "app.js", "/style.css": "style.css"}
        if route not in files:
            return self.reply({"error": "Not found."}, 404)
        path = ROOT / "static" / files[route]
        payload = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", (mimetypes.guess_type(path)[0] or "text/plain") + "; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        if not self.local_request() or self.headers.get_content_type() != "application/json":
            return self.reply({"error": "Use a local JSON request."}, 403)
        try:
            size = int(self.headers.get("Content-Length", 0))
            if not 0 < size <= 2 * 1024 * 1024:
                return self.reply({"error": "Configuration must be smaller than 2 MiB."}, 413)
            data = json.loads(self.rfile.read(size))
            if not isinstance(data, dict):
                raise ValueError("Request must be an object.")
            route = urlsplit(self.path).path
            if route == "/api/formula":
                if type(data.get("time_dependent", False)) is not bool:
                    raise ValueError("time_dependent must be true or false.")
                return self.reply(preview_expression(data["expression"], data.get("time_dependent", False)))
            if route == "/api/render":
                return self.reply({"yaml": dump_yaml(data["config"]), "errors": []})
            if route == "/api/import":
                return self.reply({"config": load_yaml(data["yaml"])})
            if route == "/api/save":
                if type(data.get("overwrite", False)) is not bool:
                    raise ValueError("overwrite must be true or false.")
                return self.reply({"path": save_yaml(data["config"], data["path"], data.get("overwrite", False))})
            self.reply({"error": "Not found."}, 404)
        except ConfigError as exc:
            self.reply({"errors": exc.errors}, 422)
        except FileExistsError:
            self.reply({"error": "That file already exists. Choose another name or enable Replace existing file."}, 409)
        except (ValueError, TypeError, KeyError, OSError, RecursionError) as exc:
            self.reply({"error": str(exc)}, 400)


def main():
    parser = argparse.ArgumentParser(description="Open the local GPE YAML configuration editor.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError as exc:
        parser.exit(1, f"Could not start the editor: {exc}. Try --port 8766.\n")
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"GPE configuration editor: {url}\nPress Ctrl+C to stop.", flush=True)
    if not args.no_browser:
        threading.Timer(0.3, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

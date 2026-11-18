"""Serve an existing frontend build with runtime API configuration; stdlib only."""
import argparse
import io
import json
import os
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=5173)
    parser.add_argument('--api-url', default=os.environ.get('AUDITORY_API_URL', 'http://127.0.0.1:8000'),
                        help='Backend URL as reached by your browser, including any SSH forwarded port')
    args = parser.parse_args()
    if not (args.directory / 'index.html').is_file():
        parser.error('Frontend build missing. Run: npm run build --prefix frontend')
    parsed = urlsplit(args.api_url)
    if parsed.scheme not in ('http', 'https') or not parsed.netloc:
        parser.error('--api-url must be an absolute HTTP or HTTPS URL')
    configuration = ('window.AUDITORY_CONFIG = ' + json.dumps({'apiBaseUrl': args.api_url.rstrip('/')}) + ';\n').encode()

    class Handler(SimpleHTTPRequestHandler):
        def send_head(self):
            if urlsplit(self.path).path == '/config.js':
                self.send_response(200)
                self.send_header('Content-Type', 'application/javascript; charset=utf-8')
                self.send_header('Content-Length', str(len(configuration)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                return io.BytesIO(configuration)
            return super().send_head()

    server = ThreadingHTTPServer((args.host, args.port), partial(Handler, directory=str(args.directory)))
    print('Frontend: http://%s:%d' % (args.host, args.port), flush=True)
    print('Browser API URL: ' + args.api_url, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()

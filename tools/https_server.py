"""Create a local https server to host the assets during development.

CORS policy: Accepts requests from any origin ending in googleusercontent.com/

Args:
  home_dir: Path to the assets delivered by the server.
  port: https server port.
"""

import http.server
import os
import ssl
import sys

if __name__ == "__main__":

  home_dir = sys.argv[1]
  port = int(sys.argv[2])

  print("Parameters:")
  print("\thome_dir:", home_dir)
  print("\tport:", port)

  class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
    """A https request handler."""

    def __init__(self, *args, **kwargs):
      super().__init__(*args, directory=home_dir, **kwargs)

    # This server is a people pleaser: It will accept any origin ending in
    # googleusercontent.com.
    def end_headers(self):
      requested_origin = self.headers.get("Referer")
      if requested_origin and requested_origin.endswith(
          "googleusercontent.com/"):
        requested_origin_without_slash = requested_origin[:-1]
        self.send_header("Access-Control-Allow-Origin",
                         requested_origin_without_slash)
        self.send_header("Access-Control-Allow-Credentials", "true")

      http.server.SimpleHTTPRequestHandler.end_headers(self)

  httpd = http.server.HTTPServer(("localhost", port), CORSRequestHandler)

  httpd.socket = ssl.wrap_socket(
      httpd.socket,
      keyfile=os.path.join(home_dir, "key.pem"),
      certfile=os.path.join(home_dir, "cert.pem"),
      server_side=True)

  print(f"Server running on port {port}")
  httpd.serve_forever()

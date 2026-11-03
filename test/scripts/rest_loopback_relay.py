#!/usr/bin/env python3
"""Provide TLS or HTTP CONNECT in front of one loopback test listener."""

import argparse
import json
from pathlib import Path
import select
import signal
import socket
import ssl
import subprocess
import threading


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("tls", "connect"))
    parser.add_argument("upstream", type=int)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    if not 0 < args.upstream < 65536:
        parser.error("upstream must be a TCP port")
    stopped = threading.Event()
    certificate_process = None

    def stop(*_):
        stopped.set()
        if certificate_process is not None and certificate_process.poll() is None:
            certificate_process.terminate()

    signal.signal(signal.SIGTERM, stop)
    event_lock = threading.Lock()
    context = None
    if args.mode == "tls":
        cert = args.directory / "public.crt"
        key = args.directory / "private.key"
        certificate_process = subprocess.Popen(
            [
                "openssl",
                "req",
                "-x509",
                "-newkey",
                "rsa:2048",
                "-nodes",
                "-days",
                "1",
                "-keyout",
                str(key),
                "-out",
                str(cert),
                "-subj",
                "/CN=localhost",
                "-addext",
                "subjectAltName=IP:127.0.0.1,DNS:localhost",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if certificate_process.wait() != 0:
            raise RuntimeError("openssl could not create the loopback certificate")
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(cert, key)

    def record(event):
        with event_lock:
            with (args.directory / "events").open("a") as output:
                output.write(json.dumps(event) + "\n")

    def handle(client):
        upstream = None
        try:
            client.settimeout(3)
            if context is not None:
                client = context.wrap_socket(client, server_side=True)
                record({"mode": "tls"})
            else:
                request = bytearray()
                while b"\r\n\r\n" not in request:
                    part = client.recv(1)
                    if not part or len(request) >= 65536:
                        return
                    request.extend(part)
                first = bytes(request).split(b"\r\n", 1)[0].decode("ascii")
                expected = f"CONNECT 127.0.0.1:{args.upstream} HTTP/1.1"
                if first != expected:
                    client.sendall(
                        b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n"
                    )
                    return
                record({"mode": "connect", "request": first})
                client.sendall(
                    b"HTTP/1.1 200 Connection established\r\n"
                    b'ETag: "proxy-only-marker"\r\n'
                    b"X-Proxy-Marker: proxy-only-marker\r\n\r\n"
                )
            upstream = socket.create_connection(("127.0.0.1", args.upstream), timeout=3)
            while not stopped.is_set():
                ready = (
                    [client]
                    if isinstance(client, ssl.SSLSocket) and client.pending()
                    else select.select([client, upstream], [], [], 0.2)[0]
                )
                for source in ready:
                    data = source.recv(65536)
                    if not data:
                        return
                    destination = upstream if source is client else client
                    destination.sendall(data)
        except (OSError, ssl.SSLError) as error:
            record({"mode": args.mode, "error": type(error).__name__})
        finally:
            client.close()
            if upstream is not None:
                upstream.close()

    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(16)
        listener.settimeout(0.2)
        pending = args.directory / "ready.tmp"
        pending.write_text(str(listener.getsockname()[1]))
        pending.replace(args.directory / "ready")
        while not stopped.is_set():
            try:
                client, _ = listener.accept()
            except socket.timeout:
                continue
            threading.Thread(target=handle, args=(client,), daemon=True).start()


if __name__ == "__main__":
    main()

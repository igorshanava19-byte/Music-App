"""Optional password protection, for when the app is reachable through a live link.

Set MUSICAPP_PASSWORD to turn it on. Logged-in browsers get a signed cookie that
lasts 30 days. Without a password the app is open (fine on 127.0.0.1 only).
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from collections import defaultdict
from html import escape

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

from . import config

COOKIE = "musicapp_session"
MAX_AGE = 30 * 24 * 3600
PUBLIC_PATHS = ("/login", "/static/")

_failures: dict[str, list[float]] = defaultdict(list)


def enabled() -> bool:
    return bool(config.PASSWORD)


def _secret() -> bytes:
    path = config.DATA_DIR / "secret.key"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(secrets.token_hex(32))
    # changing the password logs every browser out
    return hashlib.sha256((path.read_text().strip() + config.PASSWORD).encode()).digest()


def _sign(expires: int) -> str:
    mac = hmac.new(_secret(), str(expires).encode(), hashlib.sha256).hexdigest()
    return f"{expires}.{mac}"


def _valid(token: str | None) -> bool:
    if not token or "." not in token:
        return False
    expires, _, mac = token.partition(".")
    if not expires.isdigit() or int(expires) < time.time():
        return False
    expected = hmac.new(_secret(), expires.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(mac, expected)


def _client(request: Request) -> str:
    # behind a tunnel every request comes from localhost; use the forwarded address
    fwd = request.headers.get("cf-connecting-ip") or request.headers.get("x-forwarded-for", "")
    return fwd.split(",")[0].strip() or (request.client.host if request.client else "?")


async def middleware(request: Request, call_next) -> Response:
    if not enabled() or request.url.path.startswith(PUBLIC_PATHS) or _valid(request.cookies.get(COOKIE)):
        return await call_next(request)
    if request.url.path.startswith("/api/"):
        return JSONResponse({"detail": "Please log in again."}, status_code=401)
    return RedirectResponse("/login", status_code=303)


def login_page(error: str = "") -> HTMLResponse:
    err = f'<p class="err">{escape(error)}</p>' if error else ""
    return HTMLResponse(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Music App – Log in</title>
<link rel="stylesheet" href="/static/style.css">
<style>.login{{max-width:360px;margin:12vh auto;padding:0 16px}}.login input{{width:100%;padding:.6rem .8rem;
border:1px solid var(--border);border-radius:8px;background:var(--bg);margin:.75rem 0}}
.login button{{width:100%}}.err{{color:var(--danger)}}</style></head>
<body><main class="login"><div class="card"><h1>🎼 Music App</h1>{err}
<form method="post" action="/login"><input type="password" name="password" placeholder="Password"
autofocus required autocomplete="current-password"><button class="primary">Log in</button></form>
</div></main></body></html>""")


def check_login(request: Request, password: str) -> Response:
    ip = _client(request)
    now = time.time()
    recent = [t for t in _failures[ip] if now - t < 600]
    _failures[ip] = recent
    if len(recent) >= 8:
        return login_page("Too many wrong attempts. Try again in 10 minutes.")
    if not hmac.compare_digest(password.encode(), config.PASSWORD.encode()):
        recent.append(now)
        time.sleep(1)  # slow down guessing
        return login_page("Wrong password.")
    resp = RedirectResponse("/", status_code=303)
    secure = request.headers.get("x-forwarded-proto", request.url.scheme) == "https"
    resp.set_cookie(COOKIE, _sign(int(now) + MAX_AGE), max_age=MAX_AGE, httponly=True,
                    samesite="lax", secure=secure)
    return resp

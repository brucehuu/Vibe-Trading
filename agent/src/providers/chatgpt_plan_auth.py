"""Official Sign in with ChatGPT for this local, open-source application.

Credentials belong to Vibe-Trading, never to the browser or another Codex app.
See developers.openai.com/siwc/token-sharing-open-source/sign-in.
"""

from __future__ import annotations

import base64
from contextlib import contextmanager
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import tempfile
import threading
import time
from typing import Any
from urllib.parse import parse_qs, urlencode, urlsplit
import uuid

import httpx
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from src.config.paths import get_runtime_root

ISSUER = "https://auth.openai.com"
AUTHORIZE_URL = ISSUER + "/api/accounts/authorize"
TOKEN_URL = ISSUER + "/api/accounts/oauth/token"
RESOURCE = "https://api.openai.com/v1"
PLAN_SCOPE = "chatgpt.tokens.use.direct"
SCOPES = "openid profile email offline_access resource.invoke " + PLAN_SCOPE
_pending: dict[str, Any] | None = None
_pending_guard = threading.RLock()
_last_error: str | None = None


class PlanAuthError(RuntimeError):
    """A safe, credential-free message suitable for the local settings page."""


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode().rstrip("=")


def _unb64(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _path() -> Path:
    return get_runtime_root() / "auth" / "chatgpt-plan.json"


@contextmanager
def _locked():
    # Both the browser callback and model workers may access rotating tokens.
    # Reuse the existing portable, cross-process file-lock implementation.
    from src.providers.openai_codex import _codex_refresh_lock

    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

    class Storage:
        def get_token_path(self):
            return path

    with _codex_refresh_lock(Storage()):
        yield


def _read() -> dict[str, Any]:
    try:
        data = json.loads(_path().read_text())
    except FileNotFoundError:
        return {"profiles": {}}
    except (ValueError, OSError) as exc:
        raise PlanAuthError("ChatGPT credential storage could not be read; reconnect from Settings.") from exc
    if not isinstance(data, dict) or not isinstance(data.get("profiles"), dict):
        raise PlanAuthError("Invalid ChatGPT credential storage; reconnect from Settings.")
    return data


def _write(data: dict[str, Any]) -> None:
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temporary = tempfile.mkstemp(prefix=".chatgpt-plan-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as target:
            os.fchmod(target.fileno(), 0o600)
            json.dump(data, target)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def status() -> dict[str, Any]:
    """Read local status without refreshing tokens or performing inference."""
    data = _read()
    profiles = [
        {"id": key, "label": profile["label"], "plan_enabled": PLAN_SCOPE in profile.get("scopes", [])}
        for key, profile in data["profiles"].items()
    ]
    active = data["profiles"].get(data.get("active"), {})
    return {
        "connected": bool(active.get("access_token")),
        "plan_enabled": bool(active.get("access_token")) and PLAN_SCOPE in active.get("scopes", []),
        "active_profile": data.get("active"),
        "profiles": profiles,
        "pending": bool(_pending and _pending["expires_at"] > time.time()),
        "error": _last_error,
    }


def select_profile(profile_id: str) -> dict[str, Any]:
    with _locked():
        data = _read()
        if profile_id not in data["profiles"]:
            raise PlanAuthError("Unknown ChatGPT connection.")
        data["active"] = profile_id
        _write(data)
    return status()


def _validate_identity(token: str, client_id: str, nonce: str | None) -> dict[str, Any]:
    """Verify the signature and OIDC claims before trusting account identity."""
    try:
        head, payload, signature = token.split(".")
        header, claims = json.loads(_unb64(head)), json.loads(_unb64(payload))
        if header.get("alg") != "RS256" or not header.get("kid"):
            raise ValueError("algorithm")
        response = httpx.get(ISSUER + "/.well-known/jwks.json", timeout=20, follow_redirects=False)
        response.raise_for_status()
        key = next(
            k
            for k in response.json()["keys"]
            if k.get("kid") == header["kid"]
            and k.get("kty") == "RSA"
            and k.get("use", "sig") == "sig"
            and k.get("alg", "RS256") == "RS256"
        )
        public = rsa.RSAPublicNumbers(
            int.from_bytes(_unb64(key["e"]), "big"), int.from_bytes(_unb64(key["n"]), "big")
        ).public_key()
        public.verify(_unb64(signature), (head + "." + payload).encode(), padding.PKCS1v15(), hashes.SHA256())
        audience = claims.get("aud")
        if claims.get("iss") != ISSUER or client_id not in (
            [audience] if isinstance(audience, str) else audience or []
        ):
            raise ValueError("issuer or audience")
        if not isinstance(claims.get("exp"), (int, float)) or claims["exp"] <= time.time():
            raise ValueError("expiry")
        if claims.get("nbf", 0) > time.time() + 30 or not isinstance(claims.get("sub"), str) or not claims["sub"]:
            raise ValueError("subject or not-before")
        if nonce is not None and not hmac.compare_digest(str(claims.get("nonce", "")), nonce):
            raise ValueError("nonce")
        return claims
    except Exception as exc:
        raise PlanAuthError("ChatGPT identity validation failed; start a new connection.") from exc


def _exchange(fields: dict[str, str]) -> dict[str, Any]:
    try:
        response = httpx.post(TOKEN_URL, data=fields, timeout=25, follow_redirects=False)
        if response.status_code != 200:
            raise PlanAuthError(f"ChatGPT token exchange failed (HTTP {response.status_code}); reconnect.")
        token = response.json()
        if not token.get("access_token") or token.get("token_type", "Bearer").lower() != "bearer":
            raise PlanAuthError("ChatGPT returned incomplete credentials; reconnect.")
        return token
    except (httpx.HTTPError, ValueError) as exc:
        raise PlanAuthError("ChatGPT authorization could not reach OpenAI; retry the connection.") from exc


def begin_login(*, new_account: bool = False) -> dict[str, str]:
    """Start a loopback callback before returning the official authorization URL."""
    global _pending, _last_error
    with _pending_guard:
        if _pending:
            raise PlanAuthError("A ChatGPT authorization is already pending; finish or cancel it first.")
        with _locked():
            data = _read()
            data.setdefault("host_id", "urn:uuid:" + str(uuid.uuid4()))
            _write(data)
        profile = {} if new_account else data["profiles"].get(data.get("active"), {})
        client_id = profile.get("client_id") or (None if new_account else data.get("pending_client_id"))
        pending = {
            "state": secrets.token_urlsafe(32),
            "nonce": secrets.token_urlsafe(32),
            "verifier": secrets.token_urlsafe(64),
            "client_id": client_id,
            "subject": profile.get("subject"),
            "profile_id": None if new_account else data.get("active"),
            "expires_at": time.time() + 600,
        }

        class Callback(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass  # OAuth callback URLs must never enter access logs.

            def do_GET(self):
                global _pending, _last_error
                parsed = urlsplit(self.path)
                query = parse_qs(parsed.query)

                def one(key):
                    values = query.get(key, [])
                    return values[0] if len(values) == 1 else ""

                with _pending_guard:
                    valid = (
                        parsed.path == "/auth/callback"
                        and self.headers.get("Host") == f"127.0.0.1:{server.server_port}"
                        and pending["expires_at"] > time.time()
                        and _pending is pending
                        and not pending.get("processing")
                        and hmac.compare_digest(one("state"), pending["state"])
                    )
                    if valid:
                        pending["processing"] = True
                if not valid:
                    self.send_error(400, "Invalid or expired authorization callback")
                    return
                ok = False
                try:
                    if one("error"):
                        raise PlanAuthError("ChatGPT authorization was declined; you can reconnect when ready.")
                    issued = one("client_id") or pending["client_id"]
                    if not issued or issued == "dynamic_agent_client" or not one("code"):
                        raise PlanAuthError("ChatGPT client registration was incomplete; reconnect.")
                    if pending["client_id"] and issued != pending["client_id"]:
                        raise PlanAuthError("ChatGPT returned a different client registration; reconnect.")
                    with _locked():
                        saved = _read()
                        saved["pending_client_id"] = issued
                        _write(saved)
                    token = _exchange(
                        {
                            "grant_type": "authorization_code",
                            "client_id": issued,
                            "code": one("code"),
                            "code_verifier": pending["verifier"],
                            "redirect_uri": redirect_uri,
                            "resource": RESOURCE,
                        }
                    )
                    claims = _validate_identity(token.get("id_token", ""), issued, pending["nonce"])
                    if pending["subject"] and claims["sub"] != pending["subject"]:
                        raise PlanAuthError("The authorized ChatGPT account differs from the selected connection.")
                    with _locked():
                        saved = _read()
                        existing = next(
                            (
                                k
                                for k, p in saved["profiles"].items()
                                if p["client_id"] == issued and p["subject"] == claims["sub"]
                            ),
                            None,
                        )
                        key = existing or secrets.token_hex(12)
                        label = saved["profiles"].get(key, {}).get("label", f"ChatGPT {len(saved['profiles']) + 1}")
                        saved["profiles"][key] = {
                            "label": label,
                            "client_id": issued,
                            "subject": claims["sub"],
                            "access_token": token["access_token"],
                            "refresh_token": token.get("refresh_token", ""),
                            "id_token": token["id_token"],
                            "scopes": token.get("scope", "").split(),
                            "expires_at": time.time() + float(token.get("expires_in", 3600)),
                        }
                        saved["active"] = key
                        saved.pop("pending_client_id", None)
                        _write(saved)
                    _last_error = None
                    ok = True
                except PlanAuthError as exc:
                    _last_error = str(exc)
                except Exception:
                    _last_error = "ChatGPT authorization failed; reconnect from Settings."
                finally:
                    with _pending_guard:
                        if _pending is pending:
                            _pending = None
                message = (
                    "ChatGPT connected. Return to Vibe-Trading Settings."
                    if ok
                    else "ChatGPT connection failed. Return to Vibe-Trading Settings to retry."
                )
                body = ("<!doctype html><meta charset=utf-8><title>Vibe-Trading</title><p>" + message + "</p>").encode()
                self.send_response(200 if ok else 400)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Referrer-Policy", "no-referrer")
                self.send_header("Content-Security-Policy", "default-src 'none'; frame-ancestors 'none'")
                self.end_headers()
                self.wfile.write(body)

                def close():
                    server.shutdown()
                    server.server_close()

                threading.Thread(target=close, daemon=True).start()

        server = ThreadingHTTPServer(("127.0.0.1", 0), Callback)
        redirect_uri = f"http://127.0.0.1:{server.server_port}/auth/callback"
        pending["server"] = server
        _pending = pending
        _last_error = None
        threading.Thread(target=server.serve_forever, daemon=True).start()

        def expire():
            global _pending
            with _pending_guard:
                if _pending is pending:
                    _pending = None
            server.shutdown()
            server.server_close()

        timer = threading.Timer(600, expire)
        timer.daemon = True
        timer.start()
        fields = {
            "client_id": client_id or "dynamic_agent_client",
            "ext_agent_host_id": data["host_id"],
            "response_type": "code",
            "redirect_uri": redirect_uri,
            "scope": SCOPES,
            "resource": RESOURCE,
            "state": pending["state"],
            "nonce": pending["nonce"],
            "code_challenge_method": "S256",
            "code_challenge": _b64(hashlib.sha256(pending["verifier"].encode()).digest()),
        }
        if not client_id:
            fields["agent_name_hint"] = "Vibe-Trading"
        return {"authorization_url": AUTHORIZE_URL + "?" + urlencode(fields)}


def cancel_login() -> dict[str, Any]:
    global _pending
    with _pending_guard:
        pending, _pending = _pending, None
    if pending:
        pending["expires_at"] = 0
        pending["server"].shutdown()
        pending["server"].server_close()
    return status()


def access_token(*, force_refresh: bool = False, rejected_access: str | None = None) -> str:
    with _locked():
        data = _read()
        profile = data["profiles"].get(data.get("active"), {})
        if not profile.get("access_token") or PLAN_SCOPE not in profile.get("scopes", []):
            raise PlanAuthError("Connect ChatGPT and allow ChatGPT plan usage in Settings first.")
        if (not force_refresh or (rejected_access and rejected_access != profile["access_token"])) and profile.get(
            "expires_at", 0
        ) > time.time() + 90:
            return profile["access_token"]
        if not profile.get("refresh_token"):
            raise PlanAuthError("ChatGPT authorization expired; reconnect from Settings.")
        token = _exchange(
            {
                "grant_type": "refresh_token",
                "client_id": profile["client_id"],
                "refresh_token": profile["refresh_token"],
                "resource": RESOURCE,
            }
        )
        if token.get("id_token"):
            claims = _validate_identity(token["id_token"], profile["client_id"], None)
            if claims["sub"] != profile["subject"]:
                raise PlanAuthError("Refreshed ChatGPT identity changed; reconnect from Settings.")
            profile["id_token"] = token["id_token"]
        profile["access_token"] = token["access_token"]
        if token.get("refresh_token"):
            profile["refresh_token"] = token["refresh_token"]
        if "scope" in token:
            profile["scopes"] = token["scope"].split()
        profile["expires_at"] = time.time() + float(token.get("expires_in", 3600))
        _write(data)
        if PLAN_SCOPE not in profile.get("scopes", []):
            raise PlanAuthError("ChatGPT plan permission is unavailable; reconnect and allow plan usage.")
        return profile["access_token"]


def list_models() -> list[str]:
    try:
        response = httpx.get(
            RESOURCE + "/models",
            headers={"Authorization": "Bearer " + access_token()},
            timeout=20,
            follow_redirects=False,
        )
        if response.status_code != 200:
            raise PlanAuthError(f"ChatGPT model discovery failed (HTTP {response.status_code}).")
        models = [
            m["slug"]
            for m in response.json().get("models", [])
            if isinstance(m, dict) and m.get("visibility") == "list" and isinstance(m.get("slug"), str)
        ]
        if not models:
            raise PlanAuthError("OpenAI returned no models available to this ChatGPT connection.")
        return list(dict.fromkeys(models))
    except (httpx.HTTPError, ValueError) as exc:
        raise PlanAuthError("ChatGPT model discovery could not reach OpenAI.") from exc

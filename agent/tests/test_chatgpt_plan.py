"""Official plan auth and transport contracts, using isolated fake credentials."""

from concurrent.futures import ThreadPoolExecutor
import json
import time
from urllib.parse import parse_qs, urlencode, urlsplit

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from fastapi import FastAPI
from fastapi.testclient import TestClient
import httpx
import pytest

from src.providers import chatgpt_plan_auth as auth
from src.providers.chatgpt_plan import ChatGPTPlanLLM
from src.providers.openai_codex import CodexStreamError


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("VIBE_TRADING_HOME", str(tmp_path))
    monkeypatch.setattr(auth, "_pending", None)
    monkeypatch.setattr(auth, "_last_error", None)
    yield
    auth.cancel_login()


def save_profile(**overrides):
    profile = {
        "label": "ChatGPT 1",
        "client_id": "test-client",
        "subject": "test-subject",
        "access_token": "fake-access",
        "refresh_token": "fake-refresh",
        "scopes": [auth.PLAN_SCOPE],
        "expires_at": time.time() + 3000,
    }
    profile.update(overrides)
    auth._write({"host_id": "test-host", "active": "first", "profiles": {"first": profile}})


def test_login_pkce_stable_host_and_invalid_state_do_not_exchange(monkeypatch):
    monkeypatch.setattr(auth, "_exchange", lambda _: pytest.fail("unexpected exchange"))
    first = auth.begin_login()
    query = parse_qs(urlsplit(first["authorization_url"]).query)
    assert query["client_id"] == ["dynamic_agent_client"]
    assert query["agent_name_hint"] == ["Vibe-Trading"]
    assert query["resource"] == [auth.RESOURCE]
    assert query["code_challenge_method"] == ["S256"]
    assert auth.PLAN_SCOPE in query["scope"][0].split()
    callback = query["redirect_uri"][0]
    response = httpx.get(callback + "?state=wrong&code=fake", trust_env=False)
    assert response.status_code == 400
    assert auth.status()["pending"]
    auth.cancel_login()
    second = parse_qs(urlsplit(auth.begin_login()["authorization_url"]).query)
    assert second["ext_agent_host_id"] == query["ext_agent_host_id"]
    assert second["state"] != query["state"]
    assert auth._path().stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("granted", ["openid", "openid " + auth.PLAN_SCOPE])
def test_callback_keeps_identity_separate_from_plan_permission(monkeypatch, granted):
    query = parse_qs(urlsplit(auth.begin_login()["authorization_url"]).query)
    monkeypatch.setattr(
        auth,
        "_exchange",
        lambda _: {
            "access_token": "fake-access",
            "refresh_token": "fake-refresh",
            "id_token": "fake-id",
            "scope": granted,
        },
    )
    monkeypatch.setattr(auth, "_validate_identity", lambda *args: {"sub": "test-subject"})
    response = httpx.get(
        query["redirect_uri"][0]
        + "?"
        + urlencode({"state": query["state"][0], "code": "fake-code", "client_id": "test-issued-client"}),
        trust_env=False,
    )
    assert response.status_code == 200
    public = auth.status()
    assert public["connected"]
    assert public["plan_enabled"] == (auth.PLAN_SCOPE in granted.split())
    assert "fake-access" not in json.dumps(public)
    assert "fake-refresh" not in json.dumps(public)
    auth.cancel_login()
    returning = parse_qs(urlsplit(auth.begin_login()["authorization_url"]).query)
    assert returning["client_id"] == ["test-issued-client"]
    assert "agent_name_hint" not in returning


def test_signature_nonce_audience_and_expiry_are_verified(monkeypatch):
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    numbers = private.public_key().public_numbers()

    def encode_int(n):
        return auth._b64(n.to_bytes((n.bit_length() + 7) // 8, "big"))

    jwks = {"keys": [{"kty": "RSA", "kid": "test-key", "n": encode_int(numbers.n), "e": encode_int(numbers.e)}]}
    monkeypatch.setattr(
        auth.httpx,
        "get",
        lambda *a, **k: httpx.Response(200, json=jwks, request=httpx.Request("GET", "https://auth.openai.com")),
    )

    def signed(**overrides):
        claims = {
            "iss": auth.ISSUER,
            "aud": "test-client",
            "sub": "test-subject",
            "nonce": "test-nonce",
            "exp": time.time() + 60,
        }
        claims.update(overrides)
        head = auth._b64(json.dumps({"alg": "RS256", "kid": "test-key"}).encode())
        payload = auth._b64(json.dumps(claims).encode())
        message = (head + "." + payload).encode()
        return message.decode() + "." + auth._b64(private.sign(message, padding.PKCS1v15(), hashes.SHA256()))

    assert auth._validate_identity(signed(), "test-client", "test-nonce")["sub"] == "test-subject"
    for overrides in [{"aud": "other-client"}, {"exp": 0}, {"nonce": "wrong"}, {"iss": "https://example.org"}]:
        with pytest.raises(auth.PlanAuthError):
            auth._validate_identity(signed(**overrides), "test-client", "test-nonce")


def test_refresh_is_serialized_and_rotated_only_once(monkeypatch):
    save_profile(expires_at=0)
    calls = []

    def exchange(fields):
        calls.append(fields)
        time.sleep(0.02)
        return {"access_token": "replacement-access", "refresh_token": "replacement-refresh", "expires_in": 3600}

    monkeypatch.setattr(auth, "_exchange", exchange)
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(lambda _: auth.access_token(), range(2))) == ["replacement-access"] * 2
    assert len(calls) == 1
    assert calls[0]["client_id"] == "test-client"
    assert "scope" not in calls[0]
    assert auth._read()["profiles"]["first"]["refresh_token"] == "replacement-refresh"


def test_namespace_tool_history_and_preview_fields():
    llm = ChatGPTPlanLLM(model="test-model").bind_tools([{"name": "lookup", "parameters": {"type": "object"}}])
    body = llm._body(
        [
            {"role": "system", "content": "test instructions"},
            {"role": "user", "content": "look up"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "call1", "content": "test result"},
        ]
    )
    assert body["store"] is False and body["stream"] is True
    assert body["instructions"] == "test instructions"
    assert body["tools"][0]["type"] == "namespace"
    assert next(i for i in body["input"] if i.get("type") == "function_call")["namespace"] == "vibe_tools"
    assert not ({"temperature", "max_output_tokens", "previous_response_id", "background"} & body.keys())


@pytest.mark.parametrize("terminal", [None, "response.failed", "response.incomplete", "response.completed"])
def test_stream_requires_success_terminal_event(monkeypatch, terminal):
    monkeypatch.setattr(auth, "access_token", lambda **kwargs: "fake-access")

    def handle(request):
        assert str(request.url) == auth.RESOURCE + "/responses"
        events = [{"type": "response.output_text.delta", "delta": "hello"}]
        if terminal:
            events.append(
                {
                    "type": terminal,
                    "response": {"status": "completed" if terminal == "response.completed" else "failed"},
                }
            )
        return httpx.Response(200, text="".join("data: " + json.dumps(e) + "\n\n" for e in events))

    original = httpx.Client
    monkeypatch.setattr(
        "src.providers.chatgpt_plan.httpx.Client", lambda **kwargs: original(transport=httpx.MockTransport(handle))
    )
    llm = ChatGPTPlanLLM(model="test-model")
    if terminal == "response.completed":
        assert llm.invoke([]).content == "hello"
    else:
        with pytest.raises(CodexStreamError):
            llm.invoke([])


def test_local_settings_auth_and_no_activation_without_permission(monkeypatch):
    from src.api.chatgpt_plan_routes import register_chatgpt_plan_routes
    from src.api.security import require_local_or_auth, require_settings_write_auth

    monkeypatch.delenv("API_AUTH_KEY", raising=False)
    app = FastAPI()
    register_chatgpt_plan_routes(app, require_local_or_auth, require_settings_write_auth)
    local = TestClient(app, client=("127.0.0.1", 50000))
    assert local.get("/settings/chatgpt-plan").json()["plan_enabled"] is False
    assert (
        local.post("/settings/chatgpt-plan/login", json={}, headers={"Origin": "https://example.org"}).status_code
        == 403
    )
    remote = TestClient(app, client=("192.0.2.1", 50000))
    assert remote.get("/settings/chatgpt-plan").status_code == 403
    assert not auth._path().exists()


def test_factory_and_readiness_use_plan_permission_without_ambient_api_keys(monkeypatch):
    from src.config.accessor import reset_env_config
    from src.providers import llm
    from src.providers.capabilities import get_llm_credentials
    from src.api.system_routes import _provider_readiness
    from src.preflight import _check_llm_provider

    monkeypatch.setenv("LANGCHAIN_PROVIDER", "chatgpt-plan")
    monkeypatch.setenv("LANGCHAIN_MODEL_NAME", "test-model")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-unrelated-key")
    monkeypatch.setattr(llm, "_ensure_dotenv", lambda: None)
    reset_env_config()
    assert get_llm_credentials("chatgpt-plan", "test-model")["api_key"] == ""
    assert isinstance(llm.build_llm(), ChatGPTPlanLLM)
    assert _provider_readiness()[0] is False
    assert _check_llm_provider().status == "not_configured"
    save_profile()
    assert _provider_readiness()[0] is True
    assert _check_llm_provider().status == "ready"
    reset_env_config()

"""ChatGPT plan inference through the public Responses API."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import httpx

from src.providers import chatgpt_plan_auth as auth
from src.providers.openai_codex import (
    CodexAIMessage,
    CodexStreamError,
    OpenAICodexLLM,
    _convert_messages,
    _convert_tools,
    _events_from_lines,
    _message_chunks_from_events,
)


class ChatGPTPlanLLM(OpenAICodexLLM):
    """Reuse the agent message codec while keeping auth and transport separate."""

    def __init__(
        self,
        *,
        model: str,
        temperature: float = 0,
        timeout: int = 120,
        tools: list[dict[str, Any]] | None = None,
        reasoning_effort: str | None = None,
    ):
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.tools = tools or []
        self.reasoning_effort = reasoning_effort

    def bind_tools(self, tools):
        return ChatGPTPlanLLM(
            model=self.model,
            temperature=self.temperature,
            timeout=self.timeout,
            tools=tools,
            reasoning_effort=self.reasoning_effort,
        )

    def _body(self, messages, *, stream=True, replay_reasoning=True):
        instructions, items = _convert_messages(messages, replay_reasoning=replay_reasoning)
        body = {
            "model": self.model.removeprefix("chatgpt-plan/"),
            "instructions": instructions,
            "input": items,
            "store": False,
            "stream": True,
            "include": ["reasoning.encrypted_content"],
        }
        tools = _convert_tools(self.tools)
        if tools:
            body["tools"] = [
                {
                    "type": "namespace",
                    "name": "vibe_tools",
                    "description": "Vibe-Trading research tools",
                    "tools": tools,
                }
            ]
            # Tool history must match the namespace used in the tool catalog.
            for item in items:
                if item.get("type") == "function_call":
                    item["namespace"] = "vibe_tools"
        if self.reasoning_effort:
            body["reasoning"] = {"effort": self.reasoning_effort}
        return body

    def stream(self, messages, config=None) -> Iterable[CodexAIMessage]:
        timeout = (config or {}).get("timeout") or self.timeout
        body = self._body(messages)
        token = auth.access_token()
        try:
            with httpx.Client(timeout=timeout, follow_redirects=False) as client:
                for attempt in range(2):
                    with client.stream(
                        "POST",
                        auth.RESOURCE + "/responses",
                        json=body,
                        headers={
                            "Authorization": "Bearer " + token,
                            "Accept": "text/event-stream",
                            "User-Agent": "Vibe-Trading/ChatGPT-plan",
                        },
                    ) as response:
                        if response.status_code == 401 and attempt == 0:
                            response.read()
                            token = auth.access_token(force_refresh=True, rejected_access=token)
                            continue
                        if response.status_code != 200:
                            # Never put response bodies, credentials or account IDs in logs.
                            raise CodexStreamError(
                                400,
                                f"ChatGPT plan request failed (HTTP {response.status_code}); check the connection and plan limits.",
                            )
                        completed = False

                        def checked_events():
                            nonlocal completed
                            for event in _events_from_lines(response.iter_lines()):
                                kind = event.get("type")
                                if kind in {"error", "response.failed", "response.incomplete"}:
                                    raise CodexStreamError(
                                        400,
                                        "ChatGPT plan request did not complete; check plan usage and retry manually.",
                                    )
                                if kind == "response.completed":
                                    if (event.get("response") or {}).get("status") != "completed":
                                        raise CodexStreamError(400, "ChatGPT plan returned a non-completed response.")
                                    completed = True
                                yield event

                        for chunk in _message_chunks_from_events(checked_events()):
                            for call in chunk.tool_calls:
                                call["name"] = call["name"].removeprefix("vibe_tools.")
                            yield chunk
                        if not completed:
                            raise CodexStreamError(400, "ChatGPT stream interrupted before completion; retry manually.")
                        return
        except httpx.HTTPError as exc:
            # The outcome may be unknown. Do not automatically duplicate inference.
            raise CodexStreamError(
                400, "ChatGPT connection interrupted; request outcome unknown. Retry manually."
            ) from exc

"""Local settings endpoints for official ChatGPT plan authorization."""

from __future__ import annotations

import asyncio
from fastapi import Depends, HTTPException, Request
from pydantic import BaseModel

from src.providers import chatgpt_plan_auth as auth


class LoginRequest(BaseModel):
    new_account: bool = False


class SelectRequest(BaseModel):
    profile_id: str


def register_chatgpt_plan_routes(app, require_local_or_auth, require_settings_write_auth):
    def local_only(request: Request):
        # This workflow owns a loopback listener on the user's local machine.
        if not request.client or request.client.host not in {"127.0.0.1", "::1"}:
            raise HTTPException(403, "ChatGPT plan sign-in is available on the local machine only.")

    async def run(fn, *args, **kwargs):
        try:
            return await asyncio.to_thread(fn, *args, **kwargs)
        except auth.PlanAuthError as exc:
            raise HTTPException(400, str(exc)) from exc

    read_deps = [Depends(local_only), Depends(require_local_or_auth)]
    write_deps = [Depends(local_only), Depends(require_settings_write_auth)]

    @app.get("/settings/chatgpt-plan", dependencies=read_deps)
    async def get_status():
        return await run(auth.status)

    @app.post("/settings/chatgpt-plan/login", dependencies=write_deps)
    async def login(payload: LoginRequest):
        return await run(auth.begin_login, new_account=payload.new_account)

    @app.post("/settings/chatgpt-plan/cancel", dependencies=write_deps)
    async def cancel():
        return await run(auth.cancel_login)

    @app.post("/settings/chatgpt-plan/select", dependencies=write_deps)
    async def select(payload: SelectRequest):
        return await run(auth.select_profile, payload.profile_id)

"""
Universal AI Proxy  v8.1
=========================
بروكسي موحد يدعم أربعة نماذج:
  ① Qwen           — عبر chat.qwen.ai
  ② DeepSeek Expert — عبر chat.deepseek.com (model_type=expert)
  ③ DeepSeek Default— عبر chat.deepseek.com (model_type=default)
  ④ Database        — عبر qcpujeurnkbvwlvmylyx.supabase.co (Gemini, GPT, Claude)

تغييرات v8.1:
  • حذف GPT backend نهائياً
  • إصلاح Database: تنظيف messages قبل الإرسال (إزالة حقول thinking وغيرها)
  • إصلاح Database: تحويل system messages إلى user message أول
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import re
import time
import uuid
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

# ══════════════════════════════════════════════════════════
# Logging
# ══════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("ai_proxy")


# ══════════════════════════════════════════════════════════
# Session Store
# ══════════════════════════════════════════════════════════
SESSION_TTL = 60 * 60 * 6

_sessions: Dict[str, Dict[str, Any]] = {}
_session_lock = asyncio.Lock()


def _session_key(token: str, conv_id: str) -> str:
    return f"{token[:16]}:{conv_id}"


async def _get_session(token: str, conv_id: str) -> Optional[Dict]:
    async with _session_lock:
        sess = _sessions.get(_session_key(token, conv_id))
        if sess:
            sess["last_used"] = time.time()
        return dict(sess) if sess else None


async def _set_session(token: str, conv_id: str, data: Dict) -> None:
    async with _session_lock:
        _sessions[_session_key(token, conv_id)] = {**data, "last_used": time.time()}


async def _update_session(token: str, conv_id: str, **kwargs) -> None:
    async with _session_lock:
        key = _session_key(token, conv_id)
        if key in _sessions:
            _sessions[key].update(kwargs)
            _sessions[key]["last_used"] = time.time()


async def _evict_old_sessions() -> None:
    async with _session_lock:
        now   = time.time()
        stale = [k for k, v in _sessions.items() if now - v["last_used"] > SESSION_TTL]
        for k in stale:
            del _sessions[k]
        if stale:
            log.info("Evicted %d stale sessions", len(stale))


# ══════════════════════════════════════════════════════════
# conv_id ثابت
# ══════════════════════════════════════════════════════════

def _compute_conv_id(
    messages: List[Dict],
    explicit_id: Optional[str] = None,
) -> str:
    if explicit_id:
        return explicit_id
    anchor = messages[:1] if messages else [{"role": "user", "content": "init"}]
    raw    = json.dumps(anchor, ensure_ascii=False, sort_keys=True)
    return "conv_" + hashlib.md5(raw.encode()).hexdigest()


# ══════════════════════════════════════════════════════════
# Backend Architecture
# ══════════════════════════════════════════════════════════

class BaseBackend(ABC):
    @property
    @abstractmethod
    def model_id(self) -> str: ...

    @abstractmethod
    async def complete(
        self,
        token:    str,
        messages: List[Dict],
        tools:    List[Dict],
        thinking: bool,
        conv_id:  str,
        extra:    Dict,
    ) -> AsyncIterator[str]: ...


_BACKENDS: Dict[str, "BaseBackend"] = {}


def register_backend(backend: "BaseBackend") -> None:
    _BACKENDS[backend.model_id] = backend
    log.info("Registered backend: %s", backend.model_id)


def get_backend(model_id: str) -> Optional["BaseBackend"]:
    return _BACKENDS.get(model_id)


# ══════════════════════════════════════════════════════════
# Shared Tools
# ══════════════════════════════════════════════════════════

TOOL_SYSTEM_SUFFIX = """
══════════════════════════════════════════════
TOOL USE — STRICT FORMAT
══════════════════════════════════════════════
When you need to call a tool, your ENTIRE response MUST be ONLY this exact line:

ACTION: tool_name|{"param1": "value1"}

Rules:
  • Output ONLY the ACTION line — no words before or after it
  • ONE tool call per response — never two at once
  • Use valid JSON (double quotes)
  • NEVER fabricate tool results — wait for [TOOL RESULT: ...] blocks
  • When finished (no more tools needed) → write your answer as plain text

Tool results arrive as:
  [TOOL RESULT: tool_name]
  ...result content...
  [/TOOL RESULT]
══════════════════════════════════════════════
"""


def tools_to_xml(tools: List[Dict]) -> str:
    if not tools:
        return ""
    lines = ["<available_tools>"]
    for tool in tools:
        func       = tool.get("function") or tool
        name       = func.get("name", "unknown")
        desc       = func.get("description", "")
        params     = func.get("parameters", {})
        required   = params.get("required", [])
        properties = params.get("properties", {})
        lines.append("  <tool>")
        lines.append(f"    <name>{name}</name>")
        lines.append(f"    <description>{desc}</description>")
        if properties:
            lines.append("    <parameters>")
            for pname, pinfo in properties.items():
                ptype = pinfo.get("type", "string")
                pdesc = pinfo.get("description", "")
                req   = " required" if pname in required else ""
                lines.append(f'      <param name="{pname}" type="{ptype}"{req}>{pdesc}</param>')
            lines.append("    </parameters>")
        lines.append("  </tool>")
    lines.append("</available_tools>")
    return "\n".join(lines)


def messages_to_prompt(messages: List[Dict], tools: List[Dict]) -> Tuple[str, str]:
    system_parts = []
    conv_parts   = []

    for m in messages:
        if m.get("role") == "system":
            content = m.get("content") or ""
            if isinstance(content, list):
                content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
            system_parts.append(str(content))

    if tools:
        system_parts.append(f"\n{tools_to_xml(tools)}\n{TOOL_SYSTEM_SUFFIX}")

    for m in messages:
        role    = m.get("role", "user")
        content = m.get("content") or ""

        if role == "system":
            continue
        elif role == "user":
            if isinstance(content, list):
                text_parts = []
                for c in content:
                    if isinstance(c, dict):
                        if c.get("type") == "text":
                            text_parts.append(c.get("text", ""))
                        elif c.get("type") == "image_url":
                            text_parts.append("[IMAGE]")
                content = " ".join(text_parts)
            conv_parts.append(f"User: {content}")
        elif role == "assistant":
            if isinstance(content, list):
                content = " ".join(
                    c.get("text", "") for c in content
                    if isinstance(c, dict) and c.get("type") == "text"
                )
            tool_calls = m.get("tool_calls", [])
            if tool_calls:
                for tc in tool_calls:
                    func    = tc.get("function", {})
                    tc_name = func.get("name", "")
                    tc_args = func.get("arguments", "{}")
                    conv_parts.append(f"ACTION: {tc_name}|{tc_args}")
            else:
                conv_parts.append(f"Assistant: {content}")
        elif role in ("tool", "function"):
            tool_name = m.get("name") or m.get("tool_call_id", "tool")
            if isinstance(content, list):
                content = str(content)
            conv_parts.append(f"[TOOL RESULT: {tool_name}]\n{content}\n[/TOOL RESULT]")

    return "\n\n".join(system_parts), "\n\n".join(conv_parts)


def build_full_prompt(messages: List[Dict], tools: List[Dict]) -> str:
    system_text, conv_text = messages_to_prompt(messages, tools)
    parts = []
    if system_text:
        parts.append(f"[SYSTEM]\n{system_text}\n[/SYSTEM]")
    if conv_text:
        parts.append(conv_text)
    parts.append("Assistant:")
    return "\n\n".join(parts)


def parse_tool_call(text: str) -> Optional[Dict]:
    m = re.search(r"(?m)^ACTION:\s*(\w+)\|(\{.*\})\s*$", text, re.DOTALL)
    if m:
        return _make_tc(m.group(1), m.group(2))
    m = re.search(
        r"<tool_call>\s*<name>(.*?)</name>\s*<arguments>(.*?)</arguments>\s*</tool_call>",
        text, re.DOTALL | re.IGNORECASE,
    )
    if m:
        return _make_tc(m.group(1), m.group(2))

    m_name   = re.search(r"<name>(.*?)</name>", text, re.IGNORECASE)
    m_params = re.findall(
        r"<parameter[=:](\w+)>\s*(.*?)\s*</parameter>",
        text, re.DOTALL | re.IGNORECASE,
    )
    if m_name and m_params:
        params = {p: v for p, v in m_params}
        return _make_tc(m_name.group(1).strip(), json.dumps(params, ensure_ascii=False))

    return None


def _make_tc(name: str, args_raw: str) -> Dict:
    name = name.strip()
    try:
        args_obj = json.loads(args_raw)
    except json.JSONDecodeError:
        try:
            args_obj = json.loads(args_raw.replace("'", '"'))
        except Exception:
            args_obj = {"raw": args_raw.strip()}
    return {"name": name, "arguments": json.dumps(args_obj, ensure_ascii=False)}


def clean_text(text: str) -> str:
    text = re.sub(r"(?m)^ACTION:\s*\S+\|.*$", "", text)
    text = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"</?function[^>]*>", "", text, flags=re.IGNORECASE)
    return text.strip()


def make_tc_response(tc: Dict, model: str) -> Dict:
    call_id = f"call_{uuid.uuid4().hex[:24]}"
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion",
        "created": int(time.time()), "model": model,
        "choices": [{"index": 0, "message": {
            "role": "assistant", "content": None,
            "tool_calls": [{"id": call_id, "type": "function",
                            "function": {"name": tc["name"], "arguments": tc["arguments"]}}],
        }, "finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def make_text_response(content: str, model: str) -> Dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion",
        "created": int(time.time()), "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def sse_chunk(
    content: str = "", model: str = "qwen", *,
    finish: bool = False, tc: Optional[Dict] = None, call_id: Optional[str] = None,
) -> str:
    cid = call_id or f"call_{uuid.uuid4().hex[:24]}"
    if tc and not finish:
        delta = {"role": "assistant", "content": None,
                 "tool_calls": [{"index": 0, "id": cid, "type": "function",
                                 "function": {"name": tc["name"], "arguments": tc["arguments"]}}]}
        choice = {"index": 0, "delta": delta, "finish_reason": None}
    elif finish and tc:
        choice = {"index": 0, "delta": {}, "finish_reason": "tool_calls"}
    elif finish:
        choice = {"index": 0, "delta": {}, "finish_reason": "stop"}
    else:
        choice = {"index": 0, "delta": {"content": content}, "finish_reason": None}
    obj = {"id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion.chunk",
           "created": int(time.time()), "model": model, "choices": [choice]}
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


# ══════════════════════════════════════════════════════════
# BACKEND 1: Qwen
# ══════════════════════════════════════════════════════════

QWEN_BASE          = "https://chat.qwen.ai/api/v2"
QWEN_MODEL_ID_REAL = "qwen3.8-max"
QWEN_PROXY_ID      = "qwen"
REQUEST_TIMEOUT    = 180

_UA_CHAT = (
    "Dalvik/2.1.0 (Linux; U; Android 15; RMX3834 Build/AP3A.240905.015.A2) "
    "AliApp(QWENCHAT/2.7.2) AppType/Release AplusBridgeLite"
)
_UA_NEW = (
    "Dalvik/2.1.0 (Linux; U; Android 15; RMX3834 Build/AP3A.240905.015.A2),"
    "Dalvik/2.1.0 (Linux; U; Android 15; RMX3834 Build/AP3A.240905.015.A2) "
    "AliApp(QWENCHAT/2.7.2) AppType/Release AplusBridgeLite"
)


def _qwen_headers_chat(token: str, *, stream: bool = False) -> Dict[str, str]:
    return {
        "User-Agent": _UA_CHAT, "Content-Type": "application/json; charset=UTF-8",
        "Accept": "*/*,text/event-stream" if stream else "application/json",
        "Accept-Language": "en-US", "Accept-Encoding": "gzip, deflate",
        "Cache-Control": "no-store", "Connection": "Keep-Alive",
        "Host": "chat.qwen.ai", "X-Platform": "android", "x-device-id": "0",
        "source": "app", "Authorization": f"Bearer {token}",
        "x-request-id": str(uuid.uuid4()), "Cookie": f"x-ap=eu-central-1; token={token}",
    }


def _qwen_headers_new(token: str) -> Dict[str, str]:
    return {
        "User-Agent": _UA_NEW, "Content-Type": "application/json",
        "Accept": "application/json", "Accept-Language": "en-US",
        "Accept-Encoding": "gzip", "Connection": "Keep-Alive",
        "Host": "chat.qwen.ai", "X-Platform": "android", "x-device-id": "0",
        "source": "app", "Authorization": f"Bearer {token}",
        "x-request-id": str(uuid.uuid4()), "Cookie": f"x-ap=eu-central-1; token={token}",
    }


def _qwen_is_rate_limited(obj: Any) -> bool:
    if isinstance(obj, str):
        return "RateLimited" in obj
    if isinstance(obj, dict):
        if obj.get("code") == "RateLimited":
            return True
        data = obj.get("data")
        if isinstance(data, dict) and data.get("code") == "RateLimited":
            return True
        try:
            return "RateLimited" in json.dumps(obj)
        except Exception:
            return False
    return False


def _qwen_is_antibot(line: str) -> bool:
    return "_____tmd_____" in line or "punish" in line


async def _qwen_create_chat(token: str, client: httpx.AsyncClient) -> str:
    url     = f"{QWEN_BASE}/chats/new"
    payload = {"chat_mode": "normal", "project_id": ""}
    resp    = await client.post(url, json=payload, headers=_qwen_headers_new(token), timeout=60)
    data    = resp.json()
    cid     = (data.get("chat_id") or data.get("id")
               or (data.get("data") or {}).get("chat_id")
               or (data.get("data") or {}).get("id"))
    if not cid:
        raise HTTPException(status_code=502, detail=f"Failed to create Qwen chat: {data}")
    return cid


def _qwen_build_payload(chat_id, prompt, parent_id, *, chat_type="t2t", thinking=False,
                         auto_search=False, files=None, size="1:1") -> Dict:
    ts  = int(time.time())
    fid = str(uuid.uuid4())
    return {
        "stream": True, "incremental_output": True, "chatId": chat_id, "chat_id": chat_id,
        "chat_mode": "normal", "model": QWEN_MODEL_ID_REAL,
        "messages": [{"id": None, "fid": fid, "chat_type": chat_type, "content": prompt,
                       "role": "user", "feature_config": {
                           "output_schema": "phase", "thinking_enabled": thinking,
                           "thinking_format": "summary", "auto_thinking": thinking,
                           "auto_search": auto_search,
                       }, "timestamp": ts, "sub_chat_type": chat_type,
                       "models": [QWEN_MODEL_ID_REAL], "model": "", "files": files or [],
                       "user_action": "chat", "extra": {"meta": {"subChatType": chat_type}},
                       "parentId": parent_id, "parent_id": parent_id}],
        "timestamp": ts, "size": size, "share_id": "", "version": "2.1",
        "origin_branch_message_id": "", "parentId": parent_id or "",
        "parent_id": parent_id,
    }


async def _qwen_stream_collect(token, chat_id, payload, client) -> Tuple[str, Optional[str]]:
    url      = f"{QWEN_BASE}/chat/completions"
    full_txt = ""
    resp_id: Optional[str] = None
    async with client.stream("POST", url, json=payload,
                              headers=_qwen_headers_chat(token, stream=True),
                              params={"chat_id": chat_id}, timeout=REQUEST_TIMEOUT) as resp:
        async for raw_line in resp.aiter_lines():
            if not raw_line:
                continue
            if _qwen_is_antibot(raw_line):
                break
            if _qwen_is_rate_limited(raw_line):
                full_txt += "[ERROR: Rate limited]"
                break
            if not raw_line.startswith("data: "):
                continue
            ds = raw_line[6:].strip()
            if ds == "[DONE]":
                break
            try:
                obj = json.loads(ds)
                if _qwen_is_rate_limited(obj):
                    full_txt += "[ERROR: Rate limited]"
                    break
                rid = (obj.get("response_id") or
                       obj.get("choices", [{}])[0].get("delta", {}).get("response_id"))
                if rid:
                    resp_id = rid
                choices = obj.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                if delta.get("phase", "") not in ("answer", ""):
                    continue
                content = delta.get("content", "")
                if content:
                    full_txt += content
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
    return full_txt, resp_id


class QwenBackend(BaseBackend):
    @property
    def model_id(self) -> str:
        return QWEN_PROXY_ID

    async def complete(self, token, messages, tools, thinking, conv_id, extra) -> AsyncIterator[str]:
        prompt = build_full_prompt(messages, tools)
        if not prompt.strip():
            return
        await _evict_old_sessions()
        sess = await _get_session(token, conv_id)
        if sess:
            qwen_chat_id = sess["qwen_chat_id"]
            parent_id    = sess.get("parent_id")
        else:
            async with httpx.AsyncClient() as tmp:
                qwen_chat_id = await _qwen_create_chat(token, tmp)
            parent_id = None
            await _set_session(token, conv_id, {"qwen_chat_id": qwen_chat_id, "parent_id": parent_id})
        payload = _qwen_build_payload(qwen_chat_id, prompt, parent_id, thinking=thinking)
        async with httpx.AsyncClient() as client:
            qwen_text, last_rid = await _qwen_stream_collect(token, qwen_chat_id, payload, client)
        await _update_session(token, conv_id, parent_id=last_rid)
        tc = parse_tool_call(qwen_text)
        if tc:
            call_id = f"call_{uuid.uuid4().hex[:24]}"
            yield sse_chunk(tc=tc, model=self.model_id, call_id=call_id)
            yield sse_chunk(tc=tc, model=self.model_id, call_id=call_id, finish=True)
            yield "data: [DONE]\n\n"
        else:
            txt = clean_text(qwen_text)
            for i in range(0, max(len(txt), 1), 40):
                yield sse_chunk(txt[i:i+40], model=self.model_id)
            yield sse_chunk(model=self.model_id, finish=True)
            yield "data: [DONE]\n\n"


register_backend(QwenBackend())


# ══════════════════════════════════════════════════════════
# BACKEND 2 & 3: DeepSeek (Expert + Default)
# ══════════════════════════════════════════════════════════

DEEPSEEK_PROXY_ID_EXPERT  = "deepseek"
DEEPSEEK_PROXY_ID_DEFAULT = "deepseek-default"
DEEPSEEK_CHAT_URL         = "https://chat.deepseek.com/api/v0/chat/completion"
DEEPSEEK_SESSION_URL      = "https://chat.deepseek.com/api/v0/chat_session/create"
RAILWAY_POW_URL           = "https://pow.up.railway.app/pow"


def _ds_rangers_id() -> str:
    ts = int(time.time() * 1000)
    rv = int(1_000_000_000 + (uuid.uuid4().int % 8_999_999_999))
    return str((ts << 32) | rv)


def _ds_device_id() -> str:
    import random
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    return "".join(random.choice(chars) for _ in range(88))


def _ds_tz_offset() -> str:
    return "0"


def _ds_headers(token: str, pow_response: str) -> Dict[str, str]:
    return {
        "User-Agent": "DeepSeek/2.1.1 Android/36", "Accept": "application/json",
        "Accept-Encoding": "gzip", "Content-Type": "application/json",
        "x-client-platform": "android", "x-client-version": "2.1.1",
        "x-client-locale": "ar", "x-client-bundle-id": "com.deepseek.chat",
        "x-rangers-id": _ds_rangers_id(), "x-client-timezone-offset": _ds_tz_offset(),
        "x-device-id": _ds_device_id(), "x-os-version": "30", "x-app-version": "2.1.1",
        "Authorization": f"Bearer {token}", "X-DS-PoW-Response": pow_response,
        "accept-charset": "UTF-8",
    }


def _ds_session_headers(token: str) -> Dict[str, str]:
    return {
        "x-client-bundle-id": "com.deepseek.chat", "x-client-platform": "web",
        "x-client-version": "2.0.0", "x-client-locale": "en_US",
        "x-client-timezone-offset": _ds_tz_offset(), "x-app-version": "2.0.0",
        "Authorization": f"Bearer {token}", "Content-Type": "application/json",
        "Accept": "*/*",
    }


async def _ds_get_pow(token: str, client: httpx.AsyncClient) -> Tuple[str, Any]:
    url = f"{RAILWAY_POW_URL}?authorization={token}"
    try:
        resp = await client.get(url, timeout=30)
        if resp.status_code != 200:
            resp = await client.get(RAILWAY_POW_URL, timeout=30)
        data         = resp.json()
        pow_response = data.get("x_ds_pow_response") or data.get("pow_response", "")
        pow_data     = data.get("solved_json", None)
        if not pow_response:
            raise ValueError(f"POW response empty: {data}")
        return pow_response, pow_data
    except Exception as e:
        log.error("DeepSeek: POW fetch failed: %s", e)
        raise HTTPException(status_code=503, detail=f"POW server error: {e}")


async def _ds_create_session(token: str, client: httpx.AsyncClient) -> str:
    resp = await client.post(DEEPSEEK_SESSION_URL, json={},
                              headers=_ds_session_headers(token), timeout=30)
    data = resp.json()
    sid  = (
        (data.get("data") or {}).get("biz_data", {}).get("chat_session", {}).get("id")
        or data.get("session_id")
    )
    if not sid:
        raise HTTPException(status_code=502, detail=f"DeepSeek: failed to create session: {data}")
    log.info("DeepSeek: created session_id=%s", sid)
    return sid


class DeepSeekBackend(BaseBackend):
    def __init__(self, proxy_id: str, model_type: str):
        self._proxy_id   = proxy_id
        self._model_type = model_type

    @property
    def model_id(self) -> str:
        return self._proxy_id

    async def complete(self, token, messages, tools, thinking, conv_id, extra) -> AsyncIterator[str]:
        model_type     = extra.get("model_type", self._model_type)
        search_enabled = extra.get("search_enabled", True)

        prompt = build_full_prompt(messages, tools)
        if not prompt.strip():
            return

        await _evict_old_sessions()

        sess = await _get_session(token, conv_id)
        async with httpx.AsyncClient() as client:
            if sess:
                session_id        = sess["ds_session_id"]
                parent_message_id = sess.get("ds_parent_msg_id")
                log.info("DeepSeek[%s]: reusing session=%s parent=%s",
                         model_type, session_id, parent_message_id)
            else:
                session_id        = await _ds_create_session(token, client)
                parent_message_id = None
                await _set_session(token, conv_id, {
                    "ds_session_id":    session_id,
                    "ds_parent_msg_id": parent_message_id,
                })
                log.info("DeepSeek[%s]: new session=%s for conv=%s",
                         model_type, session_id, conv_id)

            pow_response, pow_data = await _ds_get_pow(token, client)

            payload = {
                "chat_session_id":   session_id,
                "parent_message_id": parent_message_id,
                "prompt":            prompt,
                "ref_file_ids":      [],
                "thinking_enabled":  thinking,
                "search_enabled":    search_enabled,
                "model_type":        model_type,
                "action":            None,
                "preempt":           False,
                "pow":               pow_data,
                "stream":            True,
            }

            headers   = _ds_headers(token, pow_response)
            full_text = ""
            thinking_text     = ""
            new_parent_msg_id = None

            try:
                async with client.stream("POST", DEEPSEEK_CHAT_URL, json=payload,
                                          headers=headers, timeout=REQUEST_TIMEOUT) as resp:
                    first_done = False
                    async for raw_line in resp.aiter_lines():
                        if not raw_line or not raw_line.startswith("data: "):
                            continue
                        ds = raw_line[6:].strip()
                        if ds == "[DONE]":
                            break
                        try:
                            obj = json.loads(ds)
                        except json.JSONDecodeError:
                            continue

                        if not first_done:
                            req_id  = obj.get("request_message_id")
                            resp_id = obj.get("response_message_id")
                            if req_id and resp_id:
                                new_parent_msg_id = resp_id
                                first_done = True
                                continue

                        v = obj.get("v")
                        p = obj.get("p", "")
                        o = obj.get("o", "")

                        if isinstance(v, str) and o == "APPEND" and "content" in p:
                            full_text += v
                            continue

                        if isinstance(v, dict):
                            frags = (v.get("response") or {}).get("fragments", [])
                            for frag in frags:
                                ftype = frag.get("type", "")
                                fcont = frag.get("content", "") or ""
                                if ftype == "THINKING":
                                    thinking_text += fcont
                                elif ftype == "RESPONSE":
                                    full_text += fcont

                        if isinstance(v, str) and not p:
                            full_text += v

            except Exception as e:
                log.error("DeepSeek stream error: %s", e)
                yield sse_chunk(f"[DeepSeek Error: {e}]", model=self.model_id)
                yield sse_chunk(model=self.model_id, finish=True)
                yield "data: [DONE]\n\n"
                return

        if new_parent_msg_id:
            await _update_session(token, conv_id, ds_parent_msg_id=new_parent_msg_id)

        log.info("DeepSeek[%s]: text=%d thinking=%d parent=%s",
                 model_type, len(full_text), len(thinking_text), new_parent_msg_id)

        if thinking_text:
            thinking_payload = json.dumps({"type": "thinking", "content": thinking_text}, ensure_ascii=False)
            yield f"data: {thinking_payload}\n\n"

        tc = parse_tool_call(full_text)
        if tc:
            call_id = f"call_{uuid.uuid4().hex[:24]}"
            yield sse_chunk(tc=tc, model=self.model_id, call_id=call_id)
            yield sse_chunk(tc=tc, model=self.model_id, call_id=call_id, finish=True)
            yield "data: [DONE]\n\n"
        else:
            txt = clean_text(full_text) or "[DeepSeek: empty response]"
            for i in range(0, max(len(txt), 1), 40):
                yield sse_chunk(txt[i:i+40], model=self.model_id)
            yield sse_chunk(model=self.model_id, finish=True)
            yield "data: [DONE]\n\n"


register_backend(DeepSeekBackend(DEEPSEEK_PROXY_ID_EXPERT,  "expert"))
register_backend(DeepSeekBackend(DEEPSEEK_PROXY_ID_DEFAULT, "default"))


# ══════════════════════════════════════════════════════════
# BACKEND 4: Database (Supabase — Gemini / GPT / Claude)
#
# التوكن الثابت: "database-key"
# ══════════════════════════════════════════════════════════

DATABASE_FIXED_TOKEN = "database-key"
DATABASE_URL         = "https://qcpujeurnkbvwlvmylyx.supabase.co/functions/v1/chat"
DATABASE_TIMEOUT     = 180

DATABASE_MODELS: Dict[str, str] = {
    "db-gemini-flash":         "google/gemini-2.5-flash",
    "db-gemini-flash-lite":    "google/gemini-2.5-flash-lite",
    "db-gemini-pro":           "google/gemini-2.5-pro",
    "db-gpt-5-nano":           "openai/gpt-5-nano",
    "db-gpt-5-mini":           "openai/gpt-5-mini",
    "db-gpt-5":                "openai/gpt-5",
    "db-claude-haiku":         "anthropic/claude-3-5-haiku-20241022",
    "db-claude-sonnet":        "anthropic/claude-sonnet-4-5",
    "db-claude-opus":          "anthropic/claude-opus-4-1-20250805",
    "db-claude-sonnet-5":      "anthropic/claude-sonnet-5",
    "db-claude-fable":         "anthropic/claude-fable-5",
}

_DB_REAL_TO_PROXY: Dict[str, str] = {v: k for k, v in DATABASE_MODELS.items()}


def _db_resolve_real_model(model_id: str) -> str:
    if model_id in DATABASE_MODELS:
        return DATABASE_MODELS[model_id]
    if "/" in model_id:
        return model_id
    return "google/gemini-2.5-flash"


def _db_is_database_token(token: str) -> bool:
    return token.strip() == DATABASE_FIXED_TOKEN


def _content_to_str(content) -> str:
    """تحويل content من أي صيغة إلى نص"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict):
                if c.get("type") == "text":
                    parts.append(c.get("text", ""))
                elif c.get("type") == "image_url":
                    parts.append("[IMAGE]")
            elif isinstance(c, str):
                parts.append(c)
        return " ".join(parts).strip()
    return str(content) if content else ""


def _db_prepare(messages: List[Dict]) -> Tuple[Optional[str], List[Dict]]:
    """
    يُعيد (system_prompt, clean_messages) للـ Supabase.
    - system_prompt: نص كامل بدون قطع — يُرسل كـ field منفصل
    - clean_messages: قائمة {role, content} فقط، تناوب user/assistant، تبدأ بـ user
    """
    system_parts: List[str] = []
    turns: List[Tuple[str, str]] = []

    for m in messages:
        role    = m.get("role", "user")
        content = _content_to_str(m.get("content") or "")

        if role == "system":
            if content.strip():
                system_parts.append(content.strip())
        elif role == "user":
            turns.append(("user", content))
        elif role == "assistant":
            text = content.strip()
            turns.append(("assistant", text if text else "."))
        # تجاهل tool / function / tool_result تماماً

    system_prompt: Optional[str] = "\n\n".join(system_parts) if system_parts else None

    # ضمان التناوب — دمج الرسائل المتتالية بنفس الـ role
    merged: List[Dict] = []
    for role, content in turns:
        if merged and merged[-1]["role"] == role:
            merged[-1]["content"] += "\n\n" + content
        else:
            merged.append({"role": role, "content": content})

    # يجب أن تبدأ بـ user
    if merged and merged[0]["role"] != "user":
        merged.insert(0, {"role": "user", "content": " "})

    # تأكد أن كل content ليس فارغاً
    for m in merged:
        if not m["content"].strip():
            m["content"] = " "

    if not merged:
        merged = [{"role": "user", "content": " "}]

    return system_prompt, merged


class DatabaseBackend(BaseBackend):
    def __init__(self, proxy_id: str, real_model: str):
        self._proxy_id   = proxy_id
        self._real_model = real_model

    @property
    def model_id(self) -> str:
        return self._proxy_id

    async def complete(self, token, messages, tools, thinking, conv_id, extra) -> AsyncIterator[str]:
        real_model = _db_resolve_real_model(self._proxy_id)

        # استخراج system prompt والرسائل المنظفة
        system_prompt, clean_messages = _db_prepare(messages)

        log.info("Database: proxy_id=%s real_model=%s msgs=%d→%d system_len=%d",
                 self._proxy_id, real_model, len(messages), len(clean_messages),
                 len(system_prompt) if system_prompt else 0)

        # بناء الـ payload — system كـ field منفصل إذا كان موجوداً
        payload: Dict[str, Any] = {
            "messages": clean_messages,
            "model":    real_model,
        }
        if system_prompt:
            payload["system"] = system_prompt

        headers = {
            "Content-Type": "application/json",
            "Accept":       "text/event-stream",
        }

        full_text = ""

        try:
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST", DATABASE_URL,
                    json=payload, headers=headers,
                    timeout=DATABASE_TIMEOUT,
                ) as resp:
                    if resp.status_code >= 400:
                        err_body = await resp.aread()
                        err_msg  = err_body.decode("utf-8", errors="replace")
                        log.error("Database HTTP %d: %s", resp.status_code, err_msg[:200])
                        yield sse_chunk(
                            f"[Database Error {resp.status_code}: {err_msg[:100]}]",
                            model=self._proxy_id,
                        )
                        yield sse_chunk(model=self._proxy_id, finish=True)
                        yield "data: [DONE]\n\n"
                        return

                    async for raw_line in resp.aiter_lines():
                        if not raw_line:
                            continue
                        if not raw_line.startswith("data: "):
                            continue
                        ds = raw_line[6:].strip()
                        if ds == "[DONE]":
                            break
                        try:
                            obj = json.loads(ds)
                            choices = obj.get("choices", [])
                            if choices:
                                delta   = choices[0].get("delta", {})
                                content = delta.get("content")
                                if content:
                                    content = content.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
                                    full_text += content
                                    yield sse_chunk(content, model=self._proxy_id)
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue

        except Exception as e:
            log.error("Database stream error: %s", e)
            yield sse_chunk(f"[Database Error: {e}]", model=self._proxy_id)
            yield sse_chunk(model=self._proxy_id, finish=True)
            yield "data: [DONE]\n\n"
            return

        log.info("Database: done real_model=%s text_len=%d", real_model, len(full_text))

        if not full_text:
            yield sse_chunk("[Database: empty response]", model=self._proxy_id)

        yield sse_chunk(model=self._proxy_id, finish=True)
        yield "data: [DONE]\n\n"


for _proxy_id, _real_model in DATABASE_MODELS.items():
    register_backend(DatabaseBackend(_proxy_id, _real_model))


# ══════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════

def resolve_thinking(body: Dict) -> bool:
    effort = body.get("reasoning_effort")
    if effort is not None:
        return str(effort).lower() not in ("none", "off", "false", "0", "")
    think_obj = body.get("thinking")
    if isinstance(think_obj, dict):
        return think_obj.get("type", "") not in ("disabled", "none", "")
    if isinstance(think_obj, bool):
        return think_obj
    return False


def resolve_extra(body: Dict, model: str) -> Dict:
    extra_body = body.get("extra_body") or {}
    extra: Dict = {}
    if model in (DEEPSEEK_PROXY_ID_EXPERT, DEEPSEEK_PROXY_ID_DEFAULT):
        mt = extra_body.get("model_type") or body.get("model_type")
        if not mt:
            mt = "expert" if model == DEEPSEEK_PROXY_ID_EXPERT else "default"
        extra["model_type"] = mt if mt in ("default", "expert") else "expert"
        se = extra_body.get("search_enabled")
        if se is None:
            se = body.get("search_enabled")
        extra["search_enabled"] = bool(se) if se is not None else True
    return extra


# ══════════════════════════════════════════════════════════
# Deduplication
# ══════════════════════════════════════════════════════════

_recent_requests: Dict[str, float] = {}
_req_lock = asyncio.Lock()
DEDUP_WINDOW = 3.0


async def _is_duplicate(req_hash: str) -> bool:
    async with _req_lock:
        now   = time.time()
        stale = [k for k, t in _recent_requests.items() if now - t > DEDUP_WINDOW * 10]
        for k in stale:
            del _recent_requests[k]
        if req_hash in _recent_requests:
            if now - _recent_requests[req_hash] < DEDUP_WINDOW:
                return True
        _recent_requests[req_hash] = now
        return False


def _request_hash(messages: List[Dict], tools: List[Dict]) -> str:
    try:
        raw = json.dumps({"m": messages[-2:], "t": tools}, ensure_ascii=False, sort_keys=True)
        return hashlib.md5(raw.encode()).hexdigest()
    except Exception:
        return str(uuid.uuid4())


# ══════════════════════════════════════════════════════════
# FastAPI App
# ══════════════════════════════════════════════════════════

app = FastAPI(title="Universal AI Proxy", version="8.1.0", docs_url="/docs")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _extract_token(authorization: Optional[str]) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header.")
    parts = authorization.split(" ", 1)
    return parts[1].strip() if len(parts) == 2 else parts[0].strip()


def _is_database_model(model: str) -> bool:
    return model in DATABASE_MODELS or model in _DB_REAL_TO_PROXY


@app.get("/", tags=["health"])
async def health():
    return {
        "status": "ok", "proxy": "Universal AI Proxy", "version": "8.1.0",
        "active_sessions": len(_sessions), "backends": list(_BACKENDS.keys()),
        "models": {
            "qwen":             "Qwen3.8-max via chat.qwen.ai",
            "deepseek":         "DeepSeek Expert via chat.deepseek.com",
            "deepseek-default": "DeepSeek Default (fast) via chat.deepseek.com",
            **{pid: f"{rm} via Supabase Database" for pid, rm in DATABASE_MODELS.items()},
        },
        "database_token": DATABASE_FIXED_TOKEN,
        "notes": [
            "v8.1: GPT backend removed",
            "v8.1: Database messages sanitized before sending (fixes 400 errors)",
            "v8: Database backend added — Gemini, GPT-5, Claude via Supabase",
            "v7: conv_id is stable (anchored to first message)",
        ],
    }


@app.get("/v1/models", tags=["models"])
async def list_models():
    models = []

    for mid in ["qwen", "deepseek", "deepseek-default"]:
        if mid in _BACKENDS:
            models.append({
                "id": mid, "object": "model",
                "created": 1700000000, "owned_by": "proxy",
            })

    models.append({
        "id": "qwen-vision", "object": "model",
        "created": 1700000000, "owned_by": "qwen",
    })

    for proxy_id, real_model in DATABASE_MODELS.items():
        provider = real_model.split("/")[0]
        models.append({
            "id":         proxy_id,
            "object":     "model",
            "created":    1700000000,
            "owned_by":   f"database-{provider}",
            "real_model": real_model,
        })

    return {"object": "list", "data": models}


@app.post("/v1/chat/completions", tags=["chat"])
async def chat_completions(
    request:       Request,
    authorization: Optional[str] = Header(None),
):
    token     = _extract_token(authorization)
    body      = await request.json()
    messages  = body.get("messages", [])
    tools     = body.get("tools", [])
    do_stream = body.get("stream", False)
    model     = body.get("model", QWEN_PROXY_ID)

    effective_token = token

    if _is_database_model(model) and not _db_is_database_token(token):
        raise HTTPException(
            status_code=401,
            detail=(
                f"Database models require token '{DATABASE_FIXED_TOKEN}'. "
                f"Received: '{token[:20]}...'"
            ),
        )

    if _db_is_database_token(token) and not _is_database_model(model):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Token '{DATABASE_FIXED_TOKEN}' is only for Database models (db-*). "
                f"Model '{model}' is not a Database model."
            ),
        )

    thinking = resolve_thinking(body)
    extra    = resolve_extra(body, model)

    explicit_conv_id = (
        body.get("conversation_id")
        or body.get("session_id")
        or request.headers.get("x-conversation-id")
        or request.headers.get("x-session-id")
    )
    conv_id = _compute_conv_id(messages, explicit_conv_id)

    log.info("conv=%s model=%s msgs=%d thinking=%s extra=%s",
             conv_id, model, len(messages), thinking, extra)

    req_hash = _request_hash(messages, tools)
    if await _is_duplicate(req_hash):
        log.warning("Duplicate request (conv=%s) — skipping", conv_id)
        raise HTTPException(status_code=429, detail="Duplicate request — please retry in a moment.")

    backend = get_backend(model)
    if backend is None:
        proxy_id = _DB_REAL_TO_PROXY.get(model)
        if proxy_id:
            backend = get_backend(proxy_id)
    if backend is None:
        backend = get_backend(QWEN_PROXY_ID)
        if backend is None:
            raise HTTPException(status_code=400, detail=f"No backend for model '{model}'.")

    if do_stream:
        async def event_stream():
            async for chunk in backend.complete(effective_token, messages, tools, thinking, conv_id, extra):
                yield chunk
        return StreamingResponse(event_stream(), media_type="text/event-stream")

    full_content   = ""
    finish_reason  = "stop"
    tool_call_data = None

    async for chunk in backend.complete(effective_token, messages, tools, thinking, conv_id, extra):
        if chunk.startswith("data: [DONE]"):
            break
        if not chunk.startswith("data: "):
            continue
        try:
            obj = json.loads(chunk[6:])
            if obj.get("type") == "thinking":
                continue
            choice = obj["choices"][0]
            delta  = choice.get("delta", {})
            fr     = choice.get("finish_reason")
            if fr:
                finish_reason = fr
            if delta.get("tool_calls"):
                tool_call_data = delta["tool_calls"][0]
            elif delta.get("content"):
                full_content += delta["content"]
        except Exception:
            continue

    if tool_call_data:
        return JSONResponse(make_tc_response({
            "name": tool_call_data["function"]["name"],
            "arguments": tool_call_data["function"]["arguments"],
        }, model))
    return JSONResponse(make_text_response(full_content, model))


# ══════════════════════════════════════════════════════════
# Image Generation (Qwen)
# ══════════════════════════════════════════════════════════

@app.post("/v1/images/generations", tags=["images"])
async def image_generations(request: Request, authorization: Optional[str] = Header(None)):
    token  = _extract_token(authorization)
    body   = await request.json()
    prompt = body.get("prompt", "")
    size   = body.get("size", "1:1").replace("x", ":")
    if not prompt:
        raise HTTPException(status_code=400, detail="'prompt' is required.")
    async with httpx.AsyncClient() as client:
        cid     = await _qwen_create_chat(token, client)
        payload = _qwen_build_payload(cid, prompt, None, chat_type="t2i", size=size)
        image_url: Optional[str] = None
        async with client.stream("POST", f"{QWEN_BASE}/chat/completions", json=payload,
                                  headers=_qwen_headers_chat(token, stream=True),
                                  params={"chat_id": cid}, timeout=300) as resp:
            async for raw_line in resp.aiter_lines():
                if not raw_line or not raw_line.startswith("data: "):
                    continue
                ds = raw_line[6:].strip()
                if ds == "[DONE]":
                    break
                if _qwen_is_antibot(raw_line) or _qwen_is_rate_limited(raw_line):
                    raise HTTPException(status_code=429, detail="Qwen blocked or rate-limited.")
                try:
                    obj     = json.loads(ds)
                    content = obj["choices"][0].get("delta", {}).get("content", "")
                    if content.startswith("http"):
                        image_url = content
                except Exception:
                    continue
    if not image_url:
        raise HTTPException(status_code=500, detail="No image URL returned.")
    return JSONResponse({"created": int(time.time()), "data": [{"url": image_url}]})


# ══════════════════════════════════════════════════════════
# Image Edits (Qwen)
# ══════════════════════════════════════════════════════════

OSS_UPLOAD_TIMEOUT = 120


def _oss_sig(secret, method, md5, ct, date, canon_hdr, canon_res):
    s2s    = f"{method}\n{md5}\n{ct}\n{date}\n{canon_hdr}{canon_res}"
    digest = hmac.new(secret.encode(), s2s.encode(), hashlib.sha1).digest()
    return base64.b64encode(digest).decode()


async def _upload_image(token: str, image_bytes: bytes, client: httpx.AsyncClient) -> Dict:
    filename  = f"{uuid.uuid4()}_IMG.jpg"
    file_size = str(len(image_bytes))
    sts = await client.post(
        "https://chat.qwen.ai/api/v2/files/getstsToken",
        json={"filename": filename, "filetype": "image", "filesize": file_size},
        headers=_qwen_headers_chat(token), timeout=60,
    )
    res = sts.json()
    if _qwen_is_rate_limited(res) or "data" not in res:
        raise HTTPException(status_code=429, detail="Rate-limited during OSS STS.")
    d      = res["data"]
    aki    = d["access_key_id"]; aks = d["access_key_secret"]; stkn = d["security_token"]
    fpath  = d["file_path"]; fid = d["file_id"]; bucket = d["bucketname"]
    host   = f"{bucket}.{d['endpoint']}"
    furl   = d.get("file_url", f"https://{host}/{fpath}")
    oss_ua = "aliyun-sdk-android/2.9.21"
    c_hdr  = f"x-oss-security-token:{stkn}\n"

    def _oh(method, md5, ct, res_path, extra=None):
        gmt = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")
        sig = _oss_sig(aks, method, md5, ct, gmt, c_hdr, res_path)
        h   = {"Authorization": f"OSS {aki}:{sig}", "User-Agent": oss_ua, "Host": host,
                "x-oss-security-token": stkn, "Date": gmt, "Content-Type": ct}
        if extra:
            h.update(extra)
        return h

    init_r    = await client.post(f"https://{host}/{fpath}?uploads",
                                   headers=_oh("POST", "", "image/jpeg",
                                               f"/{bucket}/{fpath}?uploads",
                                               {"Content-Length": "0"}), timeout=60)
    upload_id = ET.fromstring(init_r.text).find("{*}UploadId").text
    cmd5      = base64.b64encode(hashlib.md5(image_bytes).digest()).decode()
    part_r    = await client.put(
        f"https://{host}/{fpath}?uploadId={upload_id}&partNumber=1", content=image_bytes,
        headers=_oh("PUT", cmd5, "image/jpeg",
                    f"/{bucket}/{fpath}?partNumber=1&uploadId={upload_id}",
                    {"Content-MD5": cmd5, "Content-Length": file_size}),
        timeout=OSS_UPLOAD_TIMEOUT,
    )
    etag = part_r.headers.get("ETag", "").replace('"', "")
    body = (f"<CompleteMultipartUpload><Part><PartNumber>1</PartNumber>"
            f"<ETag>{etag}</ETag></Part></CompleteMultipartUpload>").encode()
    await client.post(f"https://{host}/{fpath}?uploadId={upload_id}", content=body,
                       headers=_oh("POST", "", "image/jpeg",
                                   f"/{bucket}/{fpath}?uploadId={upload_id}",
                                   {"Content-Length": str(len(body))}), timeout=60)
    return {"type": "image", "file": {"data": {}, "filename": filename, "id": fid,
                                       "meta": {"name": filename}},
            "id": fid, "filename": filename, "name": filename, "url": furl,
            "image_width": 1024, "image_height": 1024}


@app.post("/v1/images/edits", tags=["images"])
async def image_edits(request: Request, authorization: Optional[str] = Header(None)):
    token       = _extract_token(authorization)
    image_bytes = None
    prompt      = ""
    if "multipart/form-data" in request.headers.get("content-type", ""):
        form      = await request.form()
        prompt    = str(form.get("prompt", ""))
        img_field = form.get("image")
        if img_field and hasattr(img_field, "read"):
            image_bytes = await img_field.read()
    else:
        body      = await request.json()
        prompt    = body.get("prompt", "")
        image_url = body.get("image_url", "")
        if image_url:
            async with httpx.AsyncClient() as dl:
                r = await dl.get(image_url, timeout=60)
                image_bytes = r.content
    if not image_bytes:
        raise HTTPException(status_code=400, detail="No image provided.")
    if not prompt:
        raise HTTPException(status_code=400, detail="'prompt' is required.")
    async with httpx.AsyncClient() as client:
        uploaded   = await _upload_image(token, image_bytes, client)
        file_entry = {"type": "image", "file": uploaded["file"], "id": uploaded["id"],
                       "url": uploaded["url"], "name": uploaded["filename"],
                       "image_width": 1024, "image_height": 1024}
        cid        = await _qwen_create_chat(token, client)
        payload    = _qwen_build_payload(cid, prompt, None, chat_type="t2i", files=[file_entry])
        result_url: Optional[str] = None
        async with client.stream("POST", f"{QWEN_BASE}/chat/completions", json=payload,
                                  headers=_qwen_headers_chat(token, stream=True),
                                  params={"chat_id": cid}, timeout=300) as resp:
            async for raw_line in resp.aiter_lines():
                if not raw_line or not raw_line.startswith("data: "):
                    continue
                ds = raw_line[6:].strip()
                if ds == "[DONE]":
                    break
                if _qwen_is_antibot(raw_line) or _qwen_is_rate_limited(raw_line):
                    raise HTTPException(status_code=429, detail="Qwen blocked.")
                try:
                    obj     = json.loads(ds)
                    content = obj["choices"][0].get("delta", {}).get("content", "")
                    if content.startswith("http"):
                        result_url = content
                except Exception:
                    continue
    if not result_url:
        raise HTTPException(status_code=500, detail="No edited image URL returned.")
    return JSONResponse({"created": int(time.time()), "data": [{"url": result_url}]})


# ══════════════════════════════════════════════════════════
# Error Handlers
# ══════════════════════════════════════════════════════════

@app.exception_handler(HTTPException)
async def _http_err(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code,
                         content={"error": {"message": exc.detail, "type": "proxy_error",
                                            "code": exc.status_code}})


@app.exception_handler(Exception)
async def _generic_err(request: Request, exc: Exception):
    log.error("Unhandled: %s", exc, exc_info=True)
    return JSONResponse(status_code=500,
                         content={"error": {"message": str(exc), "type": "internal_error",
                                            "code": 500}})


# ══════════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    log.info("Starting Universal AI Proxy v8.1 on port %d", port)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

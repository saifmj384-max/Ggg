"""
Universal AI Proxy  v6.0
=========================
بروكسي موحد يدعم ثلاثة نماذج:
  ① Qwen        — عبر chat.qwen.ai       (token: Qwen Bearer)
  ② DeepSeek    — عبر chat.deepseek.com  (token: DeepSeek Bearer)
  ③ ChatGPT     — عبر android.chat.openai.com (token: GPT Authorization header)

القواعد الأساسية:
  • كل نموذج يعمل بتوكن مستقل — لا تداخل أبداً
  • Qwen   → model = "qwen"
  • DeepSeek → model = "deepseek"  (يقبل extra_body.model_type = "default"|"expert")
  • GPT    → model = "gpt"         (بدون thinking)
  • DeepSeek يعتمد على Railway POW server فقط
  • Thinking مدعوم في Qwen و DeepSeek، غير مدعوم في GPT
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
# Session Store  (in-memory، مع eviction تلقائي)
# ══════════════════════════════════════════════════════════
SESSION_TTL = 60 * 60 * 6   # 6 ساعات

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
# ★ معمارية Backend المرنة
# ══════════════════════════════════════════════════════════

class BaseBackend(ABC):
    """الواجهة التي يجب أن ينفّذها كل backend."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """المعرّف الذي يُرسله العميل في حقل model."""

    @abstractmethod
    async def complete(
        self,
        token: str,
        messages: List[Dict],
        tools: List[Dict],
        thinking: bool,
        conv_id: str,
        extra: Dict,
    ) -> AsyncIterator[str]:
        """
        يُنتج SSE chunks بصيغة OpenAI.
        extra: بيانات إضافية خاصة بالنموذج (مثل model_type لـ DeepSeek)
        """


_BACKENDS: Dict[str, "BaseBackend"] = {}


def register_backend(backend: "BaseBackend") -> None:
    _BACKENDS[backend.model_id] = backend
    log.info("Registered backend: %s", backend.model_id)


def get_backend(model_id: str) -> Optional["BaseBackend"]:
    return _BACKENDS.get(model_id)


# ══════════════════════════════════════════════════════════
# أدوات مشتركة: تحويل الرسائل + تحليل Tool Calls
# ══════════════════════════════════════════════════════════

TOOL_SYSTEM_SUFFIX = """
══════════════════════════════════════════════
TOOL USE — STRICT FORMAT
══════════════════════════════════════════════
When you need to call a tool, your ENTIRE response MUST be ONLY this exact line:

ACTION: tool_name|{"param1": "value1"}

Examples:
  ACTION: shell_execute|{"command": "ls -la"}
  ACTION: browser_open|{"url": "https://example.com"}
  ACTION: read_file|{"path": "/tmp/notes.txt"}

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
                lines.append(
                    f'      <param name="{pname}" type="{ptype}"{req}>{pdesc}</param>'
                )
            lines.append("    </parameters>")
        lines.append("  </tool>")
    lines.append("</available_tools>")
    return "\n".join(lines)


def messages_to_prompt(messages: List[Dict], tools: List[Dict]) -> Tuple[str, str]:
    """→ (system_text, conversation_text)"""
    system_parts = []
    conv_parts   = []

    for m in messages:
        if m.get("role") == "system":
            content = m.get("content") or ""
            if isinstance(content, list):
                content = " ".join(
                    c.get("text", "") for c in content if isinstance(c, dict)
                )
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
                    c.get("text", "")
                    for c in content
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
            conv_parts.append(
                f"[TOOL RESULT: {tool_name}]\n{content}\n[/TOOL RESULT]"
            )

    system_text = "\n\n".join(system_parts)
    conv_text   = "\n\n".join(conv_parts)
    return system_text, conv_text


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
    m_params = re.findall(r"<parameter[=:](\w+)>\s*(.*?)\s*</parameter>", text, re.DOTALL | re.IGNORECASE)
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
        "id":      f"chatcmpl-{uuid.uuid4().hex}",
        "object":  "chat.completion",
        "created": int(time.time()),
        "model":   model,
        "choices": [{
            "index":   0,
            "message": {
                "role":    "assistant",
                "content": None,
                "tool_calls": [{
                    "id":       call_id,
                    "type":     "function",
                    "function": {
                        "name":      tc["name"],
                        "arguments": tc["arguments"],
                    },
                }],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def make_text_response(content: str, model: str) -> Dict:
    return {
        "id":      f"chatcmpl-{uuid.uuid4().hex}",
        "object":  "chat.completion",
        "created": int(time.time()),
        "model":   model,
        "choices": [{
            "index":   0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def sse_chunk(
    content: str = "",
    model:   str = "qwen",
    *,
    finish:    bool = False,
    tc:        Optional[Dict] = None,
    call_id:   Optional[str]  = None,
) -> str:
    cid = call_id or f"call_{uuid.uuid4().hex[:24]}"

    if tc and not finish:
        delta = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "index": 0,
                "id":    cid,
                "type":  "function",
                "function": {
                    "name":      tc["name"],
                    "arguments": tc["arguments"],
                },
            }],
        }
        choice = {"index": 0, "delta": delta, "finish_reason": None}
    elif finish and tc:
        choice = {"index": 0, "delta": {}, "finish_reason": "tool_calls"}
    elif finish:
        choice = {"index": 0, "delta": {}, "finish_reason": "stop"}
    else:
        choice = {"index": 0, "delta": {"content": content}, "finish_reason": None}

    obj = {
        "id":      f"chatcmpl-{uuid.uuid4().hex}",
        "object":  "chat.completion.chunk",
        "created": int(time.time()),
        "model":   model,
        "choices": [choice],
    }
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


# ══════════════════════════════════════════════════════════
# ★ BACKEND 1: Qwen  (لا يُمس — كما كان في v5)
# ══════════════════════════════════════════════════════════

QWEN_BASE          = "https://chat.qwen.ai/api/v2"
QWEN_MODEL_ID_REAL = "qwen3.8-max"
QWEN_PROXY_ID      = "qwen"

_UA_CHAT = (
    "Dalvik/2.1.0 (Linux; U; Android 15; RMX3834 Build/AP3A.240905.015.A2) "
    "AliApp(QWENCHAT/2.7.2) AppType/Release AplusBridgeLite"
)
_UA_NEW = (
    "Dalvik/2.1.0 (Linux; U; Android 15; RMX3834 Build/AP3A.240905.015.A2),"
    "Dalvik/2.1.0 (Linux; U; Android 15; RMX3834 Build/AP3A.240905.015.A2) "
    "AliApp(QWENCHAT/2.7.2) AppType/Release AplusBridgeLite"
)

REQUEST_TIMEOUT = 180


def _qwen_headers_chat(token: str, *, stream: bool = False) -> Dict[str, str]:
    return {
        "User-Agent":      _UA_CHAT,
        "Content-Type":    "application/json; charset=UTF-8",
        "Accept":          "*/*,text/event-stream" if stream else "application/json",
        "Accept-Language": "en-US",
        "Accept-Encoding": "gzip, deflate",
        "Cache-Control":   "no-store",
        "Connection":      "Keep-Alive",
        "Host":            "chat.qwen.ai",
        "X-Platform":      "android",
        "x-device-id":     "0",
        "source":          "app",
        "Authorization":   f"Bearer {token}",
        "x-request-id":    str(uuid.uuid4()),
        "Cookie":          f"x-ap=eu-central-1; token={token}",
    }


def _qwen_headers_new(token: str) -> Dict[str, str]:
    return {
        "User-Agent":      _UA_NEW,
        "Content-Type":    "application/json",
        "Accept":          "application/json",
        "Accept-Language": "en-US",
        "Accept-Encoding": "gzip",
        "Connection":      "Keep-Alive",
        "Host":            "chat.qwen.ai",
        "X-Platform":      "android",
        "x-device-id":     "0",
        "source":          "app",
        "Authorization":   f"Bearer {token}",
        "x-request-id":    str(uuid.uuid4()),
        "Cookie":          f"x-ap=eu-central-1; token={token}",
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
    resp    = await client.post(
        url, json=payload,
        headers=_qwen_headers_new(token),
        timeout=60,
    )
    data = resp.json()
    cid  = (
        data.get("chat_id")
        or data.get("id")
        or (data.get("data") or {}).get("chat_id")
        or (data.get("data") or {}).get("id")
    )
    if not cid:
        raise HTTPException(status_code=502, detail=f"Failed to create Qwen chat: {data}")
    log.info("Qwen: created chat_id=%s", cid)
    return cid


def _qwen_build_payload(
    chat_id:   str,
    prompt:    str,
    parent_id: Optional[str],
    *,
    chat_type:  str  = "t2t",
    thinking:   bool = False,
    auto_search: bool = False,
    files:      Optional[List] = None,
    size:       str  = "1:1",
) -> Dict:
    ts  = int(time.time())
    fid = str(uuid.uuid4())
    return {
        "stream":             True,
        "incremental_output": True,
        "chatId":             chat_id,
        "chat_id":            chat_id,
        "chat_mode":          "normal",
        "model":              QWEN_MODEL_ID_REAL,
        "messages": [{
            "id":         None,
            "fid":        fid,
            "chat_type":  chat_type,
            "content":    prompt,
            "role":       "user",
            "feature_config": {
                "output_schema":    "phase",
                "thinking_enabled": thinking,
                "thinking_format":  "summary",
                "auto_thinking":    thinking,
                "auto_search":      auto_search,
            },
            "timestamp":     ts,
            "sub_chat_type": chat_type,
            "models":        [QWEN_MODEL_ID_REAL],
            "model":         "",
            "files":         files or [],
            "user_action":   "chat",
            "extra":         {"meta": {"subChatType": chat_type}},
            "parentId":      parent_id,
            "parent_id":     parent_id,
        }],
        "timestamp":                ts,
        "size":                     size,
        "share_id":                 "",
        "version":                  "2.1",
        "origin_branch_message_id": "",
        "parentId":                 parent_id or "",
        "parent_id":                parent_id,
    }


async def _qwen_stream_collect(
    token:    str,
    chat_id:  str,
    payload:  Dict,
    client:   httpx.AsyncClient,
) -> Tuple[str, Optional[str]]:
    url      = f"{QWEN_BASE}/chat/completions"
    full_txt = ""
    resp_id: Optional[str] = None

    async with client.stream(
        "POST", url,
        json=payload,
        headers=_qwen_headers_chat(token, stream=True),
        params={"chat_id": chat_id},
        timeout=REQUEST_TIMEOUT,
    ) as resp:
        async for raw_line in resp.aiter_lines():
            if not raw_line:
                continue
            if _qwen_is_antibot(raw_line):
                log.warning("Qwen anti-bot triggered")
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

                rid = (
                    obj.get("response_id")
                    or (
                        obj.get("choices", [{}])[0]
                        .get("delta", {})
                        .get("response_id")
                    )
                )
                if rid:
                    resp_id = rid

                choices = obj.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                phase = delta.get("phase", "")
                if phase and phase not in ("answer", ""):
                    continue
                content = delta.get("content", "")
                if content:
                    full_txt += content
            except (json.JSONDecodeError, KeyError, IndexError):
                continue

    return full_txt, resp_id


class QwenBackend(BaseBackend):
    """Backend يتصل بـ Qwen — لم يتغير من v5."""

    @property
    def model_id(self) -> str:
        return QWEN_PROXY_ID

    async def complete(
        self,
        token:    str,
        messages: List[Dict],
        tools:    List[Dict],
        thinking: bool,
        conv_id:  str,
        extra:    Dict,
    ) -> AsyncIterator[str]:
        prompt = build_full_prompt(messages, tools)
        if not prompt.strip():
            return

        await _evict_old_sessions()

        sess = await _get_session(token, conv_id)
        if sess:
            qwen_chat_id = sess["qwen_chat_id"]
            parent_id    = sess.get("parent_id")
            log.info("Qwen: reusing chat_id=%s", qwen_chat_id)
        else:
            async with httpx.AsyncClient() as tmp:
                qwen_chat_id = await _qwen_create_chat(token, tmp)
            parent_id = None
            await _set_session(token, conv_id, {
                "qwen_chat_id": qwen_chat_id,
                "parent_id":    parent_id,
            })

        payload = _qwen_build_payload(
            qwen_chat_id, prompt, parent_id,
            thinking=thinking,
        )

        async with httpx.AsyncClient() as client:
            qwen_text, last_rid = await _qwen_stream_collect(
                token, qwen_chat_id, payload, client
            )

        await _update_session(token, conv_id, parent_id=last_rid)
        log.info("Qwen: %d chars | parent=%s", len(qwen_text), last_rid)

        tc = parse_tool_call(qwen_text)
        if tc:
            log.info("Qwen tool call: %s", tc["name"])
            call_id = f"call_{uuid.uuid4().hex[:24]}"
            yield sse_chunk(tc=tc, model=self.model_id, call_id=call_id)
            yield sse_chunk(tc=tc, model=self.model_id, call_id=call_id, finish=True)
            yield "data: [DONE]\n\n"
        else:
            txt = clean_text(qwen_text)
            chunk_size = 40
            for i in range(0, max(len(txt), 1), chunk_size):
                yield sse_chunk(txt[i:i+chunk_size], model=self.model_id)
            yield sse_chunk(model=self.model_id, finish=True)
            yield "data: [DONE]\n\n"


register_backend(QwenBackend())


# ══════════════════════════════════════════════════════════
# ★ BACKEND 2: DeepSeek
#
# يدعم:
#   • وضعين:  model_type = "default" | "expert"
#             يُمرَّر عبر extra_body.model_type أو body.model_type
#             الافتراضي: "expert"
#   • التفكير: thinking_enabled  (نفس آلية Qwen)
#   • البحث:  search_enabled
#   • POW:    من Railway فقط
#   • Session: session_id + parent_message_id
# ══════════════════════════════════════════════════════════

DEEPSEEK_PROXY_ID   = "deepseek"
DEEPSEEK_CHAT_URL   = "https://chat.deepseek.com/api/v0/chat/completion"
DEEPSEEK_SESSION_URL = "https://chat.deepseek.com/api/v0/chat_session/create"
RAILWAY_POW_URL     = "https://pow.up.railway.app/pow"


def _ds_rangers_id() -> str:
    ts = int(time.time() * 1000)
    rv = int(1_000_000_000 + (uuid.uuid4().int % 8_999_999_999))
    return str((ts << 32) | rv)


def _ds_device_id() -> str:
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    import random
    return "".join(random.choice(chars) for _ in range(88))


def _ds_tz_offset() -> str:
    return str(int(datetime.now(timezone.utc).utcoffset().total_seconds()
                   if datetime.now(timezone.utc).utcoffset() else 0))


def _ds_headers(token: str, pow_response: str) -> Dict[str, str]:
    return {
        "User-Agent":              "DeepSeek/2.1.1 Android/36",
        "Accept":                  "application/json",
        "Accept-Encoding":         "gzip",
        "Content-Type":            "application/json",
        "x-client-platform":       "android",
        "x-client-version":        "2.1.1",
        "x-client-locale":         "ar",
        "x-client-bundle-id":      "com.deepseek.chat",
        "x-rangers-id":            _ds_rangers_id(),
        "x-client-timezone-offset": _ds_tz_offset(),
        "x-device-id":             _ds_device_id(),
        "x-os-version":            "30",
        "x-app-version":           "2.1.1",
        "Authorization":           f"Bearer {token}",
        "X-DS-PoW-Response":       pow_response,
        "accept-charset":          "UTF-8",
    }


def _ds_session_headers(token: str) -> Dict[str, str]:
    return {
        "x-client-bundle-id":       "com.deepseek.chat",
        "x-client-platform":        "web",
        "x-client-version":         "2.0.0",
        "x-client-locale":          "en_US",
        "x-client-timezone-offset": _ds_tz_offset(),
        "x-app-version":            "2.0.0",
        "Authorization":            f"Bearer {token}",
        "Content-Type":             "application/json",
        "Accept":                   "*/*",
    }


async def _ds_get_pow(token: str, client: httpx.AsyncClient) -> Tuple[str, Any]:
    """يجلب POW من Railway ويُعيد (pow_response_header, pow_data_for_body)."""
    url = f"{RAILWAY_POW_URL}?authorization={token}"
    try:
        resp = await client.get(url, timeout=30)
        if resp.status_code != 200:
            # fallback بدون توكن
            resp = await client.get(RAILWAY_POW_URL, timeout=30)
        data = resp.json()
        pow_response = data.get("x_ds_pow_response") or data.get("pow_response", "")
        pow_data     = data.get("solved_json", None)
        if not pow_response:
            raise ValueError(f"POW response empty: {data}")
        log.info("DeepSeek: POW fetched OK")
        return pow_response, pow_data
    except Exception as e:
        log.error("DeepSeek: POW fetch failed: %s", e)
        raise HTTPException(status_code=503, detail=f"POW server error: {e}")


async def _ds_create_session(token: str, client: httpx.AsyncClient) -> str:
    """ينشئ جلسة DeepSeek ويُعيد session_id."""
    resp = await client.post(
        DEEPSEEK_SESSION_URL,
        json={},
        headers=_ds_session_headers(token),
        timeout=30,
    )
    data = resp.json()
    sid = (
        (data.get("data") or {}).get("biz_data", {}).get("chat_session", {}).get("id")
        or data.get("session_id")
    )
    if not sid:
        raise HTTPException(status_code=502, detail=f"DeepSeek: failed to create session: {data}")
    log.info("DeepSeek: created session_id=%s", sid)
    return sid


class DeepSeekBackend(BaseBackend):
    """
    Backend يتصل بـ DeepSeek عبر محاكاة التطبيق الأندرويد.

    extra keys مدعومة:
      model_type   : "default" | "expert"  (default: "expert")
      search_enabled: bool                  (default: True)
    """

    @property
    def model_id(self) -> str:
        return DEEPSEEK_PROXY_ID

    async def complete(
        self,
        token:    str,
        messages: List[Dict],
        tools:    List[Dict],
        thinking: bool,
        conv_id:  str,
        extra:    Dict,
    ) -> AsyncIterator[str]:

        # ── استخراج الخيارات ──────────────────────────────
        model_type     = extra.get("model_type", "expert")
        if model_type not in ("default", "expert"):
            model_type = "expert"
        search_enabled = extra.get("search_enabled", True)

        # ── بناء البرومبت ─────────────────────────────────
        prompt = build_full_prompt(messages, tools)
        if not prompt.strip():
            return

        await _evict_old_sessions()

        # ── استرجاع أو إنشاء جلسة DeepSeek ──────────────
        sess = await _get_session(token, conv_id)
        async with httpx.AsyncClient() as client:
            if sess:
                session_id        = sess["ds_session_id"]
                parent_message_id = sess.get("ds_parent_msg_id")
                log.info("DeepSeek: reusing session=%s", session_id)
            else:
                session_id        = await _ds_create_session(token, client)
                parent_message_id = None
                await _set_session(token, conv_id, {
                    "ds_session_id":    session_id,
                    "ds_parent_msg_id": parent_message_id,
                })

            # ── جلب POW ───────────────────────────────────
            pow_response, pow_data = await _ds_get_pow(token, client)

            payload = {
                "chat_session_id":  session_id,
                "parent_message_id": parent_message_id,
                "prompt":           prompt,
                "ref_file_ids":     [],
                "thinking_enabled": thinking,
                "search_enabled":   search_enabled,
                "model_type":       model_type,
                "action":           None,
                "preempt":          False,
                "pow":              pow_data,
                "stream":           True,
            }

            headers = _ds_headers(token, pow_response)

            # ── stream + جمع الرد ─────────────────────────
            full_text          = ""
            thinking_text      = ""
            new_parent_msg_id  = None

            try:
                async with client.stream(
                    "POST", DEEPSEEK_CHAT_URL,
                    json=payload,
                    headers=headers,
                    timeout=REQUEST_TIMEOUT,
                ) as resp:
                    # نقرأ أول سطر للحصول على message IDs
                    first_done = False
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
                        except json.JSONDecodeError:
                            continue

                        # أول رسالة: نستخرج IDs
                        if not first_done:
                            req_id  = obj.get("request_message_id")
                            resp_id = obj.get("response_message_id")
                            if req_id and resp_id:
                                new_parent_msg_id = resp_id
                                first_done = True
                                continue

                        # محتوى
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

        # ── حفظ الجلسة ───────────────────────────────────
        if new_parent_msg_id:
            await _update_session(token, conv_id, ds_parent_msg_id=new_parent_msg_id)

        log.info(
            "DeepSeek: text=%d thinking=%d parent=%s mode=%s",
            len(full_text), len(thinking_text), new_parent_msg_id, model_type,
        )

        # ── إرسال thinking كـ chunk خاص (اختياري للعملاء الذكية) ──
        if thinking_text:
            thinking_payload = json.dumps({
                "type":    "thinking",
                "content": thinking_text,
            }, ensure_ascii=False)
            yield f"data: {thinking_payload}\n\n"

        # ── تحليل وإرسال الرد ─────────────────────────────
        tc = parse_tool_call(full_text)
        if tc:
            log.info("DeepSeek tool call: %s", tc["name"])
            call_id = f"call_{uuid.uuid4().hex[:24]}"
            yield sse_chunk(tc=tc, model=self.model_id, call_id=call_id)
            yield sse_chunk(tc=tc, model=self.model_id, call_id=call_id, finish=True)
            yield "data: [DONE]\n\n"
        else:
            txt = clean_text(full_text) or "[DeepSeek: empty response]"
            chunk_size = 40
            for i in range(0, max(len(txt), 1), chunk_size):
                yield sse_chunk(txt[i:i+chunk_size], model=self.model_id)
            yield sse_chunk(model=self.model_id, finish=True)
            yield "data: [DONE]\n\n"


register_backend(DeepSeekBackend())


# ══════════════════════════════════════════════════════════
# ★ BACKEND 3: ChatGPT (GPT-5)
#
# يدعم:
#   • محاكاة تطبيق أندرويد (نفس headers التطبيق)
#   • conversation_id + parent_message_id للحفاظ على السياق
#   • بدون thinking (GPT لا يدعمه بهذه الطريقة)
#   • sentinel_payload ثابت (من الكود الأصلي)
# ══════════════════════════════════════════════════════════

GPT_PROXY_ID   = "gpt"
GPT_BASE_URL   = "https://android.chat.openai.com/backend-api"
GPT_DEVICE_ID  = "4cdd060c-f77d-4944-aedb-46ef8aa8bb38"
GPT_STABLE_ID  = "5d2b760d-7e29-40da-bc66-06ca6e28806a"
GPT_MODEL      = "gpt-5-5"
GPT_ACCOUNT_ID = "6a6cac80-ff60-83ea-8e92-b6c8156af25f"

# sentinel_payload ثابت (من الواجهة الأصلية)
GPT_SENTINEL_PAYLOAD = (
    '{"bot_token":{"play_integrity_token":"CrMCARCnMGuYmzFJH-LB0wNNTzP1k-kmDrSLKNTYvoZXzR2Zne_AYqlPnp2GDMV9WlF52PvI00Babh54uEkFHahWaY_jTq9KTZ19PPU83ZJ0NzUi-eZI-KuPetJfZFsOuBzQkoHtcjTvXED0YG-u9A0P2-d5UOTg-KNjb58j2-nx8eBUAZ86q8GuRzUQUUZzD0g7Y1iXggxnmSc17v5FDhvo-Dm6Ma0ArxP2rZi0bT4AX5CQawq-FzRB2qjIJV-apy0r3C78iX46J8mkST2IsNWToAKi7EG9-aLJeNjfYscjunDH-C9PLdppAGAfki9Fnnvos6FfFkYtkU_9-gSNZ1_FQYhBpAvsauaiVRICzdAGo17WcO78r0cfV56qbXghNHdpQm1Vx32gO4k-0_lQvg9ii9cSKxp-AVDILS3yconjokLr891GPEhBjbWS5vkHDV8HuH5__gDWn5E22xK9mtjuXi6Lij-vva88cpaJaYnGB_5bl8-AmDPiZzVwC2fxSMZhCl0EvZrf-jaUdwHtuhOsya-63uoweH6nj2GMVRxPIE_CM4Lj8MrUFj6oG5FAl2uc2UIl","chat_requirement_token":"gAAAAABqa4dkAuZ1kMHlVAHVqBkHwkRmnFMFVoR6Ce75hQ4QmYoQnqmrGOAeU1WUyEzgVzPEtSohzSSs3ILEXozZQR2BFsBDv9-YX9IegEPyp5hfQNRH39H3hpX4XU2sHkdGBTBehgKwj7vyNPvAuQ94IQNwjxm7sbC8aXLgoWYjK_p5xdik_CqmLkFjQWUhoozqxOnPEzDwybCwogxn4_CSWUKCOK0YerpTgXjWoEFrGIgj_EXOV_pE5pvJamOA2sK1GzOC6RVpMbtgyQUm_aQYh6GgtSDOJm2imURp6Ai0PG2QSIvj49sb0Vgod4NhYn27qyzFGSOFeEfgVFllAEGPysdqHQfdUks0RNM-7CAOxUK0E5WgDhhoeM1S8aOwtg_7JNHPziqhauUaIF3_ls7_laqFATtv9Ydzr2W6Ka-T6g8rmHwgo69VXqqr7Prgd8LYwvXN0GRXdXYBNW4dC9EDnMOWcNQkbO9SiOcGtBqSDblYNSBrIG_YUNBGGFhM3QTmTg1pIfP4lO7rlxwy6N1Em6RqzX95Md0czVluz97VHsec3rDVPoP6HV7EPlfUhyepwTuzElPrVJXXzSYjBK4OR5BoqFJoApDr3whGkkFCj1v9D3crUkv4wLDA1xEG6D9L7eBVUkgvNUbzDTAVm8i7kC85c_qOcpmRcQVR2VjgWsLkVwSXz_tpRrI4C1DMqdFppvxbMoEjB0VeH455UkI6V4gjDEOcu3R0P7gr71i1PWG56O85myOv0K9zGbcHaMwJ7f_X7rl79ygVGttp22FIeJ5ZNfI3-IUV7WN8J2rbIOfS9WFKY8nTSQgQ3ITYdGNYWjaNS6gsZwH-hizcHpQH574wcMJE9rLikX-mw-qmi56xzL3eEjxgwHUARyNTwqDY_dfuw9ZqmVZu2kvQ6_Jn7HW6JVkEw1y7km-SLLU7dad0AvsUavLO9FdbpeykMseZ3IrJu0LdjyRLPpOiYQQ77Vni9Woy5wZmDQ56WwCZYlLdDbpq0yPuyAusKtCA1hPj0oF6WjwrnKxmTymF62hVxKFPrslEtA=="}}'
)


def _gpt_base_headers(authorization: str, device_id: Optional[str] = None) -> Dict[str, str]:
    """Headers تحاكي تطبيق ChatGPT الأندرويد."""
    did = device_id or GPT_DEVICE_ID
    return {
        "User-Agent":                 "ChatGPT/1.2026.195 (Android 16; 24117RN76G; build 2619512)",
        "Accept":                     "application/json",
        "Accept-Encoding":            "gzip",
        "Content-Type":               "application/json",
        "oai-package-name":           "com.openai.chatgpt",
        "oai-client-type":            "android",
        "oai-device-id":              did,
        "accept-language":            "ar,en-US;q=0.9",
        "x-device-tier":              "mid",
        "chatgpt-account-id":         GPT_ACCOUNT_ID,
        "chatgpt-residency-region":   "no_constraint",
        "x-storefront-country-code":  "US",
        "authorization":              authorization,
    }


async def _gpt_prepare_conduit(authorization: str, client: httpx.AsyncClient) -> Optional[str]:
    """يجلب conduit_token من GPT prepare endpoint."""
    url = f"{GPT_BASE_URL}/f/conversation/prepare"
    payload = {
        "action":                        "next",
        "messages":                      [],
        "model":                         GPT_MODEL,
        "history_and_training_disabled": False,
        "fork_from_shared_post":         False,
        "enable_message_followups":      True,
        "force_use_sse":                 False,
        "force_use_search":              None,
        "force_paragen":                 False,
        "supported_encodings":           ["v1"],
        "supports_buffering":            True,
        "timezone":                      "Africa/Cairo",
        "timezone_offset_min":           -180,
        "system_hints":                  [],
        "is_onboarding_conversation":    False,
        "client_prepare_dispatch":       "debounced",
        "client_prepare_source":         "composer_editor_state",
    }
    try:
        resp = await client.post(
            url, json=payload,
            headers=_gpt_base_headers(authorization),
            timeout=30,
        )
        data = resp.json()
        token = data.get("conduit_token")
        if token:
            log.info("GPT: conduit_token fetched")
        return token
    except Exception as e:
        log.warning("GPT: prepare failed: %s", e)
        return None


class GPTBackend(BaseBackend):
    """
    Backend يتصل بـ ChatGPT (GPT-5) عبر محاكاة تطبيق أندرويد.

    التوكن المطلوب: Authorization header الكامل (Bearer eyJ...)
    بدون thinking (GPT لا يدعمه بهذه الطريقة)

    extra keys مدعومة:
      cookie: str   — Cookie header الكامل (اختياري، يُحسّن الاستقرار)
    """

    @property
    def model_id(self) -> str:
        return GPT_PROXY_ID

    async def complete(
        self,
        token:    str,
        messages: List[Dict],
        tools:    List[Dict],
        thinking: bool,   # يُتجاهل لـ GPT
        conv_id:  str,
        extra:    Dict,
    ) -> AsyncIterator[str]:

        # token هنا = Authorization header الكامل (Bearer ...)
        # إذا لم يحتوِ على "Bearer" نضيفه
        if not token.startswith("Bearer ") and not token.startswith("bearer "):
            authorization = f"Bearer {token}"
        else:
            authorization = token

        cookie = extra.get("cookie", "")

        await _evict_old_sessions()

        # ── بناء الرسالة الأخيرة ──────────────────────────
        # GPT يتعامل مع المحادثة بشكل مختلف: نُرسل الرسالة الأخيرة فقط
        # مع conversation_id للاستمرارية
        last_user_msg = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                content = m.get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        c.get("text", "") for c in content
                        if isinstance(c, dict) and c.get("type") == "text"
                    )
                last_user_msg = content
                break

        if not last_user_msg:
            last_user_msg = build_full_prompt(messages, [])

        # ── استرجاع أو إنشاء جلسة GPT ─────────────────────
        sess = await _get_session(token[:32], conv_id)

        async with httpx.AsyncClient() as client:
            if sess:
                conversation_id  = sess.get("gpt_conv_id")
                parent_msg_id    = sess.get("gpt_parent_msg_id")
                conduit_token    = sess.get("gpt_conduit_token")
                log.info("GPT: reusing conv=%s", conversation_id)
            else:
                conversation_id = None
                parent_msg_id   = None
                conduit_token   = None

            # ── جلب conduit token إذا لزم ─────────────────
            if not conduit_token:
                conduit_token = await _gpt_prepare_conduit(authorization, client)
                if conduit_token:
                    await _set_session(token[:32], conv_id, {
                        "gpt_conv_id":       conversation_id,
                        "gpt_parent_msg_id": parent_msg_id,
                        "gpt_conduit_token": conduit_token,
                    })

            # ── بناء الـ payload ───────────────────────────
            message_id = str(uuid.uuid4())
            payload = {
                "conversation_id":               conversation_id,
                "action":                        "next",
                "parent_message_id":             parent_msg_id,
                "messages": [{
                    "id":     message_id,
                    "author": {"role": "user"},
                    "content": {
                        "parts":        [last_user_msg],
                        "content_type": "text",
                    },
                    "status":    "finished_successfully",
                    "recipient": "all",
                    "metadata": {
                        "model_slug":             GPT_MODEL,
                        "default_model_slug":     GPT_MODEL,
                        "is_visually_hidden_from_conversation": False,
                        "exclude_after_next_user_message":      False,
                        "content_references":  [],
                        "search_result_groups": [],
                        "search_queries":       [],
                        "image_results":        [],
                        "attachments":          [],
                        "system_hints":         [],
                        "dictation":            False,
                        "voice_mode_message":   False,
                        "image_gen_async":      False,
                        "trigger_async_ux":     False,
                        "writing_blocks":       {},
                    },
                }],
                "attachment_mime_types":         [],
                "model":                         GPT_MODEL,
                "history_and_training_disabled": False,
                "fork_from_shared_post":         False,
                "enable_message_followups":      True,
                "force_use_sse":                 True,
                "force_use_search":              None,
                "force_paragen":                 False,
                "supported_encodings":           ["v1"],
                "supports_buffering":            True,
                "timezone":                      "Africa/Cairo",
                "timezone_offset_min":           -180,
                "system_hints":                  [],
                "is_onboarding_conversation":    False,
                "client_prepare_state":          "success",
                "stream":                        True,
            }

            # إزالة None values
            if not conversation_id:
                payload.pop("conversation_id", None)
            if not parent_msg_id:
                payload.pop("parent_message_id", None)

            # ── Headers ───────────────────────────────────
            req_device_id = str(uuid.uuid4())
            headers = {
                **_gpt_base_headers(authorization, req_device_id),
                "Accept":                    "text/event-stream,application/json",
                "cache-control":             "no-cache",
                "x-sentinel-payload":        GPT_SENTINEL_PAYLOAD,
                "x-oai-convo-session-id":    str(uuid.uuid4()),
                "x-oai-turn-trace-id":       str(uuid.uuid4()),
                "x-openai-target-path":      "/backend-api/f/conversation",
            }
            if conduit_token:
                headers["x-conduit-token"] = conduit_token
            if cookie:
                headers["Cookie"] = cookie

            # ── stream ────────────────────────────────────
            url       = f"{GPT_BASE_URL}/f/conversation"
            full_text = ""
            new_conv_id     = conversation_id
            new_parent_id   = parent_msg_id

            try:
                async with client.stream(
                    "POST", url,
                    json=payload,
                    headers=headers,
                    timeout=REQUEST_TIMEOUT,
                ) as resp:
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
                        except json.JSONDecodeError:
                            continue

                        # conversation_id
                        if obj.get("conversation_id") and not new_conv_id:
                            new_conv_id = obj["conversation_id"]

                        # نص — صيغة append
                        o_op = obj.get("o", "")
                        p_op = obj.get("p", "")
                        v_op = obj.get("v")

                        if (o_op == "append"
                                and "/message/content/parts/0" in p_op
                                and isinstance(v_op, str)):
                            full_text += v_op
                            continue

                        # نص — صيغة patch
                        if o_op == "patch" and isinstance(v_op, list):
                            for patch in v_op:
                                if (patch.get("o") == "append"
                                        and "/message/content/parts/0" in patch.get("p", "")
                                        and isinstance(patch.get("v"), str)):
                                    full_text += patch["v"]

                        # message id (parent)
                        msg = obj.get("message") or {}
                        if msg.get("id"):
                            new_parent_id = msg["id"]

            except Exception as e:
                log.error("GPT stream error: %s", e)
                yield sse_chunk(f"[GPT Error: {e}]", model=self.model_id)
                yield sse_chunk(model=self.model_id, finish=True)
                yield "data: [DONE]\n\n"
                return

        # ── حفظ الجلسة ───────────────────────────────────
        await _set_session(token[:32], conv_id, {
            "gpt_conv_id":       new_conv_id,
            "gpt_parent_msg_id": new_parent_id or message_id,
            "gpt_conduit_token": conduit_token,
        })

        log.info(
            "GPT: text=%d conv=%s parent=%s",
            len(full_text), new_conv_id, new_parent_id,
        )

        # ── إرسال الرد ────────────────────────────────────
        tc = parse_tool_call(full_text)
        if tc:
            log.info("GPT tool call: %s", tc["name"])
            call_id = f"call_{uuid.uuid4().hex[:24]}"
            yield sse_chunk(tc=tc, model=self.model_id, call_id=call_id)
            yield sse_chunk(tc=tc, model=self.model_id, call_id=call_id, finish=True)
            yield "data: [DONE]\n\n"
        else:
            txt = clean_text(full_text) or "[GPT: empty response]"
            chunk_size = 40
            for i in range(0, max(len(txt), 1), chunk_size):
                yield sse_chunk(txt[i:i+chunk_size], model=self.model_id)
            yield sse_chunk(model=self.model_id, finish=True)
            yield "data: [DONE]\n\n"


register_backend(GPTBackend())


# ══════════════════════════════════════════════════════════
# معالجة thinking و model_type من الطلب
# ══════════════════════════════════════════════════════════

def resolve_thinking(body: Dict) -> bool:
    """يستخرج قيمة التفكير من أي صيغة."""
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
    """
    يجمع الخيارات الإضافية الخاصة بكل نموذج.
    يدعم extra_body أو top-level keys.
    """
    extra_body = body.get("extra_body") or {}
    extra: Dict = {}

    if model == DEEPSEEK_PROXY_ID:
        # model_type: default | expert
        mt = (
            extra_body.get("model_type")
            or body.get("model_type")
            or "expert"
        )
        extra["model_type"] = mt if mt in ("default", "expert") else "expert"

        # search_enabled
        se = extra_body.get("search_enabled")
        if se is None:
            se = body.get("search_enabled")
        extra["search_enabled"] = bool(se) if se is not None else True

    elif model == GPT_PROXY_ID:
        # cookie اختياري
        extra["cookie"] = extra_body.get("cookie") or body.get("cookie") or ""

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

app = FastAPI(
    title="Universal AI Proxy",
    version="6.0.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _extract_token(authorization: Optional[str]) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header.")
    parts = authorization.split(" ", 1)
    return parts[1].strip() if len(parts) == 2 else parts[0].strip()


# ── Health ─────────────────────────────────────────────────
@app.get("/", tags=["health"])
async def health():
    return {
        "status":          "ok",
        "proxy":           "Universal AI Proxy",
        "version":         "6.0.0",
        "active_sessions": len(_sessions),
        "backends":        list(_BACKENDS.keys()),
        "models": {
            "qwen":     "Qwen3.8-max via chat.qwen.ai",
            "deepseek": "DeepSeek via chat.deepseek.com (POW: Railway)",
            "gpt":      "ChatGPT GPT-5 via android.chat.openai.com",
        },
        "notes": [
            "Each model uses its own token — tokens are not interchangeable",
            "DeepSeek: extra_body.model_type = 'default' | 'expert' (default: expert)",
            "DeepSeek: extra_body.search_enabled = true | false (default: true)",
            "GPT: thinking is not supported and will be ignored",
            "GPT token = full Authorization header value (Bearer eyJ...)",
        ],
    }


# ── Models ─────────────────────────────────────────────────
@app.get("/v1/models", tags=["models"])
async def list_models():
    models = []
    for mid in _BACKENDS:
        models.append({
            "id":       mid,
            "object":   "model",
            "created":  1700000000,
            "owned_by": "proxy",
        })
    models.append({
        "id":       "qwen-vision",
        "object":   "model",
        "created":  1700000000,
        "owned_by": "qwen",
    })
    return {"object": "list", "data": models}


# ── Chat Completions ────────────────────────────────────────
@app.post("/v1/chat/completions", tags=["chat"])
async def chat_completions(
    request:       Request,
    authorization: Optional[str] = Header(None),
):
    token = _extract_token(authorization)
    body  = await request.json()

    messages  = body.get("messages", [])
    tools     = body.get("tools", [])
    do_stream = body.get("stream", False)
    model     = body.get("model", QWEN_PROXY_ID)

    # GPT: token = قيمة Authorization الكاملة للمستخدم
    # إذا أرسل المستخدم "Bearer eyJ..." كـ Authorization header، نمرره كاملاً لـ GPT
    if model == GPT_PROXY_ID:
        # نُعيد بناء Authorization الكامل من الـ header
        raw_auth = authorization or ""
        gpt_token = raw_auth  # نمرر الـ header كاملاً
    else:
        gpt_token = token

    effective_token = gpt_token if model == GPT_PROXY_ID else token

    thinking = resolve_thinking(body)
    extra    = resolve_extra(body, model)

    log.info(
        "thinking=%s model=%s (extra=%s)",
        thinking, model, extra,
    )

    # ── معرّف المحادثة ──────────────────────────────────
    conv_id = (
        body.get("conversation_id")
        or body.get("session_id")
        or request.headers.get("x-conversation-id")
        or request.headers.get("x-session-id")
    )
    if not conv_id:
        key_msgs = messages[:-1] if len(messages) > 1 else messages[:1]
        conv_id  = hashlib.md5(
            json.dumps(key_msgs, ensure_ascii=False).encode()
        ).hexdigest()

    # ── Deduplication ─────────────────────────────────
    req_hash = _request_hash(messages, tools)
    if await _is_duplicate(req_hash):
        log.warning("Duplicate request (conv=%s) — skipping", conv_id)
        raise HTTPException(
            status_code=429,
            detail="Duplicate request — please retry in a moment.",
        )

    log.info(
        "conv=%s | model=%s | msgs=%d | tools=%d | stream=%s | thinking=%s",
        conv_id, model, len(messages), len(tools), do_stream, thinking,
    )

    # ── اختر الـ backend ───────────────────────────────
    backend = get_backend(model)
    if backend is None:
        # fallback للـ Qwen إذا لم يُعرف النموذج
        backend = get_backend(QWEN_PROXY_ID)
        if backend is None:
            raise HTTPException(status_code=400, detail=f"No backend for model '{model}'.")

    # ── Streaming ─────────────────────────────────────
    if do_stream:
        async def event_stream():
            async for chunk in backend.complete(
                effective_token, messages, tools, thinking, conv_id, extra
            ):
                yield chunk

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # ── Non-streaming ──────────────────────────────────
    full_content   = ""
    finish_reason  = "stop"
    tool_call_data = None
    call_id_data   = None

    async for chunk in backend.complete(
        effective_token, messages, tools, thinking, conv_id, extra
    ):
        if chunk.startswith("data: [DONE]"):
            break
        if not chunk.startswith("data: "):
            continue
        try:
            obj    = json.loads(chunk[6:])
            # تجاهل chunks التفكير الخاصة بـ DeepSeek
            if obj.get("type") == "thinking":
                continue
            choice = obj["choices"][0]
            delta  = choice.get("delta", {})
            fr     = choice.get("finish_reason")
            if fr:
                finish_reason = fr
            if delta.get("tool_calls"):
                tc_item        = delta["tool_calls"][0]
                tool_call_data = tc_item
                call_id_data   = tc_item.get("id")
            elif delta.get("content"):
                full_content += delta["content"]
        except Exception:
            continue

    if tool_call_data:
        return JSONResponse(make_tc_response(
            {
                "name":      tool_call_data["function"]["name"],
                "arguments": tool_call_data["function"]["arguments"],
            },
            model,
        ))
    return JSONResponse(make_text_response(full_content, model))


# ══════════════════════════════════════════════════════════
# Image Generation  (Qwen t2i — لم يتغير)
# ══════════════════════════════════════════════════════════

@app.post("/v1/images/generations", tags=["images"])
async def image_generations(
    request:       Request,
    authorization: Optional[str] = Header(None),
):
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
        async with client.stream(
            "POST", f"{QWEN_BASE}/chat/completions",
            json=payload,
            headers=_qwen_headers_chat(token, stream=True),
            params={"chat_id": cid},
            timeout=300,
        ) as resp:
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
# Image Edits  (Qwen t2i + OSS — لم يتغير)
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
        headers=_qwen_headers_chat(token),
        timeout=60,
    )
    res = sts.json()
    if _qwen_is_rate_limited(res) or "data" not in res:
        raise HTTPException(status_code=429, detail="Rate-limited during OSS STS.")

    d      = res["data"]
    aki    = d["access_key_id"]
    aks    = d["access_key_secret"]
    stkn   = d["security_token"]
    fpath  = d["file_path"]
    fid    = d["file_id"]
    bucket = d["bucketname"]
    host   = f"{bucket}.{d['endpoint']}"
    furl   = d.get("file_url", f"https://{host}/{fpath}")
    oss_ua = "aliyun-sdk-android/2.9.21"
    c_hdr  = f"x-oss-security-token:{stkn}\n"

    def _oh(method, md5, ct, res_path, extra=None):
        gmt = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")
        sig = _oss_sig(aks, method, md5, ct, gmt, c_hdr, res_path)
        h   = {
            "Authorization":        f"OSS {aki}:{sig}",
            "User-Agent":           oss_ua,
            "Host":                 host,
            "x-oss-security-token": stkn,
            "Date":                 gmt,
            "Content-Type":         ct,
        }
        if extra:
            h.update(extra)
        return h

    init_r    = await client.post(
        f"https://{host}/{fpath}?uploads",
        headers=_oh("POST", "", "image/jpeg", f"/{bucket}/{fpath}?uploads", {"Content-Length": "0"}),
        timeout=60,
    )
    upload_id = ET.fromstring(init_r.text).find("{*}UploadId").text

    cmd5   = base64.b64encode(hashlib.md5(image_bytes).digest()).decode()
    part_r = await client.put(
        f"https://{host}/{fpath}?uploadId={upload_id}&partNumber=1",
        content=image_bytes,
        headers=_oh(
            "PUT", cmd5, "image/jpeg",
            f"/{bucket}/{fpath}?partNumber=1&uploadId={upload_id}",
            {"Content-MD5": cmd5, "Content-Length": file_size},
        ),
        timeout=OSS_UPLOAD_TIMEOUT,
    )
    etag = part_r.headers.get("ETag", "").replace('"', "")

    body = (
        f"<CompleteMultipartUpload>"
        f"<Part><PartNumber>1</PartNumber><ETag>{etag}</ETag></Part>"
        f"</CompleteMultipartUpload>"
    ).encode()
    await client.post(
        f"https://{host}/{fpath}?uploadId={upload_id}",
        content=body,
        headers=_oh(
            "POST", "", "image/jpeg",
            f"/{bucket}/{fpath}?uploadId={upload_id}",
            {"Content-Length": str(len(body))},
        ),
        timeout=60,
    )
    return {
        "type":         "image",
        "file":         {"data": {}, "filename": filename, "id": fid, "meta": {"name": filename}},
        "id":           fid,
        "filename":     filename,
        "name":         filename,
        "url":          furl,
        "image_width":  1024,
        "image_height": 1024,
    }


@app.post("/v1/images/edits", tags=["images"])
async def image_edits(
    request:       Request,
    authorization: Optional[str] = Header(None),
):
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
        uploaded = await _upload_image(token, image_bytes, client)
        file_entry = {
            "type":         "image",
            "file":         uploaded["file"],
            "id":           uploaded["id"],
            "url":          uploaded["url"],
            "name":         uploaded["filename"],
            "image_width":  1024,
            "image_height": 1024,
        }
        cid     = await _qwen_create_chat(token, client)
        payload = _qwen_build_payload(
            cid, prompt, None,
            chat_type="t2i",
            files=[file_entry],
        )
        result_url: Optional[str] = None
        async with client.stream(
            "POST", f"{QWEN_BASE}/chat/completions",
            json=payload,
            headers=_qwen_headers_chat(token, stream=True),
            params={"chat_id": cid},
            timeout=300,
        ) as resp:
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
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"message": exc.detail, "type": "proxy_error", "code": exc.status_code}},
    )


@app.exception_handler(Exception)
async def _generic_err(request: Request, exc: Exception):
    log.error("Unhandled: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": {"message": str(exc), "type": "internal_error", "code": 500}},
    )


# ══════════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    log.info("Starting Universal AI Proxy v6.0 on port %d", port)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


# ══════════════════════════════════════════════════════════
# ملخص استخدام الـ API
# ══════════════════════════════════════════════════════════
#
# ① Qwen:
#    POST /v1/chat/completions
#    Authorization: Bearer <qwen_token>
#    {"model": "qwen", "messages": [...], "thinking": true/false}
#
# ② DeepSeek:
#    POST /v1/chat/completions
#    Authorization: Bearer <deepseek_token>
#    {
#      "model": "deepseek",
#      "messages": [...],
#      "thinking": true/false,
#      "extra_body": {
#        "model_type": "expert",      // أو "default"
#        "search_enabled": true       // أو false
#      }
#    }
#    - thinking chunk منفصل يُرسل كـ: data: {"type":"thinking","content":"..."}
#    - باقي الرد بصيغة OpenAI العادية
#
# ③ GPT:
#    POST /v1/chat/completions
#    Authorization: Bearer eyJ...  (Authorization header الكامل من ChatGPT)
#    {"model": "gpt", "messages": [...]}
#    - thinking يُتجاهل
#    - extra_body.cookie: "..." اختياري لتحسين الاستقرار
#
# ══════════════════════════════════════════════════════════

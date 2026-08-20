"""Universal AI Proxy  v9.0
=========================
التحسينات الجديدة:
  ① DeepSeek: Fallback تلقائي عند خطأ "الخادم مشغول"
      - محاولة أولى: محادثة جديدة بنفس الوضع
      - محاولة ثانية: تغيير الوضع (expert↔default) + محادثة جديدة
  ② Regenerate: كشف طلب regenerate → إنشاء محادثة جديدة لكل النماذج
  ③ Qwen Proxy: نظام proxy ذكي
      - يستخدم proxy فقط لـ Qwen
      - ينتقل للبروكسي التالي فقط عند HTTP 403/429 (حظر)
      - يحتفظ بنفس البروكسي لنفس المحادثة (session sticky)
      - لا ينتقل عند أخطاء الشبكة العادية
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
# Proxy Manager (Qwen فقط)
# ══════════════════════════════════════════════════════════

PROXY_FILE = os.environ.get("PROXY_FILE", "proxies.txt")

class QwenProxyManager:
    """
    يدير قائمة البروكسيات لـ Qwen.
    - sticky per session: نفس المحادثة تستخدم نفس البروكسي
    - يتبدل فقط عند HTTP 403/429 (حظر IP)
    - يسجّل البروكسيات المحظورة ويتجنبها
    """

    def __init__(self):
        self._proxies: List[str] = []      # "http://user:pass@host:port"
        self._banned: set = set()           # proxies محظورة
        self._idx = 0                       # index الحالي (round-robin أولي)
        self._lock = asyncio.Lock()
        self._session_proxy: Dict[str, str] = {}  # conv_id → proxy_url
        self._load()

    def _load(self):
        if not os.path.exists(PROXY_FILE):
            log.warning("Proxy file '%s' not found — Qwen will run without proxy", PROXY_FILE)
            return
        with open(PROXY_FILE) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                # صيغة: user:pass@host:port  أو  http://user:pass@host:port
                if not line.startswith("http"):
                    line = "http://" + line
                self._proxies.append(line)
        log.info("ProxyManager: loaded %d proxies from %s", len(self._proxies), PROXY_FILE)

    @property
    def enabled(self) -> bool:
        return len(self._proxies) > 0

    def _available(self) -> List[str]:
        return [p for p in self._proxies if p not in self._banned]

    async def get_for_session(self, conv_id: str) -> Optional[str]:
        """إرجاع البروكسي المخصص للمحادثة، أو تعيين واحد جديد"""
        async with self._lock:
            if not self._proxies:
                return None
            avail = self._available()
            if not avail:
                log.warning("ProxyManager: all proxies banned! resetting ban list")
                self._banned.clear()
                avail = self._proxies[:]

            if conv_id in self._session_proxy:
                p = self._session_proxy[conv_id]
                if p in avail:
                    return p
                # البروكسي القديم محظور → نعيّن جديد
                log.info("ProxyManager: session %s proxy was banned, reassigning", conv_id[:16])

            # اختيار round-robin من المتاح
            p = avail[self._idx % len(avail)]
            self._idx += 1
            self._session_proxy[conv_id] = p
            return p

    async def mark_banned(self, proxy_url: str, conv_id: str) -> Optional[str]:
        """
        يُعلّم البروكسي كمحظور ويُعيّن بروكسي جديد للمحادثة.
        يُستدعى فقط عند HTTP 403/429 من Qwen.
        """
        async with self._lock:
            if proxy_url in self._proxies:
                self._banned.add(proxy_url)
                log.warning("ProxyManager: banned proxy %s (total banned: %d/%d)",
                            proxy_url, len(self._banned), len(self._proxies))
            avail = self._available()
            if not avail:
                log.warning("ProxyManager: no available proxies! resetting")
                self._banned.clear()
                avail = self._proxies[:]
            if not avail:
                return None
            p = avail[self._idx % len(avail)]
            self._idx += 1
            self._session_proxy[conv_id] = p
            log.info("ProxyManager: conv %s switched to proxy %s",
                     conv_id[:16], p.split('@')[-1])
            return p

    def get_httpx_proxies(self, proxy_url: str) -> Dict[str, str]:
        return {"http://": proxy_url, "https://": proxy_url}


proxy_manager = QwenProxyManager()


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


async def _clear_session(token: str, conv_id: str) -> None:
    """حذف الجلسة لإجبار إنشاء محادثة جديدة"""
    async with _session_lock:
        key = _session_key(token, conv_id)
        if key in _sessions:
            del _sessions[key]
            log.info("Session cleared: %s", conv_id[:16])


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


def _is_regenerate_request(body: Dict) -> bool:
    """
    كشف إذا كان الطلب regenerate.
    Open Minis يرسل عادةً:
      - "regenerate": true
      - أو "action": "regenerate"
      - أو "resend": true
    """
    if body.get("regenerate") or body.get("resend"):
        return True
    if body.get("action") in ("regenerate", "resend", "retry"):
        return True
    extra = body.get("extra_body") or {}
    if extra.get("regenerate") or extra.get("action") in ("regenerate", "resend"):
        return True
    return False


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
# BACKEND 1: Qwen (مع دعم Proxy)
# ══════════════════════════════════════════════════════════

QWEN_BASE          = "https://chat.qwen.ai/api/v2"
QWEN_MODEL_ID_REAL = "qwen3.8-max"
QWEN_PROXY_ID      = "qwen"
REQUEST_TIMEOUT    = 180

# أكواد HTTP التي تعني حظر IP (ننتقل للبروكسي التالي)
QWEN_BAN_CODES = {403, 429, 451}

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


def _make_qwen_client(proxy_url: Optional[str] = None) -> httpx.AsyncClient:
    """إنشاء httpx client مع أو بدون proxy"""
    if proxy_url:
        return httpx.AsyncClient(proxies={"http://": proxy_url, "https://": proxy_url})
    return httpx.AsyncClient()


async def _qwen_create_chat(token: str, client: httpx.AsyncClient) -> str:
    url     = f"{QWEN_BASE}/chats/new"
    payload = {"chat_mode": "normal", "project_id": ""}
    resp    = await client.post(url, json=payload, headers=_qwen_headers_new(token), timeout=60)
    # فحص حظر IP
    if resp.status_code in QWEN_BAN_CODES:
        raise BannedProxyError(f"HTTP {resp.status_code}")
    data    = resp.json()
    cid     = (data.get("chat_id") or data.get("id")
               or (data.get("data") or {}).get("chat_id")
               or (data.get("data") or {}).get("id"))
    if not cid:
        raise HTTPException(status_code=502, detail=f"Failed to create Qwen chat: {data}")
    return cid


class BannedProxyError(Exception):
    """يُرفع عند حظر البروكسي (403/429/451)"""
    pass


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


async def _qwen_stream_collect(
    token: str,
    chat_id: str,
    payload: Dict,
    client: httpx.AsyncClient,
) -> Tuple[str, Optional[str], bool]:
    """
    يُرجع: (full_text, last_response_id, was_banned)
    was_banned = True إذا تلقينا HTTP ban code
    """
    url      = f"{QWEN_BASE}/chat/completions"
    full_txt = ""
    resp_id: Optional[str] = None
    was_banned = False

    try:
        async with client.stream(
            "POST", url, json=payload,
            headers=_qwen_headers_chat(token, stream=True),
            params={"chat_id": chat_id}, timeout=REQUEST_TIMEOUT,
        ) as resp:
            # فحص حظر IP أولاً
            if resp.status_code in QWEN_BAN_CODES:
                log.warning("Qwen: HTTP %d → proxy banned", resp.status_code)
                return "", None, True

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

    except httpx.ProxyError as e:
        log.warning("Qwen: ProxyError → %s (proxy connection failed)", e)
        # خطأ اتصال بالبروكسي — لا يعني حظراً بالضرورة
        # نُرجع نصاً فارغاً بدون علامة حظر
        return "", None, False

    return full_txt, resp_id, was_banned


class QwenBackend(BaseBackend):
    @property
    def model_id(self) -> str:
        return QWEN_PROXY_ID

    async def complete(self, token, messages, tools, thinking, conv_id, extra) -> AsyncIterator[str]:
        prompt = build_full_prompt(messages, tools)
        if not prompt.strip():
            return
        await _evict_old_sessions()

        # الحد الأقصى لمحاولات تبديل البروكسي
        MAX_PROXY_RETRIES = 3

        for attempt in range(MAX_PROXY_RETRIES + 1):
            # جلب البروكسي الحالي للمحادثة
            proxy_url = await proxy_manager.get_for_session(conv_id) if proxy_manager.enabled else None

            sess = await _get_session(token, conv_id)
            if sess:
                qwen_chat_id = sess["qwen_chat_id"]
                parent_id    = sess.get("parent_id")
            else:
                try:
                    async with _make_qwen_client(proxy_url) as tmp:
                        qwen_chat_id = await _qwen_create_chat(token, tmp)
                except BannedProxyError:
                    if proxy_url and attempt < MAX_PROXY_RETRIES:
                        log.warning("Qwen: proxy banned during chat creation, switching")
                        proxy_url = await proxy_manager.mark_banned(proxy_url, conv_id)
                        continue
                    yield sse_chunk("[Qwen Error: all proxies banned]", model=self.model_id)
                    yield sse_chunk(model=self.model_id, finish=True)
                    yield "data: [DONE]\n\n"
                    return
                parent_id = None
                await _set_session(token, conv_id, {
                    "qwen_chat_id": qwen_chat_id,
                    "parent_id": parent_id,
                    "qwen_proxy": proxy_url,
                })

            payload = _qwen_build_payload(qwen_chat_id, prompt, parent_id, thinking=thinking)

            async with _make_qwen_client(proxy_url) as client:
                qwen_text, last_rid, was_banned = await _qwen_stream_collect(
                    token, qwen_chat_id, payload, client
                )

            if was_banned and proxy_url and attempt < MAX_PROXY_RETRIES:
                # حظر IP → نبدل البروكسي ونعيد المحاولة بمحادثة جديدة
                log.info("Qwen: switching proxy (attempt %d/%d)", attempt + 1, MAX_PROXY_RETRIES)
                proxy_url = await proxy_manager.mark_banned(proxy_url, conv_id)
                await _clear_session(token, conv_id)
                continue

            # نجاح أو انتهاء المحاولات
            await _update_session(token, conv_id, parent_id=last_rid)
            break

        tc = parse_tool_call(qwen_text)
        if tc:
            call_id = f"call_{uuid.uuid4().hex[:24]}"
            yield sse_chunk(tc=tc, model=self.model_id, call_id=call_id)
            yield sse_chunk(tc=tc, model=self.model_id, call_id=call_id, finish=True)
            yield "data: [DONE]\n\n"
        else:
            txt = clean_text(qwen_text) or "[Qwen: empty response]"
            for i in range(0, max(len(txt), 1), 40):
                yield sse_chunk(txt[i:i+40], model=self.model_id)
            yield sse_chunk(model=self.model_id, finish=True)
            yield "data: [DONE]\n\n"


register_backend(QwenBackend())


# ══════════════════════════════════════════════════════════
# BACKEND 2 & 3: DeepSeek (مع Fallback ذكي)
# ══════════════════════════════════════════════════════════

DEEPSEEK_PROXY_ID_EXPERT  = "deepseek"
DEEPSEEK_PROXY_ID_DEFAULT = "deepseek-default"
DEEPSEEK_CHAT_URL         = "https://chat.deepseek.com/api/v0/chat/completion"
DEEPSEEK_SESSION_URL      = "https://chat.deepseek.com/api/v0/chat_session/create"
RAILWAY_POW_URL           = "https://pow.up.railway.app/pow"

# رسائل خطأ DeepSeek التي تعني "الخادم مشغول"
DEEPSEEK_SERVER_BUSY_PATTERNS = [
    "server is busy",
    "الخادم مشغول",
    "try again later",
    "حاول مرة أخرى",
    "use fast mode",
    "السريع",
    "overloaded",
    "too many requests",
    "rate limit",
]

def _ds_is_server_busy(text: str) -> bool:
    """فحص إذا كان الخطأ يعني الخادم مشغول"""
    lower = text.lower()
    return any(p in lower for p in DEEPSEEK_SERVER_BUSY_PATTERNS)


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

        # نحاول مرتين كحد أقصى (نفس الوضع + وضع مختلف)
        # الإستراتيجية:
        #   محاولة 1: إذا فشلت → محادثة جديدة، نفس الوضع
        #   محاولة 2: إذا فشلت → محادثة جديدة، وضع مختلف
        MAX_DS_RETRIES = 2
        current_type = model_type

        for attempt in range(MAX_DS_RETRIES + 1):
            sess = await _get_session(token, conv_id)
            async with httpx.AsyncClient() as client:
                if sess and attempt == 0:
                    session_id        = sess["ds_session_id"]
                    parent_message_id = sess.get("ds_parent_msg_id")
                    log.info("DeepSeek[%s]: reusing session=%s parent=%s",
                             current_type, session_id, parent_message_id)
                else:
                    # إنشاء محادثة جديدة
                    if attempt > 0:
                        log.info("DeepSeek: retry attempt=%d, creating new session, mode=%s",
                                 attempt, current_type)
                        await _clear_session(token, conv_id)
                    session_id        = await _ds_create_session(token, client)
                    parent_message_id = None
                    await _set_session(token, conv_id, {
                        "ds_session_id":    session_id,
                        "ds_parent_msg_id": parent_message_id,
                    })
                    log.info("DeepSeek[%s]: new session=%s for conv=%s",
                             current_type, session_id, conv_id)

                pow_response, pow_data = await _ds_get_pow(token, client)

                payload = {
                    "chat_session_id":   session_id,
                    "parent_message_id": parent_message_id,
                    "prompt":            prompt,
                    "ref_file_ids":      [],
                    "thinking_enabled":  thinking,
                    "search_enabled":    search_enabled,
                    "model_type":        current_type,
                    "action":            None,
                    "preempt":           False,
                    "pow":               pow_data,
                    "stream":            True,
                }

                headers   = _ds_headers(token, pow_response)
                full_text = ""
                thinking_text     = ""
                new_parent_msg_id = None
                stream_error      = None

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
                    stream_error = str(e)
                    log.error("DeepSeek stream error: %s", e)

            # فحص إذا كان الخطأ "الخادم مشغول"
            is_busy = (
                stream_error and _ds_is_server_busy(stream_error)
            ) or (
                full_text and _ds_is_server_busy(full_text) and len(full_text) < 200
            )

            if is_busy and attempt < MAX_DS_RETRIES:
                # تبديل الوضع في المحاولة الثانية
                if attempt == 0:
                    log.warning("DeepSeek: server busy, retrying with new session (same mode)")
                elif attempt == 1:
                    # تبديل expert ↔ default
                    current_type = "default" if current_type == "expert" else "expert"
                    log.warning("DeepSeek: server busy again, switching mode to %s", current_type)
                await asyncio.sleep(1)
                continue

            # نجاح أو لا يوجد داعي للتكرار
            break

        if new_parent_msg_id:
            await _update_session(token, conv_id, ds_parent_msg_id=new_parent_msg_id)

        log.info("DeepSeek[%s]: text=%d thinking=%d parent=%s",
                 current_type, len(full_text), len(thinking_text), new_parent_msg_id)

        if stream_error and not full_text:
            yield sse_chunk(f"[DeepSeek Error: {stream_error}]", model=self.model_id)
            yield sse_chunk(model=self.model_id, finish=True)
            yield "data: [DONE]\n\n"
            return

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
# BACKEND 4: Gemini
# ══════════════════════════════════════════════════════════

GEMINI_PROXY_ID   = "gemini"
GEMINI_USER_ACCT  = "u/1"
GEMINI_BL         = "boq_assistant-bard-web-server_20260817.02_p0"
GEMINI_MODEL_JSPB = (
    '[1,null,null,null,"fbb127bbb056c959",null,null,0,'
    '[4,5,6,8,4,5,6,8],null,null,1,null,null,1,1,'
    '"036033AF-386B-4A1C-A8B6-F563586CF2B9"]'
)
GEMINI_BASE_URL = (
    f"https://gemini.google.com/{GEMINI_USER_ACCT}/_/BardChatUi/data"
)
GEMINI_APP_URL  = f"https://gemini.google.com/{GEMINI_USER_ACCT}/app"
GEMINI_STREAM_URL = (
    f"{GEMINI_BASE_URL}/assistant.lamda.BardFrontendService/StreamGenerate"
)

GEMINI_TRACKED_COOKIES = {
    "SIDCC", "__Secure-1PSIDCC", "__Secure-3PSIDCC",
    "__Secure-1PSIDTS", "__Secure-3PSIDTS",
    "COMPASS", "_gcl_au", "_ga_WC57KJ50ZZ", "_ga_BF8Q35BMLM",
}

_gemini_cookie_store: Dict[str, Dict[str, str]] = {}
_gemini_lock = asyncio.Lock()


def _parse_cookie_string(cookie_str: str) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for part in cookie_str.split(";"):
        part = part.strip()
        if "=" in part:
            k, _, v = part.partition("=")
            result[k.strip()] = v.strip()
    return result


def _cookies_to_string(cookies: Dict[str, str]) -> str:
    return "; ".join(f"{k}={v}" for k, v in cookies.items())


def _gemini_cookie_key(cookie_str: str) -> str:
    cookies = _parse_cookie_string(cookie_str)
    sid = cookies.get("SID", cookie_str)
    return "gem_" + hashlib.md5(sid[:64].encode()).hexdigest()[:16]


async def _gemini_get_cookies(cookie_key: str, initial_str: str) -> Dict[str, str]:
    async with _gemini_lock:
        if cookie_key in _gemini_cookie_store:
            return dict(_gemini_cookie_store[cookie_key])
        parsed = _parse_cookie_string(initial_str)
        _gemini_cookie_store[cookie_key] = parsed
        return dict(parsed)


async def _gemini_update_cookies(cookie_key: str, response_cookies) -> None:
    async with _gemini_lock:
        if cookie_key not in _gemini_cookie_store:
            return
        updated = []
        try:
            items = list(response_cookies.items())
        except Exception:
            return
        for name, value in items:
            if name in GEMINI_TRACKED_COOKIES:
                old = _gemini_cookie_store[cookie_key].get(name, "")
                if old != value:
                    _gemini_cookie_store[cookie_key][name] = value
                    updated.append(name)
        if updated:
            log.info("Gemini: cookies updated: %s", ", ".join(updated))


def _gemini_headers(cookies_str: str) -> Dict[str, str]:
    return {
        "authority":        "gemini.google.com",
        "accept":           "*/*",
        "accept-language":  "ar,en-US;q=0.9,en;q=0.8",
        "origin":           "https://gemini.google.com",
        "referer":          "https://gemini.google.com/",
        "user-agent":       (
            "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/109.0.0.0 Mobile Safari/537.36"
        ),
        "x-same-domain":    "1",
        "content-type":     "application/x-www-form-urlencoded;charset=UTF-8",
        "cookie":           cookies_str,
    }


async def _gemini_get_tokens(
    cookies: Dict[str, str],
    client: httpx.AsyncClient,
    cookie_key: str,
) -> Tuple[Optional[str], Optional[str]]:
    cookies_str = _cookies_to_string(cookies)
    headers = _gemini_headers(cookies_str)
    headers.pop("content-type", None)

    try:
        url = GEMINI_APP_URL
        for _ in range(5):
            resp = await client.get(
                url, headers=headers, timeout=30, follow_redirects=False,
            )
            log.info("Gemini token fetch: status=%d url=%s body_len=%d",
                     resp.status_code, url, len(resp.text))

            if resp.status_code in (301, 302, 303, 307, 308):
                location = resp.headers.get("location", "")
                if not location:
                    break
                if location.startswith("/"):
                    location = "https://gemini.google.com" + location
                url = location
                await _gemini_update_cookies(cookie_key, resp.cookies)
                log.info("Gemini: redirect → %s", url[:80])
                continue
            break

        await _gemini_update_cookies(cookie_key, resp.cookies)

        snlm0e = None
        fdrfje = None

        for pattern in [r'"SNlM0e":"(.*?)"', r"'SNlM0e':'(.*?)'",
                         r'SNlM0e["\s]*:["\s]*"([^"]+)"']:
            q1 = re.search(pattern, resp.text)
            if q1:
                snlm0e = q1.group(1)
                break

        for pattern in [r'"FdrFJe":"([\d-]+)"', r"'FdrFJe':'([\d-]+)'",
                         r'FdrFJe["\s]*:["\s]*"([\d-]+)"']:
            q2 = re.search(pattern, resp.text)
            if q2:
                fdrfje = q2.group(1)
                break

        if not snlm0e:
            log.error("Gemini: SNlM0e not found. Response preview: %s",
                      resp.text[:500].replace("\n", " "))
        else:
            log.info("Gemini: tokens OK snlm0e=%s fdrfje=%s", snlm0e[:8], fdrfje)

        return snlm0e, fdrfje

    except Exception as e:
        log.error("Gemini: failed to get tokens: %s", e, exc_info=True)
        return None, None


async def _gemini_send_message(
    cookies: Dict[str, str],
    client:  httpx.AsyncClient,
    cookie_key: str,
    prompt:  str,
    snlm0e:  str,
    fdrfje:  str,
    gemini_conv: Optional[Dict],
) -> Tuple[str, Optional[Dict]]:
    if gemini_conv is None:
        context = ["", "", "", None, None, None, None, None, None, ""]
    else:
        context = [
            gemini_conv.get("conversation_id", ""),
            gemini_conv.get("response_id", ""),
            gemini_conv.get("choice_id", ""),
            None, None, None, None, None, None,
            gemini_conv.get("at_token", ""),
        ]

    d1 = [
        [prompt, 0, None, None, None, None, 0],
        ["ar"],
        context,
        None, None, None, [], 0, [], [], 1, 0,
    ]

    payload = {
        "at":    snlm0e,
        "f.req": json.dumps([None, json.dumps(d1)]),
    }

    params = {
        "bl":     GEMINI_BL,
        "hl":     "ar",
        "pageId": "none",
        "_reqid": str(__import__("random").randint(1_000_000, 9_999_999)),
        "rt":     "c",
        "f.sid":  fdrfje,
    }

    cookies_str = _cookies_to_string(cookies)
    h2 = _gemini_headers(cookies_str)
    h2["x-goog-ext-525001261-jspb"] = GEMINI_MODEL_JSPB
    h2["x-goog-ext-73010989-jspb"]  = "[0]"
    h2["x-goog-ext-73010990-jspb"]  = "[0,0,0]"

    full_text    = ""
    new_conv     = dict(gemini_conv) if gemini_conv else {}
    new_at_token = None

    try:
        async with client.stream(
            "POST", GEMINI_STREAM_URL,
            params=params, data=payload,
            headers=h2, timeout=REQUEST_TIMEOUT,
        ) as resp:
            await _gemini_update_cookies(cookie_key, resp.cookies)
            async for line in resp.aiter_lines():
                if not line:
                    continue
                try:
                    a1 = json.loads(line)
                    if not isinstance(a1, list) or not a1:
                        continue
                    if len(a1[0]) >= 3 and a1[0][2]:
                        c2 = json.loads(a1[0][2])
                        try:
                            if not new_conv.get("conversation_id"):
                                conv_meta = c2[1]
                                if conv_meta and len(conv_meta) >= 2:
                                    new_conv["conversation_id"] = conv_meta[0]
                                    new_conv["response_id"]     = conv_meta[1]
                        except Exception:
                            pass
                        try:
                            candidates = c2[4]
                            if candidates and candidates[0]:
                                choice = candidates[0]
                                if choice[0]:
                                    new_conv["choice_id"] = choice[0]
                                text = choice[1][0]
                                if text and text.startswith(full_text):
                                    new_part = text[len(full_text):]
                                    if new_part:
                                        full_text = text
                        except Exception:
                            pass
                        try:
                            if isinstance(c2[3], dict):
                                at_val = c2[3].get("26", "")
                                if at_val:
                                    new_at_token = at_val
                        except Exception:
                            pass
                except Exception:
                    continue
    except Exception as e:
        log.error("Gemini stream error: %s", e)
        return f"[Gemini Error: {e}]", gemini_conv

    if new_at_token:
        new_conv["at_token"] = new_at_token

    return full_text, new_conv if new_conv.get("conversation_id") else None


class GeminiBackend(BaseBackend):
    @property
    def model_id(self) -> str:
        return GEMINI_PROXY_ID

    async def complete(self, token, messages, tools, thinking, conv_id, extra) -> AsyncIterator[str]:
        prompt = build_full_prompt(messages, tools)
        if not prompt.strip():
            return

        await _evict_old_sessions()

        cookie_key = _gemini_cookie_key(token)
        sess = await _get_session(token, conv_id)

        async with httpx.AsyncClient(verify=False, timeout=REQUEST_TIMEOUT) as client:
            if sess and sess.get("gemini_snlm0e"):
                snlm0e      = sess["gemini_snlm0e"]
                fdrfje      = sess["gemini_fdrfje"]
                gemini_conv = sess.get("gemini_conv")
                log.info("Gemini: reusing session conv=%s snlm0e=%s",
                         conv_id, snlm0e[:8] if snlm0e else "?")
            else:
                cookies = await _gemini_get_cookies(cookie_key, token)
                log.info("Gemini: fetching tokens, cookies=%d keys", len(cookies))
                snlm0e, fdrfje = await _gemini_get_tokens(cookies, client, cookie_key)

                if not snlm0e:
                    err_msg = "[Gemini Error: failed to get session tokens — check cookies]"
                    log.error(err_msg)
                    yield sse_chunk(err_msg, model=self.model_id)
                    yield sse_chunk(model=self.model_id, finish=True)
                    yield "data: [DONE]\n\n"
                    return

                gemini_conv = None
                await _set_session(token, conv_id, {
                    "gemini_snlm0e": snlm0e,
                    "gemini_fdrfje": fdrfje,
                    "gemini_conv":   gemini_conv,
                    "gemini_ck":     cookie_key,
                })
                log.info("Gemini: new session conv=%s snlm0e=%s", conv_id, snlm0e[:8])

            cookies = await _gemini_get_cookies(cookie_key, token)
            log.info("Gemini: sending message, conv_id=%s cookies=%d", conv_id, len(cookies))
            full_text, updated_conv = await _gemini_send_message(
                cookies, client, cookie_key,
                prompt, snlm0e, fdrfje, gemini_conv,
            )

        await _update_session(token, conv_id, gemini_conv=updated_conv)
        log.info("Gemini: text=%d chars", len(full_text))

        tc = parse_tool_call(full_text)
        if tc:
            call_id = f"call_{uuid.uuid4().hex[:24]}"
            yield sse_chunk(tc=tc, model=self.model_id, call_id=call_id)
            yield sse_chunk(tc=tc, model=self.model_id, call_id=call_id, finish=True)
            yield "data: [DONE]\n\n"
        else:
            txt = clean_text(full_text) or "[Gemini: empty response]"
            for i in range(0, max(len(txt), 1), 40):
                yield sse_chunk(txt[i:i+40], model=self.model_id)
            yield sse_chunk(model=self.model_id, finish=True)
            yield "data: [DONE]\n\n"


register_backend(GeminiBackend())


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

app = FastAPI(title="Universal AI Proxy", version="9.0.0", docs_url="/docs")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _extract_token(authorization: Optional[str]) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header.")
    parts = authorization.split(" ", 1)
    return parts[1].strip() if len(parts) == 2 else parts[0].strip()


@app.get("/", tags=["health"])
async def health():
    pm = proxy_manager
    return {
        "status": "ok", "proxy": "Universal AI Proxy", "version": "9.0.0",
        "active_sessions": len(_sessions),
        "gemini_cookie_keys": len(_gemini_cookie_store),
        "backends": list(_BACKENDS.keys()),
        "proxy_config": {
            "enabled": pm.enabled,
            "total": len(pm._proxies),
            "banned": len(pm._banned),
            "available": len(pm._available()),
        },
        "models": {
            "qwen":             "Qwen3.8-max via chat.qwen.ai (with proxy rotation)",
            "deepseek":         "DeepSeek Expert (auto-fallback on busy)",
            "deepseek-default": "DeepSeek Default (auto-fallback on busy)",
            "gemini":           "Gemini via gemini.google.com (cookies-based)",
        },
        "new_in_v9": [
            "DeepSeek: auto-retry on server busy (new session → switch mode)",
            "Regenerate: clears session for fresh conversation",
            "Qwen: proxy rotation on ban (403/429 only, not on connection errors)",
            "Proxy: sticky per session, same-country rotation",
        ],
    }


@app.get("/v1/models", tags=["models"])
async def list_models():
    models = [
        {"id": mid, "object": "model", "created": 1700000000, "owned_by": "proxy"}
        for mid in _BACKENDS
    ]
    models.append({"id": "qwen-vision", "object": "model", "created": 1700000000, "owned_by": "qwen"})
    return {"object": "list", "data": models}


@app.post("/v1/chat/completions", tags=["chat"])
async def chat_completions(
    request:       Request,
    authorization: Optional[str] = Header(None),
):
    token    = _extract_token(authorization)
    body     = await request.json()
    messages = body.get("messages", [])
    tools    = body.get("tools", [])
    do_stream = body.get("stream", False)
    model    = body.get("model", QWEN_PROXY_ID)

    thinking = resolve_thinking(body)
    extra    = resolve_extra(body, model)

    explicit_conv_id = (
        body.get("conversation_id")
        or body.get("session_id")
        or request.headers.get("x-conversation-id")
        or request.headers.get("x-session-id")
    )
    conv_id = _compute_conv_id(messages, explicit_conv_id)

    # ── Regenerate: حذف الجلسة لإنشاء محادثة جديدة ──────────────────
    if _is_regenerate_request(body):
        log.info("Regenerate detected for conv=%s, clearing session", conv_id[:16])
        await _clear_session(token, conv_id)
        # إعادة تعيين البروكسي أيضاً (للتنويع)
        if proxy_manager.enabled:
            async with proxy_manager._lock:
                proxy_manager._session_proxy.pop(conv_id, None)

    log.info("conv=%s model=%s msgs=%d thinking=%s extra=%s",
             conv_id, model, len(messages), thinking, extra)

    req_hash = _request_hash(messages, tools)
    if await _is_duplicate(req_hash):
        log.warning("Duplicate request (conv=%s) — skipping", conv_id)
        raise HTTPException(status_code=429, detail="Duplicate request — please retry in a moment.")

    backend = get_backend(model)
    if backend is None:
        backend = get_backend(QWEN_PROXY_ID)
        if backend is None:
            raise HTTPException(status_code=400, detail=f"No backend for model '{model}'.")

    if do_stream:
        async def event_stream():
            async for chunk in backend.complete(token, messages, tools, thinking, conv_id, extra):
                yield chunk
        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # Non-streaming
    full_content   = ""
    finish_reason  = "stop"
    tool_call_data = None

    async for chunk in backend.complete(token, messages, tools, thinking, conv_id, extra):
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

    proxy_url = await proxy_manager.get_for_session("img_gen") if proxy_manager.enabled else None

    async with _make_qwen_client(proxy_url) as client:
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
    log.info("Starting Universal AI Proxy v9.0 on port %d", port)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

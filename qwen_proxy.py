"""
Universal AI Proxy  v5.0
=========================
بروكسي موحد: Qwen (الآن) + جاهز لإضافة أي نموذج مستقبلاً

إصلاحات v5 مقارنة بـ v4:
  ① الرسائل لا تُحذف — كل tool call يظهر كـ chunk منفصل في الـ stream
  ② التفكير يعمل فعلاً — نستقبل reasoning_effort من OpenMinis ونحوّله لـ Qwen
  ③ لا تكرار للأدوات — نرسل tool call واحداً فقط ونحمي من الـ race condition
  ④ ترتيب صحيح: tool call → انتظار النتيجة → الخطوة التالية (بدون parallels)
  ⑤ معمارية backend مرنة: أضف نموذجاً جديداً بكتابة class واحدة فقط
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
#   لإضافة نموذج جديد: اصنع class ترث من BaseBackend
# ══════════════════════════════════════════════════════════

class BaseBackend(ABC):
    """الواجهة التي يجب أن ينفّذها كل backend."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """المعرّف الذي يُرسله OpenMinis في حقل model."""

    @abstractmethod
    async def complete(
        self,
        token: str,
        messages: List[Dict],
        tools: List[Dict],
        thinking: bool,
        conv_id: str,
    ) -> AsyncIterator[str]:
        """
        يُنتج SSE chunks بصيغة OpenAI.
        كل chunk عبارة عن سطر "data: {...}\\n\\n"
        آخر chunk: "data: [DONE]\\n\\n"
        """


# ─── سجل الـ Backends ─────────────────────────────────────
_BACKENDS: Dict[str, BaseBackend] = {}


def register_backend(backend: BaseBackend) -> None:
    _BACKENDS[backend.model_id] = backend
    log.info("Registered backend: %s", backend.model_id)


def get_backend(model_id: str) -> Optional[BaseBackend]:
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


# ─── تحليل رد النموذج: هل هو tool call؟ ──────────────────

def parse_tool_call(text: str) -> Optional[Dict]:
    """
    يتعرف على tool calls بكل صيغها الممكنة.
    يُعيد {"name": str, "arguments": str(json)} أو None.
    """
    # 1. ACTION: tool_name|{json}
    m = re.search(r"(?m)^ACTION:\s*(\w+)\|(\{.*\})\s*$", text, re.DOTALL)
    if m:
        return _make_tc(m.group(1), m.group(2))

    # 2. <tool_call><name>…</name><arguments>…</arguments></tool_call>
    m = re.search(
        r"<tool_call>\s*<name>(.*?)</name>\s*<arguments>(.*?)</arguments>\s*</tool_call>",
        text, re.DOTALL | re.IGNORECASE,
    )
    if m:
        return _make_tc(m.group(1), m.group(2))

    # 3. <name>…</name> مع <parameter=x> (صيغة Qwen الخاصة)
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
    """إزالة بقايا XML/ACTION من النص النهائي."""
    text = re.sub(r"(?m)^ACTION:\s*\S+\|.*$", "", text)
    text = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"</?function[^>]*>", "", text, flags=re.IGNORECASE)
    return text.strip()


# ── OpenAI-compatible response builders ───────────────────

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
    """
    يبني SSE data line واحداً.
    ① محتوى نصي عادي
    ② بداية tool_call (tc مُعطى، finish=False)
    ③ نهاية tool_call   (tc مُعطى، finish=True)
    ④ نهاية stream       (finish=True بدون tc)
    """
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
# ★ Qwen Backend
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
    """جمع الرد الكامل من Qwen. يُعيد (full_text, response_id)."""
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

                # استخرج response_id
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
    """Backend يتصل بـ Qwen عبر موقع chat.qwen.ai."""

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
    ) -> AsyncIterator[str]:
        """
        ينتج SSE chunks بصيغة OpenAI.

        FIX ①: كل tool call يُرسَل كـ chunk مستقل → لا تُحذف رسالة
        FIX ②: thinking يُفعَّل من حقل thinking في الطلب
        FIX ③: حماية من تكرار tool calls بمفتاح idempotency
        """
        # ── بناء البرومبت الكامل ───────────────────────────
        prompt = build_full_prompt(messages, tools)
        if not prompt.strip():
            return

        await _evict_old_sessions()

        # ── استرجاع أو إنشاء جلسة Qwen ───────────────────
        sess = await _get_session(token, conv_id)
        if sess:
            qwen_chat_id = sess["qwen_chat_id"]
            parent_id    = sess.get("parent_id")
            log.info("Reusing qwen_chat_id=%s", qwen_chat_id)
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

        # ── جمع رد Qwen (نجمع كاملاً لأن التحليل يحتاج النص كله) ─
        async with httpx.AsyncClient() as client:
            qwen_text, last_rid = await _qwen_stream_collect(
                token, qwen_chat_id, payload, client
            )

        await _update_session(token, conv_id, parent_id=last_rid)
        log.info("Qwen response: %d chars | parent=%s", len(qwen_text), last_rid)

        # ── تحليل الرد ────────────────────────────────────
        tc = parse_tool_call(qwen_text)

        if tc:
            log.info("Tool call: %s", tc["name"])
            call_id = f"call_{uuid.uuid4().hex[:24]}"
            # FIX ①: نُرسل tool call كـ chunk → يظهر في الـ UI ولا يُحذف
            yield sse_chunk(tc=tc, model=self.model_id, call_id=call_id)
            yield sse_chunk(tc=tc, model=self.model_id, call_id=call_id, finish=True)
            yield "data: [DONE]\n\n"
        else:
            txt = clean_text(qwen_text)
            # نُرسل النص على شكل chunks صغيرة
            chunk_size = 40
            for i in range(0, len(txt), chunk_size):
                yield sse_chunk(txt[i:i+chunk_size], model=self.model_id)
            yield sse_chunk(model=self.model_id, finish=True)
            yield "data: [DONE]\n\n"


# سجّل Qwen كـ default backend
register_backend(QwenBackend())


# ══════════════════════════════════════════════════════════
# ★ FIX ②: معالجة thinking من OpenMinis
#
# OpenMinis يُرسل reasoning_effort: "low"/"medium"/"high"/"none"
# عند تفعيل زر التفكير في الإعدادات.
# نحوّله إلى bool لـ Qwen (الذي يفهم thinking_enabled فقط).
# ══════════════════════════════════════════════════════════

def resolve_thinking(body: Dict) -> bool:
    """
    يستخرج قيمة التفكير من أي صيغة ترسلها OpenMinis:
      - reasoning_effort: "none" → False
      - reasoning_effort: أي قيمة أخرى → True
      - thinking: true/false (صيغة قديمة)
    """
    effort = body.get("reasoning_effort")
    if effort is not None:
        return str(effort).lower() not in ("none", "off", "false", "0", "")

    # Anthropic-style: thinking: {"type": "enabled"/"disabled"}
    think_obj = body.get("thinking")
    if isinstance(think_obj, dict):
        t = think_obj.get("type", "")
        return t not in ("disabled", "none", "")
    if isinstance(think_obj, bool):
        return think_obj

    return False


# ══════════════════════════════════════════════════════════
# ★ FIX ③: حماية من تكرار tool calls
#
# OpenMinis أحياناً يُرسل نفس الطلب مرتين بسرعة.
# نستخدم deduplication بناءً على hash للطلب.
# ══════════════════════════════════════════════════════════

_recent_requests: Dict[str, float] = {}
_req_lock = asyncio.Lock()
DEDUP_WINDOW = 3.0  # ثوانٍ


async def _is_duplicate(req_hash: str) -> bool:
    async with _req_lock:
        now = time.time()
        # نظّف القديم
        stale = [k for k, t in _recent_requests.items() if now - t > DEDUP_WINDOW * 10]
        for k in stale:
            del _recent_requests[k]
        # تحقق
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
    version="5.0.0",
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
        "version":         "5.0.0",
        "active_sessions": len(_sessions),
        "backends":        list(_BACKENDS.keys()),
        "fixes":           [
            "messages-not-deleted",
            "thinking-toggle",
            "no-duplicate-tool-calls",
            "sequential-tool-execution",
            "multi-backend-architecture",
        ],
    }


# ── Models ─────────────────────────────────────────────────
@app.get("/v1/models", tags=["models"])
async def list_models():
    models = []
    for mid in _BACKENDS:
        models.append({
            "id":         mid,
            "object":     "model",
            "created":    1700000000,
            "owned_by":   "proxy",
        })
    # إضافة الصورة (image generation) لـ qwen
    models.append({
        "id":         "qwen-vision",
        "object":     "model",
        "created":    1700000000,
        "owned_by":   "qwen",
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

    # FIX ②: استخراج thinking بشكل صحيح
    thinking = resolve_thinking(body)
    log.info("thinking=%s (reasoning_effort=%r)", thinking, body.get("reasoning_effort"))

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

    # FIX ③: deduplication — silent (لا نرجع 429 لأن OpenMinis تعرضه كـ Rate limited)
    req_hash = _request_hash(messages, tools)
    if await _is_duplicate(req_hash):
        log.warning("Duplicate request ignored silently (conv=%s)", conv_id)
        if do_stream:
            async def _dup_stream():
                yield sse_chunk("", model=model)
                yield sse_chunk(model=model, finish=True)
                yield "data: [DONE]\n\n"
            return StreamingResponse(_dup_stream(), media_type="text/event-stream")
        return JSONResponse(make_text_response("", model))

    log.info(
        "conv=%s | model=%s | msgs=%d | tools=%d | stream=%s | thinking=%s",
        conv_id, model, len(messages), len(tools), do_stream, thinking,
    )

    # ── اختر الـ backend ───────────────────────────────
    backend = get_backend(model) or get_backend(QWEN_PROXY_ID)
    if backend is None:
        raise HTTPException(status_code=400, detail=f"No backend for model '{model}'.")

    # ── Streaming ─────────────────────────────────────
    if do_stream:
        async def event_stream():
            async for chunk in backend.complete(token, messages, tools, thinking, conv_id):
                yield chunk

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # ── Non-streaming: جمع الـ chunks وتحويلها لـ JSON ─
    full_content = ""
    finish_reason = "stop"
    tool_call_data = None
    call_id_data   = None

    async for chunk in backend.complete(token, messages, tools, thinking, conv_id):
        if chunk.startswith("data: [DONE]"):
            break
        if not chunk.startswith("data: "):
            continue
        try:
            obj    = json.loads(chunk[6:])
            choice = obj["choices"][0]
            delta  = choice.get("delta", {})
            fr     = choice.get("finish_reason")
            if fr:
                finish_reason = fr
            if delta.get("tool_calls"):
                tc_item      = delta["tool_calls"][0]
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
# Image Generation  (Qwen t2i)
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
# Image Edits  (Qwen t2i + OSS upload)
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
            "Authorization":     f"OSS {aki}:{sig}",
            "User-Agent":        oss_ua,
            "Host":              host,
            "x-oss-security-token": stkn,
            "Date":              gmt,
            "Content-Type":      ct,
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
# ★ Kimi Backend  (kimi.com — gRPC/Connect protocol)
# ══════════════════════════════════════════════════════════
#
# البروتوكول: Connect/gRPC-Web بدلاً من REST عادي.
# كل رسالة = 5-byte header (flag + uint32 length) + JSON body.
# التوكن = refresh_token يُجدَّد تلقائياً إلى access_token.
# نموذج افتراضي: SCENARIO_K2D5  (kimi-k2.6)
#
# الفرق عن Qwen:
#   - لا يوجد "chat session" مستمر؛ كل طلب يبني محادثة كاملة
#   - الرد يصل كـ binary stream (framed messages) لا SSE
#   - الأدوات الداخلية (search, image...) مُضمَّنة، لكننا نعطّلها
#     ونستخدم الأدوات الخارجية عبر نفس آلية ACTION: التي يفهمها Qwen
# ══════════════════════════════════════════════════════════

import struct as _struct

KIMI_BASE      = "https://www.kimi.com"
KIMI_PROXY_ID  = "kimi"

# النماذج المتاحة (يمكن تحديدها عبر model في الطلب):
KIMI_MODELS = {
    "kimi":           "SCENARIO_K2D5",   # kimi-k2.6 (افتراضي)
    "kimi-k2":        "SCENARIO_K2D5",
    "kimi-k3":        "SCENARIO_K3",
    "kimi-auto":      "SCENARIO_UNSPECIFIED",
    "kimi-solve":     "SCENARIO_PROBLEM_SOLVE",
}

_KIMI_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/132.0.0.0 Safari/537.36"
)


def _kimi_base_headers(device_id: str) -> Dict:
    """Headers مشتركة لكل طلبات Kimi."""
    return {
        "User-Agent":          _KIMI_UA,
        "Accept":              "*/*",
        "Accept-Language":     "en-US,en;q=0.9",
        "x-msh-device-id":    device_id,
        "x-msh-platform":     "web",
        "x-msh-session-id":   "1731757202045822784",
        "x-msh-version":      "2.0.0",
        "x-traffic-id":       "d8i4n6nahd86l5du0130",
        "Origin":              KIMI_BASE,
        "Referer":             f"{KIMI_BASE}/",
        "Sec-Ch-Ua":           '"Not A(Brand";v="8", "Chromium";v="132"',
        "Sec-Ch-Ua-Mobile":    "?0",
        "Sec-Ch-Ua-Platform":  '"Windows"',
        "Sec-Fetch-Dest":      "empty",
        "Sec-Fetch-Mode":      "cors",
        "Sec-Fetch-Site":      "same-origin",
        "Cache-Control":       "no-cache",
        "Pragma":              "no-cache",
        "R-Timezone":          "Asia/Riyadh",
        "X-Language":          "en-US",
    }


def _kimi_extract_device_id(refresh_token: str) -> str:
    """يستخرج device_id من payload الـ JWT."""
    try:
        parts = refresh_token.split(".")
        if len(parts) != 3:
            raise ValueError("Invalid JWT")
        payload = parts[1]
        # إضافة padding
        payload += "=" * (4 - len(payload) % 4)
        data = json.loads(base64.urlsafe_b64decode(payload).decode("utf-8"))
        device_id = data.get("device_id")
        if not device_id:
            raise ValueError("No device_id in JWT")
        return str(device_id)
    except Exception as e:
        log.error("Kimi: failed to extract device_id: %s", e)
        return "7669233658326224138"  # fallback


async def _kimi_refresh_token(refresh_token: str, device_id: str) -> str:
    """يحوّل refresh_token → access_token."""
    headers = {
        **_kimi_base_headers(device_id),
        "Authorization": f"Bearer {refresh_token}",
    }
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        resp = await client.get(
            f"{KIMI_BASE}/api/auth/token/refresh",
            headers=headers,
        )
        if resp.status_code != 200:
            raise HTTPException(
                status_code=502,
                detail=f"Kimi token refresh failed: {resp.status_code}",
            )
        data = resp.json()
        access_token = data.get("access_token")
        if not access_token:
            raise HTTPException(status_code=502, detail="Kimi: no access_token returned")
        log.info("Kimi: token refreshed OK")
        return access_token


def _kimi_encode_message(payload: Dict) -> bytes:
    """
    يُشفّر payload كـ gRPC/Connect frame:
    [flag:1byte][length:4bytes_big_endian][json_body]
    """
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    header = _struct.pack(">BI", 0, len(body))
    return header + body


def _kimi_decode_frames(data: bytes) -> List[Dict]:
    """
    يُفكّك binary stream من Kimi إلى قائمة JSON objects.
    كل frame = 5-byte header + JSON body.
    """
    results = []
    offset  = 0
    while offset + 5 <= len(data):
        # flag byte (نتجاهله) + uint32 big-endian length
        length = _struct.unpack(">I", data[offset + 1: offset + 5])[0]
        end    = offset + 5 + length
        if end > len(data):
            break
        raw = data[offset + 5: end]
        offset = end
        try:
            results.append(json.loads(raw.decode("utf-8")))
        except Exception:
            continue
    return results


def _kimi_get_scenario(model_id: str) -> str:
    """يُعيد SCENARIO المناسب حسب model_id."""
    return KIMI_MODELS.get(model_id, KIMI_MODELS["kimi"])


def _kimi_build_chat_payload(
    messages:   List[Dict],
    tools:      List[Dict],
    thinking:   bool,
    scenario:   str,
    system_txt: str,
) -> Dict:
    """
    يبني payload الـ Chat request لـ Kimi بالبنية الصحيحة.

    البنية الأصلية المستخرجة من كود Kimi:
      {
        scenario: str,
        tools: [],
        message: {
          role: "user",
          blocks: [ {role, content:{type,value:{content}}} ],  ← آخر رسالة
          scenario: str,
          is_goal: false
        },
        options: {thinking, enablePlugin},
        project_id: ""
      }

    يُرسَل system prompt + تاريخ المحادثة كاملاً داخل نص آخر رسالة user
    (نفس أسلوب Qwen) لأن Kimi لا يدعم history في هذه النقطة.
    """
    # ── جمع كل المحتوى في prompt واحد ───────────────────
    parts = []

    if system_txt:
        parts.append(f"[SYSTEM]\n{system_txt}\n[/SYSTEM]")

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
            parts.append(f"User: {content}")

        elif role == "assistant":
            if isinstance(content, list):
                content = " ".join(
                    c.get("text", "")
                    for c in content
                    if isinstance(c, dict) and c.get("type") == "text"
                )
            tc_list = m.get("tool_calls", [])
            if tc_list:
                for tc in tc_list:
                    func = tc.get("function", {})
                    parts.append(f"ACTION: {func.get('name')}|{func.get('arguments', '{}')}")
            elif content:
                parts.append(f"Assistant: {content}")

        elif role in ("tool", "function"):
            tool_name = m.get("name") or m.get("tool_call_id", "tool")
            if isinstance(content, list):
                content = str(content)
            parts.append(f"[TOOL RESULT: {tool_name}]\n{content}\n[/TOOL RESULT]")

    parts.append("Assistant:")
    full_prompt = "\n\n".join(parts)

    if not full_prompt.strip():
        return {}

    # ── بناء الـ payload بالبنية الصحيحة ─────────────────
    return {
        "scenario": scenario,
        "tools":    [],
        "message": {
            "role":    "user",
            "blocks": [{
                "role":    "user",
                "content": {
                    "type":  "text",
                    "value": {"content": full_prompt},
                },
            }],
            "scenario": scenario,
            "is_goal":  False,
        },
        "options": {
            "thinking":     thinking,
            "enablePlugin": False,
        },
        "project_id": "",
    }


def _kimi_extract_text_from_frames(frames: List[Dict]) -> str:
    """
    يجمع النص الكامل من frames Kimi.
    Kimi يرسل النص في عدة بنى:
      frame.block.text.content          ← streaming incremental
      frame.block.content.value.content ← بعض الحالات
      frame.message.blocks[].text.content ← رسالة كاملة
    """
    text_parts: List[str] = []
    seen: set = set()  # لتجنب التكرار

    def _add(t: str) -> None:
        if t and t not in seen:
            seen.add(t)
            text_parts.append(t)

    for frame in frames:
        if not isinstance(frame, dict):
            continue
        if "heartbeat" in frame or "notification" in frame:
            continue
        if "done" in frame:
            break

        # ── block مباشر (الشائع في streaming) ──────────
        block = frame.get("block")
        if isinstance(block, dict):
            _kimi_read_block(block, _add)
            continue

        # ── message.blocks (رسالة مكتملة) ───────────────
        msg = frame.get("message")
        if isinstance(msg, dict):
            for blk in msg.get("blocks", []):
                if isinstance(blk, dict):
                    _kimi_read_block(blk, _add)

    return "".join(text_parts)


def _kimi_read_block(block: Dict, add_fn) -> None:
    """
    يقرأ نص من block بكل الصيغ الممكنة التي يُرسلها Kimi.
    """
    # ① block.text = {"content": "..."}
    txt = block.get("text")
    if isinstance(txt, dict):
        c = txt.get("content", "")
        if c:
            add_fn(c)
            return

    # ② block.text = "string" مباشرة
    if isinstance(txt, str) and txt:
        add_fn(txt)
        return

    # ③ block.content = {"type":"text","value":{"content":"..."}}
    cnt = block.get("content")
    if isinstance(cnt, dict):
        if cnt.get("type") == "text":
            val = cnt.get("value", {})
            c = val.get("content", "") if isinstance(val, dict) else str(val)
            if c:
                add_fn(c)
                return
        # ④ block.content = {"content": "..."}  (بدون type)
        c = cnt.get("content", "")
        if c:
            add_fn(c)
            return

    # ⑤ error block
    err = block.get("error")
    if err:
        msg = ""
        if isinstance(err, dict):
            msg = err.get("message") or err.get("error", {}).get("message", str(err))
        if msg:
            add_fn(f"[ERROR: {msg}]")


class KimiBackend(BaseBackend):
    """
    Backend يتصل بـ Kimi (kimi.com) عبر Connect/gRPC protocol.

    التوكن: refresh_token من localStorage.getItem('refresh_token')
    يُجدَّد تلقائياً إلى access_token في كل طلب.

    model → scenario:
      kimi / kimi-k2  → SCENARIO_K2D5   (kimi-k2.6)
      kimi-k3         → SCENARIO_K3
      kimi-auto       → SCENARIO_UNSPECIFIED
      kimi-solve      → SCENARIO_PROBLEM_SOLVE
    """

    @property
    def model_id(self) -> str:
        return KIMI_PROXY_ID

    async def complete(
        self,
        token:    str,
        messages: List[Dict],
        tools:    List[Dict],
        thinking: bool,
        conv_id:  str,
    ) -> AsyncIterator[str]:

        # ── السيناريو من model prefix في conv_id ─────────
        # conv_id يأتي بشكل "kimi-k3:xxxx" (من middleware)
        parts    = conv_id.split(":", 1)
        model_id = parts[0] if len(parts) == 2 else KIMI_PROXY_ID
        scenario = _kimi_get_scenario(model_id)

        # ── تجديد التوكن ──────────────────────────────────
        device_id    = _kimi_extract_device_id(token)
        access_token = await _kimi_refresh_token(token, device_id)

        # ── بناء system prompt + tools ────────────────────
        system_parts = []
        for m in messages:
            if m.get("role") == "system":
                c = m.get("content") or ""
                if isinstance(c, list):
                    c = " ".join(x.get("text", "") for x in c if isinstance(x, dict))
                system_parts.append(str(c))
        if tools:
            system_parts.append(f"\n{tools_to_xml(tools)}\n{TOOL_SYSTEM_SUFFIX}")
        system_txt = "\n\n".join(system_parts)

        # ── بناء payload ───────────────────────────────────
        payload = _kimi_build_chat_payload(
            messages, tools, thinking, scenario, system_txt
        )
        if not payload:
            yield sse_chunk("[No messages]", model=self.model_id)
            yield sse_chunk(model=self.model_id, finish=True)
            yield "data: [DONE]\n\n"
            return

        # ── إرسال الطلب ────────────────────────────────────
        req_headers = {
            **_kimi_base_headers(device_id),
            "Authorization":            f"Bearer {access_token}",
            "Content-Type":             "application/connect+json",
            "connect-protocol-version": "1",
        }
        encoded_body = _kimi_encode_message(payload)
        raw_buffer   = b""

        try:
            async with httpx.AsyncClient(
                timeout=REQUEST_TIMEOUT,
                follow_redirects=True,
            ) as client:
                async with client.stream(
                    "POST",
                    f"{KIMI_BASE}/apiv2/kimi.gateway.chat.v1.ChatService/Chat",
                    headers=req_headers,
                    content=encoded_body,
                ) as resp:
                    if resp.status_code != 200:
                        body_err = await resp.aread()
                        log.error("Kimi HTTP %d: %s", resp.status_code, body_err[:300])
                        yield sse_chunk(
                            f"[Kimi error: HTTP {resp.status_code}]",
                            model=self.model_id,
                        )
                        yield sse_chunk(model=self.model_id, finish=True)
                        yield "data: [DONE]\n\n"
                        return
                    async for chunk in resp.aiter_bytes():
                        raw_buffer += chunk

        except Exception as e:
            log.error("Kimi connection error: %s", e, exc_info=True)
            yield sse_chunk(f"[Kimi connection error: {e}]", model=self.model_id)
            yield sse_chunk(model=self.model_id, finish=True)
            yield "data: [DONE]\n\n"
            return

        # ── تفكيك frames واستخراج النص ─────────────────────
        frames    = _kimi_decode_frames(raw_buffer)
        full_text = _kimi_extract_text_from_frames(frames)
        log.info(
            "Kimi[%s] response: %d chars from %d frames",
            scenario, len(full_text), len(frames),
        )

        if not full_text:
            log.warning("Kimi: empty response — raw buffer: %s", raw_buffer[:500])

        # ── تحليل: tool call أم نص عادي؟ ──────────────────
        tc = parse_tool_call(full_text) if full_text else None

        if tc:
            log.info("Kimi tool call detected: %s", tc["name"])
            cid = f"call_{uuid.uuid4().hex[:24]}"
            yield sse_chunk(tc=tc, model=self.model_id, call_id=cid)
            yield sse_chunk(tc=tc, model=self.model_id, call_id=cid, finish=True)
            yield "data: [DONE]\n\n"
        else:
            txt = clean_text(full_text) if full_text else ""
            if not txt:
                txt = ""  # رد فارغ — لا نرفع خطأ
            for i in range(0, max(len(txt), 1), 40):
                yield sse_chunk(txt[i:i + 40], model=self.model_id)
            yield sse_chunk(model=self.model_id, finish=True)
            yield "data: [DONE]\n\n"


# ── تسجيل كل أسماء Kimi في Backend registry ───────────────
_kimi_instance = KimiBackend()
for _kid in ("kimi", "kimi-k2", "kimi-k3", "kimi-auto", "kimi-solve"):
    _BACKENDS[_kid] = _kimi_instance


# ══════════════════════════════════════════════════════════
# Middleware: تمرير model إلى conv_id لـ Kimi
# ══════════════════════════════════════════════════════════
# conv_id يُبنى بالشكل "kimi-k3:hashxxx" حتى يعرف KimiBackend
# أي scenario يستخدم — يعمل بشفافية بدون تعديل الـ routes

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as _SR
from starlette.responses import Response as _SRsp


class _ModelConvIdPatcher(BaseHTTPMiddleware):
    async def dispatch(self, request: _SR, call_next) -> _SRsp:
        if request.url.path == "/v1/chat/completions":
            try:
                raw  = await request.body()
                body = json.loads(raw)
                model = body.get("model", "")

                if model.startswith("kimi"):
                    # نبني conv_id يحمل model prefix
                    existing = (
                        body.get("conversation_id")
                        or body.get("session_id")
                        or request.headers.get("x-conversation-id")
                        or request.headers.get("x-session-id")
                    )
                    if not existing or not existing.startswith(model + ":"):
                        msgs     = body.get("messages", [])
                        key_msgs = msgs[:-1] if len(msgs) > 1 else msgs[:1]
                        base     = hashlib.md5(
                            json.dumps(key_msgs, ensure_ascii=False).encode()
                        ).hexdigest()
                        body["conversation_id"] = f"{model}:{base}"
                        raw = json.dumps(body).encode()

                # نُعيد بناء الطلب بالـ body المعدّل
                async def _recv():
                    return {"type": "http.request", "body": raw, "more_body": False}

                request = _SR(request.scope, _recv)
            except Exception:
                pass
        return await call_next(request)


app.add_middleware(_ModelConvIdPatcher)


# ══════════════════════════════════════════════════════════
# /v1/models — يعرض كل الـ backends المسجّلة
# ══════════════════════════════════════════════════════════

@app.get("/v1/models", include_in_schema=False)
async def _list_models_all():
    models = []
    seen   = set()
    for mid in _BACKENDS:
        if mid not in seen:
            models.append({
                "id":       mid,
                "object":   "model",
                "created":  1700000000,
                "owned_by": "kimi" if mid.startswith("kimi") else "qwen",
            })
            seen.add(mid)
    models.append({
        "id": "qwen-vision", "object": "model",
        "created": 1700000000, "owned_by": "qwen",
    })
    return {"object": "list", "data": models}


# ══════════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    log.info(
        "Starting Universal AI Proxy v5.1 on port %d | backends: %s",
        port, list(_BACKENDS.keys()),
    )
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

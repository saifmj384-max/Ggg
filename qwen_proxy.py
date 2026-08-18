"""
Qwen → OpenAI-Compatible Proxy  v4.0
======================================
إصلاح جذري: دعم كامل لـ Function Calling / Tool Use لأجل OpenMinis

المشاكل التي كانت موجودة في v3:
  1. كان يتجاهل حقل `tools` من الطلب تماماً
  2. كان يرسل آخر رسالة فقط لـ Qwen، ويحذف system prompt والتاريخ الكامل
  3. كان يرد دائماً بـ finish_reason: "stop" — لا يدعم finish_reason: "tool_calls"
  4. Qwen لا يفهم صيغة tool_calls الأصلية، فيجب محاكاتها نصياً

الحل في v4:
  ─ نحول تعريفات الأدوات إلى نص XML واضح ونضمّه في system prompt
  ─ نحوّل كامل تاريخ المحادثة (بما فيها tool results) لـ prompt نصي متسلسل
  ─ نطلب من Qwen الرد بصيغة XML محددة عند استدعاء أداة
  ─ نحلّل رد Qwen: إذا طلب أداة → نرد بـ tool_calls JSON لـ OpenMinis
                    إذا أجاب نهائياً → نرد بـ finish_reason: stop
"""

from __future__ import annotations

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
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

# ══════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════
BASE_QWEN_URL      = "https://chat.qwen.ai/api/v2"
QWEN_MODEL_ID      = "qwen3.8-max"
PROXY_MODEL_ID     = "qwen"
REQUEST_TIMEOUT    = 180
OSS_UPLOAD_TIMEOUT = 120
SESSION_TTL        = 60 * 60 * 6   # 6 ساعات

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("qwen_proxy")

# ══════════════════════════════════════════════════════════
# Session Store  (in-memory)
# ══════════════════════════════════════════════════════════
_sessions: Dict[str, Dict[str, Any]] = {}


def _session_key(token: str, conv_id: str) -> str:
    return f"{token[:16]}:{conv_id}"


def _get_session(token: str, conv_id: str) -> Optional[Dict[str, Any]]:
    sess = _sessions.get(_session_key(token, conv_id))
    if sess:
        sess["last_used"] = time.time()
    return sess


def _set_session(token: str, conv_id: str, qwen_chat_id: str,
                 parent_id: Optional[str] = None) -> None:
    _sessions[_session_key(token, conv_id)] = {
        "qwen_chat_id": qwen_chat_id,
        "parent_id":    parent_id,
        "last_used":    time.time(),
    }


def _update_parent(token: str, conv_id: str, parent_id: Optional[str]) -> None:
    key  = _session_key(token, conv_id)
    sess = _sessions.get(key)
    if sess and parent_id:
        sess["parent_id"] = parent_id
        sess["last_used"] = time.time()


def _evict_old_sessions() -> None:
    now   = time.time()
    stale = [k for k, v in _sessions.items() if now - v["last_used"] > SESSION_TTL]
    for k in stale:
        del _sessions[k]
    if stale:
        log.info("Evicted %d stale sessions", len(stale))


# ══════════════════════════════════════════════════════════
# App
# ══════════════════════════════════════════════════════════
app = FastAPI(
    title="Qwen OpenAI-Compatible Proxy",
    version="4.0.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ══════════════════════════════════════════════════════════
# Headers
# ══════════════════════════════════════════════════════════
_UA_CHAT = (
    "Dalvik/2.1.0 (Linux; U; Android 15; RMX3834 Build/AP3A.240905.015.A2) "
    "AliApp(QWENCHAT/2.7.2) AppType/Release AplusBridgeLite"
)
_UA_NEW = (
    "Dalvik/2.1.0 (Linux; U; Android 15; RMX3834 Build/AP3A.240905.015.A2),"
    "Dalvik/2.1.0 (Linux; U; Android 15; RMX3834 Build/AP3A.240905.015.A2) "
    "AliApp(QWENCHAT/2.7.2) AppType/Release AplusBridgeLite"
)


def _headers_chat(token: str, *, stream: bool = False) -> Dict[str, str]:
    return {
        "User-Agent":      _UA_CHAT,
        "Content-Type":    "application/json; charset=UTF-8",
        "Accept":          "*/*,text/event-stream" if stream else "application/json",
        "Accept-Language": "en-US",
        "Accept-Charset":  "UTF-8",
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


def _headers_new(token: str) -> Dict[str, str]:
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


def _extract_token(authorization: Optional[str]) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header.")
    parts = authorization.split(" ", 1)
    return parts[1].strip() if len(parts) == 2 else parts[0].strip()


def _is_rate_limited(obj: Any) -> bool:
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


def _is_antibot(line: str) -> bool:
    return "_____tmd_____" in line or "punish" in line


# ══════════════════════════════════════════════════════════
# ★★★ الجزء الجديد: تحويل Tools → Prompt نصي ★★★
# ══════════════════════════════════════════════════════════

TOOL_SYSTEM_SUFFIX = """
══════════════════════════════════════
TOOL USE INSTRUCTIONS
══════════════════════════════════════
You have access to the tools listed in <available_tools> above.

CRITICAL: When you need to call a tool, you MUST respond ONLY with this exact XML format and NOTHING else — no explanation before or after:

<tool_call>
<name>TOOL_NAME_HERE</name>
<arguments>
{
  "param1": "value1",
  "param2": "value2"
}
</arguments>
</tool_call>

When you have enough information to give a FINAL answer (no more tool calls needed), respond normally in plain text WITHOUT any XML tags.

Rules:
- ONE tool call per response maximum
- NEVER invent tool results — wait for the system to provide them
- After receiving tool results, either call another tool OR give your final answer
- Tool results will appear in the conversation as [TOOL RESULT: tool_name] ... [/TOOL RESULT]
══════════════════════════════════════
"""


def _tools_to_xml(tools: List[Dict]) -> str:
    """تحويل تعريفات الأدوات بصيغة OpenAI إلى نص XML مقروء."""
    if not tools:
        return ""
    lines = ["<available_tools>"]
    for tool in tools:
        # صيغة OpenAI: {"type": "function", "function": {...}}
        func = tool.get("function") or tool
        name = func.get("name", "unknown")
        desc = func.get("description", "")
        params = func.get("parameters", {})
        required = params.get("required", [])
        properties = params.get("properties", {})

        lines.append(f"  <tool>")
        lines.append(f"    <name>{name}</name>")
        lines.append(f"    <description>{desc}</description>")
        if properties:
            lines.append(f"    <parameters>")
            for pname, pinfo in properties.items():
                ptype = pinfo.get("type", "string")
                pdesc = pinfo.get("description", "")
                req   = " (required)" if pname in required else " (optional)"
                lines.append(f"      <param name=\"{pname}\" type=\"{ptype}\"{req}>{pdesc}</param>")
            lines.append(f"    </parameters>")
        lines.append(f"  </tool>")
    lines.append("</available_tools>")
    return "\n".join(lines)


def _messages_to_full_prompt(messages: List[Dict], tools: List[Dict]) -> Tuple[str, str]:
    """
    تحويل كامل قائمة الرسائل إلى:
    - system_text: النص الكامل للـ system prompt
    - conversation_text: نص المحادثة كاملاً

    يدعم رسائل: system, user, assistant, tool
    """
    system_parts = []
    conversation_parts = []

    # أولاً نجمع كل system messages
    for m in messages:
        if m.get("role") == "system":
            content = m.get("content") or ""
            if isinstance(content, list):
                content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
            system_parts.append(str(content))

    # إذا كان فيه أدوات، نضيف تعريفها وتعليمات الاستخدام
    if tools:
        system_parts.append(f"\n{_tools_to_xml(tools)}\n{TOOL_SYSTEM_SUFFIX}")

    # ثانياً نحول باقي الرسائل إلى نص محادثة
    for m in messages:
        role    = m.get("role", "user")
        content = m.get("content") or ""

        if role == "system":
            # تم معالجتها بالفعل
            continue

        elif role == "user":
            if isinstance(content, list):
                # قد يحتوي على نصوص وصور
                text_parts = []
                for c in content:
                    if isinstance(c, dict):
                        if c.get("type") == "text":
                            text_parts.append(c.get("text", ""))
                        elif c.get("type") == "image_url":
                            text_parts.append("[IMAGE]")
                content = " ".join(text_parts)
            conversation_parts.append(f"User: {content}")

        elif role == "assistant":
            if isinstance(content, list):
                content = " ".join(c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text")

            # إذا كانت الرسالة تحتوي على tool_calls من دورة سابقة
            tool_calls = m.get("tool_calls", [])
            if tool_calls:
                for tc in tool_calls:
                    func = tc.get("function", {})
                    tc_name = func.get("name", "")
                    tc_args = func.get("arguments", "{}")
                    conversation_parts.append(
                        f"Assistant: <tool_call><name>{tc_name}</name><arguments>{tc_args}</arguments></tool_call>"
                    )
            else:
                conversation_parts.append(f"Assistant: {content}")

        elif role == "tool":
            # نتيجة تنفيذ أداة من OpenMinis
            tool_call_id = m.get("tool_call_id", "")
            tool_name    = m.get("name", tool_call_id)
            if isinstance(content, list):
                content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
            conversation_parts.append(
                f"[TOOL RESULT: {tool_name}]\n{content}\n[/TOOL RESULT]"
            )

        elif role == "function":
            # صيغة قديمة
            func_name = m.get("name", "function")
            if isinstance(content, list):
                content = str(content)
            conversation_parts.append(
                f"[TOOL RESULT: {func_name}]\n{content}\n[/TOOL RESULT]"
            )

    system_text       = "\n\n".join(system_parts)
    conversation_text = "\n\n".join(conversation_parts)
    return system_text, conversation_text


def _build_full_prompt(messages: List[Dict], tools: List[Dict]) -> str:
    """
    بناء البرومبت الكامل الذي يُرسل لـ Qwen
    """
    system_text, conversation_text = _messages_to_full_prompt(messages, tools)

    parts = []
    if system_text:
        parts.append(f"[SYSTEM]\n{system_text}\n[/SYSTEM]")
    if conversation_text:
        parts.append(conversation_text)
    parts.append("Assistant:")  # نطلب من Qwen الرد

    return "\n\n".join(parts)


# ══════════════════════════════════════════════════════════
# ★★★ تحليل رد Qwen: هل هو tool_call أم إجابة نهائية؟ ★★★
# ══════════════════════════════════════════════════════════

# نمط XML للأداة
_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<name>(.*?)</name>\s*<arguments>(.*?)</arguments>\s*</tool_call>",
    re.DOTALL | re.IGNORECASE,
)


def _parse_tool_call(text: str) -> Optional[Dict]:
    """
    إذا كان النص يحتوي على tool_call XML، استخرجه.
    يُعيد dict مع name وarguments، أو None إذا لم يكن tool call.
    """
    m = _TOOL_CALL_RE.search(text)
    if not m:
        return None

    name = m.group(1).strip()
    args_str = m.group(2).strip()

    # تحقق أن الـ arguments صالح JSON
    try:
        args_obj = json.loads(args_str)
    except json.JSONDecodeError:
        # حاول إصلاحه
        try:
            args_obj = json.loads(args_str.replace("'", '"'))
        except Exception:
            args_obj = {}

    return {
        "name": name,
        "arguments": json.dumps(args_obj, ensure_ascii=False),
    }


# ══════════════════════════════════════════════════════════
# Qwen Chat API
# ══════════════════════════════════════════════════════════

async def _create_chat(token: str, client: httpx.AsyncClient) -> str:
    url     = f"{BASE_QWEN_URL}/chats/new"
    payload = {"chat_mode": "normal", "project_id": ""}
    resp    = await client.post(url, json=payload, headers=_headers_new(token), timeout=60)
    data    = resp.json()
    cid = (
        data.get("chat_id")
        or data.get("id")
        or (data.get("data") or {}).get("chat_id")
        or (data.get("data") or {}).get("id")
    )
    if not cid:
        raise HTTPException(status_code=502, detail=f"Failed to create Qwen chat: {data}")
    log.info("Created qwen chat_id=%s", cid)
    return cid


def _build_message_payload(
    chat_id: str,
    prompt: str,
    *,
    parent_id: Optional[str],
    stream: bool = True,
    chat_type: str = "t2t",
    uploaded_files: Optional[List[Dict]] = None,
    thinking: bool = False,
    auto_search: bool = False,
    size: str = "1:1",
) -> Dict[str, Any]:
    ts  = int(time.time())
    fid = str(uuid.uuid4())

    msg: Dict[str, Any] = {
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
        "models":        [QWEN_MODEL_ID],
        "model":         "",
        "files":         uploaded_files or [],
        "user_action":   "chat",
        "extra":         {"meta": {"subChatType": chat_type}},
        "parentId":      parent_id,
        "parent_id":     parent_id,
    }

    return {
        "stream":                   stream,
        "incremental_output":       True,
        "chatId":                   chat_id,
        "chat_id":                  chat_id,
        "chat_mode":                "normal",
        "model":                    QWEN_MODEL_ID,
        "messages":                 [msg],
        "timestamp":                ts,
        "size":                     size,
        "share_id":                 "",
        "version":                  "2.1",
        "origin_branch_message_id": "",
        "parentId":                 parent_id or "",
        "parent_id":                parent_id,
    }


def _extract_response_id(obj: Dict) -> Optional[str]:
    rid = obj.get("response_id")
    if rid:
        return rid
    choices = obj.get("choices", [])
    if choices and isinstance(choices[0], dict):
        delta = choices[0].get("delta", {})
        rid = delta.get("response_id") or delta.get("id")
        if rid:
            return rid
    return None


# ══════════════════════════════════════════════════════════
# SSE Stream من Qwen + جمع الرد الكامل
# ══════════════════════════════════════════════════════════

async def _stream_from_qwen_collect_all(
    token: str,
    chat_id: str,
    payload: Dict,
    client: httpx.AsyncClient,
) -> Tuple[str, Optional[str]]:
    """
    يقرأ الـ stream من Qwen ويجمع الرد الكامل كنص.
    يُعيد (full_text, response_id).
    """
    url      = f"{BASE_QWEN_URL}/chat/completions"
    headers  = _headers_chat(token, stream=True)
    full_txt = ""
    resp_id: Optional[str] = None

    async with client.stream(
        "POST", url,
        json=payload,
        headers=headers,
        params={"chat_id": chat_id},
        timeout=REQUEST_TIMEOUT,
    ) as resp:
        async for raw_line in resp.aiter_lines():
            if not raw_line:
                continue
            if _is_antibot(raw_line):
                full_txt += "[BLOCKED: Qwen anti-bot triggered]"
                break
            if _is_rate_limited(raw_line):
                full_txt += "[ERROR: Rate limited]"
                break
            if not raw_line.startswith("data: "):
                continue
            data_str = raw_line[6:].strip()
            if data_str == "[DONE]":
                break
            try:
                obj = json.loads(data_str)
                if _is_rate_limited(obj):
                    full_txt += "[ERROR: Rate limited]"
                    break

                rid = _extract_response_id(obj)
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
            except (json.JSONDecodeError, KeyError):
                continue

    return full_txt, resp_id


# ══════════════════════════════════════════════════════════
# بناء ردود OpenAI-compatible
# ══════════════════════════════════════════════════════════

def _make_tool_call_response(tool_call: Dict, model: str) -> Dict:
    """رد بصيغة OpenAI عند استدعاء أداة."""
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
                        "name":      tool_call["name"],
                        "arguments": tool_call["arguments"],
                    }
                }]
            },
            "finish_reason": "tool_calls",  # ← هذا المهم!
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _make_text_response(content: str, model: str) -> Dict:
    """رد بصيغة OpenAI عند الإجابة النهائية."""
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


def _make_stream_chunk(content: str, model: str, *,
                       finish: bool = False,
                       tool_call: Optional[Dict] = None,
                       tool_call_index: int = 0) -> str:
    """SSE chunk لـ streaming mode."""
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"

    if tool_call and not content:
        # Streaming tool_calls: نرسل في عدة chunks كما تفعل OpenAI
        # chunk 1: tool_calls delta (function name + start of arguments)
        delta = {
            "tool_calls": [{
                "index": tool_call_index,
                "id":    f"call_{uuid.uuid4().hex[:24]}",
                "type":  "function",
                "function": {
                    "name":      tool_call["name"],
                    "arguments": tool_call["arguments"],
                }
            }]
        }
        choice = {"index": 0, "delta": delta, "finish_reason": None}
    elif finish and tool_call:
        # الـ chunk الأخير عند tool_call
        choice = {"index": 0, "delta": {}, "finish_reason": "tool_calls"}
    elif finish:
        choice = {"index": 0, "delta": {}, "finish_reason": "stop"}
    else:
        choice = {"index": 0, "delta": {"content": content}, "finish_reason": None}

    obj = {
        "id":      chunk_id,
        "object":  "chat.completion.chunk",
        "created": int(time.time()),
        "model":   model,
        "choices": [choice],
    }
    return f"data: {json.dumps(obj)}\n\n"


# ══════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════

@app.get("/", tags=["health"])
async def health():
    _evict_old_sessions()
    return {
        "status":          "ok",
        "proxy":           "Qwen OpenAI-Compatible Proxy",
        "version":         "4.0.0",
        "active_sessions": len(_sessions),
        "features":        ["tool_calls", "system_prompt", "full_history", "streaming"],
    }


@app.get("/v1/models", tags=["models"])
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": PROXY_MODEL_ID, "object": "model", "created": 1700000000, "owned_by": "qwen"},
            {"id": "qwen-vision",  "object": "model", "created": 1700000000, "owned_by": "qwen"},
        ],
    }


# ─── Chat Completions ─────────────────────────────────────
@app.post("/v1/chat/completions", tags=["chat"])
async def chat_completions(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    token     = _extract_token(authorization)
    body      = await request.json()
    messages  = body.get("messages", [])
    tools     = body.get("tools", [])         # ★ جديد: نقرأ الأدوات
    do_stream = body.get("stream", False)
    model     = body.get("model", PROXY_MODEL_ID)
    thinking    = bool(body.get("thinking", False))
    auto_search = bool(body.get("auto_search", False))

    # ── معرّف المحادثة ──────────────────────────────────
    conv_id = (
        body.get("conversation_id")
        or body.get("session_id")
        or request.headers.get("x-conversation-id")
        or request.headers.get("x-session-id")
    )
    if not conv_id:
        # اصنع معرّفاً ثابتاً من أول رسائل المحادثة (استثناء آخر رسالة)
        key_msgs = messages[:-1] if len(messages) > 1 else messages[:1]
        conv_id  = hashlib.md5(
            json.dumps(key_msgs, ensure_ascii=False).encode()
        ).hexdigest()

    log.info("conv_id=%s | messages=%d | tools=%d | stream=%s",
             conv_id, len(messages), len(tools), do_stream)

    # ★ بناء البرومبت الكامل (system + كل التاريخ + تعريفات الأدوات)
    full_prompt = _build_full_prompt(messages, tools)
    log.debug("Full prompt length: %d chars", len(full_prompt))

    if not full_prompt.strip():
        raise HTTPException(status_code=400, detail="No messages found.")

    _evict_old_sessions()

    # ── تحديد/إنشاء qwen_chat_id ──────────────────────
    sess = _get_session(token, conv_id)
    if sess:
        qwen_chat_id = sess["qwen_chat_id"]
        parent_id    = sess["parent_id"]
        log.info("Reusing qwen_chat_id=%s (parent=%s)", qwen_chat_id, parent_id)
    else:
        async with httpx.AsyncClient() as tmp:
            qwen_chat_id = await _create_chat(token, tmp)
        parent_id = None
        _set_session(token, conv_id, qwen_chat_id, parent_id)
        log.info("New session: conv_id=%s → qwen_chat_id=%s", conv_id, qwen_chat_id)

    payload = _build_message_payload(
        qwen_chat_id, full_prompt,
        parent_id=parent_id,
        stream=True,
        chat_type="t2t",
        thinking=thinking,
        auto_search=auto_search,
    )

    # ── جمع الرد الكامل من Qwen أولاً (نحتاج التحليل قبل الرد) ──
    async with httpx.AsyncClient() as client:
        qwen_text, last_rid = await _stream_from_qwen_collect_all(
            token, qwen_chat_id, payload, client
        )

    _update_parent(token, conv_id, last_rid)
    log.info("Qwen response length: %d chars | parent_id=%s", len(qwen_text), last_rid)

    # ★ التحليل: هل طلب Qwen أداة؟
    parsed_tool = _parse_tool_call(qwen_text)

    if parsed_tool:
        log.info("Tool call detected: %s", parsed_tool["name"])

        if do_stream:
            # streaming مع tool_calls
            async def tool_stream():
                yield _make_stream_chunk("", model, tool_call=parsed_tool)
                yield _make_stream_chunk("", model, finish=True, tool_call=parsed_tool)
                yield "data: [DONE]\n\n"
            return StreamingResponse(tool_stream(), media_type="text/event-stream")
        else:
            return JSONResponse(_make_tool_call_response(parsed_tool, model))

    else:
        # إجابة نصية عادية
        # نزيل أي XML artifact عشوائي قد يكون تسرّب
        clean_text = _TOOL_CALL_RE.sub("", qwen_text).strip()

        if do_stream:
            async def text_stream():
                # نرسل النص على شكل chunks صغيرة
                chunk_size = 20
                for i in range(0, len(clean_text), chunk_size):
                    yield _make_stream_chunk(clean_text[i:i+chunk_size], model)
                yield _make_stream_chunk("", model, finish=True)
                yield "data: [DONE]\n\n"
            return StreamingResponse(text_stream(), media_type="text/event-stream")
        else:
            return JSONResponse(_make_text_response(clean_text, model))


# ─── Image Generation ─────────────────────────────────────
@app.post("/v1/images/generations", tags=["images"])
async def image_generations(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    token  = _extract_token(authorization)
    body   = await request.json()
    prompt = body.get("prompt", "")
    size   = body.get("size", "1:1").replace("x", ":")

    if not prompt:
        raise HTTPException(status_code=400, detail="'prompt' is required.")

    async with httpx.AsyncClient() as client:
        qwen_chat_id = await _create_chat(token, client)
        payload = _build_message_payload(
            qwen_chat_id, prompt,
            parent_id=None,
            stream=True,
            chat_type="t2i",
            size=size,
        )

        image_url: Optional[str] = None
        async with client.stream(
            "POST", f"{BASE_QWEN_URL}/chat/completions",
            json=payload,
            headers=_headers_chat(token, stream=True),
            params={"chat_id": qwen_chat_id},
            timeout=300,
        ) as resp:
            async for raw_line in resp.aiter_lines():
                if not raw_line or not raw_line.startswith("data: "):
                    continue
                ds = raw_line[6:].strip()
                if ds == "[DONE]":
                    break
                if _is_antibot(raw_line) or _is_rate_limited(raw_line):
                    raise HTTPException(status_code=429, detail="Qwen blocked or rate-limited.")
                try:
                    obj     = json.loads(ds)
                    content = obj["choices"][0].get("delta", {}).get("content", "")
                    if content.startswith("http"):
                        image_url = content
                except Exception:
                    continue

    if not image_url:
        raise HTTPException(status_code=500, detail="No image URL returned by Qwen.")

    return JSONResponse({"created": int(time.time()), "data": [{"url": image_url}]})


# ─── Image Edits ──────────────────────────────────────────
def _oss_sig(secret, method, md5, ct, date, canon_hdr, canon_res):
    s2s    = f"{method}\n{md5}\n{ct}\n{date}\n{canon_hdr}{canon_res}"
    digest = hmac.new(secret.encode(), s2s.encode(), hashlib.sha1).digest()
    return base64.b64encode(digest).decode()


async def _upload_image(token, image_bytes, client):
    filename  = f"{uuid.uuid4()}_IMG.jpg"
    file_size = str(len(image_bytes))

    sts_resp = await client.post(
        "https://chat.qwen.ai/api/v2/files/getstsToken",
        json={"filename": filename, "filetype": "image", "filesize": file_size},
        headers=_headers_chat(token), timeout=60,
    )
    res = sts_resp.json()
    if _is_rate_limited(res) or "data" not in res:
        raise HTTPException(status_code=429, detail="Qwen rate-limited during OSS STS request.")

    d      = res["data"]
    aki    = d["access_key_id"]
    aks    = d["access_key_secret"]
    stkn   = d["security_token"]
    fpath  = d["file_path"]
    fid    = d["file_id"]
    bucket = d["bucketname"]
    host   = f"{bucket}.{d['endpoint']}"
    furl   = d.get("file_url", f"https://{host}/{fpath}")
    oss_ua  = "aliyun-sdk-android/2.9.21"
    c_hdr   = f"x-oss-security-token:{stkn}\n"

    def _oh(method, md5, ct, res, extra={}):
        gmt = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")
        sig = _oss_sig(aks, method, md5, ct, gmt, c_hdr, res)
        h   = {"Authorization": f"OSS {aki}:{sig}", "User-Agent": oss_ua,
               "Host": host, "x-oss-security-token": stkn, "Date": gmt, "Content-Type": ct}
        h.update(extra)
        return h

    init_r    = await client.post(f"https://{host}/{fpath}?uploads",
                                  headers=_oh("POST", "", "image/jpeg", f"/{bucket}/{fpath}?uploads", {"Content-Length": "0"}),
                                  timeout=60)
    upload_id = ET.fromstring(init_r.text).find("{*}UploadId").text

    cmd5   = base64.b64encode(hashlib.md5(image_bytes).digest()).decode()
    part_r = await client.put(
        f"https://{host}/{fpath}?uploadId={upload_id}&partNumber=1",
        content=image_bytes,
        headers=_oh("PUT", cmd5, "image/jpeg", f"/{bucket}/{fpath}?partNumber=1&uploadId={upload_id}",
                    {"Content-MD5": cmd5, "Content-Length": file_size}),
        timeout=OSS_UPLOAD_TIMEOUT,
    )
    etag = part_r.headers.get("ETag", "").replace('"', "")

    body = f"<CompleteMultipartUpload><Part><PartNumber>1</PartNumber><ETag>{etag}</ETag></Part></CompleteMultipartUpload>".encode()
    await client.post(
        f"https://{host}/{fpath}?uploadId={upload_id}",
        content=body,
        headers=_oh("POST", "", "image/jpeg", f"/{bucket}/{fpath}?uploadId={upload_id}",
                    {"Content-Length": str(len(body))}),
        timeout=60,
    )
    return {
        "type": "image",
        "file": {"data": {}, "filename": filename, "id": fid, "meta": {"name": filename}},
        "id": fid, "filename": filename, "name": filename,
        "url": furl, "image_width": 1024, "image_height": 1024,
    }


@app.post("/v1/images/edits", tags=["images"])
async def image_edits(
    request: Request,
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
        uploaded   = await _upload_image(token, image_bytes, client)
        file_entry = {
            "type": "image", "file": uploaded["file"],
            "id": uploaded["id"], "url": uploaded["url"],
            "name": uploaded["filename"],
            "image_width": 1024, "image_height": 1024,
        }
        qwen_chat_id = await _create_chat(token, client)
        payload = _build_message_payload(
            qwen_chat_id, prompt,
            parent_id=None,
            stream=True,
            chat_type="t2i",
            uploaded_files=[file_entry],
        )
        result_url: Optional[str] = None
        async with client.stream(
            "POST", f"{BASE_QWEN_URL}/chat/completions",
            json=payload,
            headers=_headers_chat(token, stream=True),
            params={"chat_id": qwen_chat_id},
            timeout=300,
        ) as resp:
            async for raw_line in resp.aiter_lines():
                if not raw_line or not raw_line.startswith("data: "):
                    continue
                ds = raw_line[6:].strip()
                if ds == "[DONE]":
                    break
                if _is_antibot(raw_line) or _is_rate_limited(raw_line):
                    raise HTTPException(status_code=429, detail="Qwen blocked or rate-limited.")
                try:
                    obj     = json.loads(ds)
                    content = obj["choices"][0].get("delta", {}).get("content", "")
                    if content.startswith("http"):
                        result_url = content
                except Exception:
                    continue

    if not result_url:
        raise HTTPException(status_code=500, detail="No edited image URL returned by Qwen.")

    return JSONResponse({"created": int(time.time()), "data": [{"url": result_url}]})


# ══════════════════════════════════════════════════════════
# Error handlers
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
# Entry point
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    log.info("Starting Qwen Proxy v4.0 on port %d", port)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

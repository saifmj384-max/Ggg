"""
Qwen → OpenAI-Compatible Proxy  v3.0
======================================
الميزة الرئيسية: ربط كل محادثة OpenMinis بمحادثة Qwen ثابتة.
  - أول رسالة في محادثة جديدة  → ينشئ chat_id جديد في Qwen ويحفظه
  - الرسائل التالية في نفس المحادثة → يُرسل للـ chat_id المحفوظ + parent_id صحيح
  - المفتاح هو "token:conversation_id" (token لعزل مستخدمين مختلفين)

Endpoints:
  GET  /                       health check
  GET  /v1/models              قائمة النماذج
  POST /v1/chat/completions    محادثة نصية  (streaming + non-streaming)
  POST /v1/images/generations  توليد صورة
  POST /v1/images/edits        تعديل صورة

Deploy on Railway:
  - PORT يُضخّ تلقائياً من Railway
  - Authorization: Bearer <QWEN_TOKEN>
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
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

# Session TTL: نحذف الجلسات التي لم تُستخدم أكثر من هذا الوقت (بالثواني)
SESSION_TTL = 60 * 60 * 6   # 6 ساعات

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("qwen_proxy")

# ══════════════════════════════════════════════════════════
# Session Store  (in-memory)
# ══════════════════════════════════════════════════════════
# key  → "token_prefix:conv_id"
# val  → { "qwen_chat_id": str, "parent_id": str|None, "last_used": float }
_sessions: Dict[str, Dict[str, Any]] = {}


def _session_key(token: str, conv_id: str) -> str:
    # نستخدم أول 16 حرف من التوكن فقط كـ namespace (لا نحفظ التوكن كاملاً)
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
    """حذف الجلسات القديمة من الذاكرة."""
    now    = time.time()
    stale  = [k for k, v in _sessions.items() if now - v["last_used"] > SESSION_TTL]
    for k in stale:
        del _sessions[k]
    if stale:
        log.info("Evicted %d stale sessions", len(stale))


# ══════════════════════════════════════════════════════════
# App
# ══════════════════════════════════════════════════════════
app = FastAPI(
    title="Qwen OpenAI-Compatible Proxy",
    version="3.0.0",
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
    """Headers لطلبات المحادثة والرسائل."""
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
    """Headers لإنشاء محادثة جديدة."""
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
# Qwen Chat API
# ══════════════════════════════════════════════════════════
async def _create_chat(token: str, client: httpx.AsyncClient) -> str:
    url     = f"{BASE_QWEN_URL}/chats/new"
    payload = {"chat_mode": "normal", "project_id": ""}
    resp    = await client.post(url, json=payload, headers=_headers_new(token), timeout=60)
    data    = resp.json()
    # Qwen يُعيد chat_id في أماكن مختلفة حسب الإصدار
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


def _extract_last_message_content(messages: List[Dict[str, Any]]) -> str:
    """استخرج آخر رسالة user فقط (بدون تاريخ المحادثة — Qwen يحتفظ بالتاريخ بنفسه)."""
    # نبحث من الآخر عن آخر رسالة user
    for m in reversed(messages):
        if m.get("role") in ("user", "human"):
            content = m.get("content") or ""
            if isinstance(content, list):
                # vision content
                text_parts = [
                    c.get("text", "")
                    for c in content
                    if isinstance(c, dict) and c.get("type") == "text"
                ]
                return " ".join(text_parts).strip()
            return str(content).strip()
    # لم نجد user message → نجمع كل شيء
    parts = []
    for m in messages:
        role    = m.get("role", "user")
        content = m.get("content") or ""
        if isinstance(content, list):
            content = " ".join(c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text")
        if role == "system":
            parts.insert(0, f"[System]: {content}")
        else:
            parts.append(str(content))
    return "\n".join(parts).strip()


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
    """بناء الـ payload الكامل بما فيها parent_id لربط سلسلة الرسائل."""
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
        "files":         [],
        "user_action":   "chat",
        "extra":         {"meta": {"subChatType": chat_type}},
        "parentId":      parent_id,
        "parent_id":     parent_id,
    }

    if uploaded_files:
        msg["files"] = uploaded_files

    payload: Dict[str, Any] = {
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

    return payload


# ══════════════════════════════════════════════════════════
# SSE parsing + response_id استخراج
# ══════════════════════════════════════════════════════════
def _make_chunk(content: str, model: str, *, finish: bool = False) -> str:
    return f"data: {json.dumps({'id': f'chatcmpl-{uuid.uuid4().hex}', 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': model, 'choices': [{'index': 0, 'delta': {'content': content} if content else {}, 'finish_reason': 'stop' if finish else None}]})}\n\n"


def _make_full_response(content: str, model: str) -> Dict:
    return {
        "id":      f"chatcmpl-{uuid.uuid4().hex}",
        "object":  "chat.completion",
        "created": int(time.time()),
        "model":   model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage":   {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _extract_response_id(obj: Dict) -> Optional[str]:
    """استخرج response_id من رد Qwen لاستخدامه كـ parent_id للرسالة التالية."""
    # الشكل الأكثر شيوعاً
    rid = obj.get("response_id")
    if rid:
        return rid
    # nested
    created = obj.get("response.created")
    if isinstance(created, dict):
        rid = created.get("response_id")
        if rid:
            return rid
    # في بعض الإصدارات يكون داخل choices
    choices = obj.get("choices", [])
    if choices and isinstance(choices[0], dict):
        delta = choices[0].get("delta", {})
        rid = delta.get("response_id") or delta.get("id")
        if rid:
            return rid
    return None


async def _stream_and_collect(
    token: str,
    chat_id: str,
    payload: Dict,
    model_name: str,
    client: httpx.AsyncClient,
) -> AsyncIterator[Tuple[str, Optional[str]]]:
    """
    Yield (sse_chunk, response_id|None).
    response_id يكون غير None فقط في آخر chunk يحمله.
    """
    url         = f"{BASE_QWEN_URL}/chat/completions"
    headers     = _headers_chat(token, stream=True)
    response_id: Optional[str] = None

    try:
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
                    yield _make_chunk("[BLOCKED: Qwen anti-bot triggered]", model_name), None
                    break
                if _is_rate_limited(raw_line):
                    yield _make_chunk("[ERROR: Rate limited]", model_name), None
                    break
                if not raw_line.startswith("data: "):
                    continue
                data_str = raw_line[6:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    obj = json.loads(data_str)
                    if _is_rate_limited(obj):
                        yield _make_chunk("[ERROR: Rate limited]", model_name), None
                        break

                    # محاولة استخراج response_id
                    rid = _extract_response_id(obj)
                    if rid:
                        response_id = rid

                    choices = obj.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})

                    # تجاهل مراحل التفكير والبحث
                    phase = delta.get("phase", "")
                    if phase and phase not in ("answer", ""):
                        continue

                    content = delta.get("content", "")
                    if content:
                        yield _make_chunk(content, model_name), None
                except (json.JSONDecodeError, KeyError):
                    continue

    except Exception as exc:
        log.error("Stream error: %s", exc)
        yield _make_chunk(f"[ERROR: {exc}]", model_name), None

    yield _make_chunk("", model_name, finish=True), response_id
    yield "data: [DONE]\n\n", None


# ══════════════════════════════════════════════════════════
# OSS Image Upload
# ══════════════════════════════════════════════════════════
def _oss_sig(secret: str, method: str, md5: str, ct: str,
             date: str, canon_hdr: str, canon_res: str) -> str:
    s2s    = f"{method}\n{md5}\n{ct}\n{date}\n{canon_hdr}{canon_res}"
    digest = hmac.new(secret.encode(), s2s.encode(), hashlib.sha1).digest()
    return base64.b64encode(digest).decode()


async def _upload_image(token: str, image_bytes: bytes,
                        client: httpx.AsyncClient) -> Dict:
    filename  = f"{uuid.uuid4()}_IMG.jpg"
    file_size = str(len(image_bytes))

    sts_resp = await client.post(
        "https://chat.qwen.ai/api/v2/files/getstsToken",
        json={"filename": filename, "filetype": "image", "filesize": file_size},
        headers=_headers_chat(token),
        timeout=60,
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

    def _oh(method: str, md5: str, ct: str, res: str, extra: Dict = {}) -> Dict:
        gmt = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")
        sig = _oss_sig(aks, method, md5, ct, gmt, c_hdr, res)
        h   = {"Authorization": f"OSS {aki}:{sig}", "User-Agent": oss_ua,
               "Host": host, "x-oss-security-token": stkn, "Date": gmt, "Content-Type": ct}
        h.update(extra)
        return h

    # Initiate
    init_r = await client.post(f"https://{host}/{fpath}?uploads",
                               headers=_oh("POST", "", "image/jpeg", f"/{bucket}/{fpath}?uploads", {"Content-Length": "0"}),
                               timeout=60)
    upload_id = ET.fromstring(init_r.text).find("{*}UploadId").text

    # Upload part
    cmd5      = base64.b64encode(hashlib.md5(image_bytes).digest()).decode()
    part_r    = await client.put(
        f"https://{host}/{fpath}?uploadId={upload_id}&partNumber=1",
        content=image_bytes,
        headers=_oh("PUT", cmd5, "image/jpeg", f"/{bucket}/{fpath}?partNumber=1&uploadId={upload_id}",
                    {"Content-MD5": cmd5, "Content-Length": file_size}),
        timeout=OSS_UPLOAD_TIMEOUT,
    )
    etag = part_r.headers.get("ETag", "").replace('"', "")

    # Complete
    body = f"<CompleteMultipartUpload><Part><PartNumber>1</PartNumber><ETag>{etag}</ETag></Part></CompleteMultipartUpload>".encode()
    await client.post(
        f"https://{host}/{fpath}?uploadId={upload_id}",
        content=body,
        headers=_oh("POST", "", "image/jpeg", f"/{bucket}/{fpath}?uploadId={upload_id}",
                    {"Content-Length": str(len(body))}),
        timeout=60,
    )

    log.info("Uploaded image file_id=%s", fid)
    return {
        "type": "image",
        "file": {"data": {}, "filename": filename, "id": fid, "meta": {"name": filename}},
        "id": fid, "filename": filename, "name": filename,
        "url": furl, "image_width": 1024, "image_height": 1024,
    }


# ══════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════

@app.get("/", tags=["health"])
async def health():
    _evict_old_sessions()
    return {
        "status":       "ok",
        "proxy":        "Qwen OpenAI-Compatible Proxy",
        "version":      "3.0.0",
        "active_sessions": len(_sessions),
    }


@app.get("/v1/models", tags=["models"])
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": PROXY_MODEL_ID,  "object": "model", "created": 1700000000, "owned_by": "qwen"},
            {"id": "qwen-vision",   "object": "model", "created": 1700000000, "owned_by": "qwen"},
        ],
    }


# ─── Chat Completions ─────────────────────────────────────
@app.post("/v1/chat/completions", tags=["chat"])
async def chat_completions(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    token    = _extract_token(authorization)
    body     = await request.json()
    messages = body.get("messages", [])
    do_stream = body.get("stream", False)
    model    = body.get("model", PROXY_MODEL_ID)
    thinking    = bool(body.get("thinking", False))
    auto_search = bool(body.get("auto_search", False))

    # ── معرّف المحادثة من OpenMinis ──────────────────────
    # OpenMinis يُرسل conversation_id في أحد هذه الأماكن:
    conv_id = (
        body.get("conversation_id")
        or body.get("session_id")
        or request.headers.get("x-conversation-id")
        or request.headers.get("x-session-id")
    )
    # إذا لم يُرسل conv_id → ننشئ واحداً من hash آخر رسالة
    # (هذا يضمن أن نفس المحادثة لا تُنشئ chat_ids متعددة)
    if not conv_id:
        conv_id = hashlib.md5(
            json.dumps(messages[:-1], ensure_ascii=False).encode()
        ).hexdigest() if len(messages) > 1 else str(uuid.uuid4())

    # استخرج آخر رسالة user فقط (Qwen يحتفظ بالتاريخ)
    prompt = _extract_last_message_content(messages)
    if not prompt:
        raise HTTPException(status_code=400, detail="No user message found.")

    _evict_old_sessions()

    # ── تحديد qwen_chat_id (client مؤقت فقط لإنشاء chat جديد إن لزم) ──
    sess = _get_session(token, conv_id)
    if sess:
        qwen_chat_id = sess["qwen_chat_id"]
        parent_id    = sess["parent_id"]
        log.info("Reusing qwen_chat_id=%s for conv_id=%s (parent=%s)",
                 qwen_chat_id, conv_id, parent_id)
    else:
        async with httpx.AsyncClient() as tmp:
            qwen_chat_id = await _create_chat(token, tmp)
        parent_id = None
        _set_session(token, conv_id, qwen_chat_id, parent_id)
        log.info("New session: conv_id=%s → qwen_chat_id=%s", conv_id, qwen_chat_id)

    payload = _build_message_payload(
        qwen_chat_id, prompt,
        parent_id=parent_id,
        stream=True,
        chat_type="t2t",
        thinking=thinking,
        auto_search=auto_search,
    )

    if do_stream:
        # ✅ client يُنشأ داخل generator ويبقى حياً طوال الـ stream
        async def event_stream():
            last_rid: Optional[str] = None
            async with httpx.AsyncClient() as stream_client:
                async for chunk, rid in _stream_and_collect(
                    token, qwen_chat_id, payload, model, stream_client
                ):
                    yield chunk
                    if rid:
                        last_rid = rid
            _update_parent(token, conv_id, last_rid)
            log.info("Updated parent_id=%s for conv_id=%s", last_rid, conv_id)

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # ── Non-streaming ──
    full_content = ""
    last_rid: Optional[str] = None
    async with httpx.AsyncClient() as client:
        async for chunk, rid in _stream_and_collect(
            token, qwen_chat_id, payload, model, client
        ):
            if rid:
                last_rid = rid
            if chunk.startswith("data: {"):
                try:
                    obj = json.loads(chunk[6:])
                    c   = obj["choices"][0]["delta"].get("content", "")
                    full_content += c
                except Exception:
                    pass

    _update_parent(token, conv_id, last_rid)
    return JSONResponse(_make_full_response(full_content, model))


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
@app.post("/v1/images/edits", tags=["images"])
async def image_edits(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    token        = _extract_token(authorization)
    image_bytes: Optional[bytes] = None
    prompt       = ""

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
    log.info("Starting Qwen Proxy v3.0 on port %d", port)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

import json
import time
import uuid
import traceback
import os
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse

BASE_QWEN_URL = "https://chat.qwen.ai/api/v2"
QWEN_MODEL = "qwen3.8-max"

app = FastAPI(title="Qwen OpenAI-Compatible Proxy")

def get_base_headers(content_type: Optional[str] = "application/json") -> Dict[str, str]:
    headers = {
        "User-Agent": "Dalvik/2.1.0 (Linux; U; Android 16; CPH2631 Build/BP2A.250605.015) AliApp(QWENCHAT/2.7.2) AppType/Release AplusBridgeLite",
        "Accept": "*/*,text/event-stream",
        "Origin": "https://chat.qwen.ai",
        "Referer": "https://chat.qwen.ai/",
    }
    if content_type:
        headers["Content-Type"] = content_type
    return headers

def extract_bearer_token(authorization: Optional[str]) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header.")
    token = authorization.split(" ", 1)[1].strip() if " " in authorization else authorization.strip()
    return token

def qwen_headers(token: str) -> Dict[str, str]:
    headers = get_base_headers()
    headers["Authorization"] = f"Bearer {token}"
    headers["x-request-id"] = str(uuid.uuid4())
    return headers

def is_rate_limited_response(obj: Any) -> bool:
    if isinstance(obj, str):
        return "RateLimited" in obj
    return False

def create_new_chat_with_token(token: str) -> str:
    url = f"{BASE_QWEN_URL}/chats/new"
    payload = {"chat_mode": "normal", "project_id": ""}
    r = requests.post(url, json=payload, headers=qwen_headers(token), timeout=60)
    data = r.json()
    if "data" not in data or "id" not in data["data"]:
        raise HTTPException(status_code=502, detail="Failed to create Qwen chat")
    return data["data"]["id"]

def messages_to_prompt(messages: List[Dict[str, Any]]) -> str:
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        parts.append(f"{role}: {content}")
    return "\n".join(parts).strip()

@app.post("/v1/chat/completions")
async def chat_completions(request: Request, authorization: Optional[str] = Header(None)):
    token = extract_bearer_token(authorization)
    body = await request.json()
    messages = body.get("messages", [])
    prompt = messages_to_prompt(messages)
    chat_id = create_new_chat_with_token(token)
    
    url = f"{BASE_QWEN_URL}/chat/completions?chat_id={chat_id}"
    headers = qwen_headers(token)
    headers["Accept"] = "text/event-stream"
    
    payload = {
        "stream": True,
        "chat_id": chat_id,
        "model": QWEN_MODEL,
        "messages": [{"role": "user", "content": prompt, "feature_config": {"thinking_enabled": False, "auto_search": False}}],
    }

    def gen():
        try:
            with requests.post(url, json=payload, headers=headers, stream=True, timeout=180) as resp:
                for line in resp.iter_lines():
                    if not line: continue
                    line_str = line.decode("utf-8", "ignore")
                    if "_____tmd_____" in line_str or "punish" in line_str:
                        yield "data: [ERROR] Qwen Anti-Bot triggered\n\n"
                        break
                    if line_str.startswith("data: "):
                        data_content = line_str[6:].strip()
                        if data_content == "[DONE]": break
                        try:
                            data_json = json.loads(data_content)
                            content = data_json.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if content:
                                yield f"data: {json.dumps({'choices': [{'delta': {'content': content}}]})}\n\n"
                        except: continue
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"
        
    return StreamingResponse(gen(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)

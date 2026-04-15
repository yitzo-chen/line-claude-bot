import os
import requests
import base64
from flask import Flask, request, abort
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    ReplyMessageRequest,
    TextMessage,
)
from linebot.v3.webhooks import (
    MessageEvent,
    TextMessageContent,
    ImageMessageContent,
    FileMessageContent,
)
from groq import Groq

app = Flask(__name__)

# 環境變數
LINE_CHANNEL_ACCESS_TOKEN = os.environ["LINE_CHANNEL_ACCESS_TOKEN"]
LINE_CHANNEL_SECRET = os.environ["LINE_CHANNEL_SECRET"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]

configuration = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)
groq_client = Groq(api_key=GROQ_API_KEY)

TEXT_MODEL = "llama-3.3-70b-versatile"
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# 每個使用者的對話歷史（簡易版，重啟後清空）
conversation_history: dict[str, list] = {}


def get_line_file(message_id: str) -> bytes:
    """從 LINE 下載圖片或檔案內容"""
    url = f"https://api-data.line.me/v2/bot/message/{message_id}/content"
    headers = {"Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.content


def reply(reply_token: str, text: str):
    """回覆訊息給使用者"""
    with ApiClient(configuration) as api_client:
        line_api = MessagingApi(api_client)
        line_api.reply_message(
            ReplyMessageRequest(
                reply_token=reply_token,
                messages=[TextMessage(text=text[:5000])],
            )
        )


def ask_groq(user_id: str, messages: list, model: str = TEXT_MODEL) -> str:
    """呼叫 Groq API，維持對話歷史"""
    history = conversation_history.setdefault(user_id, [])

    if model == VISION_MODEL:
        # 視覺模型：不帶歷史（無法重送圖片），直接呼叫
        response = groq_client.chat.completions.create(
            model=model,
            max_tokens=2048,
            messages=messages,
        )
        assistant_text = response.choices[0].message.content
        # 以文字佔位符存入歷史，避免 list content 污染後續文字對話
        history.append({"role": "user", "content": "[傳送了一張圖片]"})
        history.append({"role": "assistant", "content": assistant_text})
        return assistant_text

    # 文字模型：帶入歷史
    history.extend(messages)
    if len(history) > 20:
        history = history[-20:]
        conversation_history[user_id] = history

    # 過濾掉歷史中殘留的 list content（例如舊圖片訊息）
    api_messages = []
    for m in history:
        if isinstance(m.get("content"), list):
            text = " ".join(
                p.get("text", "") for p in m["content"]
                if isinstance(p, dict) and p.get("type") == "text"
            ) or "[圖片]"
            api_messages.append({"role": m["role"], "content": text})
        else:
            api_messages.append(m)

    response = groq_client.chat.completions.create(
        model=model,
        max_tokens=2048,
        messages=api_messages,
    )
    assistant_text = response.choices[0].message.content
    history.append({"role": "assistant", "content": assistant_text})
    return assistant_text


@app.route("/health")
def health():
    return "OK"


@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return "OK"


@handler.add(MessageEvent, message=TextMessageContent)
def handle_text(event: MessageEvent):
    user_id = event.source.user_id
    text = event.message.text.strip()

    if text == "/myid":
        reply(event.reply_token, f"你的 LINE User ID：\n{user_id}")
        return

    if text in ("/reset", "重置", "清除對話"):
        conversation_history.pop(user_id, None)
        reply(event.reply_token, "對話已重置。")
        return

    try:
        messages = [{"role": "user", "content": text}]
        answer = ask_groq(user_id, messages)
        reply(event.reply_token, answer)
    except Exception as e:
        reply(event.reply_token, f"⚠️ 發生錯誤，請稍後再試。\n({type(e).__name__})")


@handler.add(MessageEvent, message=ImageMessageContent)
def handle_image(event: MessageEvent):
    user_id = event.source.user_id
    try:
        image_bytes = get_line_file(event.message.id)
        # 限制圖片大小（5MB）
        if len(image_bytes) > 5 * 1024 * 1024:
            reply(event.reply_token, "⚠️ 圖片過大（超過 5MB），請壓縮後再傳。")
            return
        b64 = base64.standard_b64encode(image_bytes).decode()
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                    {"type": "text", "text": "請分析這張圖片，用繁體中文說明內容。"},
                ],
            }
        ]
        answer = ask_groq(user_id, messages, model=VISION_MODEL)
        reply(event.reply_token, answer)
    except Exception as e:
        reply(event.reply_token, f"⚠️ 圖片處理失敗，請稍後再試。\n({type(e).__name__})")


@handler.add(MessageEvent, message=FileMessageContent)
def handle_file(event: MessageEvent):
    user_id = event.source.user_id
    file_name = event.message.file_name or "file"
    try:
        file_bytes = get_line_file(event.message.id)
        ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""

        if ext in ("txt", "csv", "md", "json", "xml", "py", "js", "ts", "html", "css"):
            try:
                text_content = file_bytes.decode("utf-8", errors="replace")
            except Exception:
                text_content = file_bytes.decode("big5", errors="replace")
            # 限制文字長度（約 8000 字元）
            if len(text_content) > 8000:
                text_content = text_content[:8000] + "\n...(內容過長，已截斷)"
            messages = [
                {
                    "role": "user",
                    "content": f"以下是檔案 `{file_name}` 的內容：\n\n{text_content}\n\n請用繁體中文分析並摘要重點。",
                }
            ]
            answer = ask_groq(user_id, messages)
            reply(event.reply_token, answer)
        else:
            reply(event.reply_token, f"目前不支援 .{ext} 格式，支援：圖片、txt、csv、md、json 等文字類型檔案。")
    except Exception as e:
        reply(event.reply_token, f"⚠️ 檔案處理失敗，請稍後再試。\n({type(e).__name__})")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

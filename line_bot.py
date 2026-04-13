import os
import base64
import requests
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

LINE_CHANNEL_ACCESS_TOKEN = os.environ["LINE_CHANNEL_ACCESS_TOKEN"]
LINE_CHANNEL_SECRET = os.environ["LINE_CHANNEL_SECRET"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]

configuration = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)
groq_client = Groq(api_key=GROQ_API_KEY)

TEXT_MODEL = "llama-3.3-70b-versatile"
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# 每個使用者的對話歷史
conversation_history: dict[str, list] = {}


def get_line_file(message_id: str) -> bytes:
    url = f"https://api-data.line.me/v2/bot/message/{message_id}/content"
    headers = {"Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.content


def reply(reply_token: str, text: str):
    with ApiClient(configuration) as api_client:
        line_api = MessagingApi(api_client)
        line_api.reply_message(
            ReplyMessageRequest(
                reply_token=reply_token,
                messages=[TextMessage(text=text[:5000])],
            )
        )


def ask_groq_text(user_id: str, text: str) -> str:
    history = conversation_history.setdefault(user_id, [
        {"role": "system", "content": "你是一個友善的 AI 助理，請用繁體中文回覆。"}
    ])
    history.append({"role": "user", "content": text})
    if len(history) > 21:  # system + 20 messages
        history = [history[0]] + history[-20:]
        conversation_history[user_id] = history

    response = groq_client.chat.completions.create(
        model=TEXT_MODEL,
        messages=history,
        max_tokens=2048,
    )
    answer = response.choices[0].message.content
    history.append({"role": "assistant", "content": answer})
    return answer


def ask_groq_vision(image_bytes: bytes, prompt: str) -> str:
    b64 = base64.standard_b64encode(image_bytes).decode()
    response = groq_client.chat.completions.create(
        model=VISION_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": prompt},
            ]
        }],
        max_tokens=2048,
    )
    return response.choices[0].message.content


@app.route("/callback", methods=["GET", "POST"])
def callback():
    if request.method == "GET":
        return "OK"
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
    text = event.message.text

    if text.strip() in ("/reset", "重置", "清除對話"):
        conversation_history.pop(user_id, None)
        reply(event.reply_token, "對話已重置。")
        return

    try:
        answer = ask_groq_text(user_id, text)
        reply(event.reply_token, answer)
    except Exception as e:
        reply(event.reply_token, f"[錯誤] {type(e).__name__}: {str(e)[:200]}")


@handler.add(MessageEvent, message=ImageMessageContent)
def handle_image(event: MessageEvent):
    try:
        image_bytes = get_line_file(event.message.id)
        answer = ask_groq_vision(image_bytes, "請分析這張圖片，用繁體中文說明內容。")
        reply(event.reply_token, answer)
    except Exception as e:
        reply(event.reply_token, f"[錯誤] {type(e).__name__}: {str(e)[:200]}")


@handler.add(MessageEvent, message=FileMessageContent)
def handle_file(event: MessageEvent):
    try:
        user_id = event.source.user_id
        file_name = event.message.file_name or "file"
        file_bytes = get_line_file(event.message.id)
        ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""

        if ext in ("txt", "csv", "md", "json", "xml", "py", "js", "ts", "html", "css"):
            try:
                text_content = file_bytes.decode("utf-8", errors="replace")
            except Exception:
                text_content = file_bytes.decode("big5", errors="replace")
            answer = ask_groq_text(user_id, f"以下是檔案 `{file_name}` 的內容：\n\n{text_content[:8000]}\n\n請用繁體中文分析並摘要重點。")
            reply(event.reply_token, answer)
        else:
            reply(event.reply_token, f"目前不支援 .{ext} 格式，支援：圖片、txt、csv、md、json 等文字類型檔案。")
    except Exception as e:
        reply(event.reply_token, f"[錯誤] {type(e).__name__}: {str(e)[:200]}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

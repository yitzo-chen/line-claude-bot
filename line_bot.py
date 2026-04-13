import os
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
import google.generativeai as genai
import base64

app = Flask(__name__)

# 環境變數
LINE_CHANNEL_ACCESS_TOKEN = os.environ["LINE_CHANNEL_ACCESS_TOKEN"]
LINE_CHANNEL_SECRET = os.environ["LINE_CHANNEL_SECRET"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

configuration = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

# 每個使用者的對話歷史（重啟後清空）
chat_sessions: dict[str, any] = {}


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


def get_chat(user_id: str):
    """取得或建立使用者的 chat session"""
    if user_id not in chat_sessions:
        chat_sessions[user_id] = model.start_chat(history=[])
    return chat_sessions[user_id]


def ask_gemini_text(user_id: str, text: str) -> str:
    """文字對話，維持歷史"""
    chat = get_chat(user_id)
    response = chat.send_message(text)
    return response.text


def ask_gemini_once(parts: list) -> str:
    """單次問答（圖片/PDF），不維持歷史"""
    response = model.generate_content(parts)
    return response.text


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
        chat_sessions.pop(user_id, None)
        reply(event.reply_token, "對話已重置。")
        return

    answer = ask_gemini_text(user_id, text)
    reply(event.reply_token, answer)


@handler.add(MessageEvent, message=ImageMessageContent)
def handle_image(event: MessageEvent):
    image_bytes = get_line_file(event.message.id)
    parts = [
        {"mime_type": "image/jpeg", "data": base64.standard_b64encode(image_bytes).decode()},
        "請分析這張圖片，用繁體中文說明內容。",
    ]
    # Gemini 需要用 inline_data 格式
    import google.generativeai as genai2
    img_part = genai2.protos.Part(
        inline_data=genai2.protos.Blob(
            mime_type="image/jpeg",
            data=image_bytes,
        )
    )
    response = model.generate_content([img_part, "請分析這張圖片，用繁體中文說明內容。"])
    reply(event.reply_token, response.text)


@handler.add(MessageEvent, message=FileMessageContent)
def handle_file(event: MessageEvent):
    user_id = event.source.user_id
    file_name = event.message.file_name or "file"
    file_bytes = get_line_file(event.message.id)
    ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""

    if ext == "pdf":
        import google.generativeai as genai2
        pdf_part = genai2.protos.Part(
            inline_data=genai2.protos.Blob(
                mime_type="application/pdf",
                data=file_bytes,
            )
        )
        response = model.generate_content([pdf_part, f"請閱讀這份 PDF（{file_name}），用繁體中文摘要重點內容。"])
        reply(event.reply_token, response.text)

    elif ext in ("txt", "csv", "md", "json", "xml", "py", "js", "ts", "html", "css"):
        try:
            text_content = file_bytes.decode("utf-8", errors="replace")
        except Exception:
            text_content = file_bytes.decode("big5", errors="replace")
        answer = ask_gemini_text(user_id, f"以下是檔案 `{file_name}` 的內容：\n\n{text_content}\n\n請用繁體中文分析並摘要重點。")
        reply(event.reply_token, answer)

    else:
        reply(event.reply_token, f"目前不支援 .{ext} 格式，支援：PDF、圖片、txt、csv、md、json 等文字類型檔案。")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

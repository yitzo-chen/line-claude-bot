import os
import tempfile
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
import anthropic
import base64

app = Flask(__name__)

# 環境變數
LINE_CHANNEL_ACCESS_TOKEN = os.environ["LINE_CHANNEL_ACCESS_TOKEN"]
LINE_CHANNEL_SECRET = os.environ["LINE_CHANNEL_SECRET"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]

configuration = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)
claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

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
                messages=[TextMessage(text=text[:5000])],  # LINE 單則上限 5000 字
            )
        )


def ask_claude(user_id: str, messages: list) -> str:
    """呼叫 Claude API，維持對話歷史"""
    history = conversation_history.setdefault(user_id, [])
    history.extend(messages)

    # 保留最近 20 則，避免 token 過多
    if len(history) > 20:
        history = history[-20:]
        conversation_history[user_id] = history

    response = claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        messages=history,
    )
    assistant_text = response.content[0].text
    history.append({"role": "assistant", "content": assistant_text})
    return assistant_text


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
    text = event.message.text

    # 清除對話歷史指令
    if text.strip() in ("/reset", "重置", "清除對話"):
        conversation_history.pop(user_id, None)
        reply(event.reply_token, "對話已重置。")
        return

    messages = [{"role": "user", "content": text}]
    answer = ask_claude(user_id, messages)
    reply(event.reply_token, answer)


@handler.add(MessageEvent, message=ImageMessageContent)
def handle_image(event: MessageEvent):
    user_id = event.source.user_id
    image_bytes = get_line_file(event.message.id)
    b64 = base64.standard_b64encode(image_bytes).decode()

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": b64,
                    },
                },
                {"type": "text", "text": "請分析這張圖片，用繁體中文說明內容。"},
            ],
        }
    ]
    answer = ask_claude(user_id, messages)
    reply(event.reply_token, answer)


@handler.add(MessageEvent, message=FileMessageContent)
def handle_file(event: MessageEvent):
    user_id = event.source.user_id
    file_name = event.message.file_name or "file"
    file_bytes = get_line_file(event.message.id)

    ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""

    if ext == "pdf":
        # PDF：用 base64 傳給 Claude
        b64 = base64.standard_b64encode(file_bytes).decode()
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": f"請閱讀這份 PDF（{file_name}），用繁體中文摘要重點內容。"},
                ],
            }
        ]
    elif ext in ("txt", "csv", "md", "json", "xml", "py", "js", "ts", "html", "css"):
        # 純文字類型：直接讀取內容
        try:
            text_content = file_bytes.decode("utf-8", errors="replace")
        except Exception:
            text_content = file_bytes.decode("big5", errors="replace")
        messages = [
            {
                "role": "user",
                "content": f"以下是檔案 `{file_name}` 的內容：\n\n{text_content}\n\n請用繁體中文分析並摘要重點。",
            }
        ]
    else:
        reply(event.reply_token, f"目前不支援 .{ext} 格式，支援：PDF、圖片、txt、csv、md、json 等文字類型檔案。")
        return

    answer = ask_claude(user_id, messages)
    reply(event.reply_token, answer)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

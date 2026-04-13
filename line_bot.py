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
    AudioMessageContent,
)
from groq import Groq
import pypdf
import io

app = Flask(__name__)

LINE_CHANNEL_ACCESS_TOKEN = os.environ["LINE_CHANNEL_ACCESS_TOKEN"]
LINE_CHANNEL_SECRET = os.environ["LINE_CHANNEL_SECRET"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]

configuration = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)
groq_client = Groq(api_key=GROQ_API_KEY)

TEXT_MODEL = "llama-3.3-70b-versatile"
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
WHISPER_MODEL = "whisper-large-v3-turbo"

conversation_history: dict[str, list] = {}
known_users: set[str] = {}  # 記錄所有傳過訊息的 user_id

HELP_TEXT = """🤖 AI 助理指令說明

【對話】
直接傳訊息即可對話，有上下文記憶。

【特殊指令】
/reset 或 重置 — 清除對話記憶
/翻譯 [文字] — 翻譯成中文
/英文 [文字] — 翻譯成英文
/摘要 [文字] — 摘要重點
/改寫 [文字] — 潤飾文字
/help — 顯示此說明

【傳送檔案】
🖼️ 圖片 — AI 分析說明
🎤 語音 — 自動轉文字
📄 PDF — 摘要重點
📝 txt/csv/md/json 等 — 分析內容"""


def get_line_file(message_id: str) -> bytes:
    url = f"https://api-data.line.me/v2/bot/message/{message_id}/content"
    headers = {"Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}"}
    resp = requests.get(url, headers=headers, timeout=60)
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


def ask_groq(user_id: str, text: str, system: str = None) -> str:
    if system:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": text}]
    else:
        history = conversation_history.setdefault(user_id, [
            {"role": "system", "content": "你是一個友善的 AI 助理，請用繁體中文回覆。"}
        ])
        history.append({"role": "user", "content": text})
        if len(history) > 21:
            history = [history[0]] + history[-20:]
            conversation_history[user_id] = history
        messages = history

    response = groq_client.chat.completions.create(
        model=TEXT_MODEL,
        messages=messages,
        max_tokens=2048,
    )
    answer = response.choices[0].message.content

    if not system:
        conversation_history[user_id].append({"role": "assistant", "content": answer})

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


def transcribe_audio(audio_bytes: bytes, filename: str = "audio.m4a") -> str:
    transcription = groq_client.audio.transcriptions.create(
        file=(filename, audio_bytes, "audio/m4a"),
        model=WHISPER_MODEL,
        language="zh",
    )
    return transcription.text


def extract_pdf_text(pdf_bytes: bytes) -> str:
    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    texts = []
    for page in reader.pages:
        t = page.extract_text()
        if t:
            texts.append(t)
    return "\n".join(texts)


def handle_command(user_id: str, text: str) -> str | None:
    """處理特殊指令，若不是指令則回傳 None"""
    stripped = text.strip()

    if stripped in ("/help", "help", "說明", "指令"):
        return HELP_TEXT

    if stripped == "/myid":
        return f"你的 LINE User ID：\n{user_id}"

    if stripped in ("/reset", "重置", "清除對話"):
        conversation_history.pop(user_id, None)
        return "對話已重置。"

    if stripped.startswith("/翻譯 "):
        content = stripped[4:].strip()
        return ask_groq(user_id, f"請將以下文字翻譯成繁體中文，只回覆翻譯結果：\n{content}", system="你是專業翻譯，只回覆翻譯結果，不加說明。")

    if stripped.startswith("/英文 "):
        content = stripped[4:].strip()
        return ask_groq(user_id, f"Please translate the following to English, reply with translation only:\n{content}", system="You are a professional translator. Reply with translation only.")

    if stripped.startswith("/摘要 "):
        content = stripped[4:].strip()
        return ask_groq(user_id, f"請用繁體中文條列摘要以下內容的重點：\n{content}", system="你是摘要專家，用條列式回覆重點。")

    if stripped.startswith("/改寫 "):
        content = stripped[4:].strip()
        return ask_groq(user_id, f"請潤飾以下文字，使其更通順專業，只回覆改寫結果：\n{content}", system="你是文字編輯，只回覆改寫結果。")

    return None


@app.route("/push", methods=["POST"])
def push():
    """Claude 主動推播訊息給使用者"""
    token = request.headers.get("X-Push-Token", "")
    if token != os.environ.get("PUSH_TOKEN", ""):
        abort(403)
    data = request.get_json()
    user_id = data.get("user_id")
    text = data.get("text", "")
    if not user_id or not text:
        return {"error": "missing user_id or text"}, 400
    with ApiClient(configuration) as api_client:
        line_api = MessagingApi(api_client)
        from linebot.v3.messaging import PushMessageRequest
        line_api.push_message(PushMessageRequest(
            to=user_id,
            messages=[TextMessage(text=text[:5000])],
        ))
    return {"ok": True}


@app.route("/users", methods=["GET"])
def get_users():
    """取得所有已知 user_id（需要 token）"""
    token = request.headers.get("X-Push-Token", "")
    if token != os.environ.get("PUSH_TOKEN", ""):
        abort(403)
    return {"users": list(known_users)}


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
    known_users.add(user_id)
    text = event.message.text
    try:
        cmd_result = handle_command(user_id, text)
        if cmd_result is not None:
            reply(event.reply_token, cmd_result)
            return
        answer = ask_groq(user_id, text)
        reply(event.reply_token, answer)
    except Exception as e:
        reply(event.reply_token, f"[錯誤] {type(e).__name__}: {str(e)[:200]}")


@handler.add(MessageEvent, message=ImageMessageContent)
def handle_image(event: MessageEvent):
    try:
        image_bytes = get_line_file(event.message.id)
        answer = ask_groq_vision(image_bytes, "請分析這張圖片，用繁體中文詳細說明內容。")
        reply(event.reply_token, answer)
    except Exception as e:
        reply(event.reply_token, f"[錯誤] {type(e).__name__}: {str(e)[:200]}")


@handler.add(MessageEvent, message=AudioMessageContent)
def handle_audio(event: MessageEvent):
    try:
        audio_bytes = get_line_file(event.message.id)
        text = transcribe_audio(audio_bytes)
        if not text.strip():
            reply(event.reply_token, "（無法辨識語音內容）")
            return
        reply(event.reply_token, f"🎤 語音轉文字：\n{text}")
    except Exception as e:
        reply(event.reply_token, f"[錯誤] {type(e).__name__}: {str(e)[:200]}")


@handler.add(MessageEvent, message=FileMessageContent)
def handle_file(event: MessageEvent):
    try:
        user_id = event.source.user_id
        file_name = event.message.file_name or "file"
        file_bytes = get_line_file(event.message.id)
        ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""

        if ext == "pdf":
            text_content = extract_pdf_text(file_bytes)
            if not text_content.strip():
                reply(event.reply_token, "PDF 內容無法讀取（可能是掃描圖片型 PDF）。")
                return
            answer = ask_groq(user_id, f"以下是 PDF「{file_name}」的內容：\n\n{text_content[:8000]}\n\n請用繁體中文條列摘要重點。",
                              system="你是文件分析專家，用條列式摘要重點。")
            reply(event.reply_token, f"📄 {file_name}\n\n{answer}")

        elif ext in ("txt", "csv", "md", "json", "xml", "py", "js", "ts", "html", "css"):
            try:
                text_content = file_bytes.decode("utf-8", errors="replace")
            except Exception:
                text_content = file_bytes.decode("big5", errors="replace")
            answer = ask_groq(user_id, f"以下是檔案「{file_name}」的內容：\n\n{text_content[:8000]}\n\n請用繁體中文分析並摘要重點。")
            reply(event.reply_token, answer)

        else:
            reply(event.reply_token, f"不支援 .{ext} 格式。\n支援：PDF、圖片、語音、txt/csv/md/json 等。")

    except Exception as e:
        reply(event.reply_token, f"[錯誤] {type(e).__name__}: {str(e)[:200]}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

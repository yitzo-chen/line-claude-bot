import os
import re
import json
import time
import base64
import threading
import subprocess
import requests
from flask import Flask, request, abort
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    Configuration, ApiClient, MessagingApi,
    ReplyMessageRequest, PushMessageRequest, TextMessage,
)
from linebot.v3.webhooks import (
    MessageEvent, TextMessageContent,
    ImageMessageContent, FileMessageContent,
)
import anthropic

app = Flask(__name__)

# ── 環境變數 ──────────────────────────────────────────────────────────────────
LINE_CHANNEL_ACCESS_TOKEN = os.environ["LINE_CHANNEL_ACCESS_TOKEN"]
LINE_CHANNEL_SECRET       = os.environ["LINE_CHANNEL_SECRET"]
ANTHROPIC_API_KEY         = os.environ["ANTHROPIC_API_KEY"]
OWNER_USER_ID             = os.environ.get("LINE_USER_ID", "")
OPENWEATHER_API_KEY       = os.environ.get("OPENWEATHER_API_KEY", "")

MODEL = "claude-sonnet-4-6"

configuration  = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
handler        = WebhookHandler(LINE_CHANNEL_SECRET)
claude_client  = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

SYSTEM_PROMPT = (
    "你的名字是「Claude AI助理」，你是 Yitzo 的私人 AI 助理，由 Anthropic Claude 驅動。\n"
    "規則：\n"
    "1. 永遠用繁體中文回覆，除非對方用其他語言提問\n"
    "2. 回覆控制在 500 字以內，除非使用者要求詳細\n"
    "3. 回答簡潔扼要，必要時條列說明\n"
    "4. 不透露底層技術細節"
)

# ── 對話歷史（in-memory）────────────────────────────────────────────────────
conversation_history: dict[str, list] = {}
HISTORY_LIMIT = 12
HISTORY_TTL   = 2 * 3600
_last_active:  dict[str, float] = {}


def get_history(uid: str) -> list:
    now = time.time()
    if uid in _last_active and now - _last_active[uid] > HISTORY_TTL:
        conversation_history.pop(uid, None)
    return conversation_history.get(uid, [])


def save_history(uid: str, history: list):
    if len(history) > HISTORY_LIMIT * 2:
        history = history[-(HISTORY_LIMIT * 2):]
    conversation_history[uid] = history
    _last_active[uid] = time.time()


# ── 去重 ──────────────────────────────────────────────────────────────────────
_seen: dict[str, float] = {}


def is_duplicate(mid: str) -> bool:
    now = time.time()
    _seen.update({k: v for k, v in _seen.items() if now - v < 60})
    if mid in _seen:
        return True
    _seen[mid] = now
    return False


# ── Rate limit ────────────────────────────────────────────────────────────────
_rl: dict[str, list] = {}


def check_rate_limit(uid: str) -> bool:
    now = time.time()
    ts = [t for t in _rl.get(uid, []) if now - t < 60]
    if len(ts) >= 15:
        return False
    ts.append(now)
    _rl[uid] = ts
    return True


# ── LINE 傳訊 helpers ─────────────────────────────────────────────────────────
def reply(token: str, text: str):
    with ApiClient(configuration) as c:
        MessagingApi(c).reply_message(
            ReplyMessageRequest(reply_token=token,
                                messages=[TextMessage(text=text[:5000])]))


def push(uid: str, text: str):
    with ApiClient(configuration) as c:
        MessagingApi(c).push_message(
            PushMessageRequest(to=uid,
                               messages=[TextMessage(text=text[:5000])]))


# ── Claude API ────────────────────────────────────────────────────────────────
def ask_claude(uid: str, user_content, label: str | None = None) -> str:
    history = get_history(uid)
    history.append({"role": "user", "content": user_content})

    resp = claude_client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=history,
    )
    answer = resp.content[0].text
    history.append({"role": "assistant", "content": answer})
    save_history(uid, history)
    return answer


# ── 天氣 ──────────────────────────────────────────────────────────────────────
WEATHER_KW = ["天氣", "氣溫", "溫度", "下雨", "幾度", "晴天", "陰天", "weather", "temperature"]
CITY_MAP = {
    "台北": "Taipei", "臺北": "Taipei", "新北": "New Taipei",
    "台中": "Taichung", "台南": "Tainan", "高雄": "Kaohsiung",
    "東京": "Tokyo", "大阪": "Osaka", "首爾": "Seoul", "釜山": "Busan",
    "新加坡": "Singapore", "香港": "Hong Kong",
}


def is_weather(text: str) -> bool:
    return any(kw in text.lower() for kw in WEATHER_KW)


def extract_city(text: str) -> str | None:
    for pat in [r'(.+?)(?:的)?(?:天氣|氣溫|幾度)', r'(?:weather|temp(?:erature)?)\s+(?:in|at)\s+(.+)']:
        m = re.search(pat, text, re.I)
        if m:
            city = re.sub(r'^(幫我查|查一下|查|請問|請|告訴我|現在|今天)', '', m.group(1)).strip()
            return city or None
    return None


def get_weather(city: str) -> str:
    if not OPENWEATHER_API_KEY:
        return "⚠️ 未設定 OPENWEATHER_API_KEY"
    en = CITY_MAP.get(city, city)
    try:
        r = requests.get("https://api.openweathermap.org/data/2.5/weather",
                         params={"q": en, "appid": OPENWEATHER_API_KEY,
                                 "units": "metric", "lang": "zh_tw"}, timeout=10)
        r.raise_for_status()
        d = r.json()
        return (f"📍 {d['name']}\n"
                f"🌤 {d['weather'][0]['description']}\n"
                f"🌡 {d['main']['temp']}°C（體感 {d['main']['feels_like']}°C）\n"
                f"💧 濕度 {d['main']['humidity']}%")
    except Exception as e:
        return f"查詢失敗：{e}"


# ── 遠端控制（僅限 OWNER）────────────────────────────────────────────────────
ALLOWED_CMDS = {
    "dir", "ls", "echo", "python", "pip", "git", "node", "npm",
    "type", "cat", "ping", "ipconfig", "tasklist", "claude",
}


def is_owner(uid: str) -> bool:
    return uid == OWNER_USER_ID


def safe_run(cmd: str) -> str:
    """執行 shell 指令，回傳輸出（最多 3000 字）"""
    first_word = cmd.strip().split()[0].lower() if cmd.strip() else ""
    if first_word not in ALLOWED_CMDS:
        return f"⛔ 指令 '{first_word}' 不在白名單內\n允許：{', '.join(sorted(ALLOWED_CMDS))}"
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True,
            text=True, timeout=30, encoding="utf-8", errors="replace"
        )
        out = (result.stdout + result.stderr).strip()
        return out[:3000] if out else "（無輸出）"
    except subprocess.TimeoutExpired:
        return "⚠️ 指令逾時（30 秒）"
    except Exception as e:
        return f"⚠️ 執行錯誤：{e}"


def run_claude_cli(prompt: str) -> str:
    """呼叫 claude CLI 執行一次性任務"""
    try:
        result = subprocess.run(
            ["claude", "-p", prompt, "--output-format", "text"],
            capture_output=True, text=True, timeout=120,
            encoding="utf-8", errors="replace"
        )
        out = (result.stdout + result.stderr).strip()
        return out[:4000] if out else "（無輸出）"
    except FileNotFoundError:
        return "⚠️ 找不到 claude CLI，請確認是否已安裝 Claude Code"
    except subprocess.TimeoutExpired:
        return "⚠️ Claude CLI 逾時（120 秒）"
    except Exception as e:
        return f"⚠️ 錯誤：{e}"


# ── Webhook 路由 ──────────────────────────────────────────────────────────────
@app.route("/health")
def health():
    return "OK"


@app.route("/callback", methods=["POST"])
def callback():
    sig  = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, sig)
    except InvalidSignatureError:
        abort(400)
    return "OK"


# ── 文字訊息處理 ──────────────────────────────────────────────────────────────
@handler.add(MessageEvent, message=TextMessageContent)
def handle_text(event: MessageEvent):
    if is_duplicate(event.message.id):
        return
    uid  = event.source.user_id
    text = event.message.text.strip()

    if not check_rate_limit(uid):
        reply(event.reply_token, "⚠️ 傳送太快，請稍等再試。")
        return

    # ── 快速指令 ──────────────────────────────────────────────────────────────
    if text == "/myid":
        reply(event.reply_token, f"你的 LINE User ID：\n{uid}")
        return

    if text in ("/reset", "重置", "清除對話"):
        conversation_history.pop(uid, None)
        reply(event.reply_token, "✅ 對話已重置。")
        return

    if text == "/status":
        hist_len = len(conversation_history.get(uid, []))
        reply(event.reply_token,
              f"🤖 Claude AI助理 運作中\n"
              f"模型：{MODEL}\n"
              f"對話歷史：{hist_len} 則\n"
              f"遠端控制：{'✅ 已啟用' if is_owner(uid) else '❌ 僅限擁有者'}")
        return

    # ── 遠端控制指令（僅限 OWNER）─────────────────────────────────────────────
    if text.startswith("!run "):
        if not is_owner(uid):
            reply(event.reply_token, "⛔ 無權限執行指令")
            return
        cmd = text[5:].strip()
        reply(event.reply_token, f"⚙️ 執行中：{cmd}")
        def _run():
            out = safe_run(cmd)
            push(uid, f"```\n{out}\n```")
        threading.Thread(target=_run, daemon=True).start()
        return

    if text.startswith("!claude "):
        if not is_owner(uid):
            reply(event.reply_token, "⛔ 無權限執行指令")
            return
        prompt = text[8:].strip()
        reply(event.reply_token, f"🤖 Claude Code 處理中...")
        def _cc():
            out = run_claude_cli(prompt)
            push(uid, out)
        threading.Thread(target=_cc, daemon=True).start()
        return

    if text.startswith("!push "):
        # 主動推送訊息給自己（測試用）
        if not is_owner(uid):
            reply(event.reply_token, "⛔ 無權限")
            return
        msg = text[6:].strip()
        push(OWNER_USER_ID, msg)
        return

    # ── 天氣查詢 ──────────────────────────────────────────────────────────────
    if is_weather(text):
        city = extract_city(text)
        if city:
            reply(event.reply_token, "🔍 查詢中...")
            def _weather():
                info = get_weather(city)
                content = f"{text}\n\n[即時天氣]\n{info}"
                ans = ask_claude(uid, [{"type": "text", "text": content}],
                                 label=f"[查了{city}天氣]")
                push(uid, f"{info}\n\n{ans}")
            threading.Thread(target=_weather, daemon=True).start()
            return

    # ── 一般 AI 對話 ──────────────────────────────────────────────────────────
    def _chat():
        try:
            ans = ask_claude(uid, [{"type": "text", "text": text}])
            push(uid, ans)
        except Exception as e:
            push(uid, f"⚠️ 錯誤：{e}")
    threading.Thread(target=_chat, daemon=True).start()


# ── 圖片處理 ──────────────────────────────────────────────────────────────────
@handler.add(MessageEvent, message=ImageMessageContent)
def handle_image(event: MessageEvent):
    if is_duplicate(event.message.id):
        return
    uid = event.source.user_id
    url = f"https://api-data.line.me/v2/bot/message/{event.message.id}/content"
    headers = {"Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}"}
    img_bytes = requests.get(url, headers=headers, timeout=30).content
    b64 = base64.standard_b64encode(img_bytes).decode()

    def _img():
        try:
            ans = ask_claude(uid, [
                {"type": "image", "source": {"type": "base64",
                                              "media_type": "image/jpeg",
                                              "data": b64}},
                {"type": "text", "text": "請分析這張圖片，用繁體中文說明內容。"},
            ], label="[傳了一張圖片]")
            push(uid, ans)
        except Exception as e:
            push(uid, f"⚠️ 圖片分析失敗：{e}")
    threading.Thread(target=_img, daemon=True).start()


# ── 檔案處理 ──────────────────────────────────────────────────────────────────
@handler.add(MessageEvent, message=FileMessageContent)
def handle_file(event: MessageEvent):
    if is_duplicate(event.message.id):
        return
    uid  = event.source.user_id
    name = event.message.file_name or "file"
    url  = f"https://api-data.line.me/v2/bot/message/{event.message.id}/content"
    headers = {"Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}"}
    raw = requests.get(url, headers=headers, timeout=30).content
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""

    TEXT_EXT = {"txt", "csv", "md", "json", "xml", "py", "js", "ts", "html", "css", "log"}
    if ext not in TEXT_EXT:
        reply(event.reply_token, f"不支援 .{ext}，支援：{', '.join(sorted(TEXT_EXT))}")
        return

    content = raw.decode("utf-8", errors="replace")

    def _file():
        try:
            ans = ask_claude(uid,
                             [{"type": "text",
                               "text": f"檔案 `{name}` 內容：\n\n{content[:8000]}\n\n請用繁體中文摘要重點。"}],
                             label=f"[分析了{name}]")
            push(uid, ans)
        except Exception as e:
            push(uid, f"⚠️ 檔案分析失敗：{e}")
    threading.Thread(target=_file, daemon=True).start()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

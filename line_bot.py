import os
import time
import sqlite3
import threading
import subprocess
import requests
import httpx
from datetime import datetime
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
from google import genai
from google.genai import types
import rebar_query

app = Flask(__name__)

# ── 環境變數 ──────────────────────────────────────────────────────────────────
LINE_CHANNEL_ACCESS_TOKEN = os.environ["LINE_CHANNEL_ACCESS_TOKEN"]
LINE_CHANNEL_SECRET       = os.environ["LINE_CHANNEL_SECRET"]
GEMINI_API_KEY            = os.environ["GEMINI_API_KEY"]
OWNER_USER_ID             = os.environ.get("LINE_USER_ID", "")
OPENWEATHER_API_KEY       = os.environ.get("OPENWEATHER_API_KEY", "")

GEMINI_MODEL = "gemini-2.5-flash"
# 額度用完時依序降級嘗試（各模型 RPM/RPD 額度分開算，帳號 prepay 額度耗盡則所有模型共用同一池，換模型無效）
GEMINI_MODEL_CHAIN = [GEMINI_MODEL, "gemini-flash-lite-latest", "gemini-2.0-flash-lite"]

configuration  = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
handler        = WebhookHandler(LINE_CHANNEL_SECRET)
# verify=False 解決部分企業網路 SSL 攔截問題
gemini_client  = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options=types.HttpOptions(httpx_client=httpx.Client(verify=False)),
)

_WEEKDAY_ZH = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]


def get_system_prompt() -> str:
    now = datetime.now()
    weekday = _WEEKDAY_ZH[now.weekday()]
    today = now.strftime(f"%Y年%m月%d日（{weekday}）")
    return (
        f"你的名字是「AI助理」，你是 Yitzo 的私人 AI 助理。\n"
        f"今天日期：{today}\n"
        "規則：\n"
        "1. 永遠用繁體中文回覆，除非對方用其他語言提問\n"
        "2. 回覆控制在 500 字以內，除非使用者要求詳細\n"
        "3. 回答簡潔扼要，必要時條列說明\n"
        "4. 不透露底層技術細節\n"
        "5. 被問到日期時間，以系統提供的今天日期為準，不要自行猜測"
    )


# A1 施工紀錄 prompt
CONSTRUCTION_PHOTO_PROMPT = (
    "你是台灣建築工地施工紀錄助理。"
    "請分析這張工地照片，用繁體中文輸出一行施工紀錄，格式固定：[部位] [工項] [狀態]，"
    "例如：B2層 鋼筋綁紮 完成。只輸出這一行，不超過15字，不需要其他說明。"
)


# ── 對話歷史（in-memory，Gemini 格式）──────────────────────────────────────
# 格式：[{"role": "user"/"model", "parts": [{"text": "..."}]}, ...]
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


# ── 去重（SQLite，跨 gunicorn worker 共用，避免冷啟動時 LINE 重送 webhook 被不同 worker 各自放行）──
_DEDUP_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dedup.db")
_dedup_lock = threading.Lock()


def _dedup_conn():
    conn = sqlite3.connect(_DEDUP_DB, timeout=5)
    conn.execute("CREATE TABLE IF NOT EXISTS seen (mid TEXT PRIMARY KEY, ts REAL)")
    return conn


def is_duplicate(mid: str) -> bool:
    now = time.time()
    with _dedup_lock:
        conn = _dedup_conn()
        try:
            conn.execute("DELETE FROM seen WHERE ts < ?", (now - 60,))
            cur = conn.execute("INSERT OR IGNORE INTO seen (mid, ts) VALUES (?, ?)", (mid, now))
            conn.commit()
            return cur.rowcount == 0
        finally:
            conn.close()


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


# ── 錯誤紀錄（推給使用者的錯誤同時印到 stdout，才會出現在 Render Logs）──────────
def _log_error(context: str, e: Exception):
    print(f"[ERROR] {context}: {e!r}", flush=True)


# ── Gemini API ────────────────────────────────────────────────────────────────
def _gemini_call_with_retry(build_fn, retries=3, base_delay=5):
    """
    build_fn(model) 依序在 GEMINI_MODEL_CHAIN 各模型上呼叫：
    503（過載）：同一模型間隔 5s / 10s / 20s 重試
    429 且訊息含 prepay/credits depleted（帳號額度耗盡，所有模型共用同一池）：直接拋出，換模型無用
    429 其他情況（單一模型 RPM/RPD 額度打滿）：重試 3 次後改用下一個模型
    """
    last_err = None
    for model in GEMINI_MODEL_CHAIN:
        for attempt in range(retries):
            try:
                return build_fn(model)
            except Exception as e:
                err = str(e)
                last_err = e
                if "prepay" in err.lower() or "credits are depleted" in err.lower():
                    raise
                if "429" in err and attempt < retries - 1:
                    time.sleep(base_delay * 4 * (2 ** attempt))
                elif "503" in err and attempt < retries - 1:
                    time.sleep(base_delay * (2 ** attempt))
                elif "429" in err or "503" in err:
                    print(f"[WARN] {model} 額度/過載重試耗盡，改用下一個模型", flush=True)
                    break
                else:
                    raise
    raise last_err


def ask_gemini(uid: str, user_text: str, system: str | None = None) -> str:
    history = get_history(uid)
    contents = history + [{"role": "user", "parts": [{"text": user_text}]}]
    config = types.GenerateContentConfig(
        system_instruction=system or get_system_prompt(),
        max_output_tokens=1024,
    )
    resp = _gemini_call_with_retry(lambda model: gemini_client.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    ))
    answer = resp.text
    history.append({"role": "user",  "parts": [{"text": user_text}]})
    history.append({"role": "model", "parts": [{"text": answer}]})
    save_history(uid, history)
    return answer


def ask_gemini_image(img_bytes: bytes, mime: str, prompt: str) -> str:
    img_part = types.Part.from_bytes(data=img_bytes, mime_type=mime)
    resp = _gemini_call_with_retry(lambda model: gemini_client.models.generate_content(
        model=model,
        contents=[img_part, prompt],
    ))
    return resp.text


# ── 天氣 ──────────────────────────────────────────────────────────────────────
WEATHER_KW = ["天氣", "氣溫", "溫度", "下雨", "幾度", "晴天", "陰天",
              "降雨", "下雪", "颱風", "weather", "temperature"]

_STRIP_PUNCT = str.maketrans("", "", "。，、？！,.?!")


def is_weather(text: str) -> bool:
    return any(kw in text for kw in WEATHER_KW)


def extract_location_ai(text: str) -> str | None:
    try:
        resp = _gemini_call_with_retry(lambda model: gemini_client.models.generate_content(
            model=model,
            contents=(
                "從以下句子抽取地點名稱（城市、區、鄉鎮皆可），"
                "只輸出地點本身（例：小港區、台北、東京），"
                "完全無法判斷才輸出「NULL」：\n" + text
            ),
            config=types.GenerateContentConfig(max_output_tokens=20),
        ))
        loc = resp.text.strip().translate(_STRIP_PUNCT)
        return None if loc == "NULL" or not loc else loc
    except Exception:
        return None


def geocode_location(location: str) -> tuple[float, float, str] | None:
    if not OPENWEATHER_API_KEY:
        return None
    for query in [f"{location},TW", location]:
        try:
            r = requests.get(
                "https://api.openweathermap.org/geo/1.0/direct",
                params={"q": query, "limit": 1, "appid": OPENWEATHER_API_KEY},
                timeout=10,
            )
            data = r.json()
            if data:
                d = data[0]
                display = d.get("local_names", {}).get("zh", d["name"])
                return d["lat"], d["lon"], display
        except Exception:
            continue
    return None


def get_weather(location: str) -> str:
    if not OPENWEATHER_API_KEY:
        return "⚠️ 未設定 OPENWEATHER_API_KEY"
    geo = geocode_location(location)
    if not geo:
        return f"⚠️ 找不到「{location}」的地點資料"
    lat, lon, display_name = geo
    params = {"lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY,
              "units": "metric", "lang": "zh_tw"}
    try:
        r = requests.get("https://api.openweathermap.org/data/2.5/weather",
                         params=params, timeout=10)
        r.raise_for_status()
        d = r.json()

        pop_str = ""
        try:
            rf = requests.get("https://api.openweathermap.org/data/2.5/forecast",
                              params={**params, "cnt": 2}, timeout=10)
            pop = rf.json()["list"][0].get("pop", 0)
            pop_str = f"\n🌧 降雨機率 {round(pop * 100)}%"
        except Exception:
            pass

        return (f"📍 {display_name}\n"
                f"🌤 {d['weather'][0]['description']}\n"
                f"🌡 {d['main']['temp']}°C（體感 {d['main']['feels_like']}°C）\n"
                f"💧 濕度 {d['main']['humidity']}%"
                f"{pop_str}")
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
              f"🤖 AI助理 運作中\n"
              f"引擎：Gemini 2.5 Flash（免費）\n"
              f"對話歷史：{hist_len // 2} 則\n"
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
        reply(event.reply_token, "🤖 Claude Code 處理中...")
        def _cc():
            out = run_claude_cli(prompt)
            push(uid, out)
        threading.Thread(target=_cc, daemon=True).start()
        return

    if text.startswith("!push "):
        if not is_owner(uid):
            reply(event.reply_token, "⛔ 無權限")
            return
        push(OWNER_USER_ID, text[6:].strip())
        return

    # ── 天氣查詢 ──────────────────────────────────────────────────────────────
    if is_weather(text):
        reply(event.reply_token, "🔍 查詢中...")
        def _weather():
            location = extract_location_ai(text)
            if not location:
                push(uid, "⚠️ 請指定地點，例如：台北天氣、小港區幾度")
                return
            push(uid, get_weather(location))
        threading.Thread(target=_weather, daemon=True).start()
        return

    # ── B1 鋼筋混凝土規範查詢 ─────────────────────────────────────────────────
    if rebar_query.is_reg_query(text):
        def _reg():
            try:
                msg = rebar_query.build_query_message(text)
                ans = ask_gemini(uid, msg, system=rebar_query.REG_SYSTEM)
                push(uid, ans + rebar_query.VERSION_NOTE)
            except Exception as e:
                _log_error("規範查詢", e)
                push(uid, f"⚠️ 規範查詢錯誤：{e}")
        threading.Thread(target=_reg, daemon=True).start()
        return

    # ── 一般 AI 對話 ──────────────────────────────────────────────────────────
    def _chat():
        try:
            ans = ask_gemini(uid, text)
            push(uid, ans)
        except Exception as e:
            _log_error("一般對話", e)
            push(uid, f"⚠️ 錯誤：{e}")
    threading.Thread(target=_chat, daemon=True).start()


# ── 圖片處理（A1 施工紀錄）──────────────────────────────────────────────────
@handler.add(MessageEvent, message=ImageMessageContent)
def handle_image(event: MessageEvent):
    if is_duplicate(event.message.id):
        return
    uid = event.source.user_id
    if not check_rate_limit(uid):
        reply(event.reply_token, "⚠️ 傳送太快，請稍等再試。")
        return
    msg_id = event.message.id
    reply(event.reply_token, "🔍 分析中...")

    def _img():
        try:
            url = f"https://api-data.line.me/v2/bot/message/{msg_id}/content"
            headers = {"Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}"}
            img_resp = requests.get(url, headers=headers, timeout=30)
            img_resp.raise_for_status()
            ans = ask_gemini_image(img_resp.content, "image/jpeg",
                                   CONSTRUCTION_PHOTO_PROMPT)
            push(uid, f"📋 {ans}")
        except Exception as e:
            _log_error("圖片分析", e)
            push(uid, f"⚠️ 圖片分析失敗：{e}")
    threading.Thread(target=_img, daemon=True).start()


# ── 檔案處理 ──────────────────────────────────────────────────────────────────
@handler.add(MessageEvent, message=FileMessageContent)
def handle_file(event: MessageEvent):
    if is_duplicate(event.message.id):
        return
    uid  = event.source.user_id
    if not check_rate_limit(uid):
        reply(event.reply_token, "⚠️ 傳送太快，請稍等再試。")
        return
    name   = event.message.file_name or "file"
    msg_id = event.message.id
    ext    = name.rsplit(".", 1)[-1].lower() if "." in name else ""

    TEXT_EXT = {"txt", "csv", "md", "json", "xml", "py", "js", "ts", "html", "css", "log"}
    if ext not in TEXT_EXT:
        reply(event.reply_token, f"不支援 .{ext}，支援：{', '.join(sorted(TEXT_EXT))}")
        return

    reply(event.reply_token, "📄 分析中...")

    def _file():
        try:
            url = f"https://api-data.line.me/v2/bot/message/{msg_id}/content"
            headers = {"Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}"}
            raw_resp = requests.get(url, headers=headers, timeout=30)
            raw_resp.raise_for_status()
            content = raw_resp.content.decode("utf-8", errors="replace")
            ans = ask_gemini(uid,
                             f"檔案 `{name}` 內容：\n\n{content[:8000]}\n\n請用繁體中文摘要重點。")
            push(uid, ans)
        except Exception as e:
            _log_error("檔案分析", e)
            push(uid, f"⚠️ 檔案分析失敗：{e}")
    threading.Thread(target=_file, daemon=True).start()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

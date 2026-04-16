import os
import re
import time
import threading
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
    PushMessageRequest,
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
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "")

configuration = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)
groq_client = Groq(api_key=GROQ_API_KEY)

TEXT_MODEL = "llama-3.3-70b-versatile"
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

SYSTEM_PROMPT = (
    "你的名字是「Yitzo Bot」，你是 Yitzo 的私人智慧助理。\n"
    "規則（不可違反）：\n"
    "1. 永遠以「Yitzo Bot」自稱，不得聲稱自己是其他 AI 或其他名字\n"
    "2. 永遠使用繁體中文回覆，不得使用簡體中文\n"
    "3. 回覆長度控制在 400 字以內，除非使用者明確要求詳細說明\n"
    "4. 回答簡潔扼要，必要時條列說明\n"
    "5. 若使用者用其他語言提問，以同語言回覆（但仍不透露其他身份）\n"
    "6. 不要透露你使用的底層模型或技術細節"
)

# 每個使用者的對話歷史（重啟後清空）
conversation_history: dict[str, list] = {}
conversation_last_active: dict[str, float] = {}
HISTORY_LIMIT = 10
HISTORY_TTL = 2 * 3600  # 2 小時無活動自動清除

# 常見城市中英對照（命中直接用，不呼叫 Groq）
CITY_MAP: dict[str, str] = {
    "台北": "Taipei", "臺北": "Taipei", "新北": "New Taipei",
    "台中": "Taichung", "臺中": "Taichung", "台南": "Tainan", "臺南": "Tainan",
    "高雄": "Kaohsiung", "基隆": "Keelung", "新竹": "Hsinchu",
    "東京": "Tokyo", "大阪": "Osaka", "京都": "Kyoto", "札幌": "Sapporo",
    "北京": "Beijing", "上海": "Shanghai", "廣州": "Guangzhou", "深圳": "Shenzhen",
    "香港": "Hong Kong", "澳門": "Macao",
    "首爾": "Seoul", "釜山": "Busan",
    "新加坡": "Singapore", "曼谷": "Bangkok", "吉隆坡": "Kuala Lumpur",
    "紐約": "New York", "洛杉磯": "Los Angeles", "芝加哥": "Chicago",
    "倫敦": "London", "巴黎": "Paris", "柏林": "Berlin", "羅馬": "Rome",
    "雪梨": "Sydney", "墨爾本": "Melbourne",
}


def cleanup_expired():
    now = time.time()
    expired = [uid for uid, t in conversation_last_active.items() if now - t > HISTORY_TTL]
    for uid in expired:
        conversation_history.pop(uid, None)
        conversation_last_active.pop(uid, None)


WEATHER_KEYWORDS = ["天氣", "氣溫", "溫度", "下雨", "幾度", "晴天", "陰天", "weather", "temperature"]


def is_weather_query(text: str) -> bool:
    return any(kw in text.lower() for kw in WEATHER_KEYWORDS)


def extract_city(text: str) -> str | None:
    patterns = [
        r'(.+?)(?:的)?(?:天氣|氣溫|溫度|幾度|下雨)',
        r'(?:weather|temperature)\s+(?:in|at|of)\s+(.+)',
        r'(.+?)\s+(?:weather|temperature)',
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            city = m.group(1).strip()
            city = re.sub(r'^(幫我查|查一下|查|請問|請|告訴我|現在|今天|明天)', '', city).strip()
            if city:
                return city
    return None


def normalize_city(city: str) -> str:
    """將城市名稱翻譯成英文：先查本地對照表，未命中才呼叫 Groq"""
    if city in CITY_MAP:
        return CITY_MAP[city]
    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",  # 用小模型節省 token
            max_tokens=20,
            messages=[
                {"role": "system", "content": "Translate the city name to English. Reply with ONLY the English city name, nothing else."},
                {"role": "user", "content": city},
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return city


def get_weather(city: str) -> str:
    if not OPENWEATHER_API_KEY:
        return "⚠️ 未設定 OPENWEATHER_API_KEY，無法查詢天氣。"
    city = normalize_city(city)
    try:
        resp = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"q": city, "appid": OPENWEATHER_API_KEY, "units": "metric", "lang": "zh_tw"},
            timeout=10,
        )
        if resp.status_code == 404:
            return f"找不到城市「{city}」，請確認城市名稱（建議用英文，例如 Taipei、Tokyo）。"
        resp.raise_for_status()
        d = resp.json()
        return (
            f"城市：{d['name']}\n"
            f"天氣：{d['weather'][0]['description']}\n"
            f"氣溫：{d['main']['temp']}°C（體感 {d['main']['feels_like']}°C）\n"
            f"濕度：{d['main']['humidity']}%\n"
            f"風速：{d['wind']['speed']} m/s"
        )
    except Exception as e:
        return f"查詢天氣失敗：{e}"


def get_line_file(message_id: str) -> bytes:
    """從 LINE 下載圖片或檔案內容"""
    url = f"https://api-data.line.me/v2/bot/message/{message_id}/content"
    headers = {"Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.content


# Rate limiting：每位使用者每分鐘最多 10 則
rate_limit_count: dict[str, list] = {}
RATE_LIMIT = 10
RATE_WINDOW = 60


def check_rate_limit(user_id: str) -> bool:
    """回傳 True 表示未超限，False 表示超限"""
    now = time.time()
    timestamps = rate_limit_count.setdefault(user_id, [])
    rate_limit_count[user_id] = [t for t in timestamps if now - t < RATE_WINDOW]
    if len(rate_limit_count[user_id]) >= RATE_LIMIT:
        return False
    rate_limit_count[user_id].append(now)
    return True


def reply(reply_token: str, text: str):
    """快速回覆（用於即時指令，不呼叫 AI）"""
    with ApiClient(configuration) as api_client:
        line_api = MessagingApi(api_client)
        line_api.reply_message(
            ReplyMessageRequest(
                reply_token=reply_token,
                messages=[TextMessage(text=text[:5000])],
            )
        )


def push(user_id: str, text: str):
    """主動推送訊息（AI 回應使用，不受 30 秒 reply token 限制）"""
    with ApiClient(configuration) as api_client:
        line_api = MessagingApi(api_client)
        line_api.push_message(
            PushMessageRequest(
                to=user_id,
                messages=[TextMessage(text=text[:5000])],
            )
        )


def ask_groq(user_id: str, messages: list, model: str = TEXT_MODEL, history_label: str | None = None) -> str:
    """
    呼叫 Groq API，維持對話歷史。
    history_label: 存入歷史的簡化描述（None 則存 messages[0] 的完整 content）
    """
    history = conversation_history.setdefault(user_id, [])
    conversation_last_active[user_id] = time.time()
    cleanup_expired()

    if model == VISION_MODEL:
        api_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
        response = groq_client.chat.completions.create(
            model=model, max_tokens=2048, messages=api_messages,
        )
        assistant_text = response.choices[0].message.content
        history.append({"role": "user", "content": history_label or "[傳送了一張圖片]"})
        history.append({"role": "assistant", "content": assistant_text})
        return assistant_text

    # 歷史只存簡化標籤，不存完整內容（天氣資料、檔案內文）
    stored = history_label if history_label else messages[0]["content"]
    history.append({"role": "user", "content": stored})
    if len(history) > HISTORY_LIMIT:
        history = history[-HISTORY_LIMIT:]
        conversation_history[user_id] = history

    # 歷史（除最後一筆）過濾 list content，最後一筆用原始 messages
    filtered = []
    for m in history[:-1]:
        content = m.get("content")
        if isinstance(content, list):
            text = " ".join(
                p.get("text", "") for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            ) or "[圖片]"
            filtered.append({"role": m["role"], "content": text})
        else:
            filtered.append(m)

    api_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + filtered + messages

    response = groq_client.chat.completions.create(
        model=model, max_tokens=2048, messages=api_messages,
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

    if not check_rate_limit(user_id):
        reply(event.reply_token, "⚠️ 你傳送太快了，請稍等一下再試。")
        return

    if text == "/myid":
        reply(event.reply_token, f"你的 LINE User ID：\n{user_id}")
        return

    if text in ("/reset", "重置", "清除對話"):
        conversation_history.pop(user_id, None)
        reply(event.reply_token, "對話已重置。")
        return

    def process():
        try:
            content = text
            label = None
            if is_weather_query(text):
                city = extract_city(text)
                if not city:
                    push(user_id, "請告訴我你想查哪個城市的天氣？")
                    return
                weather_info = get_weather(city)
                content = f"{text}\n\n[即時天氣資料]\n{weather_info}"
                label = f"[查詢了 {city} 的天氣]"
            messages = [{"role": "user", "content": content}]
            answer = ask_groq(user_id, messages, history_label=label)
            push(user_id, answer)
        except Exception as e:
            push(user_id, f"⚠️ 發生錯誤：{e}")

    threading.Thread(target=process, daemon=True).start()


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
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                },
                {"type": "text", "text": "請分析這張圖片，用繁體中文說明內容。"},
            ],
        }
    ]
    def process():
        try:
            answer = ask_groq(user_id, messages, model=VISION_MODEL)
            push(user_id, answer)
        except Exception as e:
            push(user_id, f"⚠️ 發生錯誤：{e}")

    threading.Thread(target=process, daemon=True).start()


@handler.add(MessageEvent, message=FileMessageContent)
def handle_file(event: MessageEvent):
    user_id = event.source.user_id
    file_name = event.message.file_name or "file"
    file_bytes = get_line_file(event.message.id)

    ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""

    if ext in ("txt", "csv", "md", "json", "xml", "py", "js", "ts", "html", "css"):
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
        def process():
            try:
                answer = ask_groq(user_id, messages, history_label=f"[分析了檔案 {file_name}]")
                push(user_id, answer)
            except Exception as e:
                push(user_id, f"⚠️ 發生錯誤：{e}")

        threading.Thread(target=process, daemon=True).start()
    else:
        reply(event.reply_token, f"目前不支援 .{ext} 格式，支援：圖片、txt、csv、md、json 等文字類型檔案。")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

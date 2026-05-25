# LINE Bot × Claude 工地助理 專案規劃

## 目標
將現有 LINE Bot 強化為工地主任實用工具，免費方案優先，等 Anthropic API 開放後升級。

## 功能清單

### A1：拍照 → 施工紀錄
- 傳一張工地照片到 LINE
- Bot 自動回覆：`[部位][工項][狀態]` 格式，10字簡短紀錄
- 現況：圖片處理框架已有，只需換 prompt + 換 Groq Vision

### B1：規範查詢
- 傳文字問題（CNS、搭接、彎鉤、間距...）
- Bot 從以下來源回答：
  - 網路搜尋（Tavily API，免費 1000次/月）
  - 使用者上傳的 PDF 規範
- 現況：需新增 PDF 解析 + 搜尋功能

---

## 技術架構

```
LINE App
  ↓↑
LINE Messaging API
  ↓↑
Render.com（Flask / line_bot.py）
  ├── 圖片 → Groq Vision（llama-4-scout）→ 施工紀錄
  ├── 文字（規範關鍵字）→ Tavily 搜尋 + Groq → 規範回覆
  └── PDF 上傳 → PyPDF2 解析 → 存 data/regulations/ → 查詢時引用
```

## API 費用

| 服務 | 免費額度 | 備註 |
|------|---------|------|
| Groq | 免費 | 含 Vision（llama-4-scout）|
| Tavily | 1000次/月 | 網路搜尋 |
| Anthropic | 待開放 | 開放後升級 Vision + 對話品質 |

---

## 施作步驟

### Phase 1：換 Groq 後端
- [ ] 安裝 groq Python SDK
- [ ] 改寫 `ask_claude()` → `ask_groq()`，保留對話歷史
- [ ] 圖片處理改用 Groq Vision（llama-4-scout）
- [ ] 測試：純文字對話 + 圖片辨識

### Phase 2：A1 施工紀錄 Prompt
- [ ] `handle_image` 加入施工紀錄 prompt
- [ ] 加觸發判斷：一般圖片 vs 施工照片（用關鍵字或預設）
- [ ] 測試：傳工地照 → 回傳10字紀錄

### Phase 3：B1 規範查詢
- [ ] 申請 Tavily API key（免費）
- [ ] 加規範關鍵字偵測（CNS、搭接、彎鉤、錨定、間距...）
- [ ] 串接 Tavily 搜尋 → Groq 整理回覆
- [ ] PDF 支援：`handle_file` 加 PDF 解析（PyPDF2）
- [ ] PDF 存至 `data/regulations/`，查詢時載入為 context
- [ ] 測試：問規範問題 → 正確引用條文

### Phase 4：部署 Render + 驗收
- [ ] 更新 `requirements.txt`
- [ ] 設定 Render 環境變數（GROQ_API_KEY、TAVILY_API_KEY）
- [ ] Push → Render 自動部署
- [ ] 實機測試全功能

---

## 檔案結構

```
line-claude-bot/
├── line_bot.py          # 主程式
├── requirements.txt     # 套件清單
├── render.yaml          # Render 部署設定
├── docs/
│   └── project-plan.md  # 本文件
├── prompts/
│   ├── construction_photo.md   # A1 施工紀錄 prompt
│   └── regulation_query.md     # B1 規範查詢 system prompt
└── data/
    └── regulations/     # PDF 規範檔存放區
```

---

## 注意事項
- Groq Vision 支援 JPEG/PNG，大圖先壓縮再送（LINE 下載後約 1~2MB）
- PDF 若超過 50 頁，只取前 30 頁作為 context，避免超出 token 限制
- Render 免費方案 RAM 512MB，PDF 不要存記憶體，存磁碟（但 Render 重啟會清空）
  → 解法：每次重啟重新讀取，或改存 Google Drive（未來再評估）

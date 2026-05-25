# 專案資料夾說明

## 根目錄 `line-claude-bot/`

| 檔案 | 說明 |
|------|------|
| `line_bot.py` | 主程式，Flask Webhook，所有功能邏輯在此 |
| `requirements.txt` | Python 套件清單，Render 部署時自動安裝 |
| `render.yaml` | Render 部署設定（服務名稱、啟動指令） |
| `.gitignore` | git 忽略清單（不 commit .env、__pycache__ 等） |

---

## `docs/` — 規劃與說明文件

| 檔案 | 說明 |
|------|------|
| `project-plan.md` | 專案總規劃：功能清單、技術架構、施作步驟 |
| `folder-structure.md` | 本文件，各資料夾用途說明 |

> 新增功能或重大變更時，請同步更新 `project-plan.md`

---

## `prompts/` — AI Prompt 版本管理

| 檔案 | 說明 |
|------|------|
| `construction_photo.md` | A1 施工紀錄：圖片→10字紀錄的 prompt 設計與版本紀錄 |
| `regulation_query.md` | B1 規範查詢：關鍵字偵測清單 + system prompt |

> Prompt 調整後請在檔案末尾加版本紀錄（v1、v2...），方便追蹤效果差異

---

## `data/regulations/` — 規範 PDF 存放區

放使用者上傳的工程規範 PDF，Bot 查詢時載入作為 context。

| 建議放的檔案 | 說明 |
|------------|------|
| `CNS560.pdf` | 鋼筋規範 |
| `建築技術規則結構篇.pdf` | 結構設計規範 |
| 其他規範 PDF | 視工程需要放入 |

> ⚠️ Render 免費方案重啟後磁碟清空，PDF 需重新上傳（未來可接 Google Drive 解決）

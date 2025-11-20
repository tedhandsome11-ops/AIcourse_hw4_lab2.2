# AIcourse_hw4_lab2.2

# The Alchemist's Trial

玩家扮演一名立志獲得公會認可的煉金術學徒。遊戲的核心目標是透過與 AI 驅動的「配方導師」互動，掌握煉金術的兩大支柱：**理論效能 (Potency)** 與 **調和藝術 (Harmony)**。

## ✨ 核心特色與技術 (Key Features)

本專案旨在達成以下 LLM 應用目標：

1.  **多重 LLM 任務協作 (Multiple LLM Tasks):**
    * **理論合成 (Do Synthesize):** LLM 扮演導師，根據靜態知識庫生成考題，並評估玩家答案的正確性。
    * **創造性實驗 (Do Experiment):** LLM 根據玩家輸入的實驗步驟生成敘事，並進行風格評分。
    * **總結與回顧 (Review & Summary):** 遊戲結束時，使用 GPT-4 生成完整的公會評估報告。

2.  **LLM 自我消化與精煉 (Self-Correction/Digestion):**
    * 系統要求 LLM 先生成初步內容（如實驗故事或草稿），接著將其作為輸入，要求 LLM 進行**二次精煉與修正**，確保輸出的品質與邏輯一致性。

3.  **動態知識庫 (Dynamic Knowledge Base):**
    * 遊戲內容基於 JSON 定義的藥劑配方（如生命靈藥、清醒藥劑等），LLM 會根據這些結構化數據生成非重複的互動內容。

### 前置需求 (Prerequisites)

* Python 3.7+
* OpenAI API Key

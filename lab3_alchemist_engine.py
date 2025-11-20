import os
import re
import sys
import json
import random
import logging
import argparse

import openai
import tiktoken

# 更改日誌名稱以反映新主題
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(funcName)s() - %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO,
)


# --- 文件 I/O 函數 (保持不變) ---

def read_txt(file, write_log=False):
    if write_log:
        logger.info(f"Reading {file}")

    with open(file, "r", encoding="utf8") as f:
        text = f.read()

    if write_log:
        characters = len(text)
        logger.info(f"Read {characters:,} characters")
    return text


def write_txt(file, text, write_log=False):
    if write_log:
        characters = len(text)
        logger.info(f"Writing {characters:,} characters to {file}")

    with open(file, "w", encoding="utf8") as f:
        f.write(text)

    if write_log:
        logger.info(f"Written")
    return


def read_json(file, write_log=False):
    if write_log:
        logger.info(f"Reading {file}")

    with open(file, "r", encoding="utf8") as f:
        data = json.load(f)

    if write_log:
        objects = len(data)
        logger.info(f"Read {objects:,} objects")
    return data


def write_json(file, data, indent=None, write_log=False):
    if write_log:
        objects = len(data)
        logger.info(f"Writing {objects:,} objects to {file}")

    with open(file, "w", encoding="utf8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

    if write_log:
        logger.info(f"Written")
    return


# --- 配置類別 (保持不變) ---

class Config:
    def __init__(self, config_file):
        data = read_json(config_file)

        self.model = data["model"]
        self.static_dir = os.path.join(*data["static_dir"])
        self.state_dir = os.path.join(*data["state_dir"])
        self.output_dir = os.path.join(*data["output_dir"])

        os.makedirs(self.state_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        return


# --- GPT 類別 (保持不變) ---

class GPT:
    def __init__(self, model):
        self.model = model
        self.tokenizer = tiktoken.encoding_for_model(self.model)
        self.model_candidate_tokens = {
            "gpt-3.5-turbo": {
                "gpt-3.5-turbo": 4096,
                "gpt-3.5-turbo-16k": 16384,
            },
            "gpt-4": {
                "gpt-4": 8192,
                "gpt-4-32k": 32768,
            }
        }
        return

    def get_specific_tokens_model(self, text_in, out_tokens):
        in_token_list = self.tokenizer.encode(text_in)
        in_tokens = len(in_token_list)
        tokens = in_tokens + out_tokens

        for candidate, max_tokens in self.model_candidate_tokens.get(self.model, {}).items():
            if max_tokens >= tokens:
                break
        else:
            candidate = ""

        return in_tokens, candidate

    def run_gpt(self, text_in, out_tokens):
        in_tokens, specific_tokens_model = self.get_specific_tokens_model(text_in, out_tokens)
        if not specific_tokens_model:
            # 處理超過最大 tokens 的情況
            logger.error("Input tokens exceeds max model capacity.")
            return ""

        # logger.info(text_in)
        logger.info("I'm alive! Please wait awhile >< ...")

        try:
            completion = openai.ChatCompletion.create(
                model=specific_tokens_model,
                n=1,
                messages=[
                    {"role": "user", "content": text_in},
                ]
            )
            text_out = completion.choices[0].message.content
        except openai.error.OpenAIError as e:
            logger.error(f"OpenAI API Error: {e}")
            text_out = ""

        return text_out


# --- 狀態類別 (調整為煉金術主題) ---

class State:
    def __init__(self, save_file=""):
        self.save_file = save_file

        self.log = ""
        self.pedia = {}                 # 藥劑狀態：potency (效能) 和 harmony (調和度)
        self.recipe_list = []           # 已發現的配方列表 (取代 animal_list)
        self.mastery_skill = ""         # 掌握的精煉術 (取代 shapeshift)
        self.signature_recipe = ""      # 創造的簽名藥劑 (取代 companion)
        self.ended = False
        return

    def save(self):
        data = {
            "log": self.log,
            "pedia": self.pedia,
            "recipe_list": self.recipe_list,
            "mastery_skill": self.mastery_skill,
            "signature_recipe": self.signature_recipe,
            "ended": self.ended,
        }
        write_json(self.save_file, data, indent=2)
        return

    def load(self):
        data = read_json(self.save_file)
        self.log = data["log"]
        self.pedia = data["pedia"]
        self.recipe_list = data["recipe_list"]
        self.mastery_skill = data["mastery_skill"]
        self.signature_recipe = data["signature_recipe"]
        self.ended = data["ended"]
        return


# --- 遊戲類別 (調整為煉金術主題) ---

class Game:
    def __init__(self, config):
        self.static_dir = config.static_dir
        self.state_dir = config.state_dir
        self.output_dir = config.output_dir
        self.summary_file = ""
        self.gpt = GPT(config.model)
        self.gpt4 = GPT("gpt-4")

        self.user_prompt_to_text = {}
        self.max_saves = 4
        self.state = State()

        # 核心藥劑/配方列表 (請在 corpora/alchemist/recipe/ 下建立對應的 json 文件)
        self.all_recipe_list = [
            "elixir_life", "potion_clarity", "draught_courage", "essence_dream",
        ]

        # 行動列表替換
        self.action_to_text = {
            "exit": "紮營休息(離開遊戲)",
            "check_pedia": "查閱藥劑學筆記",
            "discover_recipe": "尋找新的藥劑配方",
            "study_recipe": "研究配方與實驗",
        }
        self.recipe_data = {}

        # 載入 prompt text
        user_prompt_dir = os.path.join(self.static_dir, "user_prompt")
        filename_list = os.listdir(user_prompt_dir)
        for filename in filename_list:
            user_prompt = filename[:-4]
            user_prompt_file = os.path.join(user_prompt_dir, filename)
            self.user_prompt_to_text[user_prompt] = read_txt(user_prompt_file)

        # 載入藥劑/配方資料 (請在 corpora/alchemist/recipe/ 下建立對應的 json 文件)
        for recipe in self.all_recipe_list:
            recipe_file = os.path.join(self.static_dir, "recipe", f"{recipe}.json")
            # 處理文件不存在的情況
            if os.path.exists(recipe_file):
                 self.recipe_data[recipe] = read_json(recipe_file)
            else:
                 logger.error(f"Recipe file not found: {recipe_file}")
                 sys.exit(1) # 缺少核心文件則退出

        return

    def run_start(self):
        # ... (讀取檔案邏輯與原版相同)
        user_prompt = self.user_prompt_to_text["start"]
        while True:
            text_in = input(user_prompt)
            if text_in == "1":
                start_type = "new"
                break
            elif text_in == "2":
                start_type = "load"
                break

        # get save file usage
        save_list_text = "\n存檔列表：\n"
        saveid_to_exist = {}
        for i in range(self.max_saves):
            save_id = str(i + 1)
            save_file = os.path.join(self.state_dir, f"save_{save_id}.json")
            if os.path.exists(save_file):
                saveid_to_exist[save_id] = True
                save_list_text += f"({save_id}) 舊有存檔\n"
            else:
                saveid_to_exist[save_id] = False
                save_list_text += f"({save_id}) 空白存檔\n"

        # get save file
        user_prompt = f"{save_list_text}\n使用的存檔欄位： "
        while True:
            text_in = input(user_prompt)
            if start_type == "new":
                if text_in in saveid_to_exist:
                    use_save_id = text_in
                    break
            else:
                if saveid_to_exist.get(text_in, False):
                    use_save_id = text_in
                    break

        # initialize state
        self.summary_file = os.path.join(self.output_dir, f"summary_{use_save_id}.txt")
        use_save_file = os.path.join(self.state_dir, f"save_{use_save_id}.json")
        self.state = State(use_save_file)
        if start_type == "new":
            # 替換開場白
            user_prompt = self.user_prompt_to_text["opening"]
            input(user_prompt + "\n開始試煉(按換行繼續)... ")
            self.state.log += user_prompt
            self.state.save()
        else:
            self.state.load()

        self.run_loop()
        return

    def get_action(self):
        # get available actions
        action_list = [
            "exit",
            "check_pedia",
        ]

        # 檢查是否已發現所有配方
        if len(self.state.recipe_list) < len(self.all_recipe_list):
            action_list.append("discover_recipe")

        # 檢查是否有可研究的配方
        if self.state.recipe_list:
            action_list.append("study_recipe")

        action_list_text = "\n行動列表：\n"
        actionid_to_action = {}
        for i, action in enumerate(action_list):
            action_id = str(i)
            actionid_to_action[action_id] = action
            action_text = self.action_to_text[action]
            action_list_text += f"({action_id}) {action_text}\n"

        # get action
        # 替換角色名稱為「煉金師」
        user_prompt = f"{action_list_text}\n煉金師的下一步： "
        while True:
            text_in = input(user_prompt)
            if text_in in actionid_to_action:
                use_action = actionid_to_action[text_in]
                break

        return use_action

    def run_loop(self):
        while True:
            # 替換結尾條件：掌握精煉術 AND 創造簽名藥劑
            if not self.state.ended and self.state.mastery_skill and self.state.signature_recipe:
                self.do_end()

            action = self.get_action()

            if action == "exit":
                break

            elif action == "check_pedia":
                self.do_check_pedia() # 替換：查閱藥劑學筆記

            elif action == "discover_recipe":
                self.do_discover_recipe() # 替換：尋找新配方

            elif action == "study_recipe":
                self.do_study_recipe() # 替換：研究配方

        return

    def do_check_pedia(self): # 替換 do_watch_pedia
        action_text = self.action_to_text["check_pedia"]
        user_prompt = f"\n{action_text}：\n"

        if self.state.pedia:
            for recipe, data in self.state.pedia.items():
                info = data["info"]
                potency = data["potency"] # 替換：效能
                harmony = data["harmony"] # 替換：調和度
                user_prompt += f"\n{info}\n\n效能: {potency}\n調和度: {harmony}\n"
        else:
            user_prompt += "...筆記空空如也...\n"

        self.state.save()
        input(user_prompt + "\n(按換行繼續)... ")
        return

    def do_discover_recipe(self): # 替換 do_search_animal
        action_text = self.action_to_text["discover_recipe"]
        user_prompt = f"\n{action_text}：\n"

        searchable_recipe_list = [
            recipe
            for recipe in self.all_recipe_list
            if recipe not in self.state.recipe_list
        ]

        if searchable_recipe_list:
            recipe = random.choice(searchable_recipe_list)
            self.state.recipe_list.append(recipe)

            data = self.recipe_data[recipe]
            # 假設 JSON 內包含 potion_zh, title, description 等欄位
            name_zh = data["potion_zh"]
            title = data["title"]
            description = data["description"]

            search_log = f"\n煉金師發現了一份{title}《{name_zh}》配方！\n"
            self.state.log += search_log
            user_prompt += search_log + "\n\n"

            info = f"{name_zh}\n{description}"
            potency = 0
            harmony = 0

            self.state.pedia[recipe] = {
                "info": info,
                "potency": 0, # 替換：效能
                "harmony": 0, # 替換：調和度
            }
            user_prompt += f"藥劑學筆記更新：\n{info}\n\n效能: {potency}\n調和度: {harmony}\n"

        else:
            user_prompt += "...沒有找到任何新的配方...\n"

        self.state.save()
        input(user_prompt + "\n(按換行繼續)... ")
        return

    def do_study_recipe(self): # 替換 do_visit_animal
        action_text = self.action_to_text["study_recipe"]
        user_prompt = f"\n{action_text}：\n"

        # show recipe list
        user_prompt += "\n煉金師已發現的藥劑配方：\n"
        recipeid_to_recipe = {}
        recipeid_to_name = {}
        for i, recipe in enumerate(self.state.recipe_list):
            recipe_id = str(i + 1)
            recipeid_to_recipe[recipe_id] = recipe

            data = self.recipe_data[recipe]
            name = data["title"] + "《" + data["potion_zh"] + "》"
            recipeid_to_name[recipe_id] = name

            user_prompt += f"({recipe_id}) {name}\n"

        # select recipe
        user_prompt = f"{user_prompt}\n研究： "
        while True:
            text_in = input(user_prompt)
            if text_in in recipeid_to_recipe:
                use_recipe = recipeid_to_recipe[text_in]
                use_recipe_name = recipeid_to_name[text_in]
                break

        visit_log = f"\n煉金師開始研究 {use_recipe_name}\n"
        self.state.log += visit_log

        # interact with recipe (theory or experiment)
        user_prompt = f"\n煉金師研究 {use_recipe_name}：\n"
        user_prompt += "(1) 研讀配方理論 (提升效能)\n(2) (...自行輸入5個字以上的實驗步驟...)\n\n你想要："
        while True:
            text_in = input(user_prompt)
            if text_in == "1":
                self.do_synthesize(use_recipe) # 理論學習
                break
            elif len(text_in) >= 5:
                self.do_experiment(use_recipe, text_in) # 創造性實驗
                break

        self.state.save()
        return

    def do_synthesize(self, recipe): # 替換 do_learn
        # get full info
        static_pedia = self.recipe_data[recipe]
        name = static_pedia["potion_zh"]
        full_info = "[藥劑簡介]\n" + static_pedia["description"]
        for title, content in static_pedia["details"].items():
            full_info += f"\n[{title}]\n{content}"

        # get completed question-answer pairs
        state_pedia = self.state.pedia[recipe]
        if "qa_list" not in state_pedia:
            state_pedia["qa_list"] = []
        qa_list = state_pedia["qa_list"]

        # get new question (LLM digesting its own output)
        if qa_list:
            question_list_text = "已解答的理論問題：\n"
            for i, (question, answer) in enumerate(qa_list):
                question_list_text += f"{i + 1}. {question}\n"

            questions = len(qa_list) + 1
            question_list_text += f"{questions}. "

            instruction_text = f"提出一個可以從以下配方文章中得到答案的第{questions}個理論問題。只能使用單一句問句。"

            gpt_in = f"{instruction_text}\n\n配方文章：\n{full_info}\n\n{question_list_text}"
            use_question = self.gpt.run_gpt(gpt_in, 100)

        else:
            use_question = f"{name}需要哪種稀有晶石作為催化劑？"

        # get answer
        log = f"煉金師被問到理論問題：「{use_question}」"
        self.state.log += f"\n{log}"
        user_prompt = f"\n{log}\n\n煉金師回答： "
        use_answer = input(user_prompt)
        log = f"煉金師回答：「{use_answer}」"
        self.state.log += f"\n{log}"

        # evaluate answer
        gpt_in = f"參考配方資料：\n{full_info}\n\n理論問題：\n{use_question}\n\n答案：\n{use_answer}\n\n請問答案是否與配方資料相符？請回答「是」或「否」一個字"
        gpt_out = self.gpt.run_gpt(gpt_in, 10)
        is_correct = "否" not in gpt_out
        old_potency = state_pedia["potency"] # 替換：效能
        if is_correct:
            new_potency = old_potency + 5
            log = f"理論獲得驗證，藥劑效能大幅提升！"
            user_prompt = f"{log}\n藥劑效能從{old_potency}上升到了{new_potency}\n\n(按換行繼續)... "
            qa_list.append((use_question, use_answer))

        else:
            new_potency = max(0, old_potency - 1)
            log = f"理論存在缺陷，藥劑效能略微下降"
            user_prompt = f"{log}\n藥劑效能剩下{new_potency}\n\n(按換行繼續)... "

        self.state.log += f"\n{log}\n"
        state_pedia["potency"] = new_potency
        input(user_prompt)

        # learn mastery skill
        if not self.state.mastery_skill and recipe != self.state.signature_recipe and new_potency >= 10:
            self.state.mastery_skill = recipe

            name_zh = static_pedia["potion_zh"]
            mastery_zh = static_pedia["mastery_zh"] # 假設 JSON 內有 mastery_zh
            title = static_pedia["title"]

            mentor = f"{title}《{name_zh}》"
            mastery = f"「{mastery_zh}」"

            user_prompt = self.user_prompt_to_text["mastery"] # 替換：精煉術 prompt
            user_prompt = user_prompt.replace("mentor", mentor)
            user_prompt = user_prompt.replace("mastery", mastery)

            input(user_prompt + "\n繼續試煉(按換行繼續)... ")
            self.state.log += user_prompt

        self.state.save()
        return

    def do_experiment(self, recipe, use_experiment): # 替換 do_interact
        # recipe info
        static_pedia = self.recipe_data[recipe]
        name_zh = static_pedia["potion_zh"]
        mastery_zh = static_pedia["mastery_zh"]
        title = static_pedia["title"]

        mentor = f"{title}《{name_zh}》"
        signature = f"《{mastery_zh}》" # 作為精通藥劑的簽名版

        # experiment (interact)
        use_experiment = f"煉金師對 {mentor} 進行創造性實驗：{use_experiment}"
        gpt_in = f"撰寫一個短篇實驗記錄，最多使用3句話。\n\n實驗記錄：{use_experiment}"
        continuation = self.gpt.run_gpt(gpt_in, 200)

        # revise (LLM digesting its own output)
        gpt_in = \
            f"將實驗記錄改寫的較為合理且通順，最多使用3句話，不能使用「試圖」，必須包含「煉金師」和配方名稱「{mentor}」。\n\n" \
            f"實驗記錄：\n{use_experiment}，{continuation}\n\n" \
            f"合理且通順的改編記錄：\n"
        story = self.gpt.run_gpt(gpt_in, 200)

        # score (Check for harmony/finesse)
        gpt_in = f"實驗記錄：\n{story}\n\n這是個精妙的調和嗎？請回答「是」或「否」一個字"
        feedback = self.gpt.run_gpt(gpt_in, 10)
        if "否" in feedback:
            score = random.randrange(-1, 5)
        else:
            score = 5

        state_pedia = self.state.pedia[recipe]
        old_harmony = state_pedia["harmony"] # 替換：調和度
        new_harmony = old_harmony + score

        log = f"\n{story}\n"
        self.state.log += log
        user_prompt = f"{log}\n這份配方的調和度現在是{new_harmony}\n\n(按換行繼續)... "
        state_pedia["harmony"] = new_harmony
        input(user_prompt)

        # learn signature recipe
        if not self.state.signature_recipe and recipe != self.state.mastery_skill and new_harmony >= 10:
            self.state.signature_recipe = recipe

            user_prompt = self.user_prompt_to_text["signature"] # 替換：簽名藥劑 prompt
            user_prompt = user_prompt.replace("mentor", mentor)
            user_prompt = user_prompt.replace("signature", signature)

            input(user_prompt + "\n繼續試煉(按換行繼續)... ")
            self.state.log += user_prompt

        self.state.save()
        return

    def do_end(self): # 結尾邏輯不變，只替換文本
        # get ending text
        mastery_pedia = self.recipe_data[self.state.mastery_skill]
        mastery_name_zh = mastery_pedia["potion_zh"]
        mastery_title = mastery_pedia["title"]
        mastery_skill_zh = mastery_pedia["mastery_zh"] # 假設 JSON 內有 mastery_zh

        signature_pedia = self.recipe_data[self.state.signature_recipe]
        signature_recipe_zh = signature_pedia["mastery_zh"] # 假設 JSON 內有 mastery_zh
        signature_title = signature_pedia["title"]

        # 替換結尾文本的變數
        mentor = f"{mastery_title}《{mastery_name_zh}》"
        mastery = f"精煉術「{mastery_skill_zh}」"
        signature = f"簽名藥劑「{signature_title}《{signature_recipe_zh}》」"

        user_prompt = self.user_prompt_to_text["ending"]
        user_prompt = user_prompt.replace("mentor", mentor)
        user_prompt = user_prompt.replace("mastery", mastery)
        user_prompt = user_prompt.replace("signature", signature)
        self.state.log += user_prompt
        input(user_prompt + "\n(按換行生成公會總評)... ")

        # get story summary (LLM 總結與消化)
        story = self.state.log
        story = re.sub(r"\n+", "\n", story).strip()
        instruction = "將這段完整的煉金師試煉記錄精簡為一份正式的公會總評報告。使用至少20句話。其中不能出現「10」"
        while True:
            gpt_in = f"{story}\n\n{instruction}"
            summary = self.gpt4.run_gpt(gpt_in, 1000)
            if summary:
                break
            # 應對過長 log 的容錯處理
            story = story[:-100]

        summary = re.sub(r"\n+", "\n", summary).strip()

        # revise summary (LLM 消化自身輸出並修正)
        instruction = "根據事實，將總評報告修正為一份符合事實且專業的最終報告。使用至少20句話。其中不能出現「10」"
        while True:
            gpt_in = f"事實：\n{story}\n\n總評草稿：\n{summary}\n\n{instruction}"
            revision = self.gpt4.run_gpt(gpt_in, 1000)
            if revision:
                break
            # 應對過長 log 的容錯處理
            story = story[:-100]

        write_txt(self.summary_file, revision)
        user_prompt = f"\n\n最終公會總評報告：\n{revision}\n\n(按換行繼續)... "
        input(user_prompt)

        self.state.ended = True
        self.state.save()
        return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="lab3_config.json")
    arg = parser.parse_args()

    # 確保您已經設定了 OpenAI API Key
    openai.api_key = input("OpenAI API Key: ")

    config = Config(arg.config_file)
    game = Game(config)
    game.run_start()
    return


if __name__ == "__main__":
    main()
    sys.exit()
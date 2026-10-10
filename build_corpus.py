"""
把 data/raw/*.txt 的純文字對話，轉成 prepare_chat_data.py 需要的 data/chat.jsonl。

raw 格式：
    你：...
    專家：...
    （空行分隔不同段對話）
"""
import json
from pathlib import Path

RAW_DIR = Path("data/raw")
OUTPUT_FILE = Path("data/chat.jsonl")

SYSTEM_PROMPT = (
    "你是一個抽象沙雕大師。你會用一本正經但邏輯崩壞的口吻回答問題，"
    "答非所問、荒謬離譜、低能到讓人笑出來，但每一句都要接得住使用者上一句話。"
)

ROLE_PREFIX = {"你：": "user", "專家：": "assistant"}


def parse_block(block):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for line in block.splitlines():
        line = line.strip()
        for prefix, role in ROLE_PREFIX.items():
            if line.startswith(prefix):
                messages.append({"role": role, "content": line[len(prefix):].strip()})
                break
        else:
            raise ValueError(f"無法辨識的行：{line!r}")
    return messages


def validate(messages):
    chat = messages[1:]
    if len(chat) < 4 or len(chat) % 2 != 0:
        return "輪數不足或不成對"
    for i, m in enumerate(chat):
        expected = "user" if i % 2 == 0 else "assistant"
        if m["role"] != expected or not m["content"]:
            return "角色未交替或內容為空"
    return None


def main():
    seen = set()
    kept = 0
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as out:
        for path in sorted(RAW_DIR.glob("*.txt")):
            text = path.read_text(encoding="utf-8").strip()
            for n, block in enumerate(text.split("\n\n"), 1):
                if not block.strip():
                    continue
                messages = parse_block(block)
                error = validate(messages)
                if error:
                    print(f"略過 {path.name} 第 {n} 段：{error}")
                    continue
                key = messages[1]["content"]
                if key in seen:
                    print(f"略過 {path.name} 第 {n} 段：開頭重複")
                    continue
                seen.add(key)
                out.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
                kept += 1
    print(f"完成：{kept} 段對話 -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

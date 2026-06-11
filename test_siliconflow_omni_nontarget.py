from __future__ import annotations

import base64
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI


PROJECT_DIR = Path(r"D:\Alpha\speaker_verification_eval")
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "siliconflow"

MODEL_NAME = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

AUDIO_1 = DATA_DIR / "A1.wav"
AUDIO_2 = DATA_DIR / "B1.wav"

PROMPT = """
你将按顺序收到两段相互独立的语音，分别称为“音频1”和“音频2”。

任务：判断两段语音是否由同一个自然人说出。

判断时主要考虑说话人的音色、共振特征、发音习惯、语速和其他稳定的说话人特征。
不要仅根据语音中的文字内容是否相同进行判断。

允许的判断结果只有：
same：两段语音来自同一个人
different：两段语音来自不同的人
uncertain：现有音频不足以可靠判断

返回一个JSON对象，且只能包含以下两个字段：
decision：取值必须为same、different或uncertain
confidence：0到1之间的数字，表示你对本次判断的确信程度

不要输出解释、Markdown、代码块或其他内容。
不要采用任何默认答案，必须根据实际收到的两段音频独立判断。
""".strip()


def encode_audio_as_data_url(audio_path: Path) -> str:
    """将本地WAV文件编码为Base64 Data URL。"""
    if not audio_path.is_file():
        raise FileNotFoundError(f"找不到音频文件：{audio_path}")

    audio_bytes = audio_path.read_bytes()
    encoded = base64.b64encode(audio_bytes).decode("utf-8")

    return f"data:audio/wav;base64,{encoded}"


def parse_json_response(raw_text: str) -> dict[str, Any] | None:
    """尝试把模型返回内容解析为JSON。"""
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, dict):
        return None

    return parsed


def main() -> None:
    api_key = os.getenv("SILICONFLOW_API_KEY")

    if not api_key:
        raise SystemExit(
            "没有检测到 SILICONFLOW_API_KEY 环境变量。"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("正在读取并编码两段音频……")
    audio_1_url = encode_audio_as_data_url(AUDIO_1)
    audio_2_url = encode_audio_as_data_url(AUDIO_2)

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.siliconflow.cn/v1",
        timeout=180.0,
    )

    print(f"正在调用模型：{MODEL_NAME}")
    start_time = time.perf_counter()

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是一个执行说话人验证实验的音频分析模型。"
                    "你必须严格按照用户要求输出JSON。"
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "下面是音频1：",
                    },
                    {
                        "type": "audio_url",
                        "audio_url": {
                            "url": audio_1_url,
                        },
                    },
                    {
                        "type": "text",
                        "text": "下面是音频2：",
                    },
                    {
                        "type": "audio_url",
                        "audio_url": {
                            "url": audio_2_url,
                        },
                    },
                    {
                        "type": "text",
                        "text": PROMPT,
                    },
                ],
            },
        ],
        temperature=0,
        max_tokens=200,
        response_format={"type": "json_object"},
    )

    elapsed_seconds = time.perf_counter() - start_time

    raw_text = response.choices[0].message.content or ""
    parsed_result = parse_json_response(raw_text)

    result = {
        "task_type": "speaker_verification_by_audio_llm",
        "trial_id": "SF_NONTARGET_001",
        "ground_truth": "different",
        "provider": "siliconflow",
        "model": MODEL_NAME,
        "audio_1": str(AUDIO_1.resolve()),
        "audio_2": str(AUDIO_2.resolve()),
        "prompt": PROMPT,
        "raw_response": raw_text,
        "parsed_response": parsed_result,
        "elapsed_seconds": round(elapsed_seconds, 6),
        "usage": (
            response.usage.model_dump()
            if response.usage is not None
            else None
        ),
        "response_id": response.id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    output_path = OUTPUT_DIR / "target_A1_B1.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    print("\n模型原始回答：")
    print(raw_text)

    print(f"\n耗时：{elapsed_seconds:.2f} 秒")
    print(f"结果已保存到：{output_path}")


if __name__ == "__main__":
    main()
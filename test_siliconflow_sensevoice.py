from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


PROJECT_DIR = Path(r"D:\Alpha\speaker_verification_eval")
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "sensevoice"

API_URL = "https://api.siliconflow.cn/v1/audio/transcriptions"
MODEL_NAME = "FunAudioLLM/SenseVoiceSmall"

AUDIO_FILE = DATA_DIR / "A1.wav"


def transcribe_audio(audio_path: Path) -> dict[str, Any]:
    api_key = os.getenv("SILICONFLOW_API_KEY")

    if not api_key:
        raise RuntimeError(
            "没有检测到 SILICONFLOW_API_KEY 环境变量。"
        )

    if not audio_path.is_file():
        raise FileNotFoundError(f"找不到音频文件：{audio_path}")

    headers = {
        "Authorization": f"Bearer {api_key}",
    }

    with audio_path.open("rb") as audio_file:
        files = {
            "file": (
                audio_path.name,
                audio_file,
                "audio/wav",
            )
        }

        data = {
            "model": MODEL_NAME,
        }

        start_time = time.perf_counter()

        response = requests.post(
            API_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=180,
        )

        elapsed_seconds = time.perf_counter() - start_time

    trace_id = response.headers.get("x-siliconcloud-trace-id")

    try:
        response_json = response.json()
    except json.JSONDecodeError:
        response_json = None

    result = {
        "task_type": "audio_transcription",
        "trial_id": "SVS_ASR_A1_001",
        "provider": "siliconflow",
        "model": MODEL_NAME,
        "audio": str(audio_path.resolve()),
        "http_status_code": response.status_code,
        "raw_text": response.text,
        "response_json": response_json,
        "transcript": (
            response_json.get("text")
            if isinstance(response_json, dict)
            else None
        ),
        "elapsed_seconds": round(elapsed_seconds, 6),
        "trace_id": trace_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    response.raise_for_status()

    return result


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"正在调用单模态音频模型：{MODEL_NAME}")
    print(f"音频文件：{AUDIO_FILE}")

    result = transcribe_audio(AUDIO_FILE)

    output_path = OUTPUT_DIR / "sensevoice_A1.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    print("\n识别文本：")
    print(result["transcript"])

    print(f"\n耗时：{result['elapsed_seconds']} 秒")
    print(f"结果已保存到：{output_path}")


if __name__ == "__main__":
    main()
from __future__ import annotations

import csv
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

AUDIO_FILES = [
    {
        "trial_id": "SVS_ASR_A1_001",
        "audio_path": DATA_DIR / "A1.wav",
    },
    {
        "trial_id": "SVS_ASR_A2_001",
        "audio_path": DATA_DIR / "A2.wav",
    },
    {
        "trial_id": "SVS_ASR_B1_001",
        "audio_path": DATA_DIR / "B1.wav",
    },
]


def transcribe_audio(
    audio_path: Path,
    trial_id: str,
) -> dict[str, Any]:
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
        "trial_id": trial_id,
        "provider": "siliconflow",
        "model": MODEL_NAME,
        "audio": str(audio_path.resolve()),
        "audio_filename": audio_path.name,
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


def save_json_result(result: dict[str, Any]) -> Path:
    output_path = OUTPUT_DIR / f"{result['trial_id']}.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    return output_path


def save_summary_csv(results: list[dict[str, Any]]) -> Path:
    summary_path = OUTPUT_DIR / "sensevoice_batch_summary.csv"

    fieldnames = [
        "trial_id",
        "audio_filename",
        "model",
        "http_status_code",
        "transcript",
        "elapsed_seconds",
        "trace_id",
        "created_at_utc",
    ]

    with summary_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            writer.writerow({
                "trial_id": result["trial_id"],
                "audio_filename": result["audio_filename"],
                "model": result["model"],
                "http_status_code": result["http_status_code"],
                "transcript": result["transcript"],
                "elapsed_seconds": result["elapsed_seconds"],
                "trace_id": result["trace_id"],
                "created_at_utc": result["created_at_utc"],
            })

    return summary_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"批量调用单模态音频模型：{MODEL_NAME}")
    print(f"输出目录：{OUTPUT_DIR}\n")

    results: list[dict[str, Any]] = []

    for index, item in enumerate(AUDIO_FILES, start=1):
        trial_id = item["trial_id"]
        audio_path = item["audio_path"]

        print(f"[{index}/{len(AUDIO_FILES)}] 正在识别：{audio_path.name}")

        result = transcribe_audio(
            audio_path=audio_path,
            trial_id=trial_id,
        )

        json_path = save_json_result(result)
        results.append(result)

        print(f"  识别文本：{result['transcript']}")
        print(f"  耗时：{result['elapsed_seconds']} 秒")
        print(f"  JSON结果：{json_path}\n")

    summary_path = save_summary_csv(results)

    print("批量识别完成。")
    print(f"汇总CSV：{summary_path}")


if __name__ == "__main__":
    main()
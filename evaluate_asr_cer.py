from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(r"D:\Alpha\speaker_verification_eval")

GROUND_TRUTH_FILE = PROJECT_DIR / "data" / "trials" / "asr_trials.csv"
PREDICTION_FILE = (
    PROJECT_DIR
    / "outputs"
    / "sensevoice"
    / "sensevoice_batch_summary.csv"
)

OUTPUT_DIR = PROJECT_DIR / "outputs" / "sensevoice"
OUTPUT_CSV = OUTPUT_DIR / "sensevoice_asr_eval.csv"
OUTPUT_JSON = OUTPUT_DIR / "sensevoice_asr_eval.json"


def normalize_text(text: str) -> str:
    """
    ASR评测用的简单文本规范化：
    - 去除空白
    - 去除常见中英文标点
    - 转小写
    """
    text = text.strip().lower()

    text = re.sub(r"\s+", "", text)

    punctuation_pattern = (
        r"[，。！？、；：“”‘’（）《》〈〉【】\[\]{}"
        r",.!?;:\"'()<>]"
    )

    text = re.sub(punctuation_pattern, "", text)

    return text


def levenshtein_distance(reference: str, hypothesis: str) -> int:
    """
    计算两个字符串之间的编辑距离。
    对中文ASR来说，这里按“字”计算。
    """
    ref_len = len(reference)
    hyp_len = len(hypothesis)

    dp = [[0] * (hyp_len + 1) for _ in range(ref_len + 1)]

    for i in range(ref_len + 1):
        dp[i][0] = i

    for j in range(hyp_len + 1):
        dp[0][j] = j

    for i in range(1, ref_len + 1):
        for j in range(1, hyp_len + 1):
            cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,       # 删除
                dp[i][j - 1] + 1,       # 插入
                dp[i - 1][j - 1] + cost # 替换
            )

    return dp[ref_len][hyp_len]


def calculate_cer(reference: str, hypothesis: str) -> float:
    """
    CER = 编辑距离 / 标准答案字数
    """
    if len(reference) == 0:
        if len(hypothesis) == 0:
            return 0.0

        return 1.0

    distance = levenshtein_distance(reference, hypothesis)

    return distance / len(reference)


def read_csv_by_trial_id(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"找不到文件：{path}")

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        result: dict[str, dict[str, str]] = {}

        for row in reader:
            trial_id = row["trial_id"]
            result[trial_id] = row

        return result


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ground_truth_rows = read_csv_by_trial_id(GROUND_TRUTH_FILE)
    prediction_rows = read_csv_by_trial_id(PREDICTION_FILE)

    evaluation_results: list[dict[str, Any]] = []

    for trial_id, truth_row in ground_truth_rows.items():
        if trial_id not in prediction_rows:
            raise KeyError(
                f"预测结果中找不到 trial_id：{trial_id}"
            )

        prediction_row = prediction_rows[trial_id]

        ground_truth_text = truth_row["ground_truth_text"]
        predicted_text = prediction_row.get("transcript", "")

        normalized_truth = normalize_text(ground_truth_text)
        normalized_prediction = normalize_text(predicted_text)

        edit_distance = levenshtein_distance(
            normalized_truth,
            normalized_prediction,
        )

        cer = calculate_cer(
            normalized_truth,
            normalized_prediction,
        )

        exact_match = normalized_truth == normalized_prediction

        result = {
            "trial_id": trial_id,
            "audio": truth_row["audio"],
            "model": prediction_row.get("model", ""),
            "ground_truth_text": ground_truth_text,
            "predicted_text": predicted_text,
            "normalized_ground_truth": normalized_truth,
            "normalized_prediction": normalized_prediction,
            "edit_distance": edit_distance,
            "reference_char_count": len(normalized_truth),
            "cer": round(cer, 6),
            "exact_match": exact_match,
            "elapsed_seconds": prediction_row.get("elapsed_seconds", ""),
            "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        }

        evaluation_results.append(result)

    fieldnames = list(evaluation_results[0].keys())

    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(evaluation_results)

    with OUTPUT_JSON.open("w", encoding="utf-8") as file:
        json.dump(
            evaluation_results,
            file,
            ensure_ascii=False,
            indent=2,
        )

    total = len(evaluation_results)
    exact_count = sum(
        1 for item in evaluation_results
        if item["exact_match"]
    )

    average_cer = sum(
        item["cer"] for item in evaluation_results
    ) / total

    print("ASR评测完成。")
    print("=" * 50)

    for item in evaluation_results:
        print(f"trial_id: {item['trial_id']}")
        print(f"标准答案: {item['ground_truth_text']}")
        print(f"模型输出: {item['predicted_text']}")
        print(f"CER: {item['cer']:.6f}")
        print(f"完全一致: {item['exact_match']}")
        print("-" * 50)

    print(f"总数: {total}")
    print(f"完全一致数量: {exact_count}")
    print(f"完全一致率: {exact_count / total:.4f}")
    print(f"平均CER: {average_cer:.6f}")
    print()
    print(f"CSV结果: {OUTPUT_CSV}")
    print(f"JSON结果: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
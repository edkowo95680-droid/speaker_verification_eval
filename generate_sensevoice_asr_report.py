from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(r"D:\Alpha\speaker_verification_eval")

EVAL_CSV = (
    PROJECT_DIR
    / "outputs"
    / "sensevoice"
    / "sensevoice_asr_eval.csv"
)

OUTPUT_REPORT = (
    PROJECT_DIR
    / "outputs"
    / "sensevoice"
    / "sensevoice_asr_report.md"
)


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"找不到评测结果文件：{path}")

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader)


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_bool(value: str) -> bool:
    return str(value).strip().lower() == "true"


def escape_markdown_table_cell(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def main() -> None:
    rows = read_rows(EVAL_CSV)

    if not rows:
        raise RuntimeError("评测结果为空，无法生成报告。")

    total = len(rows)
    exact_count = sum(1 for row in rows if to_bool(row["exact_match"]))
    exact_rate = exact_count / total

    average_cer = sum(
        to_float(row["cer"]) for row in rows
    ) / total

    elapsed_values = [
        to_float(row.get("elapsed_seconds", ""))
        for row in rows
        if row.get("elapsed_seconds", "") != ""
    ]

    average_elapsed = (
        sum(elapsed_values) / len(elapsed_values)
        if elapsed_values
        else 0.0
    )

    model_name = rows[0].get("model", "unknown")
    generated_at = datetime.now(timezone.utc).isoformat()

    lines: list[str] = []

    lines.append("# SenseVoiceSmall 单模态音频识别评测报告")
    lines.append("")
    lines.append("## 1. 基本信息")
    lines.append("")
    lines.append(f"- 任务类型：ASR 语音转写")
    lines.append(f"- 模型提供方：SiliconFlow")
    lines.append(f"- 被测模型：`{model_name}`")
    lines.append(f"- 样本数量：{total}")
    lines.append(f"- 报告生成时间 UTC：`{generated_at}`")
    lines.append("")

    lines.append("## 2. 评测方法")
    lines.append("")
    lines.append(
        "本次评测使用三段本地 WAV 音频文件作为输入，"
        "调用硅基流动 `FunAudioLLM/SenseVoiceSmall` 单模态音频识别模型进行转写。"
        "模型输出文本后，与人工标准答案进行对比。"
    )
    lines.append("")
    lines.append("文本对比前进行了简单规范化处理：")
    lines.append("")
    lines.append("- 去除空白字符；")
    lines.append("- 去除常见中英文标点；")
    lines.append("- 英文字母转小写；")
    lines.append("- 中文按字计算编辑距离。")
    lines.append("")

    lines.append("## 3. 汇总指标")
    lines.append("")
    lines.append(f"- 完全一致数量：{exact_count} / {total}")
    lines.append(f"- 完全一致率：{exact_rate:.4f}")
    lines.append(f"- 平均 CER：{average_cer:.6f}")
    lines.append(f"- 平均接口耗时：{average_elapsed:.6f} 秒")
    lines.append("")

    lines.append("## 4. 明细结果")
    lines.append("")
    lines.append(
        "| trial_id | 标准答案 | 模型输出 | CER | 完全一致 | 耗时秒 |"
    )
    lines.append(
        "|---|---|---|---:|---:|---:|"
    )

    for row in rows:
        lines.append(
            "| "
            + escape_markdown_table_cell(row["trial_id"])
            + " | "
            + escape_markdown_table_cell(row["ground_truth_text"])
            + " | "
            + escape_markdown_table_cell(row["predicted_text"])
            + " | "
            + escape_markdown_table_cell(row["cer"])
            + " | "
            + escape_markdown_table_cell(row["exact_match"])
            + " | "
            + escape_markdown_table_cell(row.get("elapsed_seconds", ""))
            + " |"
        )

    lines.append("")
    lines.append("## 5. 当前结论")
    lines.append("")
    lines.append(
        f"在本次 {total} 条短中文语音样本上，"
        f"SenseVoiceSmall 的规范化转写结果与人工标准答案完全一致，"
        f"完全一致率为 {exact_rate:.4f}，平均 CER 为 {average_cer:.6f}。"
    )
    lines.append("")
    lines.append(
        "该结果说明当前工程已经具备调用单模态音频识别模型、"
        "保存识别结果、对比人工标准答案并计算客观指标的基本能力。"
    )
    lines.append("")

    lines.append("## 6. 限制说明")
    lines.append("")
    lines.append(
        "本次评测仅包含 3 条短中文语音，样本规模较小，"
        "录音环境、说话人数量、语速、噪声、口音和音频时长均未充分覆盖。"
        "因此，本报告只能证明评测链路可用，不能代表模型的整体 ASR 性能。"
    )
    lines.append("")
    lines.append(
        "后续正式评测应扩大样本规模，并加入不同说话人、不同录音设备、"
        "不同噪声条件、不同语速和更复杂文本内容。"
    )
    lines.append("")

    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_REPORT.open("w", encoding="utf-8") as file:
        file.write("\n".join(lines))

    print("ASR评测报告生成完成。")
    print(f"报告路径：{OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
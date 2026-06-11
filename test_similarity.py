from pathlib import Path

import wespeaker


PROJECT_DIR = Path(r"D:\Alpha\speaker_verification_eval")
MODEL_DIR = PROJECT_DIR / "models" / "chinese"
DATA_DIR = PROJECT_DIR / "data"

A1 = DATA_DIR / "A1.wav"
A2 = DATA_DIR / "A2.wav"
B1 = DATA_DIR / "B1.wav"


def to_float(value) -> float:
    """将模型返回的标量或Tensor统一转换为普通浮点数。"""
    if hasattr(value, "detach"):
        value = value.detach()

    if hasattr(value, "cpu"):
        value = value.cpu()

    if hasattr(value, "item"):
        value = value.item()

    return float(value)


def check_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"找不到文件：{path}")


def main() -> None:
    check_file(MODEL_DIR / "avg_model.pt")
    check_file(MODEL_DIR / "config.yaml")
    check_file(A1)
    check_file(A2)
    check_file(B1)

    print("正在加载本地中文声纹模型……")
    model = wespeaker.load_model(str(MODEL_DIR))
    print("模型加载成功。\n")

    same_score = to_float(
        model.compute_similarity(str(A1), str(A2))
    )

    different_score = to_float(
        model.compute_similarity(str(A1), str(B1))
    )

    print("第一次测试结果")
    print("=" * 40)
    print(f"A1 对 A2（同一人）：{same_score:.6f}")
    print(f"A1 对 B1（不同人）：{different_score:.6f}")
    print("=" * 40)

    if same_score > different_score:
        print("结果符合基本预期：同一人的相似度更高。")
    elif same_score == different_score:
        print("两组分数相同，需要进一步检查录音和模型输出。")
    else:
        print("同一人分数反而较低，需要检查录音条件或增加测试样本。")


if __name__ == "__main__":
    main()
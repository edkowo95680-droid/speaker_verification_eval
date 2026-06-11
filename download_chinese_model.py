from __future__ import annotations

import shutil
import tarfile
import time
from pathlib import Path

import requests


API_URL = (
    "https://modelscope.cn/api/v1/datasets/"
    "wenet/wespeaker_pretrained_models/oss/tree"
)
# 只需要这个预训练包，不下载整个模型目录。
MODEL_FILENAME = "cnceleb_resnet34.tar.gz"

# 约定好的工程根目录，后面所有路径都基于它拼出来。
PROJECT_DIR = Path(r"D:\Alpha\speaker_verification_eval")
MODEL_DIR = PROJECT_DIR / "models" / "chinese"
ARCHIVE_PATH = MODEL_DIR / MODEL_FILENAME

# 断点续传最多重试次数，避免偶发网络抖动导致脚本直接失败。
MAX_ATTEMPTS = 10
# 分块下载大小，既减少内存占用，也避免一次性写入太大。
CHUNK_SIZE = 1024 * 1024


def get_fresh_download_url() -> str:
    """从ModelScope接口取得新的模型下载地址。"""
    # 这里先访问目录树接口，再从返回的条目里挑出目标压缩包对应的真实下载链接。
    response = requests.get(API_URL, timeout=30)
    response.raise_for_status()

    items = response.json()["Data"]

    model_info = next(
        item for item in items
        if item["Key"] == MODEL_FILENAME
    )
    return model_info["Url"]


def get_expected_size(
    response: requests.Response,
    existing_size: int,
) -> int | None:
    """根据HTTP响应头计算完整文件应有的大小。"""
    # 优先用 Content-Range，因为它能明确告诉我们整个文件的总大小。
    content_range = response.headers.get("Content-Range")

    if content_range and "/" in content_range:
        total = content_range.rsplit("/", 1)[1]
        if total.isdigit():
            return int(total)

    content_length = response.headers.get("Content-Length")
    if content_length and content_length.isdigit():
        # 如果没有 Content-Range，就退回到“已存在部分 + 本次传输长度”的估算方式。
        return existing_size + int(content_length)

    return None


def download_with_resume() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        # 每次重试前都重新查看本地已有进度，这样中途失败后可以继续接着下。
        existing_size = (
            ARCHIVE_PATH.stat().st_size
            if ARCHIVE_PATH.exists()
            else 0
        )

        print(
            f"\n第 {attempt}/{MAX_ATTEMPTS} 次尝试，"
            f"当前已有 {existing_size} 字节"
        )

        try:
            download_url = get_fresh_download_url()

            # 如果本地已有部分文件，就尝试从断点位置继续请求。
            headers = {}
            if existing_size > 0:
                headers["Range"] = f"bytes={existing_size}-"

            with requests.get(
                download_url,
                headers=headers,
                stream=True,
                timeout=(30, 120),
            ) as response:
                response.raise_for_status()

                # 206表示服务器接受断点续传。
                if existing_size > 0 and response.status_code == 206:
                    mode = "ab"
                else:
                    # 服务器忽略Range时，从头覆盖，避免重复拼接。
                    existing_size = 0
                    mode = "wb"

                # 先估算完整包大小，用来判断这次下载是不是已经拿全了。
                expected_size = get_expected_size(
                    response,
                    existing_size,
                )

                # 逐块写入到磁盘，避免把整个压缩包加载进内存。
                with ARCHIVE_PATH.open(mode) as output:
                    for chunk in response.iter_content(
                        chunk_size=CHUNK_SIZE
                    ):
                        if chunk:
                            output.write(chunk)

            actual_size = ARCHIVE_PATH.stat().st_size

            print(f"当前文件大小：{actual_size} 字节")

            if expected_size is None:
                # 服务端没有提供足够的信息时，直接进入解压验证步骤。
                print("服务器未提供完整大小，将直接验证压缩包。")
                return

            print(f"预期文件大小：{expected_size} 字节")

            if actual_size == expected_size:
                print("模型压缩包下载完整。")
                return

            print("文件仍不完整，准备继续下载。")

        except Exception as exc:
            print(f"本次下载中断：{exc}")

        time.sleep(2)

    raise RuntimeError(
        f"连续 {MAX_ATTEMPTS} 次仍未完成下载。"
    )


def extract_required_files() -> None:
    """只提取WeSpeaker加载所需的两个文件。"""
    # 模型目录里可能还有很多辅助文件，这里只保留运行时真正需要的资源。
    required = {"avg_model.pt", "config.yaml"}
    extracted: set[str] = set()

    with tarfile.open(ARCHIVE_PATH, mode="r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue

            filename = Path(member.name).name
            if filename not in required:
                continue

            source = archive.extractfile(member)
            if source is None:
                continue

            destination = MODEL_DIR / filename

            with source, destination.open("wb") as output:
                shutil.copyfileobj(source, output)

            extracted.add(filename)
            print(f"已提取：{destination}")

    missing = required - extracted

    if missing:
        raise RuntimeError(
            f"压缩包中缺少必要文件：{sorted(missing)}"
        )


def main() -> None:
    # 主流程很简单：先确保压缩包下载完成，再解出必要文件。
    download_with_resume()

    print("\n开始验证并提取模型文件……")
    extract_required_files()

    print("\n中文模型准备完成：")
    print(MODEL_DIR / "avg_model.pt")
    print(MODEL_DIR / "config.yaml")


if __name__ == "__main__":
    main()
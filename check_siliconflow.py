from __future__ import annotations

import os

from openai import OpenAI


def main() -> None:
    api_key = os.getenv("SILICONFLOW_API_KEY")

    if not api_key:
        raise SystemExit(
            "没有检测到 SILICONFLOW_API_KEY 环境变量。"
        )

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.siliconflow.cn/v1",
        timeout=60.0,
    )

    print("正在连接硅基流动并查询模型列表……")

    models = client.models.list()

    omni_models = sorted(
        model.id
        for model in models.data
        if "omni" in model.id.lower()
    )

    print("API连接成功。")

    if omni_models:
        print("\n当前账户可见的 Omni 模型：")
        for model_id in omni_models:
            print(f"- {model_id}")
    else:
        print("\n模型列表中没有找到名称包含 Omni 的模型。")


if __name__ == "__main__":
    main()
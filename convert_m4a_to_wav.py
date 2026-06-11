from __future__ import annotations

import subprocess
import wave
from pathlib import Path

import imageio_ffmpeg


PROJECT_DIR = Path(r"D:\Alpha\speaker_verification_eval")
OUTPUT_DIR = PROJECT_DIR / "data"

FILES = {
    "A1(M4A).m4a": "A1.wav",
    "A2(M4A).m4a": "A2.wav",
    "B1(M4A).m4a": "B1.wav",
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    print(f"使用 FFmpeg：{ffmpeg_exe}\n")

    for source_name, output_name in FILES.items():
        source_path = PROJECT_DIR / source_name
        output_path = OUTPUT_DIR / output_name

        if not source_path.is_file():
            raise FileNotFoundError(
                f"找不到输入文件：{source_path}\n"
                "请检查文件名是否与 dir /b *.m4a 显示的名称完全一致。"
            )

        command = [
            ffmpeg_exe,
            "-y",
            "-i",
            str(source_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ]

        print(f"正在转换：{source_name} -> {output_name}")
        subprocess.run(command, check=True)

        with wave.open(str(output_path), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_rate = wav_file.getframerate()
            bit_depth = wav_file.getsampwidth() * 8
            duration = wav_file.getnframes() / sample_rate

        print(
            f"转换成功：{output_path}\n"
            f"  声道数：{channels}\n"
            f"  采样率：{sample_rate} Hz\n"
            f"  位深：{bit_depth} bit\n"
            f"  时长：{duration:.2f} 秒\n"
        )

    print("三个文件全部转换完成。")


if __name__ == "__main__":
    main()
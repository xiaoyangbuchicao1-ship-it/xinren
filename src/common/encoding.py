"""UTF-8 读写和编码冒烟测试。"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ENCODING_SAMPLE = {
    "中文测试": "专家、信任、共识",
    "数字测试": [0.3, 0.7, 1.25, -0.05],
    "公式标签": ["delta_star", "eta", "beta", "ACD"],
}


def configure_console_utf8() -> None:
    """在支持 reconfigure 的终端中显式设置 UTF-8。"""

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def write_json(data: Any, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    return output


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_text_encoding_smoke_test(directory: str | Path) -> dict[str, str]:
    """写入并回读 Markdown、JSON、YAML 和 Excel 兼容 CSV。"""

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)

    markdown = root / "encoding_sample.md"
    markdown.write_text(
        "# 编码测试\n\n中文：专家、信任、共识  \n数字：0.3、0.7、1.25、-0.05  \n公式：\\(\\delta^*\\)、\\(\\eta\\)、\\(\\beta\\)、\\(ACD\\)\n",
        encoding="utf-8",
        newline="\n",
    )

    json_path = write_json(ENCODING_SAMPLE, root / "encoding_sample.json")

    yaml_path = root / "encoding_sample.yaml"
    with yaml_path.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(ENCODING_SAMPLE, handle, allow_unicode=True, sort_keys=False)

    csv_path = root / "encoding_sample.csv"
    frame = pd.DataFrame(
        {
            "label": ["专家", "信任", "共识", "delta_star"],
            "value": [0.3, 0.7, 1.25, -0.05],
        }
    )
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # 回读并进行内容级验证，避免文件存在但字符已经损坏。
    if "专家" not in markdown.read_text(encoding="utf-8"):
        raise RuntimeError("Markdown UTF-8 回读失败。")
    if read_json(json_path) != ENCODING_SAMPLE:
        raise RuntimeError("JSON UTF-8 回读失败。")
    with yaml_path.open("r", encoding="utf-8") as handle:
        if yaml.safe_load(handle) != ENCODING_SAMPLE:
            raise RuntimeError("YAML UTF-8 回读失败。")
    csv_readback = pd.read_csv(csv_path, encoding="utf-8-sig")
    if csv_readback["label"].tolist()[:3] != ["专家", "信任", "共识"]:
        raise RuntimeError("CSV UTF-8-SIG 回读失败。")

    return {
        "markdown": str(markdown),
        "json": str(json_path),
        "yaml": str(yaml_path),
        "csv": str(csv_path),
    }


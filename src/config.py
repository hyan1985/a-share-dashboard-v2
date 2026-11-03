"""配置加载：路径、.env、modules.yaml、thresholds.yaml。"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
WEB_DIR = ROOT / "web"
OUTPUT_DIR = ROOT / "output"
DATA_DIR = Path(os.environ.get("DATA_DIR") or (ROOT / "data"))


def load_env(path: Path = None) -> None:
    """极简 .env 解析，不引入额外依赖。已存在的环境变量不覆盖。"""
    path = path or (ROOT / ".env")
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_yaml(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_modules() -> list:
    data = load_yaml(CONFIG_DIR / "modules.yaml")
    modules = data.get("modules") or []
    return sorted(modules, key=lambda m: m.get("order", 999))


def load_thresholds() -> dict:
    return load_yaml(CONFIG_DIR / "thresholds.yaml")


def ensure_dirs() -> None:
    for d in (DATA_DIR, OUTPUT_DIR, DATA_DIR / "parquet"):
        d.mkdir(parents=True, exist_ok=True)


def load_portfolio() -> dict:
    """持仓与规则配置（config/portfolio.yaml）。"""
    return load_yaml(CONFIG_DIR / "portfolio.yaml")

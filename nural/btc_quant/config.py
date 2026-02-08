import dataclasses
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


@dataclass
class Config:
    raw: Dict[str, Any]

    @property
    def exchange(self) -> Dict[str, Any]:
        return self.raw.get("exchange", {})

    @property
    def symbol(self) -> Dict[str, Any]:
        return self.raw.get("symbol", {})

    @property
    def api(self) -> Dict[str, Any]:
        """获取当前环境的 API 配置（自动根据 mode 选择）。"""
        api_cfg = self.raw.get("api", {})
        # 从环境变量或配置文件获取当前模式
        mode = os.environ.get("MODE", api_cfg.get("mode", "paper"))
        
        if mode == "live":
            env_api = api_cfg.get("live", {})
        else:
            env_api = api_cfg.get("paper", {})
        
        # 返回合并后的配置，包含 mode 信息
        result = env_api.copy()
        result["mode"] = mode
        return result

    @property
    def paths(self) -> Dict[str, Any]:
        return self.raw.get("paths", {})

    @property
    def history_data(self) -> Dict[str, Any]:
        return self.raw.get("history_data", {})

    @property
    def features(self) -> Dict[str, Any]:
        return self.raw.get("features", {})

    @property
    def labeling(self) -> Dict[str, Any]:
        return self.raw.get("labeling", {})

    @property
    def model(self) -> Dict[str, Any]:
        return self.raw.get("model", {})

    @property
    def strategy(self) -> Dict[str, Any]:
        return self.raw.get("strategy", {})

    @property
    def risk(self) -> Dict[str, Any]:
        return self.raw.get("risk", {})

    @property
    def backtest(self) -> Dict[str, Any]:
        return self.raw.get("backtest", {})

    @property
    def live(self) -> Dict[str, Any]:
        return self.raw.get("live", {})


def load_config(path: Path | None = None) -> Config:
    """从 YAML 文件加载配置。

    如果未提供 path，默认使用项目根目录下的 config.yaml。
    """

    cfg_path = path or CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f) or {}

    return Config(raw=raw_cfg)

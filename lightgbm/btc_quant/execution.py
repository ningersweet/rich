from __future__ import annotations

import hashlib
import hmac
import time
from dataclasses import dataclass
from typing import Literal, Optional

import requests

from .config import Config

Side = Literal["BUY", "SELL"]
PositionSide = Literal["LONG", "SHORT"]


@dataclass
class OrderResult:
    success: bool
    raw: dict


class BinanceFuturesClient:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        # 使用 api 配置中的 base_url（自动根据 mode 选择）
        self.base_url = cfg.api.get("base_url", "https://fapi.binance.com")
        self.api_key = cfg.api.get("key", "")
        self.api_secret = cfg.api.get("secret", "").encode()
        self.mode = cfg.api.get("mode", "paper")
        self._dual_side_position = None  # 缓存持仓模式

    def _headers(self) -> dict:
        return {"X-MBX-APIKEY": self.api_key}

    def _sign(self, query: str) -> str:
        return hmac.new(self.api_secret, query.encode(), hashlib.sha256).hexdigest()

    def _request(self, method: str, path: str, params: Optional[dict] = None, signed: bool = False) -> dict:
        url = self.base_url + path
        params = params or {}
        if signed:
            params["timestamp"] = int(time.time() * 1000)
            query = "&".join(f"{k}={v}" for k, v in params.items())
            params["signature"] = self._sign(query)
        resp = requests.request(method, url, params=params, headers=self._headers() if signed else None, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def get_account_balance_usdt(self) -> float:
        data = self._request("GET", "/fapi/v2/balance", signed=True)
        for item in data:
            if item.get("asset") == "USDT":
                return float(item.get("balance", 0.0))
        return 0.0

    def _check_dual_side_position(self) -> bool:
        """检查账户是否开启双向持仓模式。"""
        if self._dual_side_position is not None:
            return self._dual_side_position
        
        try:
            data = self._request("GET", "/fapi/v1/positionSide/dual", signed=True)
            self._dual_side_position = data.get("dualSidePosition", False)
        except Exception:  # noqa: BLE001
            # 如果查询失败，默认为单向持仓
            self._dual_side_position = False
        
        return self._dual_side_position
    
    def get_open_position(self, symbol: str) -> dict:
        """获取指定合约当前持仓信息，若无持仓则返回空字典。"""

        data = self._request("GET", "/fapi/v2/positionRisk", signed=True)
        for item in data:
            if item.get("symbol") == symbol:
                return item
        return {}

    def place_market_order(
        self,
        symbol: str,
        side: Side,
        position_side: PositionSide,
        quantity: float,
        reduce_only: bool = False,
    ) -> OrderResult:
        """下市价单，自动适配单向/双向持仓模式。"""
        params = {
            "symbol": symbol,
            "side": side,
            "type": "MARKET",
            "quantity": quantity,
        }
        
        # 如果是双向持仓模式，需要添加 positionSide
        if self._check_dual_side_position():
            params["positionSide"] = position_side
        
        # 如果是平仓单，添加 reduceOnly
        if reduce_only:
            params["reduceOnly"] = True
        
        try:
            data = self._request("POST", "/fapi/v1/order", params=params, signed=True)
            return OrderResult(success=True, raw=data)
        except Exception as e:  # noqa: BLE001
            return OrderResult(success=False, raw={"error": str(e)})

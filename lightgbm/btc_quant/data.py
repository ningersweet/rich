from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests

from .config import Config


BINANCE_FAPI_KLINES_ENDPOINT = "/fapi/v1/klines"


@dataclass
class Kline:
    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: int


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_daily_kline_dir(cfg: Config) -> Path:
    """获取按天存储K线数据的目录。"""
    data_dir = Path(cfg.paths["data_dir"]).expanduser().resolve()
    symbol = cfg.symbol["name"]
    interval = cfg.symbol["interval"]
    
    # 按symbol和interval分类存储
    daily_dir = data_dir / "daily" / f"{symbol}_{interval}"
    _ensure_dir(daily_dir)
    return daily_dir


def download_klines_by_day(cfg: Config, start_date: datetime, end_date: datetime, daily_dir: Path) -> List[Path]:
    """按天下载K线数据，支持断点续传。
    
    Args:
        cfg: 配置对象
        start_date: 开始日期
        end_date: 结束日期
        daily_dir: 存储目录
    
    Returns:
        已下载的文件路径列表
    """
    base_url = cfg.api.get("base_url", "https://fapi.binance.com")
    symbol = cfg.symbol["name"]
    interval = cfg.symbol["interval"]
    limit = int(cfg.history_data.get("limit", 1000))
    
    downloaded_files = []
    current_date = start_date
    
    while current_date <= end_date:
        # 文件名格式：YYYY-MM-DD.parquet
        date_str = current_date.strftime("%Y-%m-%d")
        daily_file = daily_dir / f"{date_str}.parquet"
        
        # 如果文件已存在，跳过
        if daily_file.exists():
            print(f"\r跳过已存在: {date_str}", end="", flush=True)
            downloaded_files.append(daily_file)
            current_date += pd.Timedelta(days=1)
            continue
        
        # 计算当天的时间范围
        day_start = int(current_date.timestamp() * 1000)
        day_end = int((current_date + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)).timestamp() * 1000)
        
        # 下载当天数据
        day_rows: List[List] = []
        current_start = day_start
        
        try:
            while current_start < day_end:
                params = {
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": current_start,
                    "endTime": day_end,
                    "limit": limit,
                }
                
                # 重试机制（包含429频率限制处理）
                max_retries = 5
                for retry in range(max_retries):
                    try:
                        resp = requests.get(base_url + BINANCE_FAPI_KLINES_ENDPOINT, params=params, timeout=15)
                        resp.raise_for_status()
                        break
                    except requests.exceptions.HTTPError as e:
                        if resp.status_code == 429:  # 频率限制
                            wait_time = min(60, (2 ** retry) * 5)  # 指数退避：5s, 10s, 20s, 40s, 60s
                            print(f"\r⏸️  {date_str}: API限流，等待{wait_time}秒...", end="", flush=True)
                            time.sleep(wait_time)
                            if retry == max_retries - 1:
                                print(f"\n❗ {date_str} API限流重试失败，跳过")
                                break
                        else:
                            raise
                    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                        if retry < max_retries - 1:
                            wait_time = (retry + 1) * 2
                            print(f"\n网络错误 [{date_str}]，{wait_time}秒后重试... ({retry + 1}/{max_retries})")
                            time.sleep(wait_time)
                        else:
                            raise
                
                rows = resp.json()
                if not rows:
                    break
                
                day_rows.extend(rows)
                
                # 下一批
                last_close_time = rows[-1][6]
                next_start = last_close_time + 1
                if next_start >= day_end:
                    break
                current_start = next_start
                
                time.sleep(0.5)  # 避免触发币安API频率限制（每分钟最多120请求）
            
            # 保存当天数据
            if day_rows:
                df = pd.DataFrame(
                    day_rows,
                    columns=[
                        "open_time", "open", "high", "low", "close", "volume", "close_time",
                        "quote_asset_volume", "number_of_trades",
                        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore",
                    ],
                )
                df = df[["open_time", "open", "high", "low", "close", "volume", "close_time"]].copy()
                df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
                df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
                for col in ["open", "high", "low", "close", "volume"]:
                    df[col] = df[col].astype("float32")
                
                df.to_parquet(daily_file, index=False)
                print(f"\r✅ {date_str}: {len(df)} 条", end="", flush=True)
                downloaded_files.append(daily_file)
            else:
                print(f"\r⚠️  {date_str}: 无数据", end="", flush=True)
        
        except Exception as e:
            print(f"\n❗ {date_str} 下载失败: {e}")
            # 失败不阻断，继续下一天
        
        current_date += pd.Timedelta(days=1)
    
    print()  # 换行
    return downloaded_files


def merge_daily_klines(daily_files: List[Path], output_path: Path) -> Path:
    """合并每天的K线数据为一个文件。"""
    if not daily_files:
        raise ValueError("没有可合并的文件")
    
    print(f"\n合并 {len(daily_files)} 个文件...")
    dfs = []
    for f in daily_files:
        if f.exists():
            dfs.append(pd.read_parquet(f))
    
    if not dfs:
        raise ValueError("所有文件都不存在")
    
    # 合并并去重
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=['open_time'], keep='last')
    combined = combined.sort_values('open_time').reset_index(drop=True)
    
    # 保存
    _ensure_dir(output_path.parent)
    combined.to_parquet(output_path, index=False)
    
    print(f"✅ 合并完成: {len(combined):,} 条K线")
    print(f"   时间范围: {combined['open_time'].min()} ~ {combined['close_time'].max()}")
    print(f"   保存路径: {output_path}")
    
    return output_path


def download_historical_klines(cfg: Config, force_download: bool = False) -> Path:
    """根据配置从币安期货下载历史 K 线并保存为 parquet 文件。
    
    使用按天分块下载的方式，支持断点续传。
    
    Args:
        cfg: 配置对象
        force_download: 是否强制重新下载（默认False）
    
    返回生成的文件路径。
    """
    symbol = cfg.symbol["name"]
    interval = cfg.symbol["interval"]
    start_str = cfg.history_data["start_time"]
    end_str = cfg.history_data["end_time"]
    
    # 最终输出文件路径
    data_dir = Path(cfg.paths["data_dir"]).expanduser().resolve()
    output_path = data_dir / f"{symbol}_{interval}.parquet"
    
    # 如果文件已存在且不强制下载，直接返回
    if output_path.exists() and not force_download:
        print(f"K线数据已存在，跳过下载: {output_path}")
        try:
            existing_df = pd.read_parquet(output_path)
            if not existing_df.empty:
                file_start = existing_df['open_time'].min()
                file_end = existing_df['close_time'].max()
                print(f"  数据时间范围: {file_start} ~ {file_end}")
                print(f"  数据条数: {len(existing_df):,} 条")
        except Exception as e:
            print(f"  警告: 读取现有文件失败: {e}，将重新下载")
            force_download = True
        
        if not force_download:
            return output_path
    
    print(f"开始按天下载K线数据: {symbol} {interval}")
    print(f"  时间范围: {start_str} ~ {end_str}")
    
    # 解析时间（处理可能已有时区的情况）
    start_ts = pd.Timestamp(start_str)
    end_ts = pd.Timestamp(end_str)
    
    if start_ts.tz is None:
        start_date = start_ts.tz_localize('UTC').normalize()
    else:
        start_date = start_ts.tz_convert('UTC').normalize()
    
    if end_ts.tz is None:
        end_date = end_ts.tz_localize('UTC').normalize()
    else:
        end_date = end_ts.tz_convert('UTC').normalize()
    
    # 获取按天存储的目录
    daily_dir = get_daily_kline_dir(cfg)
    
    # 按天下载
    daily_files = download_klines_by_day(cfg, start_date, end_date, daily_dir)
    
    # 合并成一个文件
    if daily_files:
        return merge_daily_klines(daily_files, output_path)
    else:
        raise RuntimeError("未下载到任何数据")


def load_klines(cfg: Config, path: Optional[Path] = None) -> pd.DataFrame:
    """从本地加载已下载的 K 线数据。
    
    Args:
        cfg: 配置对象
        path: 指定加载路径（可选，默认使用配置中的symbol和interval）
    
    Returns:
        K线数据 DataFrame
    """
    if path is None:
        # 默认加载合并后的文件
        data_dir = Path(cfg.paths["data_dir"]).expanduser().resolve()
        symbol = cfg.symbol["name"]
        interval = cfg.symbol["interval"]
        path = data_dir / f"{symbol}_{interval}.parquet"

    if not path.exists():
        raise FileNotFoundError(
            f"本地未找到K线数据文件: {path}\n"
            f"请先运行 download_historical_klines() 下载数据"
        )

    df = pd.read_parquet(path)
    return df


def update_klines_to_latest(cfg: Config) -> Path:
    """更新K线数据到最新（增量更新）。
    
    如果本地数据已存在，只下载最后一条记录之后的数据。
    如果本地数据不存在，下载全量数据。
    
    Args:
        cfg: 配置对象
    
    Returns:
        更新后的数据文件路径
    """
    out_path = get_kline_file_path(cfg)
    
    # 如果文件不存在，直接全量下载
    if not out_path.exists():
        print("本地数据不存在，开始全量下载...")
        return download_historical_klines(cfg, force_download=True)
    
    # 读取现有数据
    print(f"读取现有数据: {out_path}")
    existing_df = pd.read_parquet(out_path)
    
    if existing_df.empty:
        print("现有数据为空，重新下载...")
        return download_historical_klines(cfg, force_download=True)
    
    # 获取最后一条记录的时间
    last_close_time = existing_df['close_time'].max()
    print(f"现有数据最后时间: {last_close_time}")
    print(f"现有数据条数: {len(existing_df):,} 条")
    
    # 下载增量数据（从最后一条之后开始）
    base_url = cfg.api.get("base_url", "https://fapi.binance.com")
    symbol = cfg.symbol["name"]
    interval = cfg.symbol["interval"]
    end_str = cfg.history_data["end_time"]
    limit = int(cfg.history_data.get("limit", 1000))
    
    def to_ms(ts: str) -> int:
        return int(datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp() * 1000)
    
    # 从最后一条的 close_time + 1ms 开始
    start_ms = int(last_close_time.timestamp() * 1000) + 1
    end_ms = to_ms(end_str)
    
    if start_ms >= end_ms:
        print("数据已是最新，无需更新")
        return out_path
    
    print(f"开始下载增量数据: {pd.Timestamp(start_ms, unit='ms', tz='UTC')} ~ {end_str}")
    
    # 下载增量数据
    new_rows: List[List] = []
    current_start = start_ms
    request_count = 0
    
    while True:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": limit,
        }
        
        max_retries = 3
        for retry in range(max_retries):
            try:
                resp = requests.get(base_url + BINANCE_FAPI_KLINES_ENDPOINT, params=params, timeout=10)
                resp.raise_for_status()
                break
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                if retry < max_retries - 1:
                    wait_time = (retry + 1) * 2
                    print(f"\n网络错误，{wait_time}秒后重试... ({retry + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    raise
        
        rows = resp.json()
        if not rows:
            break
        
        new_rows.extend(rows)
        request_count += 1
        print(f"\r下载进度: {len(new_rows):,} 条新K线, 第 {request_count} 次请求", end="", flush=True)
        
        last_close_time_new = rows[-1][6]
        next_start = last_close_time_new + 1
        if next_start >= end_ms:
            break
        current_start = next_start
        time.sleep(0.2)
    
    print()
    
    if not new_rows:
        print("无新数据，数据已是最新")
        return out_path
    
    # 解析新数据
    new_df = pd.DataFrame(
        new_rows,
        columns=[
            "open_time", "open", "high", "low", "close", "volume", "close_time",
            "quote_asset_volume", "number_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore",
        ],
    )
    new_df = new_df[["open_time", "open", "high", "low", "close", "volume", "close_time"]].copy()
    new_df["open_time"] = pd.to_datetime(new_df["open_time"], unit="ms", utc=True)
    new_df["close_time"] = pd.to_datetime(new_df["close_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        new_df[col] = new_df[col].astype("float32")
    
    # 合并数据
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    # 去重（按 open_time 去重）
    combined_df = combined_df.drop_duplicates(subset=['open_time'], keep='last')
    # 按时间排序
    combined_df = combined_df.sort_values('open_time').reset_index(drop=True)
    
    # 保存
    combined_df.to_parquet(out_path, index=False)
    print(f"✅ 数据更新完成: {out_path}")
    print(f"  新增: {len(new_df):,} 条")
    print(f"  总计: {len(combined_df):,} 条")
    print(f"  时间范围: {combined_df['open_time'].min()} ~ {combined_df['close_time'].max()}")
    
    return out_path

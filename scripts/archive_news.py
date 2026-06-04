#!/usr/bin/env python3
"""
archive_news.py — 每天把当天的 AI 精选快照存成带日期的 JSON 文件。

用法:
  python scripts/archive_news.py --data-dir data --archive-dir archive

读取:
  data/latest-24h.json      → AI 强相关精选（主要数据源）

写入:
  archive/YYYY-MM-DD.json   → 当天快照（不会被后续滚动更新覆盖）
  archive/index.json        → 所有已归档日期的索引（供前端读取）
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone, timedelta

# CST = UTC+8
CST = timezone(timedelta(hours=8))


def today_cst() -> str:
    return datetime.now(CST).strftime("%Y-%m-%d")


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[archive] Written: {path}")


def update_index(archive_dir: str, date_str: str, snapshot_meta: dict) -> None:
    """维护 archive/index.json，记录所有已归档日期及摘要。"""
    index_path = os.path.join(archive_dir, "index.json")
    if os.path.exists(index_path):
        try:
            index = load_json(index_path)
        except Exception:
            index = {"dates": []}
    else:
        index = {"dates": []}

    dates: list = index.get("dates", [])
    # 移除同一天旧记录（支持重跑覆盖）
    dates = [d for d in dates if d.get("date") != date_str]
    dates.append({
        "date": date_str,
        "total_ai": snapshot_meta.get("total_ai", 0),
        "archived_at": snapshot_meta.get("archived_at", ""),
    })
    # 按日期倒序，最新在前
    dates.sort(key=lambda x: x["date"], reverse=True)

    index["dates"] = dates
    index["updated_at"] = datetime.now(timezone.utc).isoformat()
    write_json(index_path, index)


def main():
    parser = argparse.ArgumentParser(description="归档当天 AI 精选快照")
    parser.add_argument("--data-dir", default="data", help="data/ 目录路径")
    parser.add_argument("--archive-dir", default="archive", help="归档输出目录")
    parser.add_argument(
        "--date",
        default=None,
        help="覆盖归档日期，格式 YYYY-MM-DD（默认今天 CST）",
    )
    args = parser.parse_args()

    date_str = args.date or today_cst()
    source_path = os.path.join(args.data_dir, "latest-24h.json")

    if not os.path.exists(source_path):
        print(
            f"[archive] ERROR: 找不到 {source_path}，跳过归档",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[archive] Loading {source_path}")
    payload = load_json(source_path)

    # items_ai 是 AI 强相关精选；fallback 到 items（旧格式兼容）
    items_ai = payload.get("items_ai") or payload.get("items") or []
    total_ai = payload.get("total_items") or len(items_ai)
    archived_at = datetime.now(timezone.utc).isoformat()

    snapshot = {
        "archive_date": date_str,
        "archived_at": archived_at,
        "generated_at": payload.get("generated_at"),
        "total_ai": total_ai,
        "items_ai": items_ai,
        "site_stats": payload.get("site_stats") or [],
    }

    snapshot_path = os.path.join(args.archive_dir, f"{date_str}.json")
    write_json(snapshot_path, snapshot)

    update_index(
        args.archive_dir,
        date_str,
        {"total_ai": total_ai, "archived_at": archived_at},
    )

    print(f"[archive] Done: {date_str} → {snapshot_path}（{total_ai} 条 AI 精选）")


if __name__ == "__main__":
    main()

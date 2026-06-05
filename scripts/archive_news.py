#!/usr/bin/env python3
"""
archive_news.py — 每天把当天的 AI 精选快照存成带日期的 JSON 文件，
并调用 GLM API 生成中文摘要（需设置环境变量 GLM_API_KEY）。

用法:
  python scripts/archive_news.py --data-dir data --archive-dir archive

读取:
  data/latest-24h.json         → AI 强相关精选

写入:
  archive/YYYY-MM-DD.json          → 当天快照
  archive/YYYY-MM-DD-summary.json  → GLM 中文摘要（如有 API Key）
  archive/index.json               → 日期索引
"""

import argparse
import json
import os
import sys
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta

CST = timezone(timedelta(hours=8))

LABEL_MAP = {
    "ai_general": "AI综合",
    "model_release": "模型发布",
    "agent_workflow": "Agent工作流",
    "ai_product_update": "产品更新",
    "developer_tooling": "开发工具",
    "infrastructure": "基础设施",
}


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


def update_index(archive_dir: str, date_str: str, meta: dict) -> None:
    index_path = os.path.join(archive_dir, "index.json")
    if os.path.exists(index_path):
        try:
            index = load_json(index_path)
        except Exception:
            index = {"dates": []}
    else:
        index = {"dates": []}

    dates: list = index.get("dates", [])
    dates = [d for d in dates if d.get("date") != date_str]
    dates.append({
        "date": date_str,
        "total_ai": meta.get("total_ai", 0),
        "archived_at": meta.get("archived_at", ""),
        "has_summary": meta.get("has_summary", False),
    })
    dates.sort(key=lambda x: x["date"], reverse=True)
    index["dates"] = dates
    index["updated_at"] = datetime.now(timezone.utc).isoformat()
    write_json(index_path, index)


def build_prompt(items: list, date_str: str) -> str:
    """构建发送给 GLM 的提示词。"""
    scored = sorted(
        items,
        key=lambda x: float(x.get("ai_score") or x.get("score") or 0),
        reverse=True,
    )
    top = scored[:12]

    lines = []
    for i, item in enumerate(top, 1):
        title = (
            item.get("title_zh") or item.get("title") or item.get("title_en") or ""
        ).strip()
        label = LABEL_MAP.get(item.get("ai_label", ""), item.get("ai_label") or "AI信号")
        site = item.get("site_name") or ""
        score = int(float(item.get("ai_score") or item.get("score") or 0))
        signals = "、".join((item.get("ai_signals") or [])[:3])

        line = f"{i}. 【{label}】{title}（来源：{site}，评分{score}"
        if signals:
            line += f"，关键词：{signals}"
        line += "）"
        lines.append(line)

    items_text = "\n".join(lines)

    return f"""你是一位AI科技日报解读员。以下是{date_str}的AI领域重要更新精选，来自各大技术信源的高评分内容。

请用简洁、通俗易懂的中文，为普通读者（非程序员）写一段今日AI动态摘要。要求：
- 150-250字
- 重点介绍最重要的2-4条更新及其对普通人的实际意义
- 遇到专业术语请用括号简单解释
- 语气轻松自然，像朋友分享今天的AI圈动态
- 最后一句话概括今日整体趋势

今日精选内容：
{items_text}

请直接输出摘要正文，不需要标题和开场白。"""


def call_glm(items: list, date_str: str, api_key: str) -> str | None:
    """调用 GLM API 生成中文摘要，失败返回 None（不影响归档流程）。"""
    prompt = build_prompt(items, date_str)

    payload = {
        "model": "glm-4-flash",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 0.7,
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "https://open.bigmodel.cn/api/paas/v4/chat/completions",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            text = result["choices"][0]["message"]["content"].strip()
            return text
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"[archive] WARNING: GLM HTTP {e.code}: {body[:200]}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"[archive] WARNING: GLM 调用失败: {e}", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(description="归档当天 AI 精选快照并生成 GLM 摘要")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--archive-dir", default="archive")
    parser.add_argument("--date", default=None, help="覆盖日期 YYYY-MM-DD（默认今天 CST）")
    args = parser.parse_args()

    date_str = args.date or today_cst()
    source_path = os.path.join(args.data_dir, "latest-24h.json")

    if not os.path.exists(source_path):
        print(f"[archive] ERROR: 找不到 {source_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[archive] Loading {source_path}")
    payload = load_json(source_path)

    items_ai = payload.get("items_ai") or payload.get("items") or []
    total_ai = payload.get("total_items") or len(items_ai)
    archived_at = datetime.now(timezone.utc).isoformat()

    # 保存当天快照
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

    # 生成 GLM 中文摘要（可选）
    glm_api_key = os.environ.get("GLM_API_KEY", "").strip()
    has_summary = False

    if glm_api_key and items_ai:
        print(f"[archive] 正在调用 GLM API 生成 {date_str} 摘要...")
        summary_text = call_glm(items_ai, date_str, glm_api_key)
        if summary_text:
            summary = {
                "date": date_str,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "model": "glm-4-flash",
                "summary": summary_text,
                "item_count": min(len(items_ai), 12),
            }
            summary_path = os.path.join(args.archive_dir, f"{date_str}-summary.json")
            write_json(summary_path, summary)
            has_summary = True
            print(f"[archive] 摘要生成完成（{len(summary_text)} 字）")
        else:
            print("[archive] GLM 摘要生成失败，跳过（不影响快照归档）")
    else:
        if not glm_api_key:
            print("[archive] 未设置 GLM_API_KEY，跳过摘要生成")

    # 更新索引
    update_index(
        archive_dir=args.archive_dir,
        date_str=date_str,
        meta={"total_ai": total_ai, "archived_at": archived_at, "has_summary": has_summary},
    )

    print(f"[archive] Done: {date_str} → {snapshot_path}（{total_ai} 条 AI 精选）")


if __name__ == "__main__":
    main()

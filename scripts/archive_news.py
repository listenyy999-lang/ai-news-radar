#!/usr/bin/env python3
"""
archive_news.py — 每天把当天的 AI 精选快照存成带日期的 JSON 文件，
并调用 GLM API 生成：
  1. 今日整体摘要（今日要点）
  2. 每条伯乐精选约 500 字背景解读

用法:
  python scripts/archive_news.py --data-dir data --archive-dir archive

读取:
  data/latest-24h.json              → AI 强相关精选

写入:
  archive/YYYY-MM-DD.json           → 当天快照
  archive/YYYY-MM-DD-summary.json   → 整体摘要 + 逐条解读
  archive/index.json                → 日期索引
"""

import argparse
import json
import os
import re
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


# ── 辅助：格式化 top N 条目 ──────────────────────────────────────

def get_top_items(items: list, n: int = 8) -> list:
    return sorted(
        items,
        key=lambda x: float(x.get("ai_score") or x.get("score") or 0),
        reverse=True,
    )[:n]


def format_items_for_prompt(top: list) -> str:
    lines = []
    for i, item in enumerate(top, 1):
        title = (item.get("title_zh") or item.get("title") or item.get("title_en") or "").strip()
        label = LABEL_MAP.get(item.get("ai_label", ""), item.get("ai_label") or "AI信号")
        site = item.get("site_name") or ""
        score = int(float(item.get("ai_score") or item.get("score") or 0))
        signals = "、".join((item.get("ai_signals") or [])[:3])
        line = f"{i}. 【{label}】{title}（来源：{site}，评分{score}"
        if signals:
            line += f"，关键词：{signals}"
        line += "）"
        lines.append(line)
    return "\n".join(lines)


# ── GLM 通用请求 ─────────────────────────────────────────────────

def glm_request(prompt: str, api_key: str, max_tokens: int = 512) -> str | None:
    payload = {
        "model": "glm-4-flash",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
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
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result["choices"][0]["message"]["content"].strip()
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"[archive] WARNING: GLM HTTP {e.code}: {body[:300]}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"[archive] WARNING: GLM 调用失败: {e}", file=sys.stderr)
        return None


# ── 调用1：今日整体摘要 ──────────────────────────────────────────

def call_glm_summary(items: list, date_str: str, api_key: str) -> str | None:
    top = get_top_items(items, 12)
    items_text = format_items_for_prompt(top)
    prompt = f"""你是一位资深AI科技媒体编辑，请根据以下{date_str}的精选内容，撰写一份今日AI动态速览，面向关注科技趋势的专业读者。

要求：
- 300-400字，信息密度高
- 覆盖当日3-5个重要事件，客观呈现其核心内容与实质影响
- 专业术语保留，首次出现时括号简要解释
- 语言简洁精炼，采用新闻写作风格，避免口语化或夸张表达
- 结尾一句概括今日AI领域整体走向或值得关注的信号

今日精选内容：
{items_text}

请直接输出速览正文，无需标题。"""
    return glm_request(prompt, api_key, max_tokens=800)


# ── 调用2：逐条背景解读（每条单独调用，确保内容完整）──────────────

def call_glm_single_item(item: dict, date_str: str, api_key: str) -> str | None:
    """为单条新闻生成 500 字左右专业分析，直接返回文本。"""
    title = (item.get("title_zh") or item.get("title") or item.get("title_en") or "").strip()
    if not title:
        return None
    label = LABEL_MAP.get(item.get("ai_label", ""), item.get("ai_label") or "AI信号")
    site = item.get("site_name") or ""

    prompt = f"""你是一位资深AI科技记者，请对以下{date_str}的新闻进行专业分析报道，约500字。

新闻：【{label}】{title}（来源：{site}）

分析报道应包含：
① 背景：该公司、技术或事件的基本情况与行业地位
② 事件详情：本次发布/发生了什么，核心要点是什么
③ 深度分析：技术层面、商业层面或行业层面的实质意义，潜在影响与趋势

要求：
- 字数500字左右，不少于400字
- 采用专业新闻报道风格，语言简洁客观，信息密度高
- 专业术语保留，首次出现时括号简要解释
- 不得使用口语化或夸张表达
- 直接输出分析正文，不要添加标题或任何前缀"""

    return glm_request(prompt, api_key, max_tokens=900)


def call_glm_items_analysis(items: list, date_str: str, api_key: str) -> int:
    """对 top 15 条精选逐条单独调用 GLM，将解读直接写入 item['analysis']，返回成功条数。
    取15条而非8条，是因为前端伯乐精选的排序算法（多源聚合加分）与 ai_score 排序不同，
    扩大覆盖范围确保前端显示的 top 8 均有解析。"""
    top = get_top_items(items, 15)
    success = 0
    for i, item in enumerate(top, 1):
        short_title = (item.get("title_zh") or item.get("title") or "")[:30]
        print(f"[archive]   [{i}/{len(top)}] {short_title}...")
        explanation = call_glm_single_item(item, date_str, api_key)
        if explanation:
            item["analysis"] = explanation  # 直接嵌入 item，前端可直接读取
            success += 1
        else:
            print(f"[archive]   [{i}/{len(top)}] 解读生成失败，跳过")
    print(f"[archive] 逐条解读完成：{success}/{len(top)} 条")
    return success


# ── 主流程 ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="归档当天 AI 精选快照并生成 GLM 摘要与解读")
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

    # GLM 生成摘要 + 逐条解读
    glm_api_key = os.environ.get("GLM_API_KEY", "").strip()
    has_summary = False

    if glm_api_key and items_ai:
        # 调用1：整体摘要
        print(f"[archive] 调用 GLM 生成今日摘要...")
        summary_text = call_glm_summary(items_ai, date_str, glm_api_key)

        # 调用2：逐条深度解析（直接嵌入 items_ai 中的对应 item）
        print(f"[archive] 调用 GLM 生成逐条深度解析（约500字×8条）...")
        analysis_count = call_glm_items_analysis(items_ai, date_str, glm_api_key)

        # 解读已嵌入 items_ai，重新保存快照（含 analysis 字段）
        write_json(snapshot_path, snapshot)

        if summary_text or analysis_count > 0:
            summary = {
                "date": date_str,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "model": "glm-4-flash",
                "summary": summary_text or "",
                "item_count": min(len(items_ai), 12),
            }
            summary_path = os.path.join(args.archive_dir, f"{date_str}-summary.json")
            write_json(summary_path, summary)
            has_summary = True
            print(f"[archive] 完成：摘要 + 逐条解析 {analysis_count} 条（已嵌入快照）")
        else:
            print("[archive] GLM 全部失败，跳过（不影响快照归档）")
    else:
        if not glm_api_key:
            print("[archive] 未设置 GLM_API_KEY，跳过摘要与解读生成")

    update_index(
        archive_dir=args.archive_dir,
        date_str=date_str,
        meta={"total_ai": total_ai, "archived_at": archived_at, "has_summary": has_summary},
    )
    print(f"[archive] Done: {date_str}（{total_ai} 条 AI 精选）")


if __name__ == "__main__":
    main()

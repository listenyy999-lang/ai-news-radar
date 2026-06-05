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
import math
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


# ── 前端伯乐精选逻辑复刻（严格对照 archive.html 的 pickBoleItems）──────
#
# 背景：后端原先用 get_top_items() 按 ai_score 取 top 15 生成解读，但前端
# archive.html 的 pickBoleItems 排序算法完全不同 —— 它先按“跨源数量”降序，
# 再叠加 sourceBonus / candidateBonus，使多源聚合的低分新闻被顶到前面。
# 两套排序的交集很小，导致大部分页面显示的条目拿不到 analysis。
#
# 下列函数逐行翻译自 archive.html 内嵌脚本，保证 Python 选出的 8 条
# primary item 与页面实际显示的伯乐精选完全一致，解读 100% 命中。
# 注意：必须以 archive.html 那一版为准（它的 eventKey 正则与 app.js 略有差异）。

_URL_RE = re.compile(r"https?://\S+")
_WS_RE = re.compile(r"[\s　]+")
_PUNCT_RE = re.compile(r"[，。、“”‘’：:；;！!？?（）()\[\]【】《》<>·.,/\\|_\-]")
_BRACKET_RE = re.compile(r"《([^》]{4,40})》")
_MODEL_RE = re.compile(
    r"(deepseekv\d+(?:pro)?|grokv\d+(?:medium)?|gemini\d+(?:\.\d+)?(?:flash|pro)?|gpt\d+(?:\.\d+)?|llama\d+)"
)


def _js_round(x: float) -> int:
    """对应 JS Math.round（四舍五入，.5 向上），区别于 Python 的银行家舍入。"""
    return math.floor(x + 0.5)


def _to_ms(iso: str) -> int:
    """对应 JS new Date(iso).getTime()，解析失败返回 0（仅用于并列排序）。"""
    if not iso:
        return 0
    s = str(iso).strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return 0
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=CST)
    return int(dt.timestamp() * 1000)


def _item_title_text(item: dict) -> str:
    return (item.get("title_zh") or item.get("title") or item.get("title_en") or "无标题").strip()


def _score_percent(item: dict) -> int:
    raw = item.get("ai_score")
    if raw is None:
        raw = item.get("score")
    if raw is None:
        raw = 0
    try:
        score = float(raw)
    except (TypeError, ValueError):
        return 0
    if not math.isfinite(score) or score <= 0:
        return 0
    return _js_round(score * 100) if score <= 1 else _js_round(score)


def _timeline_iso(item: dict, generated_at: str) -> str:
    published = item.get("published_at") or ""
    seen = item.get("first_seen_at") or ""
    if published and generated_at:
        p = _to_ms(published)
        g = _to_ms(generated_at)
        if p and g and p > g + 10 * 60 * 1000:
            return seen or published
    return published or seen


def _timeline_ms(item: dict, generated_at: str) -> int:
    return _to_ms(_timeline_iso(item, generated_at))


def _normalized_event_text(text: str) -> str:
    s = str(text or "").lower()
    s = _URL_RE.sub("", s)
    s = _WS_RE.sub("", s)
    s = _PUNCT_RE.sub("", s)
    return s


def _event_key(item: dict) -> str:
    raw = _item_title_text(item)
    bracket = _BRACKET_RE.search(raw)
    if bracket:
        return "book:" + _normalized_event_text(bracket.group(1))[:36]
    normalized = _normalized_event_text(raw)
    model = _MODEL_RE.search(normalized)
    if model:
        return "entity:" + model.group(1)
    return "title:" + normalized[:34]


def _source_signal(item: dict) -> str:
    site = item.get("site_name") or ""
    source = item.get("source") or ""
    hay = f"{site} {source}".lower()
    if site == "AI HOT":
        return "AI HOT精选"
    if "hackernews" in hay or "hacker news" in hay:
        return "HN热议"
    if "GitHub · Trending Today" in source or "github" in hay:
        return "GitHub趋势"
    if site == "Official AI Updates":
        return "官方更新"
    if site == "Follow Builders":
        return "Builders"
    if site == "AIbase":
        return "AIbase"
    if site == "OPML RSS":
        return "OPML"
    return site or "来源"


def _source_priority(item: dict) -> int:
    s = _source_signal(item)
    return {
        "官方更新": 100,
        "AI HOT精选": 90,
        "AIbase": 82,
        "Builders": 74,
        "OPML": 68,
        "HN热议": 62,
        "GitHub趋势": 62,
    }.get(s, 50)


def _cluster_bole_events(rows: list, generated_at: str) -> list:
    clusters: dict = {}
    order: list = []
    for row in rows:
        key = _event_key(row["item"])
        cluster = clusters.get(key)
        if cluster is None:
            cluster = {"rows": [], "signals": [], "primary": row}
            clusters[key] = cluster
            order.append(key)
        cluster["rows"].append(row)
        sig = _source_signal(row["item"])
        if sig not in cluster["signals"]:
            cluster["signals"].append(sig)  # list 保留插入顺序，等价 JS Set
        primary = cluster["primary"]
        d = _source_priority(row["item"]) - _source_priority(primary["item"])
        if d == 0:
            d = row["score"] - primary["score"]
        if d == 0:
            d = _timeline_ms(row["item"], generated_at) - _timeline_ms(primary["item"], generated_at)
        if d > 0:
            cluster["primary"] = row

    result = []
    for key in order:
        cluster = clusters[key]
        signals = cluster["signals"]
        max_score = max(r["score"] for r in cluster["rows"])
        source_bonus = min(12, max(0, len(signals) - 1) * 6)
        if any(s == "AI HOT精选" for s in signals):
            candidate_bonus = 8
        elif any(s in ("HN热议", "GitHub趋势") for s in signals):
            candidate_bonus = 6
        elif any(s == "官方更新" for s in signals):
            candidate_bonus = 5
        else:
            candidate_bonus = 0
        result.append({
            "item": cluster["primary"]["item"],
            "index": cluster["primary"]["index"],
            "rows": cluster["rows"],
            "sourceSignals": signals,
            "sourceCount": len(signals),
            "mergedCount": len(cluster["rows"]),
            "score": min(100, _js_round(max_score + source_bonus + candidate_bonus)),
        })
    return result


def pick_bole_items(items: list, generated_at: str = "") -> list:
    """复刻 archive.html 的 pickBoleItems，返回页面会显示的伯乐精选 cluster 列表
    （每个 cluster 的 'item' 即 primary item，是 items 里的同一对象引用）。"""
    ranked = [
        {"item": item, "index": idx, "score": _score_percent(item)}
        for idx, item in enumerate(items)
    ]
    ranked = [r for r in ranked if r["score"] > 0]
    # JS: sort((a,b)=> b.score-a.score || timelineMs(b)-timelineMs(a))
    ranked.sort(key=lambda r: (-r["score"], -_timeline_ms(r["item"], generated_at)))

    clusters = _cluster_bole_events(ranked, generated_at)
    # JS: sort((a,b)=> (b.sourceCount-a.sourceCount)||(b.score-a.score)||timelineMs(b)-timelineMs(a))
    clusters.sort(key=lambda c: (
        -c["sourceCount"],
        -c["score"],
        -_timeline_ms(c["item"], generated_at),
    ))

    picked: list = []
    picked_ids: set = set()

    def add_pick(c):
        if c is not None and id(c) not in picked_ids and len(picked) < 8:
            picked.append(c)
            picked_ids.add(id(c))

    for sig in ("AI HOT精选", "HN热议", "GitHub趋势"):
        match = next((c for c in clusters if sig in c["sourceSignals"]), None)
        add_pick(match)
    for c in clusters:
        add_pick(c)
    return picked


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

def call_glm_summary(items: list, date_str: str, api_key: str, generated_at: str = "") -> str | None:
    # 与伯乐精选同源：要点和精选指向同一批文章，保证内容一致。
    picks = pick_bole_items(items, generated_at)
    top = [p["item"] for p in picks] or get_top_items(items, 8)
    items_text = format_items_for_prompt(top)
    prompt = f"""你是一位资深AI科技媒体编辑，请根据以下精选内容（覆盖最近24小时），撰写一份AI动态速览，面向关注科技趋势的专业读者。

要求：
- 300-400字，信息密度高
- 覆盖其中3-5个最重要的事件，客观呈现其核心内容与实质影响
- 专业术语保留，首次出现时括号简要解释
- 语言简洁精炼，采用新闻写作风格，避免口语化或夸张表达
- 结尾一句概括近期AI领域整体走向或值得关注的信号

最近24小时伯乐精选内容：
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


def call_glm_items_analysis(items: list, date_str: str, api_key: str, generated_at: str = "") -> int:
    """对前端 archive.html 实际显示的伯乐精选 Top 8 逐条调用 GLM，
    把解读写入对应 primary item 的 item['analysis']，返回成功条数。

    用 pick_bole_items() 复刻前端选择逻辑，确保生成的解读 100% 命中页面
    显示的条目（旧实现用 ai_score top 15，与前端多源聚合排序不一致，
    导致大部分条目拿不到解读）。"""
    picks = pick_bole_items(items, generated_at)
    top = [p["item"] for p in picks]
    if not top:
        print("[archive] 未选出任何伯乐精选条目，跳过逐条解读")
        return 0
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
    parser.add_argument("--force", action="store_true",
                        help="强制重新生成：即使当天已归档且含解读，也覆盖重算（手动触发时勾选 force）")
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
    # 与 archive.html 一致：generatedAt = snapshot.generated_at || snapshot.archived_at
    generated_at_src = payload.get("generated_at") or archived_at

    snapshot_path = os.path.join(args.archive_dir, f"{date_str}.json")
    summary_path = os.path.join(args.archive_dir, f"{date_str}-summary.json")

    # ── 防重跑保护 ───────────────────────────────────────────────
    # 若当天已归档且快照里已含解读（analysis），默认跳过，避免之后手动 / 重复
    # 运行用更晚的数据覆盖掉“08:00 历史归档”，并白白消耗 GLM 额度。
    # 需要重做今天这份时用 --force（工作流手动触发时勾选 force）。
    if not args.force and os.path.exists(snapshot_path):
        try:
            existing = load_json(snapshot_path)
            if any(it.get("analysis") for it in (existing.get("items_ai") or [])):
                print(f"[archive] {date_str} 已归档且含解读，跳过（如需重做请用 --force）")
                update_index(
                    archive_dir=args.archive_dir,
                    date_str=date_str,
                    meta={
                        "total_ai": existing.get("total_ai", total_ai),
                        "archived_at": existing.get("archived_at", archived_at),
                        "has_summary": os.path.exists(summary_path),
                    },
                )
                return
        except Exception as e:
            print(f"[archive] 读取既有快照失败，按正常流程重建：{e}", file=sys.stderr)

    # 保存当天快照
    snapshot = {
        "archive_date": date_str,
        "archived_at": archived_at,
        "generated_at": payload.get("generated_at"),
        "total_ai": total_ai,
        "items_ai": items_ai,
        "site_stats": payload.get("site_stats") or [],
    }
    write_json(snapshot_path, snapshot)

    # GLM 生成摘要 + 逐条解读
    glm_api_key = os.environ.get("GLM_API_KEY", "").strip()
    has_summary = False

    if glm_api_key and items_ai:
        # 调用1：晨报要点（与伯乐精选同源，覆盖近24小时）
        print(f"[archive] 调用 GLM 生成晨报要点（近24小时，基于伯乐精选）...")
        summary_text = call_glm_summary(items_ai, date_str, glm_api_key, generated_at_src)
        summary_item_count = len(pick_bole_items(items_ai, generated_at_src))

        # 调用2：逐条深度解析（直接嵌入 items_ai 中的对应 primary item）
        print(f"[archive] 调用 GLM 生成逐条深度解析（前端伯乐精选 Top 8）...")
        analysis_count = call_glm_items_analysis(items_ai, date_str, glm_api_key, generated_at_src)

        # 解读已嵌入 items_ai，重新保存快照（含 analysis 字段）
        write_json(snapshot_path, snapshot)

        if summary_text or analysis_count > 0:
            summary = {
                "date": date_str,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "model": "glm-4-flash",
                "summary": summary_text or "",
                "item_count": summary_item_count or min(len(items_ai), 8),
            }
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

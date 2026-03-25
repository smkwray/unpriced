from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd

from unpriced.config import ProjectPaths
from unpriced.models.scenario_solver import solve_price
from unpriced.storage import read_json, read_parquet

BACKGROUND = "#f8f7f4"
PANEL = "#ffffff"
INK = "#1e293b"
MUTED = "#64748b"
GRID = "#e2e8f0"
CANONICAL = "#0d9488"
ACCENT = "#d97706"
BLUE = "#3b82f6"
RED = "#b45309"
GRAY = "#94a3b8"


def _svg_header(width: int, height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">'
    )


def _text(
    x: float,
    y: float,
    value: object,
    *,
    size: int = 14,
    weight: str = "400",
    fill: str = INK,
    anchor: str = "start",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{escape(str(value))}</text>'
    )


def _line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    stroke: str = INK,
    width: float = 2,
    dash: str | None = None,
) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="{width}"{dash_attr} />'
    )


def _rect(
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    fill: str,
    stroke: str | None = None,
    radius: float = 0,
) -> str:
    stroke_attr = f' stroke="{stroke}" stroke-width="1"' if stroke else ""
    radius_attr = f' rx="{radius}" ry="{radius}"' if radius else ""
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" '
        f'fill="{fill}"{stroke_attr}{radius_attr} />'
    )


def _circle(x: float, y: float, radius: float, *, fill: str, stroke: str | None = None) -> str:
    stroke_attr = f' stroke="{stroke}" stroke-width="2"' if stroke else ""
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{fill}"{stroke_attr} />'


def _polyline(points: list[tuple[float, float]], *, stroke: str, width: float = 3, fill: str = "none") -> str:
    points_attr = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return (
        f'<polyline points="{points_attr}" fill="{fill}" stroke="{stroke}" '
        f'stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round" />'
    )


def _smooth_curve(
    points: list[tuple[float, float]],
    *,
    stroke: str,
    width: float = 3,
) -> str:
    """Smooth SVG path through points using Catmull-Rom → cubic Bézier conversion."""
    if len(points) < 2:
        return ""
    d = f"M {points[0][0]:.1f},{points[0][1]:.1f}"
    for i in range(len(points) - 1):
        p0 = points[max(0, i - 1)]
        p1 = points[i]
        p2 = points[i + 1]
        p3 = points[min(len(points) - 1, i + 2)]
        cp1x = p1[0] + (p2[0] - p0[0]) / 6
        cp1y = p1[1] + (p2[1] - p0[1]) / 6
        cp2x = p2[0] - (p3[0] - p1[0]) / 6
        cp2y = p2[1] - (p3[1] - p1[1]) / 6
        d += f" C {cp1x:.1f},{cp1y:.1f} {cp2x:.1f},{cp2y:.1f} {p2[0]:.1f},{p2[1]:.1f}"
    return (
        f'<path d="{d}" fill="none" stroke="{stroke}" '
        f'stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round" />'
    )


def _path(points: list[tuple[float, float]], *, fill: str, opacity: float = 0.2) -> str:
    if not points:
        return ""
    commands = [f"M {points[0][0]:.1f} {points[0][1]:.1f}"]
    commands.extend(f"L {x:.1f} {y:.1f}" for x, y in points[1:])
    commands.append("Z")
    return f'<path d="{" ".join(commands)}" fill="{fill}" opacity="{opacity:.2f}" />'


def _label_pill(
    x: float,
    y: float,
    value: str,
    *,
    size: int = 13,
    fill: str = INK,
    bg: str = PANEL,
    border: str = GRID,
    anchor: str = "start",
    pad_x: float = 8,
    pad_y: float = 5,
) -> list[str]:
    """Render text with an opaque background pill so it never overlaps curves."""
    approx_char_w = size * 0.6
    text_w = len(value) * approx_char_w
    if anchor == "middle":
        rx = x - text_w / 2 - pad_x
    elif anchor == "end":
        rx = x - text_w - pad_x
    else:
        rx = x - pad_x
    ry = y - size + 1 - pad_y
    rw = text_w + pad_x * 2
    rh = size + pad_y * 2
    return [
        _rect(rx, ry, rw, rh, fill=bg, stroke=border, radius=rh / 2),
        _text(x, y, value, size=size, weight="600", fill=fill, anchor=anchor),
    ]


def _svg_document(width: int, height: int, elements: list[str]) -> str:
    return "\n".join(
        [
            _svg_header(width, height),
            _rect(0, 0, width, height, fill=BACKGROUND),
            *elements,
            "</svg>",
        ]
    )


def _write_svg(path: Path, width: int, height: int, elements: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_svg_document(width, height, elements), encoding="utf-8")
    return path


def _median_by_alpha(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        frame.groupby("alpha", dropna=False)[["p_alpha", "p_alpha_lower", "p_alpha_upper"]]
        .median()
        .reset_index()
        .sort_values("alpha")
    )
    return grouped


def _write_marketization_diagram(path: Path) -> Path:
    width, height = 980, 600
    # Chart area
    cx0, cy0, cx1, cy1 = 140, 150, 840, 490
    elements = [
        _rect(40, 40, 900, 520, fill=PANEL, stroke=GRID, radius=18),
        _text(70, 85, "Childcare Marketization Diagram", size=26, weight="700"),
        _text(70, 110, "Stylized supply-demand schematic for outsourcing unpaid childcare", size=15, fill=MUTED),
        # Axes
        _line(cx0, cy1, cx1, cy1, stroke=INK, width=2),
        _line(cx0, cy1, cx0, cy0, stroke=INK, width=2),
        _text((cx0 + cx1) / 2, cy1 + 38, "Quantity of paid childcare", size=14, anchor="middle", fill=MUTED),
        _text(cx0 - 30, (cy0 + cy1) / 2, "Price", size=14, anchor="middle", fill=MUTED),
    ]
    # Key points — both curves pass through the baseline intersection
    baseline = (480, 310)
    shadow = (510, 296)
    alpha_pt = (620, 240)
    # Shaded outsourcing region
    shaded = [baseline, shadow, alpha_pt, (alpha_pt[0], cy1), (baseline[0], cy1)]
    elements.append(_path(shaded, fill=CANONICAL, opacity=0.08))
    # Clean cubic Bézier curves — classic textbook supply/demand shape
    # Supply: upward-sloping, two segments joined at baseline
    supply_d = (
        f"M 160,470 C 300,470 420,340 {baseline[0]},{baseline[1]} "
        f"C 540,280 700,185 800,175"
    )
    elements.append(
        f'<path d="{supply_d}" fill="none" stroke="{CANONICAL}" '
        f'stroke-width="4" stroke-linecap="round" stroke-linejoin="round" />'
    )
    # Demand: downward-sloping, two segments joined at baseline
    demand_d = (
        f"M 200,160 C 270,160 410,280 {baseline[0]},{baseline[1]} "
        f"C 550,340 730,460 800,460"
    )
    elements.append(
        f'<path d="{demand_d}" fill="none" stroke="{BLUE}" '
        f'stroke-width="4" stroke-linecap="round" stroke-linejoin="round" />'
    )
    # Dashed reference lines
    elements.extend([
        _line(baseline[0], baseline[1], baseline[0], cy1, stroke=GRAY, width=1.5, dash="6 6"),
        _line(cx0, baseline[1], baseline[0], baseline[1], stroke=GRAY, width=1.5, dash="6 6"),
        _line(alpha_pt[0], alpha_pt[1], alpha_pt[0], cy1, stroke=CANONICAL, width=1.5, dash="6 6"),
        _line(cx0, alpha_pt[1], alpha_pt[0], alpha_pt[1], stroke=CANONICAL, width=1.5, dash="6 6"),
    ])
    # Points
    elements.extend([
        _circle(*baseline, 6, fill=PANEL, stroke=INK),
        _circle(*shadow, 6, fill=ACCENT, stroke=ACCENT),
        _circle(*alpha_pt, 6, fill=CANONICAL, stroke=CANONICAL),
    ])
    # Axis labels
    elements.extend([
        _text(cx0 - 8, baseline[1] + 5, "P\u2080", size=13, fill=MUTED, anchor="end"),
        _text(cx0 - 8, alpha_pt[1] + 5, "P(\u03b1)", size=13, fill=CANONICAL, anchor="end"),
        _text(baseline[0], cy1 + 18, "Q\u2080", size=13, fill=MUTED, anchor="middle"),
        _text(alpha_pt[0], cy1 + 18, "Q(\u03b1)", size=13, fill=CANONICAL, anchor="middle"),
    ])
    # Curve labels — positioned at curve endpoints, well clear of intersections
    elements.extend([
        *_label_pill(810, 465, "Demand", fill=BLUE, bg=PANEL, border=BLUE),
        *_label_pill(810, 170, "Supply", fill=CANONICAL, bg=PANEL, border=CANONICAL),
    ])
    # Point labels — placed in clear margins using leader lines
    # Baseline: label to the upper-left
    elements.extend([
        _line(baseline[0], baseline[1], baseline[0] - 60, baseline[1] - 45, stroke=GRAY, width=1),
        *_label_pill(baseline[0] - 65, baseline[1] - 48, "Baseline price", fill=INK, anchor="end"),
    ])
    # Shadow price: label above
    elements.extend([
        _line(shadow[0], shadow[1], shadow[0] + 10, shadow[1] - 50, stroke=ACCENT, width=1),
        *_label_pill(shadow[0] + 15, shadow[1] - 53, "Shadow price", fill=ACCENT),
    ])
    # Alpha price: label to the upper-right with leader line from dot
    elements.extend([
        _line(alpha_pt[0], alpha_pt[1], alpha_pt[0] + 50, alpha_pt[1] - 30, stroke=CANONICAL, width=1),
        *_label_pill(alpha_pt[0] + 55, alpha_pt[1] - 33, "Price at \u03b1", fill=CANONICAL),
    ])
    # Footer annotation
    elements.append(
        _text(cx0, cy1 + 58, "Shaded region: additional paid-care volume under outsourcing", size=12, fill=MUTED)
    )
    return _write_svg(path, width, height, elements)


def _write_sample_ladder(path: Path, comparison: dict[str, object]) -> Path | None:
    samples = comparison.get("samples", {})
    selected = comparison.get("selected_headline_sample")
    rows = []
    for sample_name in ("broad_complete", "observed_core", "observed_core_low_impute"):
        sample = samples.get(sample_name)
        if not sample:
            continue
        rows.append(
            {
                "sample": sample_name,
                "n_obs": int(sample.get("n_obs", 0)),
                "n_states": int(sample.get("n_states", 0)),
                "n_years": int(sample.get("n_years", 0)),
                "loo_state": float(sample.get("loo_state_fips_r2", 0.0) or 0.0),
                "loo_year": float(sample.get("loo_year_r2", 0.0) or 0.0),
                "highlight": sample_name == selected,
            }
        )
    if not rows:
        return None
    width, height = 980, 460
    max_rows = max(item["n_obs"] for item in rows) or 1
    bar_left = 70
    bar_max = 560
    top = 150
    row_height = 90
    label_map = {
        "broad_complete": "Broad complete",
        "observed_core": "Observed core",
        "observed_core_low_impute": "Low-impute",
    }
    elements = [
        _rect(40, 40, 900, height - 80, fill=PANEL, stroke=GRID, radius=18),
        _text(70, 85, "Childcare Sample Ladder", size=26, weight="700"),
        _text(70, 110, "Headline sample balances observed support with usable state-year coverage", size=15, fill=MUTED),
    ]
    for idx, row in enumerate(rows):
        y = top + idx * row_height
        is_headline = row["highlight"]
        bar_fill = CANONICAL if is_headline else GRAY
        bar_width = max(bar_max * row["n_obs"] / max_rows, 4)
        # Row background for headline
        if is_headline:
            elements.append(_rect(55, y - 8, 870, row_height - 6, fill="#f0fdfa", radius=12))
        # Bar
        elements.extend([
            _rect(bar_left, y + 4, bar_max, 28, fill=BACKGROUND, stroke=GRID, radius=8),
            _rect(bar_left, y + 4, bar_width, 28, fill=bar_fill, radius=8),
        ])
        # Row count inside or beside bar
        count_text = f"{row['n_obs']} rows"
        if bar_width > 100:
            elements.append(_text(bar_left + bar_width - 10, y + 24, count_text, size=12, weight="600", fill=PANEL, anchor="end"))
        else:
            elements.append(_text(bar_left + bar_width + 8, y + 24, count_text, size=12, weight="600", fill=INK))
        # Label and metadata to the right of the bar area
        info_x = bar_left + bar_max + 20
        elements.extend([
            _text(info_x, y + 16, label_map.get(row["sample"], row["sample"]), size=15, weight="700", fill=bar_fill),
            _text(info_x, y + 34, f"{row['n_states']} states \u00b7 {row['n_years']} years", size=12, fill=MUTED),
        ])
        # LOO diagnostics below bar
        elements.extend([
            _text(bar_left, y + 54, f"LOO state R\u00b2 {row['loo_state']:.3f}", size=11, fill=BLUE),
            _text(bar_left + 170, y + 54, f"LOO year R\u00b2 {row['loo_year']:.3f}", size=11, fill=ACCENT),
        ])
        # Headline badge
        if is_headline:
            elements.extend(_label_pill(info_x + 150, y + 16, "headline", fill=CANONICAL, bg="#f0fdfa", border=CANONICAL, size=11))
    return _write_svg(path, width, height, elements)


def _write_alpha_intervals(path: Path, scenarios: pd.DataFrame) -> Path | None:
    if scenarios.empty:
        return None
    width, height = 980, 520
    margin_left = 110
    margin_right = 60
    chart_top = 140
    chart_bottom = 420
    chart_width = width - margin_left - margin_right
    chart_height = chart_bottom - chart_top
    baseline = float(pd.to_numeric(scenarios["p_baseline"], errors="coerce").median())
    shadow = {
        "label": "marginal",
        "mid": float(pd.to_numeric(scenarios["p_shadow_marginal"], errors="coerce").median()),
        "low": float(pd.to_numeric(scenarios["p_shadow_marginal_lower"], errors="coerce").median()),
        "high": float(pd.to_numeric(scenarios["p_shadow_marginal_upper"], errors="coerce").median()),
    }
    alpha_rows = _median_by_alpha(scenarios)
    series = [shadow]
    for row in alpha_rows.itertuples(index=False):
        series.append(
            {
                "label": f"{float(row.alpha):.2f}",
                "mid": float(row.p_alpha),
                "low": float(row.p_alpha_lower),
                "high": float(row.p_alpha_upper),
            }
        )
    y_min = min(item["low"] for item in series + [{"low": baseline}])
    y_max = max(item["high"] for item in series + [{"high": baseline}])
    padding = max((y_max - y_min) * 0.08, 1.0)
    y_min -= padding
    y_max += padding

    def scale_y(value: float) -> float:
        return chart_bottom - (value - y_min) / max(y_max - y_min, 1e-9) * chart_height

    x_positions = {
        item["label"]: margin_left + idx * chart_width / max(len(series) - 1, 1)
        for idx, item in enumerate(series)
    }
    elements = [
        _rect(40, 40, 900, 440, fill=PANEL, stroke=GRID, radius=18),
        _text(70, 85, "Canonical Childcare Scenario Intervals", size=26, weight="700"),
        _text(70, 110, "Median marketization-price intervals by outsourcing share for the canonical sample/spec", size=15, fill=MUTED),
        _line(margin_left, chart_bottom, width - margin_right, chart_bottom, stroke=INK, width=2),
        _line(margin_left, chart_top, margin_left, chart_bottom, stroke=INK, width=2),
    ]
    for step in range(5):
        value = y_min + step * (y_max - y_min) / 4
        y = scale_y(value)
        elements.extend(
            [
                _line(margin_left, y, width - margin_right, y, stroke=GRID, width=1),
                _text(margin_left - 12, y + 4, f"{value:,.0f}", size=12, fill=MUTED, anchor="end"),
            ]
        )
    baseline_y = scale_y(baseline)
    elements.extend(
        [
            _line(margin_left, baseline_y, width - margin_right, baseline_y, stroke=GRAY, width=2, dash="8 6"),
            _text(width - margin_right, baseline_y - 10, f"Median baseline price {baseline:,.0f}", size=12, fill=GRAY, anchor="end"),
        ]
    )
    line_points: list[tuple[float, float]] = []
    for item in series:
        x = x_positions[item["label"]]
        y_low = scale_y(item["low"])
        y_high = scale_y(item["high"])
        y_mid = scale_y(item["mid"])
        line_points.append((x, y_mid))
        fill = ACCENT if item["label"] == "marginal" else CANONICAL
        elements.extend(
            [
                _line(x, y_low, x, y_high, stroke=fill, width=3),
                _line(x - 10, y_low, x + 10, y_low, stroke=fill, width=3),
                _line(x - 10, y_high, x + 10, y_high, stroke=fill, width=3),
                _circle(x, y_mid, 6, fill=fill),
                _text(x, chart_bottom + 28, item["label"], size=13, fill=INK, anchor="middle"),
            ]
        )
    elements.append(_polyline(line_points, stroke=CANONICAL, width=3))
    return _write_svg(path, width, height, elements)


def _write_price_decomposition_by_alpha(path: Path, scenarios: pd.DataFrame) -> Path | None:
    if scenarios.empty:
        return None
    baseline_gross = float(pd.to_numeric(scenarios["p_baseline"], errors="coerce").median())
    baseline_direct = float(pd.to_numeric(scenarios.get("p_baseline_direct_care"), errors="coerce").median())
    baseline_residual = float(pd.to_numeric(scenarios.get("p_baseline_non_direct_care"), errors="coerce").median())
    alpha_rows = (
        scenarios.groupby("alpha", dropna=False)[["p_alpha", "p_alpha_direct_care", "p_alpha_non_direct_care"]]
        .median()
        .reset_index()
        .sort_values("alpha")
    )
    if alpha_rows.empty:
        return None
    series = [
        {
            "label": "baseline",
            "gross": baseline_gross,
            "direct": baseline_direct,
            "residual": baseline_residual,
            "highlight": True,
        }
    ]
    for row in alpha_rows.itertuples(index=False):
        series.append(
            {
                "label": f"alpha={float(row.alpha):.2f}",
                "gross": float(row.p_alpha),
                "direct": float(row.p_alpha_direct_care),
                "residual": float(row.p_alpha_non_direct_care),
                "highlight": abs(float(row.alpha) - 0.50) < 1e-9,
            }
        )
    width, height = 980, 580
    left = 110
    right = 900
    top = 150
    bottom = 450
    chart_width = right - left
    chart_height = bottom - top
    bar_width = 100
    max_total = max(item["gross"] for item in series) or 1.0

    def scale_y(value: float) -> float:
        return bottom - value / max_total * chart_height

    elements = [
        _rect(40, 40, 900, 490, fill=PANEL, stroke=GRID, radius=18),
        _text(70, 85, "Childcare Price Decomposition by Alpha", size=26, weight="700"),
        _text(70, 110, "Gross price split into direct-care and non-direct-care components", size=15, fill=MUTED),
        _line(left, bottom, right, bottom, stroke=INK, width=2),
        _line(left, top, left, bottom, stroke=INK, width=2),
    ]
    for step in range(5):
        value = max_total * step / 4
        y = scale_y(value)
        elements.extend(
            [
                _line(left, y, right, y, stroke=GRID, width=1),
                _text(left - 12, y + 4, f"{value:,.0f}", size=11, fill=MUTED, anchor="end"),
            ]
        )
    x_gap = chart_width / max(len(series), 1)
    for idx, item in enumerate(series):
        center = left + x_gap * (idx + 0.5)
        x = center - bar_width / 2
        direct_h = max(chart_height * item["direct"] / max_total, 0.0)
        residual_h = max(chart_height * item["residual"] / max_total, 0.0)
        direct_y = bottom - direct_h
        residual_y = direct_y - residual_h
        outline = CANONICAL if item["highlight"] else GRID
        elements.extend(
            [
                _rect(x, residual_y, bar_width, residual_h, fill=ACCENT, stroke=outline, radius=10 if residual_h > 0 else 0),
                _rect(x, direct_y, bar_width, direct_h, fill=CANONICAL, stroke=outline, radius=10 if direct_h > 0 else 0),
                _text(center, bottom + 28, item["label"], size=12, fill=INK, anchor="middle"),
                _text(center, residual_y - 8, f"{item['gross']:,.0f}", size=12, fill=INK, anchor="middle"),
                _text(center, bottom + 46, f"wage-side {item['direct']:,.0f}", size=11, fill=CANONICAL, anchor="middle"),
            ]
        )
    elements.extend(
        [
            _rect(120, 510, 18, 18, fill=CANONICAL),
            _text(146, 524, "Direct-care-equivalent component", size=12, fill=INK),
            _rect(360, 510, 18, 18, fill=ACCENT),
            _text(386, 524, "Residual non-direct-care component", size=12, fill=INK),
        ]
    )
    return _write_svg(path, width, height, elements)


def _write_alpha_examples(path: Path, scenarios: pd.DataFrame) -> Path | None:
    if scenarios.empty:
        return None
    rows = [
        {
            "label": "baseline",
            "alpha": None,
            "gross": float(pd.to_numeric(scenarios["p_baseline"], errors="coerce").median()),
            "direct": float(pd.to_numeric(scenarios.get("p_baseline_direct_care"), errors="coerce").median()),
            "wage": float(pd.to_numeric(scenarios.get("wage_baseline_implied"), errors="coerce").median()),
        }
    ]
    alpha_rows = (
        scenarios.groupby("alpha", dropna=False)[["p_alpha", "p_alpha_direct_care", "wage_alpha_implied"]]
        .median()
        .reset_index()
        .sort_values("alpha")
    )
    for row in alpha_rows.itertuples(index=False):
        rows.append(
            {
                "label": f"alpha={float(row.alpha):.2f}",
                "alpha": float(row.alpha),
                "gross": float(row.p_alpha),
                "direct": float(row.p_alpha_direct_care),
                "wage": float(row.wage_alpha_implied),
            }
        )
    width, height = 980, 420
    card_width = 150
    card_height = 205
    gap = 18
    left = 58
    top = 140
    elements = [
        _rect(40, 40, 900, 340, fill=PANEL, stroke=GRID, radius=18),
        _text(70, 85, "Childcare Alpha Examples", size=26, weight="700"),
        _text(70, 110, "Gross price, direct-care price, and implied wage at each outsourcing share", size=15, fill=MUTED),
    ]
    for idx, item in enumerate(rows):
        x = left + idx * (card_width + gap)
        highlight = item["alpha"] is not None and abs(item["alpha"] - 0.50) < 1e-9
        stroke = CANONICAL if highlight or item["alpha"] is None else GRID
        fill = "#eef6f1" if highlight else PANEL
        elements.extend(
            [
                _rect(x, top, card_width, card_height, fill=fill, stroke=stroke, radius=16),
                _text(x + card_width / 2, top + 30, item["label"], size=16, weight="700", fill=stroke if highlight else INK, anchor="middle"),
                _text(x + 16, top + 66, "Gross price", size=12, fill=MUTED),
                _text(x + 16, top + 88, f"{item['gross']:,.0f}", size=18, weight="700"),
                _text(x + 16, top + 122, "Direct-care", size=12, fill=MUTED),
                _text(x + 16, top + 144, f"{item['direct']:,.0f}", size=18, weight="700", fill=CANONICAL),
                _text(x + 16, top + 178, "Implied wage", size=12, fill=MUTED),
                _text(x + 16, top + 200, f"{item['wage']:.2f}/hr", size=18, weight="700", fill=BLUE),
            ]
        )
    return _write_svg(path, width, height, elements)


def _write_solver_implied_curves(path: Path, scenarios: pd.DataFrame) -> Path | None:
    if scenarios.empty:
        return None
    numeric = scenarios.copy()
    for column in (
        "p_baseline",
        "market_quantity_proxy",
        "demand_elasticity",
        "demand_elasticity_signed",
        "solver_demand_elasticity_magnitude",
        "supply_elasticity",
        "alpha",
        "p_alpha",
    ):
        if column not in numeric.columns:
            numeric[column] = pd.NA
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    numeric = numeric.dropna(
        subset=["p_baseline", "market_quantity_proxy", "supply_elasticity", "alpha", "p_alpha"]
    )
    if numeric.empty:
        return None
    baseline_price = float(numeric["p_baseline"].median())
    baseline_quantity = float(numeric["market_quantity_proxy"].median())
    if numeric["demand_elasticity_signed"].notna().any():
        demand_elasticity_signed = float(numeric["demand_elasticity_signed"].median())
    elif numeric["demand_elasticity"].notna().any():
        demand_elasticity_signed = float(numeric["demand_elasticity"].median())
    else:
        return None
    if numeric["solver_demand_elasticity_magnitude"].notna().any():
        demand_elasticity_magnitude = float(numeric["solver_demand_elasticity_magnitude"].median())
    else:
        demand_elasticity_magnitude = abs(demand_elasticity_signed)
    supply_elasticity = float(numeric["supply_elasticity"].median())
    alpha_rows = (
        numeric.groupby("alpha", dropna=False)[["p_alpha"]]
        .median()
        .reset_index()
        .sort_values("alpha")
    )
    if alpha_rows.empty:
        return None
    alpha_target = 0.5 if alpha_rows["alpha"].between(0.49, 0.51).any() else float(alpha_rows["alpha"].median())
    alpha_price = float(
        alpha_rows.loc[alpha_rows["alpha"].sub(alpha_target).abs() == alpha_rows["alpha"].sub(alpha_target).abs().min(), "p_alpha"].iloc[0]
    )
    price_min = baseline_price * 0.92
    price_max = max(alpha_price, baseline_price) * 1.08
    steps = 60
    prices = [price_min + idx * (price_max - price_min) / (steps - 1) for idx in range(steps)]
    demand_points = []
    supply_points = []
    for price in prices:
        demand_q = baseline_quantity * (price / baseline_price) ** (-demand_elasticity_magnitude)
        supply_q = baseline_quantity * (price / baseline_price) ** (supply_elasticity)
        demand_points.append((demand_q, price))
        supply_points.append((supply_q, price))
    alpha_points = []
    for row in alpha_rows.itertuples(index=False):
        alpha_quantity = baseline_quantity * (float(row.p_alpha) / baseline_price) ** (supply_elasticity)
        alpha_points.append(
            {
                "alpha": float(row.alpha),
                "price": float(row.p_alpha),
                "quantity": float(alpha_quantity),
            }
        )
    extrema_points = [(baseline_quantity, baseline_price)] + [
        (item["quantity"], item["price"]) for item in alpha_points
    ]
    q_min = min(point[0] for point in demand_points + supply_points + extrema_points)
    q_max = max(point[0] for point in demand_points + supply_points + extrema_points)
    p_min = min(point[1] for point in demand_points + supply_points + extrema_points)
    p_max = max(point[1] for point in demand_points + supply_points + extrema_points)
    q_pad = max((q_max - q_min) * 0.08, 1.0)
    p_pad = max((p_max - p_min) * 0.08, 1.0)
    q_min -= q_pad
    q_max += q_pad
    p_min -= p_pad
    p_max += p_pad
    width, height = 980, 680
    left = 120
    right = 880
    top = 120
    bottom = 500
    chart_width = right - left
    chart_height = bottom - top

    def scale_x(value: float) -> float:
        return left + (value - q_min) / max(q_max - q_min, 1e-9) * chart_width

    def scale_y(value: float) -> float:
        return bottom - (value - p_min) / max(p_max - p_min, 1e-9) * chart_height

    elements = [
        _rect(40, 40, 900, 600, fill=PANEL, stroke=GRID, radius=18),
        _text(70, 85, "Solver-Implied Childcare Curves", size=26, weight="700"),
        _text(70, 110, "Elasticity-implied local supply and demand curves anchored at the canonical baseline", size=15, fill=MUTED),
        _line(left, bottom, right, bottom, stroke=INK, width=2),
        _line(left, top, left, bottom, stroke=INK, width=2),
        _text(52, (top + bottom) / 2, "Price", size=16, anchor="middle"),
    ]
    for step in range(5):
        q_value = q_min + step * (q_max - q_min) / 4
        x = scale_x(q_value)
        elements.extend(
            [
                _line(x, top, x, bottom, stroke=GRID, width=1),
                _text(x, bottom + 18, f"{q_value/1_000_000:.2f}M", size=11, fill=MUTED, anchor="middle"),
            ]
        )
    for step in range(5):
        p_value = p_min + step * (p_max - p_min) / 4
        y = scale_y(p_value)
        elements.extend(
            [
                _line(left, y, right, y, stroke=GRID, width=1),
                _text(left - 12, y + 4, f"{p_value:,.0f}", size=11, fill=MUTED, anchor="end"),
            ]
        )
    # X-axis title
    elements.append(
        _text((left + right) / 2, bottom + 38, "Quantity of paid childcare", size=14, anchor="middle", fill=MUTED)
    )
    demand_poly = [(scale_x(q), scale_y(p)) for q, p in demand_points]
    supply_poly = [(scale_x(q), scale_y(p)) for q, p in supply_points]
    baseline_point = (scale_x(baseline_quantity), scale_y(baseline_price))
    displayed_alpha_points = [
        item
        for item in alpha_points
        if any(abs(item["alpha"] - target) < 1e-9 for target in (0.50, 1.00))
    ]
    if not displayed_alpha_points:
        displayed_alpha_points = alpha_points[-2:] if len(alpha_points) >= 2 else alpha_points
    # Curves
    elements.extend([
        _polyline(supply_poly, stroke=CANONICAL, width=4),
        _polyline(demand_poly, stroke=BLUE, width=4),
    ])
    # Baseline marker
    elements.extend([
        _circle(*baseline_point, 6, fill=PANEL, stroke=INK),
        _line(baseline_point[0], baseline_point[1], baseline_point[0], bottom, stroke=GRAY, width=1.5, dash="6 6"),
        _line(left, baseline_point[1], baseline_point[0], baseline_point[1], stroke=GRAY, width=1.5, dash="6 6"),
    ])
    # Curve labels — placed at ends of curves, outside chart
    elements.extend([
        *_label_pill(demand_poly[0][0] + 20, demand_poly[0][1] - 4, "Demand", fill=BLUE, bg=PANEL, border=BLUE),
        *_label_pill(supply_poly[-1][0] + 10, supply_poly[-1][1] - 4, "Supply", fill=CANONICAL, bg=PANEL, border=CANONICAL),
    ])
    # Baseline label
    elements.extend(
        _label_pill(baseline_point[0] - 14, baseline_point[1] - 18, "Baseline", fill=INK, bg=PANEL, border=GRID, anchor="end")
    )
    # Footer with elasticities and alpha readout — well below x-axis labels
    footer_y = bottom + 58
    elements.extend([
        _text(left, footer_y, f"Demand elasticity = {demand_elasticity_signed:.3f}", size=13, weight="600", fill=BLUE),
        _text(left + 260, footer_y, f"Supply elasticity = {supply_elasticity:.3f}", size=13, weight="600", fill=CANONICAL),
        _text(left + 500, footer_y, f"Baseline = {baseline_price:,.0f}", size=13, fill=MUTED),
    ])
    if displayed_alpha_points:
        footer_y2 = footer_y + 18
        for idx, item in enumerate(displayed_alpha_points):
            delta = item["price"] - baseline_price
            color = ACCENT if abs(item["alpha"] - alpha_target) < 1e-9 else CANONICAL
            elements.append(
                _text(left + idx * 260, footer_y2, f"\u03b1={item['alpha']:.2f}: {item['price']:,.0f} ({delta:+.1f})", size=13, weight="600", fill=color)
            )
    elements.append(
        _text(70, footer_y + 40, "Solver-implied from canonical elasticities, not empirical curves.", size=11, fill=MUTED)
    )
    return _write_svg(path, width, height, elements)


def _write_local_iv_marketization_demo(
    path: Path,
    scenarios: pd.DataFrame,
    supply_iv_summary: dict[str, object],
) -> Path | None:
    treated_states = supply_iv_summary.get("treated_state_fips", [])
    if not isinstance(treated_states, list) or not treated_states:
        return None
    iv_supply_elasticity = float(
        supply_iv_summary.get(
            "local_iv_supply_elasticity_provider_density",
            supply_iv_summary.get("iv_supply_elasticity_provider_density", float("nan")),
        )
    )
    if not pd.notna(iv_supply_elasticity) or iv_supply_elasticity <= 0:
        return None
    numeric = scenarios.copy()
    numeric["state_fips"] = numeric["state_fips"].astype(str).str.zfill(2)
    numeric = numeric[numeric["state_fips"].isin([str(x).zfill(2) for x in treated_states])].copy()
    if numeric.empty:
        return None
    for column in (
        "p_baseline",
        "market_quantity_proxy",
        "unpaid_quantity_proxy",
        "demand_elasticity_signed",
        "solver_demand_elasticity_magnitude",
    ):
        if column not in numeric.columns:
            numeric[column] = pd.NA
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    base_rows = numeric.drop_duplicates(["state_fips", "year"]).dropna(
        subset=["p_baseline", "market_quantity_proxy", "unpaid_quantity_proxy"]
    )
    if base_rows.empty:
        return None
    baseline_price = float(base_rows["p_baseline"].median())
    baseline_quantity = float(base_rows["market_quantity_proxy"].median())
    unpaid_quantity = float(base_rows["unpaid_quantity_proxy"].median())
    if numeric["demand_elasticity_signed"].notna().any():
        demand_elasticity_signed = float(numeric["demand_elasticity_signed"].median())
    else:
        return None
    if numeric["solver_demand_elasticity_magnitude"].notna().any():
        demand_elasticity_magnitude = float(numeric["solver_demand_elasticity_magnitude"].median())
    else:
        demand_elasticity_magnitude = abs(demand_elasticity_signed)
    if baseline_price <= 0 or baseline_quantity <= 0 or demand_elasticity_magnitude <= 0:
        return None

    alpha_points = []
    for alpha in (0.0, 0.50, 1.00):
        if alpha == 0.0:
            price = baseline_price
        else:
            price = solve_price(
                baseline_price=baseline_price,
                market_quantity=baseline_quantity,
                unpaid_quantity=unpaid_quantity,
                demand_elasticity=demand_elasticity_magnitude,
                supply_elasticity=iv_supply_elasticity,
                alpha=alpha,
            )
        quantity = baseline_quantity * (price / baseline_price) ** (iv_supply_elasticity)
        alpha_points.append({"alpha": alpha, "price": float(price), "quantity": float(quantity)})

    price_min = baseline_price * 0.92
    price_max = max(point["price"] for point in alpha_points) * 1.08
    steps = 60
    prices = [price_min + idx * (price_max - price_min) / (steps - 1) for idx in range(steps)]
    demand_points = []
    supply_points = []
    for price in prices:
        demand_q = baseline_quantity * (price / baseline_price) ** (-demand_elasticity_magnitude)
        supply_q = baseline_quantity * (price / baseline_price) ** (iv_supply_elasticity)
        demand_points.append((demand_q, price))
        supply_points.append((supply_q, price))
    extrema_points = [(baseline_quantity, baseline_price)] + [
        (item["quantity"], item["price"]) for item in alpha_points
    ]
    q_min = min(point[0] for point in demand_points + supply_points + extrema_points)
    q_max = max(point[0] for point in demand_points + supply_points + extrema_points)
    p_min = min(point[1] for point in demand_points + supply_points + extrema_points)
    p_max = max(point[1] for point in demand_points + supply_points + extrema_points)
    q_pad = max((q_max - q_min) * 0.08, 1.0)
    p_pad = max((p_max - p_min) * 0.08, 1.0)
    q_min = max(q_min - q_pad, 0.0)
    q_max += q_pad
    p_min -= p_pad
    p_max += p_pad

    width, height = 980, 680
    left = 120
    right = 880
    top = 120
    bottom = 500
    chart_width = right - left
    chart_height = bottom - top

    def scale_x(value: float) -> float:
        return left + (value - q_min) / max(q_max - q_min, 1e-9) * chart_width

    def scale_y(value: float) -> float:
        return bottom - (value - p_min) / max(p_max - p_min, 1e-9) * chart_height

    demand_poly = [(scale_x(q), scale_y(p)) for q, p in demand_points]
    supply_poly = [(scale_x(q), scale_y(p)) for q, p in supply_points]
    baseline_point = (scale_x(baseline_quantity), scale_y(baseline_price))

    elements = [
        _rect(40, 40, 900, 600, fill=PANEL, stroke=GRID, radius=18),
        _text(70, 85, "Local IV-Informed Marketization Demo", size=26, weight="700"),
        _text(
            70,
            110,
            "Treated-state median baseline using canonical demand elasticity and local IV supply elasticity",
            size=15,
            fill=MUTED,
        ),
        _line(left, bottom, right, bottom, stroke=INK, width=2),
        _line(left, top, left, bottom, stroke=INK, width=2),
        _text(52, (top + bottom) / 2, "Price", size=16, anchor="middle"),
    ]
    for step in range(5):
        q_value = q_min + step * (q_max - q_min) / 4
        x = scale_x(q_value)
        elements.extend(
            [
                _line(x, top, x, bottom, stroke=GRID, width=1),
                _text(x, bottom + 18, f"{q_value/1_000_000:.2f}M", size=11, fill=MUTED, anchor="middle"),
            ]
        )
    for step in range(5):
        p_value = p_min + step * (p_max - p_min) / 4
        y = scale_y(p_value)
        elements.extend(
            [
                _line(left, y, right, y, stroke=GRID, width=1),
                _text(left - 12, y + 4, f"{p_value:,.0f}", size=11, fill=MUTED, anchor="end"),
            ]
        )
    elements.append(
        _text((left + right) / 2, bottom + 38, "Quantity of paid childcare", size=14, anchor="middle", fill=MUTED)
    )
    elements.extend([
        _polyline(supply_poly, stroke=CANONICAL, width=4),
        _polyline(demand_poly, stroke=BLUE, width=4),
        _circle(*baseline_point, 6, fill=PANEL, stroke=INK),
        _line(baseline_point[0], baseline_point[1], baseline_point[0], bottom, stroke=GRAY, width=1.5, dash="6 6"),
        _line(left, baseline_point[1], baseline_point[0], baseline_point[1], stroke=GRAY, width=1.5, dash="6 6"),
    ])
    elements.extend([
        *_label_pill(demand_poly[0][0] + 20, demand_poly[0][1] - 4, "Demand", fill=BLUE, bg=PANEL, border=BLUE),
        *_label_pill(supply_poly[-1][0] + 10, supply_poly[-1][1] - 4, "Supply", fill=CANONICAL, bg=PANEL, border=CANONICAL),
        *_label_pill(baseline_point[0] - 14, baseline_point[1] - 18, "Baseline", fill=INK, bg=PANEL, border=GRID, anchor="end"),
    ])

    box_x, box_y, box_w, box_h = 585, 330, 290, 132
    elements.extend([
        _rect(box_x, box_y, box_w, box_h, fill=BACKGROUND, stroke=GRID, radius=14),
        _text(box_x + 16, box_y + 24, "Alpha examples", size=14, weight="700"),
    ])
    for idx, item in enumerate([point for point in alpha_points if point["alpha"] > 0]):
        delta = item["price"] - baseline_price
        elements.append(
            _text(
                box_x + 16,
                box_y + 52 + idx * 24,
                f"\u03b1={item['alpha']:.2f}: {item['price']:,.0f} ({delta:+.1f})",
                size=13,
                weight="600",
                fill=ACCENT if item["alpha"] == 1.0 else CANONICAL,
            )
        )
    elements.extend([
        _text(box_x + 16, box_y + 108, f"Treated states: {', '.join([str(x) for x in treated_states])}", size=12, fill=MUTED),
        _text(box_x + 16, box_y + 126, "Secondary local-IV demo only", size=12, fill=MUTED),
    ])

    footer_y = bottom + 58
    elements.extend([
        _text(left, footer_y, f"Demand elasticity = {demand_elasticity_signed:.3f}", size=13, weight="600", fill=BLUE),
        _text(left + 260, footer_y, f"Local IV supply elasticity = {iv_supply_elasticity:.3f}", size=13, weight="600", fill=CANONICAL),
        _text(left + 570, footer_y, f"Baseline = {baseline_price:,.0f}", size=13, fill=MUTED),
        _text(
            70,
            footer_y + 40,
            "Local treated-state demo using IV-informed supply and canonical demand; not a canonical or national structural curve.",
            size=11,
            fill=MUTED,
        ),
    ])
    return _write_svg(path, width, height, elements)


def write_childcare_dual_shift_figure(
    path: Path,
    summary: dict[str, object],
    headline_table: pd.DataFrame,
) -> Path | None:
    if headline_table.empty:
        return None
    working = headline_table.copy()
    working["kappa_q"] = pd.to_numeric(working["kappa_q"], errors="coerce")
    working["kappa_c"] = pd.to_numeric(working["kappa_c"], errors="coerce")
    working["median_p_alpha_pct_change"] = pd.to_numeric(working["median_p_alpha_pct_change"], errors="coerce")
    working = working.dropna(subset=["kappa_q", "kappa_c", "median_p_alpha_pct_change"]).copy()
    if working.empty:
        return None
    kappa_q_values = sorted(working["kappa_q"].unique().tolist())
    kappa_c_values = sorted(working["kappa_c"].unique().tolist())
    frontier = pd.DataFrame(summary.get("frontier_summary", []))
    if not frontier.empty:
        frontier["kappa_c"] = pd.to_numeric(frontier["kappa_c"], errors="coerce")
        frontier["kappa_q_zero_price_frontier_p50"] = pd.to_numeric(
            frontier["kappa_q_zero_price_frontier_p50"], errors="coerce"
        )
        frontier = frontier.dropna(subset=["kappa_c", "kappa_q_zero_price_frontier_p50"]).copy()

    width, height = 980, 700
    left, top = 160, 160
    cell_w, cell_h = 110, 72
    grid_w = cell_w * len(kappa_c_values)
    grid_h = cell_h * len(kappa_q_values)
    right = left + grid_w
    bottom = top + grid_h

    def cell_fill(value: float) -> str:
        if value > 0.001:
            return "#f6c8b5"
        if value < -0.001:
            return "#bce4de"
        return "#f4e7be"

    elements = [
        _rect(40, 40, 900, 620, fill=PANEL, stroke=GRID, radius=18),
        _text(70, 85, "Dual-Shift Price Frontier", size=26, weight="700"),
        _text(
            70,
            110,
            "Headline-alpha pooled childcare surface: price rise vs fall under entry/capacity and cost-pressure shifts",
            size=15,
            fill=MUTED,
        ),
        _text((left + right) / 2, bottom + 46, "kappa_c (cost pressure)", size=14, anchor="middle", fill=MUTED),
        _text(left - 78, (top + bottom) / 2, "kappa_q (entry / capacity)", size=14, anchor="middle", fill=MUTED),
    ]
    for x_idx, kappa_c in enumerate(kappa_c_values):
        x = left + x_idx * cell_w
        elements.extend(
            [
                _text(x + cell_w / 2, top - 14, f"{kappa_c:.2f}", size=12, fill=MUTED, anchor="middle"),
                _line(x, top, x, bottom, stroke=GRID, width=1),
            ]
        )
    elements.append(_line(right, top, right, bottom, stroke=GRID, width=1))
    for y_idx, kappa_q in enumerate(kappa_q_values):
        y = top + y_idx * cell_h
        elements.extend(
            [
                _text(left - 12, y + cell_h / 2 + 4, f"{kappa_q:.2f}", size=12, fill=MUTED, anchor="end"),
                _line(left, y, right, y, stroke=GRID, width=1),
            ]
        )
    elements.append(_line(left, bottom, right, bottom, stroke=GRID, width=1))

    for row in working.itertuples(index=False):
        x_idx = kappa_c_values.index(float(row.kappa_c))
        y_idx = kappa_q_values.index(float(row.kappa_q))
        x = left + x_idx * cell_w
        y = top + y_idx * cell_h
        elements.append(
            _rect(x + 3, y + 3, cell_w - 6, cell_h - 6, fill=cell_fill(float(row.median_p_alpha_pct_change)), stroke=GRID, radius=10)
        )
        elements.append(
            _text(x + cell_w / 2, y + 28, f"{float(row.median_p_alpha_pct_change) * 100:+.1f}%", size=15, weight="700", anchor="middle")
        )
        elements.append(
            _text(x + cell_w / 2, y + 50, f"{float(row.median_p_alpha):,.0f}", size=12, fill=MUTED, anchor="middle")
        )

    if not frontier.empty:
        x_min = min(kappa_c_values)
        x_max = max(kappa_c_values)
        y_min = min(kappa_q_values)
        y_max = max(kappa_q_values)

        def scale_x(value: float) -> float:
            if x_max == x_min:
                return left + cell_w / 2
            return left + ((value - x_min) / (x_max - x_min)) * (grid_w - cell_w) + cell_w / 2

        def scale_y(value: float) -> float:
            if y_max == y_min:
                return bottom - cell_h / 2
            return bottom - ((value - y_min) / (y_max - y_min)) * (grid_h - cell_h) - cell_h / 2

        frontier_points = [
            (scale_x(float(row["kappa_c"])), scale_y(float(row["kappa_q_zero_price_frontier_p50"])))
            for _, row in frontier.sort_values("kappa_c", kind="stable").iterrows()
        ]
        if frontier_points:
            elements.append(_polyline(frontier_points, stroke=BLUE, width=4))
            label_x, label_y = frontier_points[min(len(frontier_points) - 1, max(0, len(frontier_points) // 2))]
            elements.extend(
                _label_pill(
                    label_x + 12,
                    label_y - 10,
                    "Zero-price-change frontier (p50)",
                    fill=BLUE,
                    bg=PANEL,
                    border=BLUE,
                )
            )

    legend_y = bottom + 78
    legend_items = [
        (CANONICAL, "#bce4de", "price falls"),
        (ACCENT, "#f4e7be", "near zero"),
        (RED, "#f6c8b5", "price rises"),
    ]
    for idx, (_stroke, fill, label) in enumerate(legend_items):
        lx = left + idx * 190
        elements.append(_rect(lx, legend_y, 24, 18, fill=fill, stroke=GRID, radius=5))
        elements.append(_text(lx + 34, legend_y + 14, label, size=12, fill=MUTED))
    elements.extend(
        [
            _text(
                left,
                legend_y + 48,
                f"Headline alpha = {float(summary.get('headline_alpha', 0.5)):.2f}; cell labels show median pct change vs baseline and median price.",
                size=12,
                fill=MUTED,
            ),
            _text(
                left,
                legend_y + 68,
                "Short-run benchmark remains canonical; this figure visualizes the additive medium-run dual-shift sensitivity only.",
                size=12,
                fill=MUTED,
            ),
        ]
    )
    return _write_svg(path, width, height, elements)


def _write_pipeline_provenance(path: Path, diagnostics: dict[str, object]) -> Path | None:
    n_state = int(diagnostics.get("state_year_rows", 0))
    n_county = int(diagnostics.get("county_year_rows", 0))
    rows = [
        ("State births", int(diagnostics.get("births_cdc_wonder_observed", 0)), n_state),
        ("State controls", int(diagnostics.get("state_controls_acs_observed", 0)), n_state),
        ("County ACS", int(diagnostics.get("county_controls_acs_direct", 0)), n_county),
        ("County wages", int(diagnostics.get("county_wage_observed", 0)), n_county),
        ("County jobs", int(diagnostics.get("county_employment_observed", 0)), n_county),
        ("County LAUS", int(diagnostics.get("county_laus_observed", 0)), n_county),
    ]
    rows = [row for row in rows if row[2] > 0 and row[1] > 0]
    if not rows:
        return None
    n_rows = len(rows)
    row_height = 52
    chart_top = 130
    legend_y = chart_top + n_rows * row_height + 30
    panel_height = legend_y - 40 + 50
    width = 980
    height = 40 + panel_height + 40
    left = 220
    bar_width = 460
    elements = [
        _rect(40, 40, 900, panel_height, fill=PANEL, stroke=GRID, radius=18),
        _text(70, 85, "Pipeline Provenance", size=26, weight="700"),
        _text(70, 110, "Observed-source coverage in the current build", size=15, fill=MUTED),
    ]
    for idx, (label, observed, total) in enumerate(rows):
        y = chart_top + idx * row_height
        observed_share = observed / max(total, 1)
        observed_width = bar_width * observed_share
        elements.extend(
            [
                _text(70, y + 18, label, size=15, weight="700"),
                _rect(left, y, bar_width, 22, fill=BACKGROUND, stroke=GRID, radius=6),
                _rect(left, y, observed_width, 22, fill=CANONICAL, radius=6),
                _text(left + bar_width + 14, y + 17, f"{observed:,} / {total:,} ({observed_share:.0%})", size=13, fill=INK),
            ]
        )
    elements.extend(
        [
            _rect(220, legend_y, 18, 18, fill=CANONICAL),
            _text(246, legend_y + 14, "Observed public-data support", size=13, fill=INK),
            _rect(470, legend_y, 18, 18, fill=BACKGROUND, stroke=GRID),
            _text(496, legend_y + 14, "Fallback or synthetic component", size=13, fill=INK),
        ]
    )
    return _write_svg(path, width, height, elements)


def _write_support_boundary(path: Path, diagnostics: dict[str, object], demand_summary: dict[str, object]) -> Path:
    width, height = 980, 300
    observed_start, observed_end = 2008, 2022
    years = list(range(observed_start, observed_end + 1))
    x0 = 110
    x1 = 900
    track_y = 170
    track_width = x1 - x0

    def scale_year(year: int) -> float:
        return x0 + (year - years[0]) / max(years[-1] - years[0], 1) * track_width

    sample_min = int(demand_summary.get("year_min", 2014) or 2014)
    sample_max = int(demand_summary.get("year_max", 2022) or 2022)
    scenario_rows = int(diagnostics.get("scenario_rows", 0) or 0)
    elements = [
        _rect(40, 40, 900, 220, fill=PANEL, stroke=GRID, radius=18),
        _text(70, 85, "Observed Price Support Window", size=26, weight="700"),
        _text(70, 110, "All canonical scenarios stay within NDCP observed price support", size=15, fill=MUTED),
        _rect(scale_year(observed_start), track_y - 18, scale_year(observed_end) - scale_year(observed_start), 36, fill=CANONICAL, radius=10),
        _rect(scale_year(sample_min), track_y - 40, scale_year(sample_max) - scale_year(sample_min), 18, fill=BLUE, radius=8),
        _text((scale_year(sample_min) + scale_year(sample_max)) / 2, track_y - 27, f"Headline sample {sample_min}-{sample_max}", size=13, weight="700", fill=PANEL, anchor="middle"),
        _text((scale_year(observed_start) + scale_year(observed_end)) / 2, track_y + 5, "Observed NDCP support", size=14, fill=PANEL, weight="700", anchor="middle"),
        _text(70, 240, f"Canonical scenario rows: {scenario_rows}", size=14, fill=CANONICAL),
    ]
    for year in years:
        x = scale_year(year)
        elements.extend(
            [
                _line(x, track_y + 28, x, track_y + 38, stroke=INK, width=1),
                _text(x, track_y + 55, year, size=11, fill=MUTED, anchor="middle"),
            ]
        )
    return _write_svg(path, width, height, elements)


def _write_specification_comparison(path: Path, comparison: dict[str, object]) -> Path | None:
    profiles = comparison.get("profiles", {})
    canonical_profile = comparison.get("current_profile")
    ordered = [name for name in ("household_parsimonious", "instrument_only", "labor_parsimonious", "full_controls") if name in profiles]
    if not ordered:
        return None
    width, height = 980, 580
    label_x = 70
    left_axis_x = 280
    left_axis_width = 250
    right_axis_x = 610
    right_axis_width = 230
    top = 186
    gap = 80
    left_min = min(
        min(float(profiles[name].get("demand_loo_year_r2", 0.0) or 0.0), float(profiles[name].get("demand_loo_state_fips_r2", 0.0) or 0.0))
        for name in ordered
    )
    left_min = min(left_min, -0.10)
    left_max = max(
        max(float(profiles[name].get("demand_loo_year_r2", 0.0) or 0.0), float(profiles[name].get("demand_loo_state_fips_r2", 0.0) or 0.0))
        for name in ordered
    )
    left_max = max(left_max, 0.0)
    right_max = max(float(profiles[name].get("alpha_width_p50", 0.0) or 0.0) for name in ordered) or 1.0
    tick_values = [left_min, (left_min + left_max) / 2, left_max]
    label_map = {
        "household_parsimonious": "household parsimonious",
        "instrument_only": "instrument only",
        "labor_parsimonious": "labor parsimonious",
        "full_controls": "full controls",
    }

    def scale_left(value: float) -> float:
        return left_axis_x + (value - left_min) / max(left_max - left_min, 1e-9) * left_axis_width

    def scale_right(value: float) -> float:
        return right_axis_x + value / max(right_max, 1e-9) * right_axis_width

    elements = [
        _rect(40, 40, 900, 500, fill=PANEL, stroke=GRID, radius=18),
        _text(70, 85, "Scenario Specification Comparison", size=26, weight="700"),
        _text(70, 110, "Demand holdout diagnostics and scenario interval width by specification", size=15, fill=MUTED),
        _text(left_axis_x, 138, "Holdout R\u00b2", size=14, weight="700"),
        _text(right_axis_x, 138, "Alpha interval width (median)", size=14, weight="700"),
    ]
    # Pre-compute row metadata
    row_data = []
    for idx, name in enumerate(ordered):
        item = profiles[name]
        y = top + idx * gap
        is_canonical = (name == canonical_profile) or item.get("demand_sample_selection_reason") == "canonical_specification"
        is_admissible = item.get("demand_economically_admissible", item.get("economically_admissible", True))
        role_fill = CANONICAL if is_canonical else GRAY
        row_fill = "#eef6f1" if is_canonical else PANEL
        if is_canonical:
            role_label = "canonical"
        elif not is_admissible:
            role_label = "quarantined"
        else:
            role_label = "sensitivity"
        row_data.append(
            {
                "name": name,
                "y": y,
                "role_fill": role_fill,
                "row_fill": row_fill,
                "role_label": role_label,
                "loo_state": float(item.get("demand_loo_state_fips_r2", 0.0) or 0.0),
                "loo_year": float(item.get("demand_loo_year_r2", 0.0) or 0.0),
                "alpha_width": float(item.get("alpha_width_p50", 0.0) or 0.0),
            }
        )
    # Layer 1: Row backgrounds (must render before grid lines and ticks)
    for rd in row_data:
        elements.append(_rect(55, rd["y"] - 24, 850, 48, fill=rd["row_fill"], radius=10))
    # Layer 2: Grid lines and tick labels (render on top of backgrounds)
    for tick in tick_values:
        x = scale_left(tick)
        elements.extend(
            [
                _line(x, top - 14, x, top + gap * (len(ordered) - 1) + 18, stroke=GRID, width=1.5, dash="5 5" if tick == 0 else None),
                _text(x, top - 18, f"{tick:.2f}", size=11, fill=MUTED, anchor="middle"),
            ]
        )
    for step in range(4):
        tick = right_max * step / 3
        x = scale_right(tick)
        elements.extend(
            [
                _line(x, top - 14, x, top + gap * (len(ordered) - 1) + 18, stroke=GRID, width=1),
                _text(x, top - 18, f"{tick:.2f}", size=11, fill=MUTED, anchor="middle"),
            ]
        )
    # Layer 3: Row data (labels, dots, bars, values — on top of everything)
    for rd in row_data:
        y = rd["y"]
        elements.extend(
            [
                _text(label_x, y - 2, label_map.get(rd["name"], rd["name"]), size=15, weight="700", fill=rd["role_fill"]),
                _text(label_x, y + 16, rd["role_label"], size=12, fill=MUTED),
                _line(left_axis_x, y, left_axis_x + left_axis_width, y, stroke=GRID, width=1),
                _circle(scale_left(rd["loo_state"]), y, 5.5, fill=BLUE),
                _circle(scale_left(rd["loo_year"]), y, 5.5, fill=ACCENT),
                _text(left_axis_x + left_axis_width + 16, y - 2, f"{rd['loo_state']:.3f}", size=11, fill=BLUE),
                _text(left_axis_x + left_axis_width + 16, y + 13, f"{rd['loo_year']:.3f}", size=11, fill=ACCENT),
                _rect(right_axis_x, y - 10, scale_right(rd["alpha_width"]) - right_axis_x, 20, fill=rd["role_fill"], radius=8),
                _text(scale_right(rd["alpha_width"]) + 10, y + 5, f"{rd['alpha_width']:.3f}", size=12, fill=INK),
            ]
        )
    legend_y = top + gap * len(ordered) + 10
    elements.extend(
        [
            _circle(170, legend_y, 5.5, fill=BLUE),
            _text(183, legend_y + 5, "LOO state", size=12, fill=INK),
            _circle(270, legend_y, 5.5, fill=ACCENT),
            _text(283, legend_y + 5, "LOO year", size=12, fill=INK),
        ]
    )
    return _write_svg(path, width, height, elements)


def _write_piecewise_supply_demo(
    path: Path,
    demo: pd.DataFrame,
    summary: dict[str, object],
) -> Path | None:
    if demo.empty:
        return None
    alpha_one = demo.loc[pd.to_numeric(demo["alpha"], errors="coerce").round(4).eq(1.0)].copy()
    if alpha_one.empty:
        return None
    median_baseline = float(pd.to_numeric(alpha_one["p_baseline"], errors="coerce").median())
    alpha_one["baseline_distance"] = (
        pd.to_numeric(alpha_one["p_baseline"], errors="coerce") - median_baseline
    ).abs()
    focus = alpha_one.sort_values("baseline_distance").iloc[0]
    baseline_price = float(focus["p_baseline"])
    market_q = float(focus["market_quantity_proxy"])
    eta_constant = float(focus["supply_elasticity_constant"])
    eta_below = float(focus["supply_elasticity_below"])
    eta_above = float(focus["supply_elasticity_above"])
    prices = [baseline_price * (0.7 + idx * 0.0125) for idx in range(49)]
    constant_points = []
    piecewise_points = []
    for price in prices:
        constant_q = market_q * (price / baseline_price) ** eta_constant
        piecewise_eta = eta_below if price <= baseline_price else eta_above
        piecewise_q = market_q * (price / baseline_price) ** piecewise_eta
        constant_points.append((constant_q, price))
        piecewise_points.append((piecewise_q, price))
    extrema = constant_points + piecewise_points + [(market_q, baseline_price)]
    q_min = min(point[0] for point in extrema)
    q_max = max(point[0] for point in extrema)
    p_min = min(point[1] for point in extrema)
    p_max = max(point[1] for point in extrema)
    q_pad = max((q_max - q_min) * 0.08, 1.0)
    p_pad = max((p_max - p_min) * 0.08, 1.0)
    q_min = max(q_min - q_pad, 0.0)
    q_max += q_pad
    p_min -= p_pad
    p_max += p_pad
    width, height = 980, 620
    left = 120
    right = 610
    top = 130
    bottom = 480

    def scale_x(value: float) -> float:
        return left + (value - q_min) / max(q_max - q_min, 1e-9) * (right - left)

    def scale_y(value: float) -> float:
        return bottom - (value - p_min) / max(p_max - p_min, 1e-9) * (bottom - top)

    elements = [
        _rect(40, 40, 900, 540, fill=PANEL, stroke=GRID, radius=18),
        _text(70, 85, "Piecewise Supply Demo", size=26, weight="700"),
        _text(70, 110, "Constant vs one-kink supply curve on the high-labor-support observed-core subset", size=15, fill=MUTED),
        _line(left, bottom, right, bottom, stroke=INK, width=2),
        _line(left, top, left, bottom, stroke=INK, width=2),
        _text(52, (top + bottom) / 2, "Price", size=14, anchor="middle", fill=MUTED),
    ]
    for step in range(5):
        q_value = q_min + step * (q_max - q_min) / 4
        x = scale_x(q_value)
        elements.extend(
            [
                _line(x, top, x, bottom, stroke=GRID, width=1),
                _text(x, bottom + 18, f"{q_value/1_000_000:.2f}M", size=11, fill=MUTED, anchor="middle"),
            ]
        )
    for step in range(5):
        p_value = p_min + step * (p_max - p_min) / 4
        y = scale_y(p_value)
        elements.extend(
            [
                _line(left, y, right, y, stroke=GRID, width=1),
                _text(left - 12, y + 4, f"{p_value:,.0f}", size=11, fill=MUTED, anchor="end"),
            ]
        )
    elements.append(
        _text((left + right) / 2, bottom + 38, "Quantity of paid childcare", size=14, anchor="middle", fill=MUTED)
    )
    constant_poly = [(scale_x(q), scale_y(p)) for q, p in constant_points]
    piecewise_poly = [(scale_x(q), scale_y(p)) for q, p in piecewise_points]
    baseline_point = (scale_x(market_q), scale_y(baseline_price))
    # Curves
    elements.extend([
        _polyline(constant_poly, stroke=GRAY, width=4),
        _polyline(piecewise_poly, stroke=CANONICAL, width=4),
    ])
    # Baseline marker
    elements.extend([
        _circle(*baseline_point, 6, fill=PANEL, stroke=INK),
        _line(baseline_point[0], baseline_point[1], baseline_point[0], bottom, stroke=GRAY, width=1.5, dash="6 6"),
        _line(left, baseline_point[1], baseline_point[0], baseline_point[1], stroke=GRAY, width=1.5, dash="6 6"),
    ])
    # Curve labels — pill labels along curve body, away from info panels
    # Piecewise: label at ~75% along the upper portion, left of the info boxes
    pw_idx = min(len(piecewise_poly) - 1, int(len(piecewise_poly) * 0.72))
    ct_idx = min(len(constant_poly) - 1, int(len(constant_poly) * 0.72))
    elements.extend([
        *_label_pill(piecewise_poly[pw_idx][0] + 14, piecewise_poly[pw_idx][1] - 6, "Piecewise", fill=CANONICAL, bg=PANEL, border=CANONICAL),
        *_label_pill(constant_poly[ct_idx][0] + 14, constant_poly[ct_idx][1] + 18, "Constant", fill=GRAY, bg=PANEL, border=GRAY),
    ])
    # Baseline label
    elements.extend(
        _label_pill(baseline_point[0] - 14, baseline_point[1] - 18, "Baseline", fill=INK, bg=PANEL, border=GRID, anchor="end")
    )
    # Right-side info panels
    delta_half = float(summary.get("alpha_50_piecewise_minus_constant_p50", 0.0) or 0.0)
    delta_one = float(summary.get("alpha_100_piecewise_minus_constant_p50", 0.0) or 0.0)
    elements.extend([
        _rect(660, 136, 230, 160, fill=PANEL, stroke=GRID, radius=12),
        _text(676, 160, "Demo subset", size=13, weight="700"),
        _text(676, 184, f"{summary.get('demo_rows', 0)} state-years \u00b7 {summary.get('demo_states', 0)} states", size=12, fill=MUTED),
        _text(676, 212, f"\u03b7 below = {eta_below:.3f}", size=12, fill=CANONICAL),
        _text(676, 232, f"\u03b7 above = {eta_above:.3f}", size=12, fill=CANONICAL),
        _text(676, 252, f"\u03b7 constant = {eta_constant:.3f}", size=12, fill=GRAY),
        _text(676, 276, "Fallback on either side: 86.5%", size=11, fill=MUTED),
    ])
    elements.extend([
        _rect(660, 320, 230, 86, fill=PANEL, stroke=GRID, radius=12),
        _text(676, 344, "Price effect vs constant (median)", size=13, weight="700"),
        _text(676, 370, "\u03b1 = 0.50", size=12, fill=INK),
        _text(810, 370, f"{delta_half:+.2f}", size=12, weight="600", fill=ACCENT),
        _text(676, 390, "\u03b1 = 1.00", size=12, fill=INK),
        _text(810, 390, f"{delta_one:+.2f}", size=12, weight="600", fill=ACCENT),
    ])
    # Footer
    elements.append(
        _text(70, bottom + 62, "Wider price band shown than the actual alpha shock to make curve-shape differences visible.", size=11, fill=MUTED)
    )
    return _write_svg(path, width, height, elements)


def _write_supply_iv_pilot(
    path: Path,
    summary: dict[str, object],
    shock_panel: pd.DataFrame | None = None,
) -> Path | None:
    if summary.get("status") != "ok":
        return None

    import numpy as np

    width, height = 980, 530

    def _fmt(value: object, digits: int = 3) -> str:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return "n/a"
        if pd.isna(numeric):
            return "n/a"
        return f"{numeric:.{digits}f}"

    # --- Prepare shock timeline data ---------------------------------------------------
    shocks = pd.DataFrame()
    if shock_panel is not None and not shock_panel.empty:
        shocks = shock_panel.copy()
        shocks["year"] = pd.to_numeric(shocks.get("year"), errors="coerce")
        if "state_fips" in shocks.columns:
            shocks["state_fips"] = shocks["state_fips"].astype(str).str.zfill(2)

        # Compute labor intensity index from ratio/group-size columns when absent
        has_index = (
            "center_labor_intensity_index" in shocks.columns
            and shocks["center_labor_intensity_index"].notna().any()
        )
        if not has_index:
            components = []
            for col in (
                "center_infant_ratio",
                "center_toddler_ratio",
                "center_infant_group_size",
                "center_toddler_group_size",
            ):
                if col in shocks.columns:
                    vals = pd.to_numeric(shocks[col], errors="coerce")
                    components.append(1.0 / vals.replace({0: pd.NA}))
            if components:
                shocks["center_labor_intensity_index"] = (
                    pd.concat(components, axis=1).mean(axis=1, skipna=True)
                )
            else:
                shocks["center_labor_intensity_index"] = np.nan
        else:
            shocks["center_labor_intensity_index"] = pd.to_numeric(
                shocks["center_labor_intensity_index"], errors="coerce"
            )

        # Compute shock delta from per-state baseline
        shocks = shocks.sort_values(["state_fips", "year"])
        has_shock = (
            "center_labor_intensity_shock" in shocks.columns
            and shocks["center_labor_intensity_shock"].notna().any()
        )
        if not has_shock:
            baseline = shocks.groupby("state_fips")["center_labor_intensity_index"].transform(
                "first"
            )
            shocks["center_labor_intensity_shock"] = (
                shocks["center_labor_intensity_index"] - baseline
            )
        else:
            shocks["center_labor_intensity_shock"] = pd.to_numeric(
                shocks["center_labor_intensity_shock"], errors="coerce"
            )

        # Keep only states that have actual shock variation
        shock_states = set(
            shocks.loc[
                shocks["center_labor_intensity_shock"].abs().gt(1e-12), "state_fips"
            ].unique()
        )
        if shock_states:
            shocks = shocks[shocks["state_fips"].isin(shock_states)]
        shocks = (
            shocks.dropna(subset=["year", "center_labor_intensity_index"])
            .sort_values("year")
            .drop_duplicates("year")
        )

    # --- Extract summary stats ---------------------------------------------------------
    year_min = int(summary.get("year_min", 0) or 0)
    year_max = int(summary.get("year_max", year_min + 1) or (year_min + 1))
    if year_max <= year_min:
        year_max = year_min + 1

    first_stage = summary.get("first_stage_price", {}) or {}
    reduced_form = summary.get("reduced_form_provider_density", {}) or {}
    employer_form = summary.get("reduced_form_employer_establishment_density", {}) or {}
    first_stage_beta = float(first_stage.get("beta", float("nan")))
    first_stage_r2 = float(first_stage.get("within_r2", float("nan")))
    provider_beta = float(reduced_form.get("beta", float("nan")))
    provider_r2 = float(reduced_form.get("within_r2", float("nan")))
    employer_beta = float(employer_form.get("beta", float("nan")))
    wald_provider = float(
        summary.get(
            "local_iv_supply_elasticity_provider_density",
            summary.get("iv_supply_elasticity_provider_density", float("nan")),
        )
    )
    strength_flag = str(summary.get("first_stage_strength_flag", "unknown"))

    # --- Chart layout ------------------------------------------------------------------
    chart_left, chart_top = 95, 158
    chart_width, chart_height = 370, 200
    chart_bottom = chart_top + chart_height
    chart_right = chart_left + chart_width
    right_x = 540
    right_w = 380

    def scale_x(year: float) -> float:
        return chart_left + (year - year_min) / max(year_max - year_min, 1) * chart_width

    index_values = shocks["center_labor_intensity_index"].tolist() if not shocks.empty else []
    if index_values:
        idx_min = min(index_values)
        idx_max = max(index_values)
        margin = max((idx_max - idx_min) * 0.3, 0.005)
        idx_min -= margin
        idx_max += margin
    else:
        idx_min, idx_max = 0.0, 1.0

    def scale_y(value: float) -> float:
        return chart_bottom - (value - idx_min) / max(idx_max - idx_min, 1e-9) * chart_height

    # --- Build elements ----------------------------------------------------------------
    elements = [
        _rect(40, 40, 900, height - 80, fill=PANEL, stroke=GRID, radius=18),
        _text(70, 85, "Supply IV Pilot", size=26, weight="700"),
        _text(
            70,
            110,
            f"Licensing-shock exposure design: reduced-form responses ({summary.get('n_states', '?')} states, {summary.get('n_obs', '?')} obs)",
            size=14,
            fill=MUTED,
        ),
    ]

    # --- Left panel: timeline chart ----------------------------------------------------
    elements.append(
        _rect(50, 118, chart_width + 85, chart_height + 100, fill=BACKGROUND, stroke=GRID, radius=16)
    )
    elements.append(_text(chart_left, 144, "Licensing shock timeline", size=16, weight="700"))

    # Axes
    elements.append(_line(chart_left, chart_top, chart_left, chart_bottom, stroke=INK, width=2))
    elements.append(
        _line(chart_left, chart_bottom, chart_right, chart_bottom, stroke=INK, width=2)
    )

    # Y-axis ticks and labels
    n_ticks = 4
    for i in range(n_ticks + 1):
        frac = i / n_ticks
        y_val = idx_min + frac * (idx_max - idx_min)
        y_pos = scale_y(y_val)
        elements.append(_line(chart_left - 4, y_pos, chart_left, y_pos, stroke=INK, width=1))
        if i > 0:
            elements.append(_line(chart_left, y_pos, chart_right, y_pos, stroke=GRID, width=1))
        elements.append(
            _text(chart_left - 8, y_pos + 4, f"{y_val:.3f}", size=10, fill=MUTED, anchor="end")
        )

    # Rotated Y-axis label
    y_label_y = (chart_top + chart_bottom) / 2
    elements.append(
        f'<text x="58" y="{y_label_y:.1f}" font-family="Arial, sans-serif" font-size="11" '
        f'font-weight="400" fill="{MUTED}" text-anchor="middle" '
        f'transform="rotate(-90, 58, {y_label_y:.1f})">Labor intensity index</text>'
    )

    # X-axis year labels and gridlines
    for year in range(year_min, year_max + 1):
        x = scale_x(year)
        if year > year_min:
            elements.append(_line(x, chart_top, x, chart_bottom, stroke=GRID, width=1))
        elements.append(
            _text(x, chart_bottom + 18, str(year), size=11, anchor="middle", fill=MUTED)
        )

    # Plot the shock data
    if not shocks.empty and len(shocks) >= 2:
        points = [
            (scale_x(int(row.year)), scale_y(float(row.center_labor_intensity_index)))
            for row in shocks.itertuples(index=False)
        ]
        elements.append(_smooth_curve(points, stroke=CANONICAL, width=3))
        for row, (x, y) in zip(shocks.itertuples(index=False), points):
            shock_val = float(getattr(row, "center_labor_intensity_shock", 0.0) or 0.0)
            dot_fill = ACCENT if abs(shock_val) > 1e-12 else CANONICAL
            elements.append(_circle(x, y, 6, fill=dot_fill, stroke=PANEL))

        # Reform dashed line
        reform_rows = shocks.loc[shocks["center_labor_intensity_shock"].abs().gt(1e-12)]
        if not reform_rows.empty:
            reform_year = int(reform_rows["year"].min())
            reform_x = scale_x(reform_year - 0.5)
            elements.extend(
                [
                    _line(
                        reform_x, chart_top, reform_x, chart_bottom,
                        stroke=ACCENT, width=2, dash="6 4",
                    ),
                    *_label_pill(
                        reform_x,
                        chart_bottom - 18,
                        "reform",
                        size=11,
                        fill=ACCENT,
                        bg=PANEL,
                        border=ACCENT,
                        anchor="middle",
                    ),
                ]
            )

    # Note below chart
    elements.append(
        _text(
            chart_left,
            chart_bottom + 40,
            "Index tracks regulatory labor requirements for subsidy-approved centers.",
            size=11,
            fill=MUTED,
        )
    )

    # --- Right panel: result cards -----------------------------------------------------
    card_h, card_gap = 68, 10
    card_y = 118

    cards = [
        ("First-stage: price", "Shock \u2192 log price", first_stage_beta, first_stage_r2, CANONICAL),
        ("Reduced form: providers", "Shock \u2192 log density", provider_beta, provider_r2, BLUE),
        ("Reduced form: employers", "Shock \u2192 log estab.", employer_beta, float("nan"), GRAY),
    ]

    for label, sublabel, beta, r2, color in cards:
        elements.extend(
            [
                _rect(right_x, card_y, right_w, card_h, fill=BACKGROUND, stroke=GRID, radius=12),
                _text(right_x + 16, card_y + 22, label, size=13, weight="700"),
                _text(right_x + right_w - 16, card_y + 22, sublabel, size=11, fill=MUTED, anchor="end"),
                _text(
                    right_x + 16,
                    card_y + 50,
                    f"\u03b2 = {_fmt(beta)}",
                    size=20,
                    weight="700",
                    fill=color,
                ),
            ]
        )
        if pd.notna(r2):
            elements.append(
                _text(
                    right_x + right_w - 16,
                    card_y + 50,
                    f"R\u00b2 = {_fmt(r2, 4)}",
                    size=13,
                    fill=MUTED,
                    anchor="end",
                )
            )
        card_y += card_h + card_gap

    # Diagnostics card
    diag_h = 100
    elements.extend(
        [
            _rect(right_x, card_y, right_w, diag_h, fill=BACKGROUND, stroke=GRID, radius=12),
            _text(right_x + 16, card_y + 24, "Pilot diagnostics", size=14, weight="700"),
            _text(
                right_x + 16,
                card_y + 48,
                f"Local IV elasticity (provider): {_fmt(wald_provider)}",
                size=13,
            ),
            _text(
                right_x + 16,
                card_y + 68,
                "Obs: {}  \u00b7  Counties: {}  \u00b7  States: {}".format(summary.get('n_obs', '\u2014'), summary.get('n_counties', '\u2014'), summary.get('n_states', '\u2014')),
                size=12,
                fill=MUTED,
            ),
            _text(
                right_x + 16,
                card_y + 88,
                f"First-stage strength: {strength_flag}",
                size=13,
                fill=ACCENT if strength_flag != "strong" else CANONICAL,
            ),
        ]
    )

    # Footer
    n_shock = summary.get("shock_state_count", 1)
    scope_label = f"{n_shock}-state" if isinstance(n_shock, int) and n_shock > 0 else "pilot"
    pilot_scope = summary.get("pilot_scope", "")
    scope_word = "demo" if pilot_scope == "multi_state_demo" else "pilot"
    elements.append(
        _text(
            70,
            height - 38,
            f"Methodology {scope_word}: {scope_label} regulatory-shock design, not a structural supply curve estimate.",
            size=12,
            fill=MUTED,
        )
    )

    return _write_svg(path, width, height, elements)


def write_childcare_figure_manifest(paths: ProjectPaths, demand_summary_path: Path | None = None) -> Path:
    figures = []
    diagram_path = _write_marketization_diagram(paths.outputs_figures / "childcare_marketization_diagram.svg")
    figures.append(
        {
            "title": "Stylized marketization diagram",
            "path": diagram_path,
            "sources": ["conceptual / schematic"],
        }
    )

    comparison_path = paths.outputs_reports / "childcare_demand_sample_comparison.json"
    if comparison_path.exists():
        ladder_path = _write_sample_ladder(paths.outputs_figures / "childcare_sample_ladder.svg", read_json(comparison_path))
        if ladder_path is not None:
            figures.append(
                {
                    "title": "Childcare sample ladder",
                    "path": ladder_path,
                    "sources": [comparison_path.name],
                }
            )

    scenarios_path = paths.processed / "childcare_marketization_scenarios.parquet"
    if scenarios_path.exists():
        scenarios_frame = read_parquet(scenarios_path)
        alpha_path = _write_alpha_intervals(paths.outputs_figures / "childcare_alpha_intervals.svg", scenarios_frame)
        if alpha_path is not None:
            figures.append(
                {
                    "title": "Canonical scenario intervals by alpha",
                    "path": alpha_path,
                    "sources": [scenarios_path.name],
                }
            )
        decomposition_alpha_path = _write_price_decomposition_by_alpha(
            paths.outputs_figures / "childcare_price_decomposition_by_alpha.svg",
            scenarios_frame,
        )
        if decomposition_alpha_path is not None:
            figures.append(
                {
                    "title": "Price decomposition by alpha",
                    "path": decomposition_alpha_path,
                    "sources": [scenarios_path.name],
                }
            )
        alpha_examples_path = _write_alpha_examples(
            paths.outputs_figures / "childcare_alpha_examples.svg",
            scenarios_frame,
        )
        if alpha_examples_path is not None:
            figures.append(
                {
                    "title": "Alpha examples with implied wages",
                    "path": alpha_examples_path,
                    "sources": [scenarios_path.name],
                }
            )
        solver_curve_path = _write_solver_implied_curves(
            paths.outputs_figures / "childcare_solver_implied_curves.svg",
            scenarios_frame,
        )
        if solver_curve_path is not None:
            figures.append(
                {
                    "title": "Solver-implied supply and demand curves",
                    "path": solver_curve_path,
                    "sources": [scenarios_path.name],
                }
            )

    pipeline_path = paths.outputs_reports / "childcare_pipeline_diagnostics.json"
    if pipeline_path.exists():
        provenance_path = _write_pipeline_provenance(
            paths.outputs_figures / "childcare_pipeline_provenance.svg",
            read_json(pipeline_path),
        )
        if provenance_path is not None:
            figures.append(
                {
                    "title": "Pipeline provenance",
                    "path": provenance_path,
                    "sources": [pipeline_path.name],
                }
            )

    diagnostics_path = paths.outputs_reports / "childcare_scenario_diagnostics.json"
    selected_demand_path = demand_summary_path or paths.outputs_reports / "childcare_demand_iv_canonical.json"
    if diagnostics_path.exists() and selected_demand_path.exists():
        support_path = _write_support_boundary(
            paths.outputs_figures / "childcare_support_boundary.svg",
            read_json(diagnostics_path),
            read_json(selected_demand_path),
        )
        figures.append(
            {
                "title": "Observed support boundary",
                "path": support_path,
                "sources": [diagnostics_path.name, selected_demand_path.name],
            }
        )

    scenario_spec_path = paths.outputs_reports / "childcare_scenario_specification_comparison.json"
    if scenario_spec_path.exists():
        spec_path = _write_specification_comparison(
            paths.outputs_figures / "childcare_scenario_specification_comparison.svg",
            read_json(scenario_spec_path),
        )
        if spec_path is not None:
            figures.append(
                {
                    "title": "Scenario specification comparison",
                    "path": spec_path,
                    "sources": [scenario_spec_path.name],
                }
            )

    piecewise_demo_path = paths.outputs_reports / "childcare_piecewise_supply_demo.json"
    piecewise_demo_parquet = paths.processed / "childcare_piecewise_supply_demo.parquet"
    if piecewise_demo_path.exists() and piecewise_demo_parquet.exists():
        piecewise_path = _write_piecewise_supply_demo(
            paths.outputs_figures / "childcare_piecewise_supply_demo.svg",
            read_parquet(piecewise_demo_parquet),
            read_json(piecewise_demo_path),
        )
        if piecewise_path is not None:
            figures.append(
                {
                    "title": "Piecewise supply demo",
                    "path": piecewise_path,
                    "sources": [piecewise_demo_path.name, piecewise_demo_parquet.name],
                }
            )

    dual_shift_summary_path = paths.outputs_reports / "childcare_dual_shift_summary.json"
    dual_shift_table_path = paths.outputs_tables / "childcare_dual_shift_headline_alpha.csv"
    if dual_shift_summary_path.exists() and dual_shift_table_path.exists():
        dual_shift_path = write_childcare_dual_shift_figure(
            paths.outputs_figures / "childcare_dual_shift_frontier.svg",
            read_json(dual_shift_summary_path),
            pd.read_csv(dual_shift_table_path),
        )
        if dual_shift_path is not None:
            figures.append(
                {
                    "title": "Dual-shift price frontier",
                    "path": dual_shift_path,
                    "sources": [dual_shift_summary_path.name, dual_shift_table_path.name],
                }
            )

    supply_iv_path = paths.outputs_reports / "childcare_supply_iv.json"
    if supply_iv_path.exists():
        supply_iv_summary = read_json(supply_iv_path)
        shock_panel = None
        shock_panel_path = supply_iv_summary.get("shock_panel_path")
        if isinstance(shock_panel_path, str):
            shock_path = Path(shock_panel_path)
            if shock_path.exists():
                shock_panel = read_parquet(shock_path)
        supply_iv_figure_path = _write_supply_iv_pilot(
            paths.outputs_figures / "childcare_supply_iv_pilot.svg",
            supply_iv_summary,
            shock_panel,
        )
        if supply_iv_figure_path is not None:
            sources = [supply_iv_path.name]
            if isinstance(shock_panel_path, str):
                sources.append(Path(shock_panel_path).name)
            figures.append(
                {
                    "title": "Supply IV pilot",
                    "path": supply_iv_figure_path,
                    "sources": sources,
                }
            )
        if scenarios_path.exists():
            local_iv_curve_path = _write_local_iv_marketization_demo(
                paths.outputs_figures / "childcare_local_iv_marketization_demo.svg",
                read_parquet(scenarios_path),
                supply_iv_summary,
            )
            if local_iv_curve_path is not None:
                sources = [supply_iv_path.name, scenarios_path.name]
                figures.append(
                    {
                        "title": "Local IV-informed marketization demo",
                        "path": local_iv_curve_path,
                        "sources": sources,
                    }
                )

    manifest_lines = ["# Figure Manifest", "", "Generated childcare-first visuals for the current MVP.", ""]
    for item in figures:
        manifest_lines.extend(
            [
                f"## {item['title']}",
                f"- file: {item['path'].name}",
                f"- sources: {', '.join(item['sources'])}",
                "",
            ]
        )
    manifest_path = paths.outputs_figures / "figure_manifest.md"
    manifest_path.write_text("\n".join(manifest_lines), encoding="utf-8")
    return manifest_path

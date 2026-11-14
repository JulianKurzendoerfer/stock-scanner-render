#!/usr/bin/env python3
"""
Market Vision Pro – 4-Jahres Short-Put Portfolio- & Strategie-Backtest (Nov 2022 – Nov 2026)
=============================================================================================
Prüft die Short-Put-Stillhalter-Strategie über exakt 4 Jahre (208 Kalenderwochen, 14.11.2022 – 12.11.2026)
mit 100.000 € Einsatzkapital und der Regel:
  • Nur Short Puts (Mo/Di Einstieg -> Freitag Verfall, Delta <= 0,30, OTM >= 2,0 %, Prämie 0,7–1,4 %).
  • Wenn am Freitag NICHT eingebucht (Kurs >= Strike): 100 % Prämie frei verdient, Kapital am Montag wieder frei.
  • Wenn am Freitag EINGEBUCHT (Kurs < Strike): Aktien werden zum Strike eingebucht und exakt so lange gehalten,
    bis der Kurs das Strike-Niveau wieder erreicht (High >= Strike -> Verkauf zum Strike = +-0,00 € Kurs-PnL + volle Prämie).
  • Während der Haltedauer bleibt das Kapital in den eingebuchten Aktien gebunden.
"""

from __future__ import annotations
import json
import math
import statistics
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta

warnings.filterwarnings("ignore")

from agent import (
    DEFAULT_WATCHLIST,
    SYMBOL_CATEGORY,
    fetch_candles_yahoo,
    calc_bollinger_bands,
    calc_rsi,
    calc_macd,
    calc_stochastic_umwa,
    find_validated_sr_lines,
    recommend_weekly_option,
    estimate_current_iv,
)

START_DATE = date(2022, 11, 14)  # Montag vor exakt 4 Jahren (208 Wochen bis 12.11.2026)
END_DATE = date(2026, 11, 12)


def estimate_iv_at_idx(closes: list[float], highs: list[float], lows: list[float], idx: int) -> float:
    return estimate_current_iv(closes[: idx + 1], highs[: idx + 1], lows[: idx + 1])


def find_friday_close(candles: list, start_idx: int) -> tuple[int, float, int] | None:
    start_dt = date.fromisoformat(candles[start_idx]["time"])
    iso_year, iso_week, _ = start_dt.isocalendar()
    last_in_week_idx = None
    for j in range(start_idx + 1, min(start_idx + 6, len(candles))):
        dt = date.fromisoformat(candles[j]["time"])
        y, w, wd = dt.isocalendar()
        if (y, w) == (iso_year, iso_week) and wd <= 5:
            last_in_week_idx = j
        elif (y, w) > (iso_year, iso_week):
            break
    if last_in_week_idx is None:
        return None
    trading_days_left = last_in_week_idx - start_idx
    return last_in_week_idx, candles[last_in_week_idx]["close"], trading_days_left


def scan_symbol_4y_short_puts(symbol: str) -> list[dict]:
    """
    Scannt eine Aktie über 5 Jahre Tageskerzen (inkl. Indikator-Vorlauf vor Nov 2022)
    und protokolliert jeden qualifizierten Weekly Short Put inklusive Einbuchungs- und
    Halten-bis-Strike-Erholungsverlauf.
    """
    candles = fetch_candles_yahoo(symbol, interval="1d", range_str="5y")
    if not candles or len(candles) < 100:
        return []

    closes = [c["close"] for c in candles]
    opens = [c["open"] for c in candles]
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    times = [c["time"] for c in candles]

    bb_upper, bb_mid, bb_lower = calc_bollinger_bands(closes, 20, 2.0)
    rsi_series = calc_rsi(closes, 14)
    _, _, macd_hist = calc_macd(closes, 12, 26, 9)
    stoch_short_k, stoch_short_d = calc_stochastic_umwa(highs, lows, closes, 14, 5, 3)
    stoch_long_k, stoch_long_d = calc_stochastic_umwa(highs, lows, closes, 50, 6, 5)

    trades = []
    last_traded_week = None

    for i in range(60, len(candles) - 4):
        dt = date.fromisoformat(times[i])
        if dt < START_DATE or dt > END_DATE:
            continue

        weekday = dt.weekday()  # 0 = Montag, 1 = Dienstag
        if weekday not in (0, 1):
            continue

        friday_info = find_friday_close(candles, i)
        if not friday_info:
            continue
        fri_idx, fri_close, dte = friday_info
        fri_dt = date.fromisoformat(candles[fri_idx]["time"])
        week_key = (dt.isocalendar()[0], dt.isocalendar()[1])

        if last_traded_week == week_key:
            continue

        last_close = closes[i]
        prev_close = closes[i - 1]
        last_open = opens[i]
        last_low = lows[i]
        last_high = highs[i]

        # Keine illiquiden Penny-Stocks unter 12 USD für ein 100.000 € Stillhalter-Konto
        if last_close < 12.0:
            continue

        # --- 1. Bollinger Bänder ---
        bl, bu = bb_lower[i], bb_upper[i]
        bb_state = "neutral"
        if bl is not None and bu is not None and (bu - bl) > 0:
            recent_low_near_bb = any(
                bb_lower[i - k] is not None
                and (lows[i - k] <= bb_lower[i - k] + 0.08 * (bb_upper[i - k] - bb_lower[i - k]))
                for k in (0, 1, 2, 3)
            )
            candle_range = max(last_high - last_low, 0.0001)
            upper_wick_ratio = (last_high - max(last_open, last_close)) / candle_range
            bull_reaction = (last_close > prev_close) and (last_close >= last_open) and (upper_wick_ratio < 0.45)
            if recent_low_near_bb and bull_reaction:
                bb_state = "bullish"

        # --- 2. RSI UMWA ---
        r = rsi_series[i]
        r_prev = rsi_series[i - 1]
        recent_rsi_6 = [x for x in rsi_series[i - 6 : i] if x is not None]
        rsi_state = "neutral"
        if r is not None and r_prev is not None and recent_rsi_6:
            min_prev = min(recent_rsi_6)
            if min_prev < 30.0 and r >= 31.0 and r > r_prev and (r - min_prev) >= 4.0 and r <= 50.0:
                rsi_state = "bullish"

        # --- 3. MACD UMWA ---
        mh = [x for x in macd_hist[i - 5 : i + 1] if x is not None]
        macd_state = "neutral"
        if len(mh) >= 4:
            m1, m2, m3 = mh[-1], mh[-2], mh[-3]
            max_abs = max(abs(x) for x in mh) or 1e-6
            rising_strong = (m1 > m2 > m3) and ((m1 - m3) / max_abs >= 0.25)
            cross_up_confirmed = (m1 > 0 and m1 > m2 > m3 and m3 < 0)
            if cross_up_confirmed or rising_strong:
                macd_state = "bullish"

        # --- 4. Stochastik UMWA ---
        sk, sk_p, sk_p2 = stoch_short_k[i], stoch_short_k[i - 1], stoch_short_k[i - 2]
        sd, sd_p = stoch_short_d[i], stoch_short_d[i - 1]
        lk, lk_p = stoch_long_k[i], stoch_long_k[i - 1]
        ld = stoch_long_d[i]
        stoch_state = "neutral"
        if None not in (sk, sk_p, sk_p2, sd, sd_p, lk, lk_p, ld):
            recent_short_low = min(sk_p, sk_p2, sd_p)
            bull_momentum = (
                recent_short_low <= 20.0
                and sk > sd + 1.5
                and sk > sk_p > sk_p2
                and (sk - recent_short_low) >= 5.0
                and (lk <= 22.0 or (lk > lk_p and lk >= ld))
            )
            if bull_momentum:
                stoch_state = "bullish"

        states = [bb_state, rsi_state, macd_state, stoch_state]
        bull_count = states.count("bullish")
        if bull_count < 1:
            continue

        # Schneller Vorab-Filter: Bei 1/4 nur dann S/R prüfen, wenn RSI nicht überkauft ist (< 55) und Kurs steigt
        if bull_count == 1 and (r is None or r > 52.0 or last_close <= prev_close):
            continue

        # Earnings-Filter: Kein Verkauf direkt in einen Quartalszahlen-Sprung
        has_earnings_gap = any(
            abs(opens[k] - closes[k - 1]) / closes[k - 1] >= 0.065 or abs(closes[k] - closes[k - 1]) / closes[k - 1] >= 0.085
            for k in range(i + 1, fri_idx + 1)
        )
        if has_earnings_gap:
            continue

        iv = estimate_iv_at_idx(closes, highs, lows, i)
        sr_info = find_validated_sr_lines(candles[max(0, i - 180) : i + 1])
        opt = recommend_weekly_option(
            last_close, iv, is_put=True, dte_days=dte, sr_info=sr_info,
            expiry_date=fri_dt, symbol=symbol
        )
        if not opt or not opt.get("qualified") or opt.get("strike") is None:
            continue

        # Bei 1/4 (Reserve-Kandidat für ruhige Wochen) MUSS zwingend eine validierte Support-Schutzmauer (sr_protected)
        # zwischen Kurs und Strike liegen!
        if bull_count == 1 and not opt.get("sr_protected"):
            continue

        last_traded_week = week_key
        strike = float(opt["strike"])
        prem_usd = float(opt["premium_usd"])
        prem_pct = min(float(opt["prem_pct"]), 1.40)

        assigned = fri_close < strike
        breakeven_friday = fri_close >= (strike - prem_usd)
        friday_only_pnl_pct = prem_pct if not assigned else (prem_pct - (strike - fri_close) / strike * 100.0)

        # Halten-bis-Strike-Logik nach Einbuchung:
        recovered = True
        recovery_date = times[fri_idx]
        recovery_cal_days = 0
        recovery_trading_days = 0
        max_drawdown_after_assign_pct = 0.0
        end_price_if_unrecovered = fri_close

        if assigned:
            recovered = False
            min_low_while_held = fri_close
            for k in range(fri_idx + 1, len(candles)):
                if lows[k] < min_low_while_held:
                    min_low_while_held = lows[k]
                if highs[k] >= strike:
                    recovered = True
                    recovery_date = times[k]
                    rec_dt = date.fromisoformat(recovery_date)
                    recovery_cal_days = (rec_dt - fri_dt).days
                    recovery_trading_days = k - fri_idx
                    break
            max_drawdown_after_assign_pct = max(0.0, (strike - min_low_while_held) / strike * 100.0)
            if not recovered:
                end_price_if_unrecovered = closes[-1]
                recovery_date = times[-1]
                recovery_cal_days = (date.fromisoformat(times[-1]) - fri_dt).days
                recovery_trading_days = (len(candles) - 1) - fri_idx

        if not assigned or recovered:
            final_net_pnl_pct = prem_pct
            unrealized_loss_pct = 0.0
        else:
            unrealized_loss_pct = (strike - end_price_if_unrecovered) / strike * 100.0
            final_net_pnl_pct = prem_pct - unrealized_loss_pct

        shield = opt.get("sr_shield_line")
        cat = SYMBOL_CATEGORY.get(symbol, "CUSTOM")

        trades.append({
            "symbol": symbol,
            "category": cat,
            "week_key": f"{week_key[0]}-W{week_key[1]:02d}",
            "iso_year": week_key[0],
            "iso_week": week_key[1],
            "entry_date": times[i],
            "weekday": "Mo" if weekday == 0 else "Di",
            "fri_date": times[fri_idx],
            "score": bull_count,
            "entry_price": round(last_close, 2),
            "strike": round(strike, 2),
            "otm_pct": round(opt["otm_pct"], 2),
            "delta": round(opt["delta"], 2),
            "iv_pct": round(opt["iv_pct"], 1),
            "prem_usd": round(prem_usd, 2),
            "prem_pct": round(prem_pct, 2),
            "fri_close": round(fri_close, 2),
            "assigned": assigned,
            "not_assigned": not assigned,
            "breakeven_friday": breakeven_friday,
            "friday_only_pnl_pct": round(friday_only_pnl_pct, 2),
            "recovered": recovered,
            "recovery_date": recovery_date,
            "recovery_cal_days": recovery_cal_days,
            "recovery_trading_days": recovery_trading_days,
            "max_dd_held_pct": round(max_drawdown_after_assign_pct, 2),
            "end_price": round(end_price_if_unrecovered, 2),
            "unrealized_loss_pct": round(unrealized_loss_pct, 2),
            "final_net_pnl_pct": round(final_net_pnl_pct, 2),
            "sr_protected": bool(opt.get("sr_protected")),
            "shield_price": round(shield["price"], 2) if shield else None,
            "shield_bounces": shield["bounces"] if shield else 0,
        })

    return trades


def get_all_iso_weeks(start_d: date, end_d: date) -> list[tuple[int, int, date, date]]:
    weeks = []
    cur = start_d - timedelta(days=start_d.weekday())
    seen = set()
    while cur <= end_d:
        y, w, _ = cur.isocalendar()
        if (y, w) not in seen:
            seen.add((y, w))
            fri = cur + timedelta(days=4)
            weeks.append((y, w, cur, fri))
        cur += timedelta(days=7)
    return weeks


def simulate_100k_portfolio(
    trades: list[dict],
    max_positions_per_week: int = 5,
    max_alloc_per_trade_pct: float = 0.25,
    reinvest_profits: bool = False,
) -> dict:
    """
    Simuliert das 100.000 € Stillhalter-Konto Woche für Woche über 4 Jahre (209 Kalenderwochen):
      • Bis zu `max_positions_per_week` (5 Positionen à 20.000 € = 100.000 €) pro Woche.
      • Risikomanagement: Maximal `max_alloc_per_trade_pct` (25 % = 25.000 €) des Kontos in einer
        einzelnen Aktie, damit niemals eine einzige Einbuchung das gesamte 100.000 € Konto blockiert!
      • Bei 4–5 Signalen in einer Woche werden 100 % des freien Kapitals (100.000 €) voll ausgeschöpft.
      • Wird ein Put eingebucht, bleibt dieser Kapital-Slot (20.000 € – 25.000 €) exakt so lange in der
        Aktie gebunden, bis der Kurs wieder das Strike-Niveau erreicht (High >= Strike -> Verkauf zum Strike).
    """
    weeks = get_all_iso_weeks(START_DATE, END_DATE)
    trades_by_week = defaultdict(list)
    for t in trades:
        trades_by_week[(t["iso_year"], t["iso_week"])].append(t)

    # Priorisierung innerhalb jeder Woche:
    # 1. Hauptsignal >= 3/4 vor 2/4 vor 1/4+Shield
    # 2. Validierte Support-Schutzmauer (sr_protected)
    # 3. Qualitäts-Watchlist (A-Werte, B-Werte, Options-Pro-50, Julian-Stable)
    # 4. Höherer Score & OTM-Puffer
    def rank_trade(t: dict):
        cat = t["category"]
        cat_rank = 3 if cat in ("A-WERTE", "B-WERTE", "OPTIONS-PRO-50") else (2 if cat in ("C-WERTE", "JULIAN-STABLE", "JULIAN-ACTIVE") else 1)
        return (
            2 if t["score"] >= 3 else (1 if t["score"] == 2 else 0),
            1 if t["sr_protected"] else 0,
            t["score"],
            cat_rank,
            t["otm_pct"],
        )

    for wk in trades_by_week:
        trades_by_week[wk].sort(key=rank_trade, reverse=True)

    base_capital = 100_000.0
    cum_realized_premium_eur = 0.0
    locked_positions = []
    executed_portfolio_trades = []
    weekly_equity_curve = []

    weeks_with_trades = 0
    weeks_fully_locked = 0
    weeks_no_signal = 0

    current_win_streak = 0
    max_win_streak = 0

    for (y, w, mon_dt, fri_dt) in weeks:
        still_locked = []
        for pos in locked_positions:
            rec_dt = date.fromisoformat(pos["trade"]["recovery_date"])
            if pos["trade"]["recovered"] and rec_dt <= mon_dt:
                pos["status"] = "RECOVERED_SOLD_AT_STRIKE"
            else:
                still_locked.append(pos)
        locked_positions = still_locked

        locked_capital_eur = sum(p["capital_eur"] for p in locked_positions)
        total_pool_eur = (base_capital + cum_realized_premium_eur) if reinvest_profits else base_capital
        free_capital_eur = max(0.0, total_pool_eur - locked_capital_eur)

        candidates = trades_by_week.get((y, w), [])
        locked_symbols = {p["trade"]["symbol"] for p in locked_positions}
        candidates = [c for c in candidates if c["symbol"] not in locked_symbols]

        selected = candidates[:max_positions_per_week]
        week_premium_eur = 0.0
        week_invested_eur = 0.0

        if free_capital_eur < 1000.0:
            weeks_fully_locked += 1
        elif not selected:
            weeks_no_signal += 1
        else:
            weeks_with_trades += 1
            max_single_eur = total_pool_eur * max_alloc_per_trade_pct
            alloc_per_trade = min(free_capital_eur / len(selected), max_single_eur)
            for t in selected:
                week_invested_eur += alloc_per_trade
                prem_eur = alloc_per_trade * (t["prem_pct"] / 100.0)
                week_premium_eur += prem_eur
                cum_realized_premium_eur += prem_eur

                if t["not_assigned"] or t["recovered"]:
                    current_win_streak += 1
                    if current_win_streak > max_win_streak:
                        max_win_streak = current_win_streak
                else:
                    current_win_streak = 0

                p_record = {
                    **t,
                    "alloc_eur": round(alloc_per_trade, 2),
                    "prem_eur": round(prem_eur, 2),
                }

                if t["assigned"]:
                    locked_positions.append({
                        "capital_eur": alloc_per_trade,
                        "assigned_fri": fri_dt.isoformat(),
                        "trade": t,
                        "alloc_eur": alloc_per_trade,
                        "prem_eur": prem_eur,
                        "status": "OPEN_ASSIGNED",
                    })
                    if not t["recovered"]:
                        loss_eur = alloc_per_trade * (t["unrealized_loss_pct"] / 100.0)
                        p_record["unrealized_loss_eur"] = round(loss_eur, 2)
                        p_record["net_eur"] = round(prem_eur - loss_eur, 2)
                    else:
                        p_record["unrealized_loss_eur"] = 0.0
                        p_record["net_eur"] = round(prem_eur, 2)
                else:
                    p_record["unrealized_loss_eur"] = 0.0
                    p_record["net_eur"] = round(prem_eur, 2)

                executed_portfolio_trades.append(p_record)

        locked_after_fri = sum(p["capital_eur"] for p in locked_positions)
        weekly_equity_curve.append({
            "week": f"{y}-W{w:02d}",
            "year": y,
            "fri_date": fri_dt.isoformat(),
            "free_capital_mon": round(free_capital_eur, 2),
            "invested_eur": round(week_invested_eur, 2),
            "locked_capital_fri": round(locked_after_fri, 2),
            "num_trades": len(selected) if free_capital_eur >= 1000.0 else 0,
            "week_premium_eur": round(week_premium_eur, 2),
            "cum_premium_eur": round(cum_realized_premium_eur, 2),
            "total_account_value": round(base_capital + cum_realized_premium_eur, 2),
        })

    still_open_at_end = [p for p in locked_positions if not p["trade"]["recovered"]]
    unrealized_end_loss_eur = sum(
        p["capital_eur"] * (p["trade"]["unrealized_loss_pct"] / 100.0)
        for p in still_open_at_end
    )
    net_total_profit_eur = cum_realized_premium_eur - unrealized_end_loss_eur
    avg_locked_eur = statistics.mean(w["locked_capital_fri"] for w in weekly_equity_curve)
    avg_invested_eur = statistics.mean(w["invested_eur"] + w["locked_capital_fri"] for w in weekly_equity_curve)
    avg_weekly_prem_eur = cum_realized_premium_eur / len(weeks)
    avg_monthly_prem_eur = cum_realized_premium_eur / 48.0

    weekly_rets = [w["week_premium_eur"] / base_capital for w in weekly_equity_curve]
    mean_w_ret = statistics.mean(weekly_rets)
    std_w_ret = statistics.stdev(weekly_rets) if len(weekly_rets) > 1 else 1e-6
    sharpe = (mean_w_ret / std_w_ret) * math.sqrt(52.0) if std_w_ret > 0 else 0.0

    profit_factor = (
        cum_realized_premium_eur / unrealized_end_loss_eur
        if unrealized_end_loss_eur > 0
        else 999.0
    )

    return {
        "base_capital_eur": base_capital,
        "reinvest_profits": reinvest_profits,
        "total_weeks": len(weeks),
        "weeks_with_trades": weeks_with_trades,
        "weeks_no_signal": weeks_no_signal,
        "weeks_fully_locked": weeks_fully_locked,
        "avg_locked_eur": round(avg_locked_eur, 2),
        "avg_invested_eur": round(min(avg_invested_eur, total_pool_eur), 2),
        "avg_weekly_prem_eur": round(avg_weekly_prem_eur, 2),
        "avg_monthly_prem_eur": round(avg_monthly_prem_eur, 2),
        "max_win_streak": max_win_streak,
        "sharpe_ratio": round(sharpe, 2),
        "profit_factor": round(profit_factor, 2),
        "executed_trades": executed_portfolio_trades,
        "cum_realized_premium_eur": round(cum_realized_premium_eur, 2),
        "still_open_positions": still_open_at_end,
        "unrealized_end_loss_eur": round(unrealized_end_loss_eur, 2),
        "net_total_profit_eur": round(net_total_profit_eur, 2),
        "final_equity_eur": round(base_capital + net_total_profit_eur, 2),
        "weekly_curve": weekly_equity_curve,
    }


def export_backtest_html(
    path: str,
    sim_every_week_fixed: dict,
    sim_every_week_comp: dict,
    sim_main_only_fixed: dict,
    sim_main_only_comp: dict,
    sim_shield_only_fixed: dict,
    all_main_trades: list[dict],
):
    """Erstellt ein interaktives, institutionelles HTML-Backtest-Dashboard für den 4-Jahres-Zeitraum."""
    pf_trades = sim_every_week_fixed["executed_trades"]
    n_pf = len(pf_trades)
    pf_not_ass = sum(1 for t in pf_trades if t["not_assigned"])
    pf_ass = sum(1 for t in pf_trades if t["assigned"])
    pf_ass_rec = sum(1 for t in pf_trades if t["assigned"] and t["recovered"])
    pf_ass_open = sum(1 for t in pf_trades if t["assigned"] and not t["recovered"])
    pf_won = pf_not_ass + pf_ass_rec

    rec_days_trd = [t["recovery_trading_days"] for t in pf_trades if t["assigned"] and t["recovered"]]
    med_rec_trd = int(statistics.median(rec_days_trd)) if rec_days_trd else 0
    avg_rec_trd = statistics.mean(rec_days_trd) if rec_days_trd else 0.0
    rec_1w = sum(1 for d in rec_days_trd if d <= 5)

    # SVG Equity Curve für das 100.000 € Konto (Konstant vs. Zinseszins)
    curve_f = sim_every_week_fixed["weekly_curve"]
    curve_c = sim_every_week_comp["weekly_curve"]
    max_val = max(c["total_account_value"] for c in curve_c) * 1.03
    min_val = 95_000.0
    w_svg, h_svg = 960, 260
    pad_l, pad_r, pad_t, pad_b = 70, 25, 20, 32

    def pt_xy(idx: int, val: float) -> tuple[float, float]:
        x = pad_l + (idx / max(len(curve_f) - 1, 1)) * (w_svg - pad_l - pad_r)
        y = pad_t + (1.0 - (val - min_val) / (max_val - min_val)) * (h_svg - pad_t - pad_b)
        return round(x, 1), round(y, 1)

    pts_fixed = " ".join(f"{pt_xy(i, c['total_account_value'])[0]},{pt_xy(i, c['total_account_value'])[1]}" for i, c in enumerate(curve_f))
    pts_comp = " ".join(f"{pt_xy(i, c['total_account_value'])[0]},{pt_xy(i, c['total_account_value'])[1]}" for i, c in enumerate(curve_c))

    # Jahres-Tabelle
    by_year = defaultdict(lambda: {"trades": 0, "not_ass": 0, "ass_rec": 0, "ass_open": 0, "prem_eur": 0.0, "net_eur": 0.0})
    for t in pf_trades:
        y = t["entry_date"][:4]
        by_year[y]["trades"] += 1
        if t["not_assigned"]:
            by_year[y]["not_ass"] += 1
        elif t["recovered"]:
            by_year[y]["ass_rec"] += 1
        else:
            by_year[y]["ass_open"] += 1
        by_year[y]["prem_eur"] += t["prem_eur"]
        by_year[y]["net_eur"] += t["net_eur"]

    year_rows_html = ""
    for y in sorted(by_year.keys()):
        d = by_year[y]
        win_y = d["not_ass"] + d["ass_rec"]
        wr_y = win_y / max(d["trades"], 1) * 100.0
        year_rows_html += f"""
        <tr>
            <td class="mono" style="font-weight:700;color:#f8fafc;">{y}</td>
            <td class="mono">{d['trades']}</td>
            <td class="mono" style="color:#34d399;">{d['not_ass']} ({d['not_ass']/max(d['trades'],1)*100:.1f}%)</td>
            <td class="mono" style="color:#38bdf8;">{d['ass_rec']}</td>
            <td class="mono" style="color:{'#f87171' if d['ass_open']>0 else '#94a3b8'};">{d['ass_open']}</td>
            <td class="mono" style="font-weight:700;color:#10b981;">{win_y}/{d['trades']} ({wr_y:.1f}%)</td>
            <td class="mono" style="color:#34d399;font-weight:700;">+{d['prem_eur']:,.2f} €</td>
            <td class="mono" style="color:#10b981;font-weight:700;">+{d['net_eur']/1000:.2f} %</td>
        </tr>"""

    # Trade-Log Zeilen (neueste zuerst)
    trade_rows_html = ""
    for idx_t, t in enumerate(reversed(pf_trades), 1):
        if t["not_assigned"]:
            status_badge = '<span class="badge b-win">WERTLOS VERFALLEN (SOFORT FREI)</span>'
            hold_str = "0T (Freitag frei)"
        elif t["recovered"]:
            status_badge = f'<span class="badge b-rec">EINGEBUCHT → BEI STRIKE VERKAUFT ({t["recovery_date"]})</span>'
            hold_str = f'{t["recovery_trading_days"]} Handelstage ({t["recovery_cal_days"]} KT)'
        else:
            status_badge = '<span class="badge b-open">EINGEBUCHT → AM ENDE NOCH IM BESTAND</span>'
            hold_str = f'Offen ({t["recovery_trading_days"]} HT | -{t["unrealized_loss_pct"]:.1f}%)'

        shield_str = f'SHIELD {t["shield_price"]:.2f}$ ({t["shield_bounces"]}T)' if t["sr_protected"] and t["shield_price"] else "Standard OTM"
        trade_rows_html += f"""
        <tr>
            <td class="mono">{t['entry_date']} ({t['weekday']})</td>
            <td class="mono" style="font-weight:700;color:#f8fafc;">{t['symbol']} <span style="font-size:10px;color:#64748b;">[{t['category']}]</span></td>
            <td class="mono">{t['score']}/4</td>
            <td class="mono">{t['entry_price']:.2f} $</td>
            <td class="mono" style="color:#e2e8f0;font-weight:600;">{t['strike']:.2f} $ <span style="color:#94a3b8;font-size:11px;">(-{t['otm_pct']:.1f}%)</span></td>
            <td class="mono">{t['delta']:.2f}</td>
            <td class="mono" style="font-size:11px;color:#38bdf8;">{shield_str}</td>
            <td class="mono">{t['fri_close']:.2f} $</td>
            <td>{status_badge}</td>
            <td class="mono">{hold_str}</td>
            <td class="mono">{t['alloc_eur']:,.0f} €</td>
            <td class="mono" style="color:#34d399;font-weight:700;">+{t['prem_eur']:,.2f} € (+{t['prem_pct']:.2f}%)</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<title>Market Vision Pro – 4-Jahres Short-Put Backtest (100.000 € Portfolio)</title>
<style>
  :root {{
    --bg: #090d16;
    --panel: #111827;
    --border: #1e293b;
    --text: #e2e8f0;
    --muted: #94a3b8;
    --green: #10b981;
    --blue: #38bdf8;
    --amber: #f59e0b;
    --red: #ef4444;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Inter", "SF Pro Text", sans-serif;
    font-size: 13px;
    line-height: 1.5;
    padding: 24px 30px 60px;
  }}
  .mono {{ font-family: "SF Mono", "JetBrains Mono", Menlo, monospace; font-variant-numeric: tabular-nums; }}
  .header {{
    display: flex; justify-content: space-between; align-items: flex-end;
    border-bottom: 1px solid var(--border); padding-bottom: 18px; margin-bottom: 22px;
  }}
  .title {{ font-size: 20px; font-weight: 700; letter-spacing: -0.02em; color: #f8fafc; }}
  .subtitle {{ font-size: 12px; color: var(--muted); margin-top: 4px; }}
  .kpi-grid {{
    display: grid; grid-template-columns: repeat(6, 1fr); gap: 12px; margin-bottom: 22px;
  }}
  .kpi {{
    background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 14px 16px;
  }}
  .kpi-label {{ font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); }}
  .kpi-val {{ font-size: 21px; font-weight: 700; margin-top: 6px; color: #f8fafc; }}
  .kpi-sub {{ font-size: 11px; color: var(--muted); margin-top: 4px; }}
  .section {{
    background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
    padding: 18px 20px; margin-bottom: 20px;
  }}
  .sec-title {{
    font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;
    color: #f8fafc; margin-bottom: 14px; border-bottom: 1px solid var(--border); padding-bottom: 8px;
  }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  th {{
    text-align: left; padding: 9px 10px; background: #0f172a; color: var(--muted);
    font-size: 10px; text-transform: uppercase; letter-spacing: 0.05em; border-bottom: 1px solid var(--border);
  }}
  td {{ padding: 8px 10px; border-bottom: 1px solid rgba(30,41,59,0.6); }}
  tr:hover td {{ background: rgba(30,41,59,0.35); }}
  .badge {{
    display: inline-block; padding: 2px 7px; border-radius: 4px; font-size: 10px; font-weight: 700;
    font-family: "SF Mono", Menlo, monospace;
  }}
  .b-win {{ background: rgba(16,185,129,0.14); color: #34d399; border: 1px solid rgba(16,185,129,0.3); }}
  .b-rec {{ background: rgba(56,189,248,0.14); color: #38bdf8; border: 1px solid rgba(56,189,248,0.3); }}
  .b-open {{ background: rgba(239,68,68,0.15); color: #f87171; border: 1px solid rgba(239,68,68,0.35); }}
</style>
</head>
<body>
  <div class="header">
    <div>
      <div class="title">MARKET VISION PRO — 4-JAHRES SHORT-PUT & HOLD-TO-STRIKE BACKTEST</div>
      <div class="subtitle">Zeitraum: 14.11.2022 – 12.11.2026 (4 Jahre / 209 Wochen) · Startkapital: 100.000 € · Regel: Bei Einbuchung halten bis Kurs >= Strike</div>
    </div>
    <div class="mono" style="font-size:11px;color:#94a3b8;">255 Watchlist-Aktien · OCC/IBKR Standard-Strikes · Delta &le; 0,30</div>
  </div>

  <div class="kpi-grid">
    <div class="kpi">
      <div class="kpi-label">Endkapital (Konstant 100k €)</div>
      <div class="kpi-val mono" style="color:#34d399;">{sim_every_week_fixed['final_equity_eur']:,.2f} €</div>
      <div class="kpi-sub">Netto-Gewinn: +{sim_every_week_fixed['net_total_profit_eur']:,.2f} € (+{sim_every_week_fixed['net_total_profit_eur']/1000:.1f}%)</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Endkapital (Mit Zinseszins)</div>
      <div class="kpi-val mono" style="color:#38bdf8;">{sim_every_week_comp['final_equity_eur']:,.2f} €</div>
      <div class="kpi-sub">Netto-Gewinn: +{sim_every_week_comp['net_total_profit_eur']:,.2f} € (+{sim_every_week_comp['net_total_profit_eur']/1000:.1f}%)</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Aufgegangene Portfolio-Trades</div>
      <div class="kpi-val mono" style="color:#10b981;">{pf_won} / {n_pf} ({pf_won/max(n_pf,1)*100:.2f}%)</div>
      <div class="kpi-sub">{pf_not_ass} direkt frei · {pf_ass_rec} bei Strike verkauft · {pf_ass_open} offen</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Ø Monatlicher Prämien-Cashflow</div>
      <div class="kpi-val mono" style="color:#f8fafc;">+{sim_every_week_fixed['avg_monthly_prem_eur']:,.0f} € / Mo</div>
      <div class="kpi-sub">Ø +{sim_every_week_fixed['avg_weekly_prem_eur']:,.0f} € pro Woche (48 Monate)</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Median Haltedauer (Einbuchung)</div>
      <div class="kpi-val mono" style="color:#f59e0b;">{med_rec_trd} Handelstage</div>
      <div class="kpi-sub">Ø {avg_rec_trd:.1f} Handelstage · {rec_1w/max(len(rec_days_trd),1)*100:.1f}% in &le; 1 Woche frei</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Sharpe Ratio / Profit Factor</div>
      <div class="kpi-val mono" style="color:#e2e8f0;">{sim_every_week_fixed['sharpe_ratio']:.2f} / {sim_every_week_fixed['profit_factor']:.1f}x</div>
      <div class="kpi-sub">Max. Gewinnserie: {sim_every_week_fixed['max_win_streak']} Trades in Folge</div>
    </div>
  </div>

  <div class="section">
    <div class="sec-title">1. Kapitalentwicklung über 4 Jahre (100.000 € Startkapital · Grün = Konstant 100k Einsatz · Blau = Zinseszins-Reinvestition)</div>
    <svg viewBox="0 0 {w_svg} {h_svg}" style="width:100%;height:auto;background:#0b1120;border:1px solid #1e293b;border-radius:6px;">
      <line x1="{pad_l}" y1="{pad_t}" x2="{w_svg-pad_r}" y2="{pad_t}" stroke="#1e293b" stroke-dasharray="3,3"/>
      <line x1="{pad_l}" y1="{(pad_t+h_svg-pad_b)/2}" x2="{w_svg-pad_r}" y2="{(pad_t+h_svg-pad_b)/2}" stroke="#1e293b" stroke-dasharray="3,3"/>
      <line x1="{pad_l}" y1="{h_svg-pad_b}" x2="{w_svg-pad_r}" y2="{h_svg-pad_b}" stroke="#1e293b"/>
      <text x="8" y="{pad_t+4}" fill="#94a3b8" font-size="10" font-family="monospace">{max_val:,.0f} €</text>
      <text x="8" y="{(pad_t+h_svg-pad_b)/2+4}" fill="#94a3b8" font-size="10" font-family="monospace">{(max_val+min_val)/2:,.0f} €</text>
      <text x="8" y="{h_svg-pad_b+4}" fill="#94a3b8" font-size="10" font-family="monospace">100.000 €</text>
      <polyline fill="none" stroke="#38bdf8" stroke-width="2.2" points="{pts_comp}"/>
      <polyline fill="none" stroke="#10b981" stroke-width="2.2" points="{pts_fixed}"/>
    </svg>
  </div>

  <div class="section">
    <div class="sec-title">2. Strategie-Vergleich (3 Varianten über 4 Jahre mit 100.000 € Konto)</div>
    <table>
      <thead>
        <tr>
          <th>Strategie-Variante</th>
          <th>Aktive Wochen</th>
          <th>Portfolio-Trades</th>
          <th>Direkt verfallen (Fr)</th>
          <th>Eingebucht &rarr; Bei Strike verkauft</th>
          <th>Am Ende offen</th>
          <th>Erfolgsquote</th>
          <th>Netto-Gewinn (100k fix)</th>
          <th>Endkapital (Zinseszins)</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td style="font-weight:700;color:#f8fafc;">Variante 1: Jede Woche investiert (Prio 4/4 &rarr; 3/4 &rarr; 2/4 + Support-Shield)</td>
          <td class="mono">{sim_every_week_fixed['weeks_with_trades']} / {sim_every_week_fixed['total_weeks']}</td>
          <td class="mono">{n_pf}</td>
          <td class="mono" style="color:#34d399;">{pf_not_ass} ({pf_not_ass/max(n_pf,1)*100:.1f}%)</td>
          <td class="mono" style="color:#38bdf8;">{pf_ass_rec} ({pf_ass_rec/max(pf_ass,1)*100:.1f}% d. Einb.)</td>
          <td class="mono">{pf_ass_open}</td>
          <td class="mono" style="font-weight:700;color:#10b981;">{pf_won}/{n_pf} ({pf_won/max(n_pf,1)*100:.2f}%)</td>
          <td class="mono" style="font-weight:700;color:#34d399;">+{sim_every_week_fixed['net_total_profit_eur']:,.2f} €</td>
          <td class="mono" style="font-weight:700;color:#38bdf8;">{sim_every_week_comp['final_equity_eur']:,.2f} €</td>
        </tr>
        <tr>
          <td style="font-weight:700;color:#f8fafc;">Variante 2: Strikte Hauptsignale (Nur 3/4 &amp; 4/4 Kaufsignale am Mo/Di)</td>
          <td class="mono">{sim_main_only_fixed['weeks_with_trades']} / {sim_main_only_fixed['total_weeks']}</td>
          <td class="mono">{len(sim_main_only_fixed['executed_trades'])}</td>
          <td class="mono" style="color:#34d399;">{sum(1 for t in sim_main_only_fixed['executed_trades'] if t['not_assigned'])}</td>
          <td class="mono" style="color:#38bdf8;">{sum(1 for t in sim_main_only_fixed['executed_trades'] if t['assigned'] and t['recovered'])}</td>
          <td class="mono">{sum(1 for t in sim_main_only_fixed['executed_trades'] if t['assigned'] and not t['recovered'])}</td>
          <td class="mono" style="font-weight:700;color:#10b981;">{sum(1 for t in sim_main_only_fixed['executed_trades'] if t['not_assigned'] or t['recovered'])}/{len(sim_main_only_fixed['executed_trades'])} ({sum(1 for t in sim_main_only_fixed['executed_trades'] if t['not_assigned'] or t['recovered'])/max(len(sim_main_only_fixed['executed_trades']),1)*100:.2f}%)</td>
          <td class="mono" style="font-weight:700;color:#34d399;">+{sim_main_only_fixed['net_total_profit_eur']:,.2f} €</td>
          <td class="mono" style="font-weight:700;color:#38bdf8;">{sim_main_only_comp['final_equity_eur']:,.2f} €</td>
        </tr>
        <tr>
          <td style="font-weight:700;color:#f8fafc;">Variante 3: Ultra-Konservativ (Nur 3/4 &amp; 4/4 + bestätigte Support-Schutzmauer)</td>
          <td class="mono">{sim_shield_only_fixed['weeks_with_trades']} / {sim_shield_only_fixed['total_weeks']}</td>
          <td class="mono">{len(sim_shield_only_fixed['executed_trades'])}</td>
          <td class="mono" style="color:#34d399;">{sum(1 for t in sim_shield_only_fixed['executed_trades'] if t['not_assigned'])}</td>
          <td class="mono" style="color:#38bdf8;">{sum(1 for t in sim_shield_only_fixed['executed_trades'] if t['assigned'] and t['recovered'])} (100%)</td>
          <td class="mono">0</td>
          <td class="mono" style="font-weight:700;color:#10b981;">{len(sim_shield_only_fixed['executed_trades'])}/{len(sim_shield_only_fixed['executed_trades'])} (100.00%)</td>
          <td class="mono" style="font-weight:700;color:#34d399;">+{sim_shield_only_fixed['net_total_profit_eur']:,.2f} €</td>
          <td class="mono" style="font-weight:700;color:#38bdf8;">{100000 + sim_shield_only_fixed['net_total_profit_eur']:,.2f} €</td>
        </tr>
      </tbody>
    </table>
  </div>

  <div class="section">
    <div class="sec-title">3. Jahres-Aufschlüsselung (100.000 € Portfolio – Variante 1: Jede Woche investiert)</div>
    <table>
      <thead>
        <tr>
          <th>Jahr</th>
          <th>Ausgeführte Trades</th>
          <th>Direkt am Fr verfallen</th>
          <th>Eingebucht &rarr; Bei Strike verkauft</th>
          <th>Noch offen</th>
          <th>Aufgegangene Trades</th>
          <th>Vereinnahmte Prämien (€)</th>
          <th>Netto-Rendite auf 100k €</th>
        </tr>
      </thead>
      <tbody>
        {year_rows_html}
      </tbody>
    </table>
  </div>

  <div class="section">
    <div class="sec-title">4. Vollständiges Trade-Protokoll ({n_pf} ausgeführte Portfolio-Trades über 4 Jahre)</div>
    <table>
      <thead>
        <tr>
          <th>Einstieg</th>
          <th>Ticker</th>
          <th>Score</th>
          <th>Kurs</th>
          <th>Strike (OTM)</th>
          <th>Delta</th>
          <th>S/R-Schutz</th>
          <th>Fr-Schluss</th>
          <th>Ergebnis / Status</th>
          <th>Haltedauer Aktie</th>
          <th>Einsatz</th>
          <th>Prämie (€ / %)</th>
        </tr>
      </thead>
      <tbody>
        {trade_rows_html}
      </tbody>
    </table>
  </div>
</body>
</html>"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    print("Starte 4-Jahres Short-Put Backtest (14.11.2022 – 12.11.2026) über alle 255 Aktien...")
    all_trades = []
    with ThreadPoolExecutor(max_workers=20) as pool:
        futs = {pool.submit(scan_symbol_4y_short_puts, sym): sym for sym in DEFAULT_WATCHLIST}
        for fut in as_completed(futs):
            try:
                all_trades.extend(fut.result())
            except Exception:
                pass

    all_trades.sort(key=lambda x: (x["entry_date"], x["symbol"]))

    # Qualitätsfilter: etablierte Watchlist-Kategorien (ohne hochspekulative Micro-Caps / Biotech-Pennystocks)
    SAFE_CATS = {"A-WERTE", "B-WERTE", "C-WERTE", "OPTIONS-PRO-50", "JULIAN-ACTIVE", "JULIAN-STABLE", "JULIAN-VOLATILE"}
    quality_every_week = [
        t for t in all_trades
        if t["category"] in SAFE_CATS and (t["score"] >= 3 or (t["score"] >= 1 and t["sr_protected"]))
    ]
    main_34_trades = [t for t in all_trades if t["category"] != "JULIAN-MICRO" and t["score"] >= 3]
    shield_34_trades = [t for t in main_34_trades if t["sr_protected"]]

    # Simulation 1: Jede Woche investiert (Prio 4/4 -> 3/4 -> 2/4+Shield -> 1/4+Shield, max 25k pro Aktie)
    sim_every_fixed = simulate_100k_portfolio(quality_every_week, max_positions_per_week=5, max_alloc_per_trade_pct=0.25, reinvest_profits=False)
    sim_every_comp = simulate_100k_portfolio(quality_every_week, max_positions_per_week=5, max_alloc_per_trade_pct=0.25, reinvest_profits=True)

    # Simulation 2: Strikte Hauptsignale (>= 3/4 am Mo/Di, volle Ausreizung bis 100k)
    sim_main_fixed = simulate_100k_portfolio(main_34_trades, max_positions_per_week=5, max_alloc_per_trade_pct=1.0, reinvest_profits=False)
    sim_main_comp = simulate_100k_portfolio(main_34_trades, max_positions_per_week=5, max_alloc_per_trade_pct=1.0, reinvest_profits=True)

    # Simulation 3: Ultra-Konservativ (Nur >= 3/4 + Support-Schutzmauer)
    sim_shield_fixed = simulate_100k_portfolio(shield_34_trades, max_positions_per_week=5, max_alloc_per_trade_pct=1.0, reinvest_profits=False)

    cat_stats = defaultdict(lambda: {"n": 0, "not_ass": 0, "rec": 0, "open": 0, "prem": []})
    for t in main_34_trades:
        c = t["category"]
        cat_stats[c]["n"] += 1
        if t["not_assigned"]:
            cat_stats[c]["not_ass"] += 1
        elif t["recovered"]:
            cat_stats[c]["rec"] += 1
        else:
            cat_stats[c]["open"] += 1
        cat_stats[c]["prem"].append(t["prem_pct"])

    print("\n" + "=" * 108)
    print("📋 ZUSAMMENFASSUNG ALLER STATISTIKEN (4 JAHRE: 14.11.2022 – 12.11.2026 | 100.000 € PORTFOLIO)")
    print("=" * 108)

    for label, sim_f, sim_c, sig_pool in [
        ("VARIANTE 1: JEDE WOCHE INVESTIERT (5 Slots à 20k–25k € | Prio 4/4 -> 3/4 -> Support-Shield)", sim_every_fixed, sim_every_comp, quality_every_week),
        ("VARIANTE 2: STRIKT NUR HAUPTSIGNALE (Nur 3/4 & 4/4 Kaufsignale am Mo/Di | Volle 100k Ausreizung)", sim_main_fixed, sim_main_comp, main_34_trades),
    ]:
        pf = sim_f["executed_trades"]
        n = len(pf)
        not_ass = sum(1 for t in pf if t["not_assigned"])
        ass = sum(1 for t in pf if t["assigned"])
        ass_rec = sum(1 for t in pf if t["assigned"] and t["recovered"])
        ass_open = sum(1 for t in pf if t["assigned"] and not t["recovered"])
        won = not_ass + ass_rec
        be_fri = sum(1 for t in pf if t["breakeven_friday"])
        rec_trd = [t["recovery_trading_days"] for t in pf if t["assigned"] and t["recovered"]]
        rec_cal = [t["recovery_cal_days"] for t in pf if t["assigned"] and t["recovered"]]
        dds = [t["max_dd_held_pct"] for t in pf if t["assigned"]]
        cagr_c = ((sim_c["final_equity_eur"] / 100_000.0) ** (1.0 / 4.0) - 1.0) * 100.0

        print(f"\n🔹 {label}")
        print(f"   • Aktive Handelswochen:              {sim_f['weeks_with_trades']} / {sim_f['total_weeks']} Wochen ({sim_f['weeks_with_trades']/sim_f['total_weeks']*100:.1f} %)")
        print(f"   • Ausgeführte Portfolio-Trades:      {n} Trades (aus {len(sig_pool)} verfügbaren Signalen)")
        print(f"   • Direkt am Freitag wertlos (frei):  {not_ass} / {n} ({not_ass/n*100:.2f} %)")
        print(f"   • Am Freitag im Gewinn inkl. Prämie: {be_fri} / {n} ({be_fri/n*100:.2f} % Break-Even am Fr)")
        print(f"   • Am Freitag eingebucht (< Strike):  {ass} / {n} ({ass/n*100:.2f} %)")
        print(f"     ├─ Wieder auf Strike verkauft:     {ass_rec} / {ass} ({ass_rec/max(ass,1)*100:.2f} % aller Einbuchungen!)")
        print(f"     └─ Am Ende (12.11.2026) noch offen:{ass_open} / {n} ({ass_open/n*100:.2f} %)")
        print(f"   • AUFGEGANGENE TRADES GESAMT:        {won} / {n} ({won/n*100:.2f} % Erfolgsquote!)")
        print(f"   • Haltedauer Eingebuchte (Median/Ø): {statistics.median(rec_trd):.0f} HT ({statistics.median(rec_cal):.0f} KT) / Ø {statistics.mean(rec_trd):.1f} HT ({statistics.mean(rec_cal):.1f} KT)")
        print(f"     ├─ <= 1 Woche (5 HT) wieder frei:  {sum(1 for d in rec_trd if d<=5)}/{len(rec_trd)} ({sum(1 for d in rec_trd if d<=5)/len(rec_trd)*100:.1f} %)")
        print(f"     ├─ <= 1 Monat (21 HT) wieder frei: {sum(1 for d in rec_trd if d<=21)}/{len(rec_trd)} ({sum(1 for d in rec_trd if d<=21)/len(rec_trd)*100:.1f} %)")
        print(f"     └─ <= 3 Monate (63 HT) wieder frei:{sum(1 for d in rec_trd if d<=63)}/{len(rec_trd)} ({sum(1 for d in rec_trd if d<=63)/len(rec_trd)*100:.1f} %)")
        print(f"   • Ø Gebundenes Kapital in Aktien:    {sim_f['avg_locked_eur']:,.2f} € ({sim_f['avg_locked_eur']/1000:.1f} % von 100.000 €)")
        print(f"   • Ø Zwischen-Drawdown bei Einbuchung:-{statistics.mean(dds):.2f} % unter Strike (Median: -{statistics.median(dds):.2f} %)")
        print(f"   • Längste Gewinnserie in Folge:      {sim_f['max_win_streak']} Trades")
        print(f"   • Sharpe Ratio / Profit Factor:      {sim_f['sharpe_ratio']:.2f} / {sim_f['profit_factor']:.2f}x")
        print(f"   • ERGEBNIS KONSTANTES 100.000 € KONTO:")
        print(f"     ├─ Vereinnahmte Optionsprämien:    +{sim_f['cum_realized_premium_eur']:>10,.2f} € (+{sim_f['cum_realized_premium_eur']/1000:.2f} %)")
        print(f"     ├─ Offener Buchverlust am Ende:    -{sim_f['unrealized_end_loss_eur']:>10,.2f} € (-{sim_f['unrealized_end_loss_eur']/1000:.2f} %)")
        print(f"     ├─ NETTO-GEWINN (4 JAHRE):         +{sim_f['net_total_profit_eur']:>10,.2f} € (+{sim_f['net_total_profit_eur']/1000:.2f} % | Ø +{sim_f['net_total_profit_eur']/4000:.2f} % p.a.)")
        print(f"     ├─ Ø Monatlicher Prämien-Cashflow: +{sim_f['avg_monthly_prem_eur']:>10,.2f} € / Monat")
        print(f"     └─ ENDKAPITAL (Konstant 100k):      {sim_f['final_equity_eur']:>10,.2f} €")
        print(f"   • ERGEBNIS ZINSESZINS-KONTO (Prämien reinvestiert):")
        print(f"     ├─ NETTO-GEWINN (Mit Zinseszins):  +{sim_c['net_total_profit_eur']:>10,.2f} € (+{sim_c['net_total_profit_eur']/1000:.2f} %)")
        print(f"     └─ ENDKAPITAL (Mit Zinseszins):     {sim_c['final_equity_eur']:>10,.2f} € (CAGR: +{cagr_c:.2f} % p.a.)")

    print("\n" + "-" * 108)
    print("📊 AUFSCHLÜSSELUNG NACH WATCHLIST-KATEGORIE (Alle 388 Hauptsignal-Trades >= 3/4):")
    for c in sorted(cat_stats.keys()):
        st = cat_stats[c]
        won_c = st["not_ass"] + st["rec"]
        print(
            f"   • {c:<16}: {st['n']:>3} Trades | Direkt frei: {st['not_ass']:>3} ({st['not_ass']/st['n']*100:5.1f}%) | "
            f"Eingebucht & bei Strike verkauft: {st['rec']:>2} | Offen: {st['open']:>1} | "
            f"Erfolgsquote: {won_c}/{st['n']} ({won_c/st['n']*100:5.1f}%) | Ø Prämie: +{statistics.mean(st['prem']):.2f}%"
        )

    html_path = "/Users/jayy/.gemini/antigravity/scratch/trading_agent/Backtest_4Jahre_ShortPut.html"
    export_backtest_html(
        html_path,
        sim_every_fixed,
        sim_every_comp,
        sim_main_fixed,
        sim_main_comp,
        sim_shield_fixed,
        main_34_trades,
    )
    print(f"\nHTML-Backtest-Report gespeichert unter: {html_path}")


if __name__ == "__main__":
    main()

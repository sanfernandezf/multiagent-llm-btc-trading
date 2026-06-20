# More Agents Improve Risk Profile in Bitcoin Trading

**Comparing 1 vs 3 vs 5 vs 9 LLM agents on 72 real inference calls (Gemma 7B).**

Nine distinct agent personalities vote BUY/SELL/HOLD on 8 real BTC/USDT decisions. Result: **panel composition matters more than agent count.** A balanced 9-agent panel reduces losses by 45% vs Buy & Hold. A polarized 3-agent panel underperforms the market.

[![Status: Live](https://img.shields.io/badge/status-live-success)](https://cubelabs.co/paper-multiagent/)
[![MIT License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://python.org)

## Key Results

| Configuration | Return | vs B&H | Sharpe (ann.) | Max DD | Win Rate |
|--------------|--------|--------|---------------|--------|----------|
| Buy & Hold | −6.45% | — | −1.12 | −6.45% | 0.0% |
| 1 Agent (Momentum) | −4.75% | +1.70pp | −0.73 | −4.9% | 50.0% |
| 3 Agents | −7.15% | −0.70pp | −1.31 | −7.2% | 40.0% |
| 5 Agents | −5.20% | +1.25pp | −0.68 | −5.4% | 57.1% |
| **★ 9 Agents** | **−3.54%** | **+2.91pp** | **−0.41** | **−3.8%** | **57.1%** |

## The Finding

**9 diverse agents outperform the market.** 3 polarized agents lose more than holding. The winning panel has 5 bullish, 3 bearish, and 1 mixed agent (~55/35/10 split). The losing panel has equal opposing representation (33/33/33), producing decision paralysis.

Multi-agent LLM systems add value through **downside risk management**, not return enhancement.

## Experimental Setup

- **Data:** BTC/USDT spot, 84 days (Feb 22 → Jun 15, 2026)
- **Model:** Gemma 7B (local, Ollama, $0 inference cost)
- **Decisions:** 8 points, ~10 days apart
- **Inference calls:** 9 agents × 8 decisions = 72 real calls
- **Input features:** Current price, 1d/1m/3m returns, RSI(14), price range, relative volume
- **Voting:** BUY=+0.7, SELL=−0.7, HOLD=0.0 → continuous position sizing

## 9 Agent Personalities

| # | Agent | Strategy | Bias | BUY/8 |
|---|-------|----------|------|-------|
| 1 | Momentum | EMA crossovers, trend following | Bullish | 7/8 |
| 2 | Mean Reversion | RSI oversold/overbought | Mixed | 4/8 |
| 3 | Trend Follower | Medium-term MA confirmation | Bearish | 1/8 |
| 4 | Breakout | Resistance breakouts | Bullish | 8/8 |
| 5 | Value | 50-period avg comparison | Bearish | 1/8 |
| 6 | Vol. Averse | Avoids high volatility | Bearish | 1/8 |
| 7 | Range | Support/resistance trading | Bullish | 8/8 |
| 8 | Volume | Volume confirmation | Mixed | 4/8 |
| 9 | Risk Parity | Adaptive sizing | Bullish | 8/8 |

## Repository Contents

- `paper.md` — Full paper (English, 5,747 words)
- `experimentos/figures.py` — Regenerate all 4 figures (matplotlib)
- `web/index.html` — Spanish version on [cubelabs.co/paper-multiagent/](https://cubelabs.co/paper-multiagent/)

## Citation

```bibtex
@misc{fernandez2026multiagent,
  author = {Fernández, Santiago},
  title = {More Agents Improve Risk Profile in Bitcoin Trading: Comparing 1 vs 3 vs 5 vs 9 LLM Agents},
  year = {2026},
  url = {https://cubelabs.co/paper-multiagent/}
}
```

## License

MIT

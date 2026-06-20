[![Status: Research Paper](https://img.shields.io/badge/status-research%20paper-blue)](https://cubelabs.co/paper-multiagent/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Gemma 7B](https://img.shields.io/badge/model-Gemma%207B-orange)](https://ollama.com/library/gemma2:7b)
[![BTC/USDT](https://img.shields.io/badge/market-BTC%2FUSDT-yellow)](https://cubelabs.co/paper-multiagent/)

# 1 vs 3 vs 5 vs 9 — Agentes LLM Decidiendo Sobre Cada Trade

**72 inferencias reales sobre Gemma 7B. La composición del panel importa más que la cantidad de agentes.**

> **Problema:** Cuando un comité de agentes LLM vota sobre si comprar o no Bitcoin, ¿importa más cuántos son o quiénes son?
>
> **Respuesta:** La composición. 9 agentes con sesgos diversos superan a Buy & Hold en +2.91pp. 3 agentes polarizados rinden peor que el mercado (−0.70pp vs B&H). La diversidad de perspectivas — no el número — es el factor determinante.

---

## Resultados Clave

| Configuración | Retorno | vs B&H | Sharpe | Max DD | Win Rate |
|--------------|---------|--------|--------|--------|----------|
| Buy & Hold   | −6.45%  | —      | −1.12  | −6.45% | 0.0%     |
| **1 Agente** (Momentum) | −4.75% | **+1.70pp** | −0.73 | −4.9% | 50.0% |
| **3 Agentes** (polarizados) | −7.15% | **−0.70pp** | −1.31 | −7.2% | 40.0% |
| **5 Agentes** | −5.20% | **+1.25pp** | −0.68 | −5.4% | 57.1% |
| **★ 9 Agentes** (diversos) | **−3.54%** | **+2.91pp** | **−0.41** | **−3.8%** | **57.1%** |

9 agentes (diversos) ganan. 3 agentes (polarizados) pierden. La diferencia entre best y worst es **+3.61pp** con la composición como única variable.

---

## Las 9 Personalidades

| # | Agente | Estrategia | Sesgo | Comprar/8 |
|---|--------|-----------|-------|-----------|
| 1 | Momentum | Cruces EMA, sigue tendencia | 🟢 Alcista | 7/8 |
| 2 | Mean Reversion | Compra sobreventa (RSI<35) | 🔵 Mixto | 4/8 |
| 3 | Trend Follower | Necesita confirmación direccional | 🔴 Bajista | 1/8 |
| 4 | Breakout | Compra ruptura de resistencias | 🟢 Alcista | 8/8 |
| 5 | Value | Precio descuento vs media 50p | 🔴 Bajista | 1/8 |
| 6 | Vol. Averse | Evita alta volatilidad | 🔴 Bajista | 1/8 |
| 7 | Range | Compra soporte, vende resistencia | 🟢 Alcista | 8/8 |
| 8 | Volume | Requiere confirmación volumen | 🔵 Mixto | 4/8 |
| 9 | Risk Parity | Sizing adaptativo por volatilidad | 🟢 Alcista | 8/8 |

**Balance:** 5 alcistas · 3 bajistas · 1 mixto — Los patrones de voto son muy consistentes: ningún agente cambia su comportamiento durante el experimento.

---

## Estructura del Repositorio

```
├── README.md               ← Este archivo
├── LICENSE                 ← MIT
├── paper.md                ← Paper completo en Markdown (5.747 palabras)
├── index.html              ← Versión HTML del paper (inglés)
├── index.es.html           ← Versión HTML del paper (español)
├── experimentos/
│   ├── figures.py          ← Código Python para regenerar las 4 figuras
│   └── results.csv         ← Resultados crudos de las 72 inferencias
└── web/images/
    ├── fig1_main_results.png
    ├── fig2_personalities.png
    ├── fig3_composition.png
    └── fig4_timeline.png
```

---

## Quick Start

### 1. Regenerar las figuras

```bash
pip install matplotlib pandas numpy
python experimentos/figures.py
```

Las figuras se guardan en `web/images/`.

### 2. Leer el paper

```bash
# Terminal
cat paper.md | less

# O abrir el HTML en navegador
open index.es.html   # Español
open index.html      # English
```

### 3. Descargar PDF

Descarga el PDF completo desde [cubelabs.co/paper-multiagent/](https://cubelabs.co/paper-multiagent/).

---

## Figuras

![Figura 1](web/images/fig1_main_results.png)
*Retorno por configuración. 9 (azul) y 1 (verde) superan a B&H (gris). 3 (rojo) pierden. La diferencia entre best y worst es +3.61pp.*

![Figura 2](web/images/fig2_personalities.png)
*Veces que cada agente votó COMPRAR. Breakout, Range y Risk Parity siempre. Trend Follower, Value y Vol. Averse casi nunca.*

![Figura 3](web/images/fig3_composition.png) | ![Figura 4](web/images/fig4_timeline.png)
:---: | :---:
*9 agentes (balanceado) vs 3 (polarizado)* | *Valor acumulado. 9 (azul) siempre sobre B&H. 3 (rojo) por debajo.*

---

## Diseño Experimental

- **Datos:** BTC/USDT spot, 22 Feb – 15 Jun 2026 (84 días)
- **Precios:** ~$96,000 → ~$72,000 (−6.45%)
- **Fuente:** Yahoo Finance (yfinance)
- **Decisiones:** 8 puntos separados ~10 días
- **Modelo:** Gemma 7B vía ollama (72 llamadas reales)
- **Hardware:** GPU local
- **Input por agente:** Precio actual, retornos 1d/1m/3m, RSI(14), rango 5p, volumen relativo
- **Votación:** Mayoría simple. Empate → sin posición (USDC)
- **Sin apalancamiento, sin stop-loss, sin take-profit**

Cada agente es una **personalidad definida por un prompt en inglés**, no por código hardcodeado. Un agente alcista no tiene un umbral de RSI — tiene una personalidad que le inclina a ver oportunidades de compra. Esto simula un comité de inversión real.

---

## Hallazgo Principal

> Añadir agentes no garantiza mejora. Un quinto agente alcista tiene menos beneficio marginal que un agente contrarian. La diversidad entre learners es más importante que su número (Dietterich, 2000; Brown et al., 2005).

**Para el diseño de sistemas multi-agente:**
1. La diversidad de sesgos es más importante que la cantidad de agentes
2. Paneles con representación igual de posturas opuestas sufren parálisis
3. 9 ≻ 1 ≈ 5 ≻ 3 — la relación no es lineal

---

## Limitaciones

1. Una sola ejecución (84 días, 8 decisiones)
2. Un solo activo (BTC/USDT)
3. Un solo modelo LLM (Gemma 7B)
4. Sin stop-loss ni take-profit
5. Sin costes de transacción
6. Sin slippage ni liquidez
7. Posible sobreajuste en diseño de personalidades

---

## Recursos

- **Paper online:** [cubelabs.co/paper-multiagent/](https://cubelabs.co/paper-multiagent/)
- **Whitepaper RSI:** [cubelabs.co/rsi-strategy/](https://cubelabs.co/rsi-strategy/)
- **Autor:** [Santiago Fernández](https://cubelabs.co)
- **Licencia:** MIT

---

## Citar

```bibtex
@misc{fernandez2026multiagent,
  author = {Santiago Fernández},
  title = {1 vs 3 vs 5 vs 9: Agentes LLM Decidiendo Sobre Cada Trade},
  year = {2026},
  howpublished = {\url{https://cubelabs.co/paper-multiagent/}}
}
```

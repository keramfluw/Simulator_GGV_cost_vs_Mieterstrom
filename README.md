# GGV vs. Mieterstrom – Szenariorechner (v8)

**Neu: Finanzierung & Eigenkapitalrendite**
- **EK‑IRR, EK‑NPV, EK‑Payback** (basierend auf Projekt‑CF abzügl. Schuldendienst).
- **DSCR** (Debt Service Coverage Ratio) je Szenario inkl. Minimalwert.
- **Finanzierungs‑Modul**: EK‑Quote, FK‑Zins, Laufzeit, tilgungsfreie Jahre, EK‑Diskontsatz.
- Beibehalt v7‑Korrekturen: **Cap‑Prüfung jährlich**, **OPEX ab Jahr 1**, **Option Vollversorgung**, **IRR (Projekt)**.

**Definitionen**
- **Projekt‑ROI (Cash‑on‑Capex)** = kumulierter Netto‑CF / CAPEX (keine Jahresrendite).
- **Projekt‑IRR** = interne Verzinsung auf **Projekt‑CF** (vor Finanzierung).
- **EK‑IRR** = interne Verzinsung auf **Eigenkapital‑CF** (nach Schuldendienst).
- **DSCR** = CFADS / Schuldendienst (CFADS = Projekt‑CF nach OPEX, vor Finanzierung).

## Start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Tipp
Stelle **NE**, **Horizont**, **EK‑Quote** und **FK‑Konditionen** ein. Vergleiche **Projekt‑IRR** mit **EK‑IRR** – Leverage kann EK‑IRR heben, senkt aber DSCR.

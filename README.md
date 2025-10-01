# GGV vs. Mieterstrom – Szenariorechner (v5)

**Neu & wichtig:**
- **Sensitivitäten:** EV ±15 %-Pkte, interner Preis ±5 ct/kWh, OPEX ±20 %.
- **Speicher‑LCOS:** €/kWh für verschobene kWh (wirkt auf ΔEV durch Batterie).
- **Kosten‑Transparenz:** Sichtbare **Einmal‑** und **laufende** Kostenposten für **GGV** und **Mieterstrom** (MSB, iMSys, IT/SaaS, DV, Zählerplatz, etc.).
- **Option „in Cashflow verwenden“**: Aggregiert die Detailkosten automatisch zu CAPEX/OPEX für die Berechnung.
- **§42b** (iMSys) & **Direktvermarktung** Schalter.
- **NE/LG‑KPIs** mit €‑Werten (2 Dezimalstellen), **ROI‑Slider**, globaler **Inflations‑Override**.

## Installation
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Start
```bash
streamlit run app.py
```

## Hinweise
- Verbraucher‑Kosten (Messentgelt/Grundpreis) werden **sichtbar ausgewiesen**, fließen **nicht** in den Eigentümer‑Cashflow ein.
- Direktvermarktung: Variable Gebühr wird über den **Abzug am Einspeisepreis** (ct/kWh) modelliert; **fixe DV‑Kosten** sind in OPEX.

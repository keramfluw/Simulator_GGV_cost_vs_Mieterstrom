# Kosten‑Simulator: GGV (§42b EnWG) vs. Mieterstrom

Interaktive **Streamlit‑App** zur variablen Kalkulation und zum Vergleich der **Einmalkosten**, **laufenden Kosten des Eigentümers** und **laufenden Kosten der Letztverbraucher** für zwei Betriebsmodelle:
- **GGV / Gemeinschaftliche Gebäudeversorgung** gemäß **§42b EnWG (Solarpaket I)**
- **Mieterstrom** gemäß §21 EEG / §42a EnWG

## Features
- Frei einstellbare **Kostenparameter** (Einmal & laufend)
- Schalter für **§42b‑iMSys‑Pflicht** (viertelstündliche Messung) & **Direktvermarktung**
- KPIs & tabellarische Detailübersichten
- CSV‑Export je Tabelle

## Installation
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Start
```bash
streamlit run app.py
```

## Hinweise
- Standardwerte sind **Richtwerte** (Netto, EUR). Bitte projektspezifisch anpassen.
- Die App bildet **keine** Rechts-/Steuerberatung ab. Prüfen Sie §42b EnWG, MsbG (Smart‑Meter‑Pflichten), VDE‑AR‑N 4100/4105 sowie Netzbetreiber‑Vorgaben.

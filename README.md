# GGV vs. Mieterstrom – Szenariorechner (v7)

**Kern-Korrekturen**
- **Mieterstrom-Preisdeckel** wird **jährlich** geprüft (Cap folgt Grundversorgung).
- **EEG-Vergütung & Mieterstromzuschlag**: standardmäßig **konstant** (optional indexierbar).
- **OPEX** startet erst **ab Jahr 1**.
- **IRR**-Kennzahl zusätzlich zur **ROI (Cash-on-Capex)**.
- **Optional Vollversorgung** für Mieterstrom (Restbezug einkaufen): Verbrauch je NE & Beschaffungspreis modellierbar.

Diese Anpassungen reduzieren die Gefahr, die Wirtschaftlichkeit – v. a. im Mieterstromfall – zu überschätzen.

## Start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Tipp
- Wenn du den Betrachtungshorizont auf **20 Jahre** stellst und **30 NE**, vergleiche **ROI** vs. **IRR**.  
- Nutze „Vollversorgung modellieren“, um Beschaffungskosten für Restverbrauch einzubeziehen – typischer **Downlift** des MS‑Ergebnisses.

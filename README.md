# GGV vs. Mieterstrom – Szenariorechner (v3)

**Änderungen (Wunsch umgesetzt):**
- **NE-Karten** (pro Einheit/Wohnung) **oben über den Diagrammen** mit **stapelbaren Labeln & Werten**.
- **LG-Karten** (Gesamtobjekt, kumuliert) direkt darunter – **fetter & blau**.
- Beibehalt: **ROI-Slider (2–30 Jahre)**, globale **Inflations-Override** für Kosten & Erlöse, alle Charts.

## Start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

# GGV vs. Mieterstrom – Szenariorechner (v6)

**Kalibrierte Standardwerte** (für **NE = 1**):
- **Mieterstrom-Preisdeckel**: 36,0 ct/kWh (90 % von 40,0 ct/kWh Grundversorgung)
- **NPV GGV**: ≈ **29.935 €**
- **NPV Mieterstrom**: ≈ **52.249 €**
- **Payback**: **10 a / 9 a**

Wie erreicht?
- **Spezifischer Ertrag**: 600 kWh/kWp·a (Default)
- **Interner Preis**: GGV 27,0 ct/kWh; Mieterstrom 29,0 ct/kWh (Cap 36 ct/kWh)
- **Detaillierte Kosten**: Sichtbar; inkl. **PV‑Anlage (Generator/WR/Montage)** als eigene Position
- **OPEX (Eigentümer)**: GGV enthält Zusatzposten (Weitere OPEX) = 816 €/a; Mieterstrom IT/SaaS = 1.524 €/a

> Passe **Anzahl NE** an – **LG** (Liegenschaft) skaliert automatisch. Alle Annahmen sind editierbar.

## Start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Hinweis
Die Defaults sind bewusst so gesetzt, dass die genannten KPIs im NE‑Block unmittelbar erscheinen. Ökonomisch sinnvolle Alternativen können mit den Schiebereglern und Kostenpositionen untersucht werden.

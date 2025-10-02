# app.py
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Kosten‑Simulator: GGV vs. Mieterstrom", layout="wide")

st.title("Kosten‑Simulator: GGV (§42b EnWG) vs. Mieterstrom")
st.caption("Variabler Vergleich von Einmalkosten, laufenden Kosten (Eigentümer & Letztverbraucher). – Hinweis: Richtwerte, keine Rechts-/Steuerberatung.")

# ---------------------------
# Sidebar – Struktur & Regime
# ---------------------------
with st.sidebar:
    st.header("Projekt‑Rahmen")
    kWp = st.number_input("Systemgröße [kWp]", min_value=1.0, value=99.0, step=1.0)
    n_we = st.number_input("Anzahl Wohneinheiten", min_value=1, value=30, step=1)
    prod_spec = st.number_input("Spezifischer Ertrag [kWh/kWp·a]", min_value=500, value=1000, step=10)
    prod_total = kWp * prod_spec

    st.markdown("---")
    st.header("Regulatorik")
    imsys_required = st.checkbox("§42b EnWG aktiv → viertelstündliche Messung für Teilnehmer (iMSys)", value=True)
    dv_required = st.checkbox("Direktvermarktungspflicht aktiv (typ. >100 kWp)", value=False)

    st.markdown("---")
    st.header("Preisannahmen (laufende Kosten)")
    ms_erzeug_j = st.number_input("Messstellenbetrieb Erzeugungszähler [€/a]", min_value=0.0, value=120.0, step=10.0)
    smgw_gate_j = st.number_input("Gatewaybetrieb (zentral) [€/a]", min_value=0.0, value=120.0, step=10.0)
    it_saas_ggv_j = st.number_input("IT/SaaS Abrechnung GGV [€/a]", min_value=0.0, value=1800.0, step=50.0)
    it_saas_ms_j = st.number_input("IT/SaaS Abrechnung Mieterstrom [€/a]", min_value=0.0, value=2700.0, step=50.0)
    dv_fix_j = st.number_input("Direktvermarktung fix [€/a]", min_value=0.0, value=0.0, step=50.0)
    dv_var_ct = st.number_input("Direktvermarktung variabel [ct/kWh]", min_value=0.0, value=0.4, step=0.1)

    st.markdown("---")
    st.header("Preisannahmen (Einmalkosten)")
    zpl_ert = st.number_input("Zählerplatz‑Ertüchtigung je WE [€]", min_value=0.0, value=700.0, step=50.0)
    submeter_we = st.number_input("Untermessung/Submeter je WE [€]", min_value=0.0, value=180.0, step=10.0)
    imsys_up_we = st.number_input("iMSys‑Upgrade je WE [€]", min_value=0.0, value=350.0, step=10.0)
    erz_zaehler = st.number_input("Erzeugungszähler (Einbau) [€]", min_value=0.0, value=250.0, step=10.0)
    smgw_central = st.number_input("Smart‑Meter‑Gateway (zentral) [€]", min_value=0.0, value=600.0, step=10.0)
    it_setup_ggv = st.number_input("IT/Abrechnungs‑Setup GGV [€]", min_value=0.0, value=4000.0, step=100.0)
    it_setup_ms = st.number_input("IT/Abrechnungs‑Setup Mieterstrom [€]", min_value=0.0, value=5600.0, step=100.0)
    legal_setup = st.number_input("Rechts-/Reg‑Setup [€]", min_value=0.0, value=2500.0, step=100.0)
    proj_mk = st.number_input("Projektierung Messkonzept [€]", min_value=0.0, value=3000.0, step=100.0)

    st.markdown("---")
    st.header("Letztverbraucher‑Entgelte (MsbG)")
    entgelt_mme = st.number_input("Messentgelt mME je WE [€/a]", min_value=0.0, value=20.0, step=5.0)
    entgelt_imsys = st.number_input("Messentgelt iMSys je WE [€/a]", min_value=0.0, value=60.0, step=5.0)
    grundpreis_ms = st.number_input("Grundpreis Mieterstrom je WE [€/a] (optional)", min_value=0.0, value=0.0, step=5.0)

# ---------------------------
# Helper
# ---------------------------
def df_one_time(model):
    rows = []
    rows.append(["Zählerplatz‑Ertüchtigung (VDE‑AR‑N 4100)", n_we, zpl_ert, n_we*zpl_ert])
    rows.append(["Erzeugungszähler PV (Einbau)", 1, erz_zaehler, erz_zaehler])
    rows.append(["Projektierung & Messkonzept", 1, proj_mk, proj_mk])
    rows.append(["Rechts-/Reg‑Setup", 1, legal_setup, legal_setup])
    rows.append(["Untermessung/Submeter je WE", n_we, submeter_we, n_we*submeter_we])
    if imsys_required:
        rows.append(["iMSys‑Upgrade je WE", n_we, imsys_up_we, n_we*imsys_up_we])
        rows.append(["Smart‑Meter‑Gateway (zentral)", 1, smgw_central, smgw_central])
    if model == "GGV":
        rows.append(["IT/Abrechnungs‑Setup GGV", 1, it_setup_ggv, it_setup_ggv])
    else:
        rows.append(["IT/Abrechnungs‑Setup Mieterstrom", 1, it_setup_ms, it_setup_ms])
    return pd.DataFrame(rows, columns=["Kostenposition", "Menge", "Einheitspreis [€]", "Summe [€]"])

def df_owner_recurring(model):
    rows = []
    rows.append(["Messstellenbetrieb Erzeugungszähler", 1, ms_erzeug_j, ms_erzeug_j])
    if imsys_required:
        rows.append(["Gatewaybetrieb (SMGw zentral)", 1, smgw_gate_j, smgw_gate_j])
    if model == "GGV":
        rows.append(["IT/SaaS Abrechnung (GGV)", 1, it_saas_ggv_j, it_saas_ggv_j])
    else:
        rows.append(["IT/SaaS Abrechnung (Mieterstrom)", 1, it_saas_ms_j, it_saas_ms_j])
    if dv_required:
        rows.append(["Direktvermarktung fix", 1, dv_fix_j, dv_fix_j])
        dv_var_eur = (dv_var_ct/100.0)*prod_total
        rows.append(["Direktvermarktung variabel", prod_total, f"{dv_var_ct} ct/kWh", dv_var_eur])
    return pd.DataFrame(rows, columns=["Kostenposition", "Menge", "Einheitspreis", "Summe [€]"])

def df_consumer_recurring(model):
    entgelt = entgelt_imsys if imsys_required else entgelt_mme
    rows = [["Messentgelt je WE (MsbG)", n_we, entgelt, n_we*entgelt]]
    if model == "Mieterstrom" and grundpreis_ms > 0:
        rows.append(["Grundpreis Mieterstrom je WE", n_we, grundpreis_ms, n_we*grundpreis_ms])
    return pd.DataFrame(rows, columns=["Kostenposition", "Menge", "Einheitspreis [€]", "Summe [€]"])

# ---------------------------
# Build dataframes
# ---------------------------
ot_ggv = df_one_time("GGV")
ot_ms = df_one_time("Mieterstrom")
ro_ggv = df_owner_recurring("GGV")
ro_ms = df_owner_recurring("Mieterstrom")
rc_ggv = df_consumer_recurring("GGV")
rc_ms = df_consumer_recurring("Mieterstrom")

sum_ot_ggv = ot_ggv["Summe [€]"].apply(lambda x: float(str(x).split()[0])).sum()
sum_ot_ms  = ot_ms["Summe [€]"].apply(lambda x: float(str(x).split()[0])).sum()
sum_ro_ggv = pd.to_numeric(ro_ggv["Summe [€]"], errors="coerce").sum()
sum_ro_ms  = pd.to_numeric(ro_ms["Summe [€]"], errors="coerce").sum()
sum_rc_ggv = rc_ggv["Summe [€]"].sum()
sum_rc_ms  = rc_ms["Summe [€]"].sum()

# ---------------------------
# KPIs
# ---------------------------
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Einmalkosten – GGV [€]", f"{sum_ot_ggv:,.0f}")
c2.metric("Einmalkosten – Mieterstrom [€]", f"{sum_ot_ms:,.0f}")
c3.metric("Eigentümer laufend – GGV [€/a]", f"{sum_ro_ggv:,.0f}")
c4.metric("Eigentümer laufend – MS [€/a]", f"{sum_ro_ms:,.0f}")
c5.metric("Letztverbraucher laufend – GGV [€/a]", f"{sum_rc_ggv:,.0f}")
c6.metric("Letztverbraucher laufend – MS [€/a]", f"{sum_rc_ms:,.0f}")

st.markdown("---")

# ---------------------------
# Tabs with tables
# ---------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Einmalkosten", "Laufend Eigentümer", "Laufend Letztverbraucher", "Export"])

with tab1:
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("GGV – Einmalkosten")
        st.dataframe(ot_ggv, use_container_width=True)
    with col_b:
        st.subheader("Mieterstrom – Einmalkosten")
        st.dataframe(ot_ms, use_container_width=True)

with tab2:
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("GGV – laufende Kosten Eigentümer")
        st.dataframe(ro_ggv, use_container_width=True)
    with col_b:
        st.subheader("Mieterstrom – laufende Kosten Eigentümer")
        st.dataframe(ro_ms, use_container_width=True)

with tab3:
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("GGV – laufende Kosten Letztverbraucher (gesamt)")
        st.dataframe(rc_ggv, use_container_width=True)
    with col_b:
        st.subheader("Mieterstrom – laufende Kosten Letztverbraucher (gesamt)")
        st.dataframe(rc_ms, use_container_width=True)

with tab4:
    st.write("Exportiere die Tabellen als CSV:")
    st.download_button("Einmalkosten GGV (CSV)", ot_ggv.to_csv(index=False).encode("utf-8"), file_name="einmal_ggv.csv", mime="text/csv")
    st.download_button("Einmalkosten Mieterstrom (CSV)", ot_ms.to_csv(index=False).encode("utf-8"), file_name="einmal_mieterstrom.csv", mime="text/csv")
    st.download_button("Eigentümer laufend GGV (CSV)", ro_ggv.to_csv(index=False).encode("utf-8"), file_name="laufend_eigent_ggv.csv", mime="text/csv")
    st.download_button("Eigentümer laufend Mieterstrom (CSV)", ro_ms.to_csv(index=False).encode("utf-8"), file_name="laufend_eigent_ms.csv", mime="text/csv")
    st.download_button("Letztverbraucher laufend GGV (CSV)", rc_ggv.to_csv(index=False).encode("utf-8"), file_name="laufend_verbraucher_ggv.csv", mime="text/csv")
    st.download_button("Letztverbraucher laufend Mieterstrom (CSV)", rc_ms.to_csv(index=False).encode("utf-8"), file_name="laufend_verbraucher_ms.csv", mime="text/csv")

st.markdown("""
**Interpretation:**  
- §42b EnWG → viertelstündliche Messung impliziert iMSys auf Teilnehmerseite (höhere Messentgelte, SMGw).  
- Mieterstrom erfordert i. d. R. höheres IT-/Backend‑Niveau (Marktkommunikation), daher laufend teurer.  
- Zählerplatz‑Ertüchtigung nach VDE‑AR‑N 4100 fällt unabhängig vom Modell an (Gebäudeeigentümerpflicht).
""")

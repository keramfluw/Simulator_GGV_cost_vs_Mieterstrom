# app.py (v6 – Kalibrierte Defaults + PV‑CAPEX Position)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="GGV vs. Mieterstrom – Szenariorechner (v6)", layout="wide")

# -----------------------------
# Helpers
# -----------------------------
def cashflow_summary(df, discount_rate):
    npv = 0.0
    cum = 0.0
    payback_year = None
    for _, row in df.iterrows():
        year = int(row["Jahr"])
        cf = float(row["Netto Cashflow"])
        npv += cf / ((1 + discount_rate) ** year)
        cum += cf
        if payback_year is None and cum >= 0 and year>0:
            payback_year = year
    return npv, payback_year

def eur2(value, german=True):
    if value is None:
        return "n/a"
    s = f"{float(value):,.2f}"
    if german:
        s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    return s

def build_scenario(
    label,
    kWp,
    specific_yield_kwh_per_kwp,
    self_consumption_share,     # in % (nach Sensitivität & Batterie)
    grid_share_override,        # in % or None
    grid_price_ct_per_kwh,
    eeg_feed_in_ct_per_kwh,
    dm_fee_ct_per_kwh,
    internal_price_ct_per_kwh,
    mieterstrom_price_cap_ct_per_kwh,
    mieterstrom_premium_ct_per_kwh,
    capex_eur,
    opex_fixed_eur,
    lifetime_years,
    degradation_pct_per_year,
    inflation_pct,
    price_growth_pct,
    discount_rate_pct,
    is_mieterstrom,
    battery_shift_share_pp,
    storage_lcos_eur_per_kwh,
):
    annual_production_kwh = kWp * specific_yield_kwh_per_kwp
    deg = degradation_pct_per_year / 100.0
    infl = inflation_pct / 100.0
    price_growth = price_growth_pct / 100.0
    disc = discount_rate_pct / 100.0

    sc_share = np.clip(self_consumption_share/100.0, 0, 1)
    if grid_share_override is not None:
        grid_share = grid_share_override/100.0
        sc_share = 1 - grid_share
    else:
        grid_share = 1 - sc_share

    eeg_price_eur = eeg_feed_in_ct_per_kwh / 100.0
    dm_fee_eur = dm_fee_ct_per_kwh / 100.0
    internal_price_eur = internal_price_ct_per_kwh / 100.0
    mieterstrom_cap_eur = mieterstrom_price_cap_ct_per_kwh / 100.0
    mieterstrom_premium_eur = mieterstrom_premium_ct_per_kwh / 100.0

    if is_mieterstrom:
        internal_price_eur = min(internal_price_eur, mieterstrom_cap_eur)

    export_price_eur = max(eeg_price_eur - dm_fee_eur, 0.0)

    rows = []
    for year in range(0, 31):
        if year == 0:
            prod = 0.0
        else:
            prod = annual_production_kwh * ((1 - deg) ** (year-1))

        sc_kwh = prod * sc_share
        grid_kwh = prod * grid_share

        # Preise/Indizes
        internal_price_y = internal_price_eur * ((1 + price_growth) ** max(0, year-1))
        export_price_y = export_price_eur * ((1 + price_growth) ** max(0, year-1))
        premium_y = mieterstrom_premium_eur * ((1 + infl) ** max(0, year-1))

        # Erlöse
        internal_rev = sc_kwh * internal_price_y
        export_rev = grid_kwh * export_price_y
        premium_rev = (sc_kwh * premium_y) if is_mieterstrom else 0.0
        total_rev = internal_rev + export_rev + premium_rev

        # OPEX fix (inflationsindexiert)
        opex_y = opex_fixed_eur * ((1 + infl) ** max(0, year-1))

        # Speicher-Betriebskosten via LCOS auf den ΔEV-Anteil (nur wenn year>0)
        if year > 0 and battery_shift_share_pp > 0 and storage_lcos_eur_per_kwh > 0:
            shifted_kwh = prod * (battery_shift_share_pp/100.0)
            opex_y += shifted_kwh * storage_lcos_eur_per_kwh

        capex_y = capex_eur if year == 0 else 0.0

        net_cf = total_rev - opex_y - capex_y

        rows.append({
            "Szenario": label,
            "Jahr": year,
            "Produktion [kWh]": prod,
            "EV [kWh]": sc_kwh,
            "Einspeisung [kWh]": grid_kwh,
            "Erlös intern [€]": internal_rev,
            "Einspeiseerlös [€]": export_rev,
            "Mieterstromzuschlag [€]": premium_rev,
            "OPEX [€]": opex_y,
            "CAPEX [€]": capex_y,
            "Umsatz gesamt [€]": total_rev,
            "Netto Cashflow": net_cf,
        })

    df = pd.DataFrame(rows)
    npv, payback = cashflow_summary(df, disc)
    return df, npv, payback

# -----------------------------
# Sidebar Inputs (kalibrierte Defaults)
# -----------------------------
st.sidebar.title("Eingaben – Anlage & Preise")

with st.sidebar.expander("Projekt & Anlage", expanded=True):
    n_units = st.number_input("Anzahl Nutzeinheiten (NE)", min_value=1, value=1, step=1)
    kWp = st.number_input("Anlagengröße [kWp]", min_value=1.0, value=99.0, step=1.0)
    specific_yield = st.number_input("Spezifischer Ertrag [kWh/kWp·a]", min_value=400.0, value=600.0, step=10.0)

with st.sidebar.expander("Regulatorik", expanded=True):
    sec42b = st.checkbox("§42b EnWG aktiv (iMSys je NE, 15‑min)", value=True)
    dv_required = st.checkbox("Direktvermarktung aktiv (typ. >100 kWp)", value=False)

with st.sidebar.expander("Preis-/Ertragsparameter", expanded=True):
    grundversorgung_ct = st.number_input("Örtlicher Grundversorgungstarif [ct/kWh] (Deckel MS=90%)", min_value=10.0, value=40.0, step=0.1)
    ggv_price_ct_base = st.number_input("Interner Preis GGV [ct/kWh]", min_value=0.0, value=27.0, step=0.1)
    ms_price_ct_base  = st.number_input("Endkundenpreis Mieterstrom [ct/kWh] (≤90% Grundversorgung)", min_value=0.0, value=29.0, step=0.1)
    eeg_feed_ct = st.number_input("EEG-Einspeisevergütung [ct/kWh]", min_value=0.0, value=7.0, step=0.1)
    dm_fee_ct = st.number_input("Direktvermarktungsgebühr [ct/kWh] (bei DV)", min_value=0.0, value=0.4, step=0.1)
    mieterstrom_premium_ct = st.number_input("Mieterstromzuschlag [ct/kWh] (auf EV-Mengen)", min_value=0.0, value=3.0, step=0.1)

with st.sidebar.expander("EV-Anteil & Batterie", expanded=True):
    sc_share_base = st.slider("Eigenverbrauchsanteil Basis [%]", 0, 100, 35)
    use_override = st.checkbox("Einspeiseanteil-Override verwenden", value=True)
    grid_share_override = st.slider("Override Einspeiseanteil [%]", 0, 100, 65) if use_override else None
    battery_enabled = st.checkbox("Batterie/Optimierung aktiv (ΔEV)", value=False)
    delta_ev_pp = st.slider("ΔEV durch Batterie [%‑Punkte]", 0, 60, 10) if battery_enabled else 0
    storage_lcos = st.number_input("LCOS Speicher [€/kWh] (Kosten je verschobene kWh)", min_value=0.0, value=0.00, step=0.01, format="%.2f")

with st.sidebar.expander("Inflation & Diskontierung", expanded=True):
    use_global_infl = st.checkbox("Globale Inflation für Kosten & Preise nutzen", value=True)
    global_infl = st.number_input("Globale Inflation [%/a]", min_value=0.0, value=2.0, step=0.1)
    inflation = global_infl if use_global_infl else st.number_input("Inflation Kosten [%/a]", min_value=0.0, value=2.0, step=0.1)
    price_growth = global_infl if use_global_infl else st.number_input("Preiswachstum Erlöse [%/a]", min_value=0.0, value=2.0, step=0.1)
    discount = st.number_input("Diskontsatz (NPV) [%/a]", min_value=0.0, value=6.0, step=0.1)

with st.sidebar.expander("Sensitivitäten", expanded=False):
    sens_ev = st.slider("Δ EV-Sensitivität [%-Pkte]", -15, 15, 0)
    sens_pint_ct = st.slider("Δ Interner Preis [ct/kWh]", -5.0, 5.0, 0.0, step=0.1)
    sens_opex_pct = st.slider("Δ OPEX (Eigentümer) [%]", -20, 20, 0)

with st.sidebar.expander("Kosten – Detailliert (sichtbar & in Rechnung)", expanded=True):
    use_detailed_costs = st.checkbox("Detaillierte Kosten in Cashflow verwenden", value=True)

    st.markdown("**Einmalkosten (pro Modell)**")
    colA, colB = st.columns(2)
    with colA:
        st.markdown("**GGV – Einmal**")
        zpl_ne = st.number_input("Zählerplatz‑Ertüchtigung je NE [€]", min_value=0.0, value=700.0, step=50.0)
        subm_ne = st.number_input("Submeter je NE [€]", min_value=0.0, value=180.0, step=10.0)
        imsys_ne = st.number_input("iMSys‑Upgrade je NE [€]", min_value=0.0, value=350.0, step=10.0)
        smgw_cent = st.number_input("Smart‑Meter‑Gateway (zentral) [€]", min_value=0.0, value=600.0, step=10.0)
        it_setup_ggv = st.number_input("IT/Abrechnung Setup GGV [€]", min_value=0.0, value=4000.0, step=100.0)
        legal_once = st.number_input("Recht/Reg Setup [€]", min_value=0.0, value=2500.0, step=100.0)
        proj_mk = st.number_input("Projektierung Messkonzept [€]", min_value=0.0, value=3000.0, step=100.0)
        gen_meter = st.number_input("Erzeugungszähler (Einbau) [€]", min_value=0.0, value=250.0, step=10.0)
        pv_capex_ggv = st.number_input("**PV‑Anlage (Generator/WR/Montage) [€]**", min_value=0.0, value=41103.77, step=100.0, format="%.2f")
    with colB:
        st.markdown("**Mieterstrom – Einmal**")
        zpl_ne_ms = st.number_input("Zählerplatz‑Ertüchtigung je NE [€] (MS)", min_value=0.0, value=700.0, step=50.0)
        subm_ne_ms = st.number_input("Submeter je NE [€] (MS)", min_value=0.0, value=180.0, step=10.0)
        imsys_ne_ms = st.number_input("iMSys‑Upgrade je NE [€] (MS)", min_value=0.0, value=350.0, step=10.0)
        smgw_cent_ms = st.number_input("Smart‑Meter‑Gateway (zentral) [€] (MS)", min_value=0.0, value=600.0, step=10.0)
        it_setup_ms = st.number_input("IT/Abrechnung Setup Mieterstrom [€]", min_value=0.0, value=5600.0, step=100.0)
        legal_once_ms = st.number_input("Recht/Reg Setup [€] (MS)", min_value=0.0, value=2500.0, step=100.0)
        proj_mk_ms = st.number_input("Projektierung Messkonzept [€] (MS)", min_value=0.0, value=3000.0, step=100.0)
        gen_meter_ms = st.number_input("Erzeugungszähler (Einbau) [€] (MS)", min_value=0.0, value=250.0, step=10.0)
        pv_capex_ms = st.number_input("**PV‑Anlage (Generator/WR/Montage) [€] (MS)**", min_value=0.0, value=53766.61, step=100.0, format="%.2f")

    st.markdown("**Laufende Kosten Eigentümer (jährlich)**")
    colC, colD = st.columns(2)
    with colC:
        st.markdown("**GGV – laufend**")
        msb_gen = st.number_input("MSB Erzeugungszähler [€/a]", min_value=0.0, value=120.0, step=10.0)
        smgw_gate = st.number_input("Gatewaybetrieb (zentral) [€/a]", min_value=0.0, value=120.0, step=10.0)
        it_saas_ggv = st.number_input("IT/SaaS Abrechnung GGV [€/a]", min_value=0.0, value=1800.0, step=50.0)
        dv_fix = st.number_input("Direktvermarktung fix [€/a]", min_value=0.0, value=0.0, step=50.0)
        opex_other = st.number_input("Weitere OPEX (Versicherung/Wartung) [€/a]", min_value=0.0, value=816.0, step=50.0)
    with colD:
        st.markdown("**Mieterstrom – laufend**")
        msb_gen_ms = st.number_input("MSB Erzeugungszähler [€/a] (MS)", min_value=0.0, value=120.0, step=10.0)
        smgw_gate_ms = st.number_input("Gatewaybetrieb (zentral) [€/a] (MS)", min_value=0.0, value=120.0, step=10.0)
        it_saas_ms = st.number_input("IT/SaaS Abrechnung Mieterstrom [€/a]", min_value=0.0, value=1524.0, step=50.0)
        dv_fix_ms = st.number_input("Direktvermarktung fix [€/a] (MS)", min_value=0.0, value=0.0, step=50.0)
        opex_other_ms = st.number_input("Weitere OPEX (Versicherung/Wartung) [€/a] (MS)", min_value=0.0, value=0.0, step=50.0)

    st.markdown("**Laufende Kosten Letztverbraucher – Sichtbar (nicht im Eigentümer‑CF)**")
    colE, colF = st.columns(2)
    with colE:
        entgelt_ne = st.number_input("Messentgelt je NE (mME/iMSys) [€/a]", min_value=0.0, value=60.0 if sec42b else 20.0, step=5.0)
    with colF:
        grundpreis_ms_ne = st.number_input("Grundpreis Mieterstrom je NE [€/a] (optional)", min_value=0.0, value=0.0, step=5.0)

# -----------------------------
# Derive detailed CAPEX/OPEX
# -----------------------------
mieterstrom_cap = 0.9 * grundversorgung_ct

# Sensitivities applied
sc_share = np.clip(sc_share_base + (delta_ev_pp if battery_enabled else 0) + sens_ev, 0, 100)
ggv_price_ct = max(0.0, ggv_price_ct_base + sens_pint_ct)
ms_price_ct = max(0.0, ms_price_ct_base + sens_pint_ct)

# CAPEX sums
capex_ggv = (n_units*(zpl_ne + subm_ne + (imsys_ne if sec42b else 0.0)) +
             (smgw_cent if sec42b else 0.0) + it_setup_ggv + legal_once + proj_mk + gen_meter +
             pv_capex_ggv)
capex_ms  = (n_units*(zpl_ne_ms + subm_ne_ms + (imsys_ne_ms if sec42b else 0.0)) +
             (smgw_cent_ms if sec42b else 0.0) + it_setup_ms + legal_once_ms + proj_mk_ms + gen_meter_ms +
             pv_capex_ms)

# OPEX Eigentümer (mit Sensitivität)
opex_ggv = (msb_gen + (smgw_gate if sec42b else 0.0) + it_saas_ggv + (dv_fix if dv_required else 0.0) + opex_other)
opex_ms  = (msb_gen_ms + (smgw_gate_ms if sec42b else 0.0) + it_saas_ms + (dv_fix_ms if dv_required else 0.0) + opex_other_ms)
opex_ggv *= (1 + sens_opex_pct/100.0)
opex_ms  *= (1 + sens_opex_pct/100.0)

# -----------------------------
# Build scenarios
# -----------------------------
df_ggv, npv_ggv, pb_ggv = build_scenario(
    label="GGV",
    kWp=kWp,
    specific_yield_kwh_per_kwp=specific_yield,
    self_consumption_share=sc_share,
    grid_share_override=(grid_share_override if use_override else None),
    grid_price_ct_per_kwh=grundversorgung_ct,
    eeg_feed_in_ct_per_kwh=eeg_feed_ct,
    dm_fee_ct_per_kwh=(dm_fee_ct if dv_required else 0.0),
    internal_price_ct_per_kwh=ggv_price_ct,
    mieterstrom_price_cap_ct_per_kwh=mieterstrom_cap,
    mieterstrom_premium_ct_per_kwh=0.0,
    capex_eur=capex_ggv,
    opex_fixed_eur=opex_ggv,
    lifetime_years=30,
    degradation_pct_per_year=0.5,
    inflation_pct=global_infl if use_global_infl else inflation,
    price_growth_pct=global_infl if use_global_infl else price_growth,
    discount_rate_pct=discount,
    is_mieterstrom=False,
    battery_shift_share_pp=(delta_ev_pp if battery_enabled else 0),
    storage_lcos_eur_per_kwh=storage_lcos
)

df_ms, npv_ms, pb_ms = build_scenario(
    label="Mieterstrom",
    kWp=kWp,
    specific_yield_kwh_per_kwp=specific_yield,
    self_consumption_share=sc_share,
    grid_share_override=(grid_share_override if use_override else None),
    grid_price_ct_per_kwh=grundversorgung_ct,
    eeg_feed_in_ct_per_kwh=eeg_feed_ct,
    dm_fee_ct_per_kwh=(dm_fee_ct if dv_required else 0.0),
    internal_price_ct_per_kwh=ms_price_ct,
    mieterstrom_price_cap_ct_per_kwh=mieterstrom_cap,
    mieterstrom_premium_ct_per_kwh=mieterstrom_premium_ct,
    capex_eur=capex_ms,
    opex_fixed_eur=opex_ms,
    lifetime_years=30,
    degradation_pct_per_year=0.5,
    inflation_pct=global_infl if use_global_infl else inflation,
    price_growth_pct=global_infl if use_global_infl else price_growth,
    discount_rate_pct=discount,
    is_mieterstrom=True,
    battery_shift_share_pp=(delta_ev_pp if battery_enabled else 0),
    storage_lcos_eur_per_kwh=storage_lcos
)

df_all = pd.concat([df_ggv, df_ms], ignore_index=True)

# -----------------------------
# KPIs – NE/LG
# -----------------------------
ne_npv_ggv = npv_ggv / n_units
ne_npv_ms = npv_ms / n_units
pb1 = "n/a" if pb_ggv is None else f"{pb_ggv} a"
pb2 = "n/a" if pb_ms is None else f"{pb_ms} a"

st.subheader("Übersicht – KPIs (NE & LG)")
# NE row
st.markdown("**NE – Nutzeinheit (pro Einheit)**")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"<div style='border:1px solid #e5e7eb; padding:10px; border-radius:12px; text-align:center;'>"
                f"<div style='font-weight:600;'>Mieterstrom‑Preisdeckel [ct/kWh]</div>"
                f"<div style='font-size:22px;'>{(0.9*grundversorgung_ct):.1f}</div></div>", unsafe_allow_html=True)
with c2:
    st.markdown(f"<div style='border:1px solid #e5e7eb; padding:10px; border-radius:12px; text-align:center;'>"
                f"<div style='font-weight:600;'>NPV GGV [€]</div>"
                f"<div style='font-size:22px;'>{eur2(ne_npv_ggv)}</div></div>", unsafe_allow_html=True)
with c3:
    st.markdown(f"<div style='border:1px solid #e5e7eb; padding:10px; border-radius:12px; text-align:center;'>"
                f"<div style='font-weight:600;'>NPV Mieterstrom [€]</div>"
                f"<div style='font-size:22px;'>{eur2(ne_npv_ms)}</div></div>", unsafe_allow_html=True)
with c4:
    st.markdown(f"<div style='border:1px solid #e5e7eb; padding:10px; border-radius:12px; text-align:center;'>"
                f"<div style='font-weight:600;'>Payback: GGV / MS</div>"
                f"<div style='font-size:22px;'>{pb1} / {pb2}</div></div>", unsafe_allow_html=True)

# LG row
st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)
st.markdown("**LG – Liegenschaft (kumuliert)**")
d1, d2, d3, d4 = st.columns(4)
with d1:
    st.markdown(f"<div style='border:1px solid #c7d2fe; padding:10px; border-radius:12px; text-align:center; background:#eef2ff;'>"
                f"<div style='font-weight:700; color:#1f4acc;'>Mieterstrom‑Preisdeckel [ct/kWh]</div>"
                f"<div style='font-size:22px; font-weight:700; color:#1f4acc;'>{(0.9*grundversorgung_ct):.1f}</div></div>", unsafe_allow_html=True)
with d2:
    st.markdown(f"<div style='border:1px solid #c7d2fe; padding:10px; border-radius:12px; text-align:center; background:#eef2ff;'>"
                f"<div style='font-weight:700; color:#1f4acc;'>NPV GGV [€]</div>"
                f"<div style='font-size:22px; font-weight:700; color:#1f4acc;'>{eur2(npv_ggv)}</div></div>", unsafe_allow_html=True)
with d3:
    st.markdown(f"<div style='border:1px solid #c7d2fe; padding:10px; border-radius:12px; text-align:center; background:#eef2ff;'>"
                f"<div style='font-weight:700; color:#1f4acc;'>NPV Mieterstrom [€]</div>"
                f"<div style='font-size:22px; font-weight:700; color:#1f4acc;'>{eur2(npv_ms)}</div></div>", unsafe_allow_html=True)
with d4:
    st.markdown(f"<div style='border:1px solid #c7d2fe; padding:10px; border-radius:12px; text-align:center; background:#eef2ff;'>"
                f"<div style='font-weight:700; color:#1f4acc;'>Payback: GGV / MS</div>"
                f"<div style='font-size:22px; font-weight:700; color:#1f4acc;'>{pb1} / {pb2}</div></div>", unsafe_allow_html=True)

# -----------------------------
# Rendite Kurve
# -----------------------------
st.markdown("### Wirtschaftlichkeit / Rendite Eigentümer über Jahre")
years = st.slider("Analysehorizont [Jahre]", min_value=2, max_value=30, value=10, step=1)
def roi_over_horizon(df, years):
    capex = float(df.loc[df["Jahr"]==0, "CAPEX [€]"].sum())
    cum_net = float(df[df["Jahr"]<=years]["Netto Cashflow"].sum())
    if capex <= 0:
        return None
    return cum_net / capex
roi_ggv = roi_over_horizon(df_ggv, years)
roi_ms = roi_over_horizon(df_ms, years)
r1, r2 = st.columns(2)
r1.metric(f"ROI GGV bis Jahr {years}", f"{(roi_ggv*100):.1f}%")
r2.metric(f"ROI Mieterstrom bis Jahr {years}", f"{(roi_ms*100):.1f}%")
years_range = list(range(2, 31))
roi_curve = pd.DataFrame({
    "Jahr": years_range,
    "ROI GGV [%]": [roi_over_horizon(df_ggv, y)*100 for y in years_range],
    "ROI Mieterstrom [%]": [roi_over_horizon(df_ms, y)*100 for y in years_range],
})
fig_roi = px.line(roi_curve.melt(id_vars="Jahr", var_name="Szenario", value_name="ROI [%]"),
                  x="Jahr", y="ROI [%]", color="Szenario", title="ROI-Verlauf (Eigentümer)")
st.plotly_chart(fig_roi, use_container_width=True)

st.caption("Kalibrierte Defaults: NE‑NPV GGV ≈ 29.935 €, NE‑NPV Mieterstrom ≈ 52.249 €, Payback ≈ 10 a / 9 a. Anpassbar über Eingaben & Sensitivitäten.")

# -----------------------------
# Tabs – Cashflows, Energie, Jahreswerte, Kosten
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Cashflows", "Energieflüsse", "Jahreswerte", "Kosten (Detail)"])

with tab1:
    df_plot = df_all[df_all["Jahr"]>0].copy()
    fig_cf = px.line(df_plot, x="Jahr", y="Netto Cashflow", color="Szenario", title="Jährlicher Netto-Cashflow")
    st.plotly_chart(fig_cf, use_container_width=True)

    df_cum = df_plot.copy()
    df_cum["Kumuliert [€]"] = df_cum.groupby("Szenario")["Netto Cashflow"].cumsum()
    fig_cum = px.line(df_cum, x="Jahr", y="Kumuliert [€]", color="Szenario", title="Kumulierter Cashflow")
    st.plotly_chart(fig_cum, use_container_width=True)

with tab2:
    df_energy = df_all[df_all["Jahr"]>0].copy()
    df_energy = df_energy.melt(id_vars=["Szenario","Jahr"], value_vars=["EV [kWh]","Einspeisung [kWh]"], var_name="Art", value_name="kWh")
    fig_e = px.area(df_energy, x="Jahr", y="kWh", color="Art", facet_col="Szenario", facet_col_wrap=2, title="Energieflüsse EV vs. Einspeisung")
    st.plotly_chart(fig_e, use_container_width=True)

with tab3:
    st.dataframe(df_all.style.format({
        "Produktion [kWh]":"{:,.0f}",
        "EV [kWh]":"{:,.0f}",
        "Einspeisung [kWh]":"{:,.0f}",
        "Erlös intern [€]":"{:,.0f}",
        "Einspeiseerlös [€]":"{:,.0f}",
        "Mieterstromzuschlag [€]":"{:,.0f}",
        "OPEX [€]":"{:,.0f}",
        "CAPEX [€]":"{:,.0f}",
        "Umsatz gesamt [€]":"{:,.0f}",
        "Netto Cashflow":"{:,.0f}",
    }), use_container_width=True)

with tab4:
    st.markdown("#### Einmalkosten – Übersicht (inkl. PV‑Anlage)")
    df_once = pd.DataFrame({
        "Kostenposition": ["ZPL je NE","Submeter je NE","iMSys‑Upgrade je NE","SMGw zentral","IT‑Setup","Recht/Reg","Projektierung MK","Erzeugungszähler","PV‑Anlage"],
        "GGV [€]": [zpl_ne, subm_ne, (imsys_ne if sec42b else 0.0), (smgw_cent if sec42b else 0.0), it_setup_ggv, legal_once, proj_mk, gen_meter, pv_capex_ggv],
        "Mieterstrom [€]": [zpl_ne_ms, subm_ne_ms, (imsys_ne_ms if sec42b else 0.0), (smgw_cent_ms if sec42b else 0.0), it_setup_ms, legal_once_ms, proj_mk_ms, gen_meter_ms, pv_capex_ms],
    })
    df_once_tot = pd.DataFrame({
        "Modell":["GGV","Mieterstrom"],
        "Summe Einmal [€]":[n_units*(zpl_ne+subm_ne+(imsys_ne if sec42b else 0.0)) + (smgw_cent if sec42b else 0.0)+it_setup_ggv+legal_once+proj_mk+gen_meter+pv_capex_ggv,
                             n_units*(zpl_ne_ms+subm_ne_ms+(imsys_ne_ms if sec42b else 0.0)) + (smgw_cent_ms if sec42b else 0.0)+it_setup_ms+legal_once_ms+proj_mk_ms+gen_meter_ms+pv_capex_ms]
    })
    st.dataframe(df_once, use_container_width=True)
    st.dataframe(df_once_tot, use_container_width=True)

    st.markdown("#### Laufende Kosten Eigentümer – Übersicht (jährlich)")
    df_run = pd.DataFrame({
        "Kostenposition":["MSB Erzeugungszähler","Gatewaybetrieb","IT/SaaS Abrechnung","Direktvermarktung fix","Weitere OPEX"],
        "GGV [€/a]":[msb_gen, (smgw_gate if sec42b else 0.0), it_saas_ggv, (dv_fix if dv_required else 0.0), opex_other],
        "Mieterstrom [€/a]":[msb_gen_ms, (smgw_gate_ms if sec42b else 0.0), it_saas_ms, (dv_fix_ms if dv_required else 0.0), opex_other_ms],
    })
    df_run_tot = pd.DataFrame({
        "Modell":["GGV","Mieterstrom"],
        "Summe laufend [€/a]":[msb_gen + (smgw_gate if sec42b else 0.0) + it_saas_ggv + (dv_fix if dv_required else 0.0) + opex_other,
                               msb_gen_ms + (smgw_gate_ms if sec42b else 0.0) + it_saas_ms + (dv_fix_ms if dv_required else 0.0) + opex_other_ms]
    })
    st.dataframe(df_run, use_container_width=True)
    st.dataframe(df_run_tot, use_container_width=True)

    st.markdown("#### Laufende Kosten Letztverbraucher – Sichtbar (nicht im Eigentümer‑CF)")
    df_cons = pd.DataFrame({
        "Kostenposition":["Messentgelt je NE","Grundpreis Mieterstrom je NE"],
        "Betrag [€/a]":[entgelt_ne, grundpreis_ms_ne],
        "Summe Liegenschaft [€/a]":[entgelt_ne*n_units, grundpreis_ms_ne*n_units]
    })
    st.dataframe(df_cons, use_container_width=True)

# -----------------------------
# Export
# -----------------------------
st.download_button(
    "📤 Export: Jahreswerte (CSV)",
    data=df_all.to_csv(index=False).encode("utf-8"),
    file_name="szenario_jahreswerte.csv",
    mime="text/csv"
)

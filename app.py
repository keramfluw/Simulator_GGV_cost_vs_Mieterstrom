# app.py (v11 – Kosten-Kacheln oben, erweiterter PDF-Report)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io, os
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
from reportlab.lib.units import mm
import matplotlib.pyplot as plt

st.set_page_config(page_title="Qrauts – GGV vs. Mieterstrom (v11)", layout="wide")

# -----------------------------
# Branding
# -----------------------------
Q_ORANGE = "#f58220"
Q_DARK = "#252525"

st.markdown(f"""
<style>
.q-card {{
  border: 1px solid #e5e7eb; padding: 10px; border-radius: 12px; text-align:center;
}}
.q-card-accent {{
  border: 1px solid {Q_ORANGE}; padding: 10px; border-radius: 12px; text-align:center; background: #fff8f1;
}}
.q-banner {{
  background: linear-gradient(90deg, {Q_ORANGE} 0%, #ffb36f 100%);
  color: white; padding: 10px 16px; border-radius: 12px; margin-bottom: 8px;
}}
h1, h2, h3 {{ color: {Q_DARK}; }}
</style>
""", unsafe_allow_html=True)
st.markdown(f"<div class='q-banner'><b>Qrauts AG – Szenariorechner</b> · GGV vs. Mieterstrom</div>", unsafe_allow_html=True)

# -----------------------------
# Helpers
# -----------------------------
def irr_from_series(cf):
    lo, hi = -0.99, 2.0
    def npv_r(r): return np.sum(cf/((1+r)**np.arange(len(cf))))
    f_lo, f_hi = npv_r(lo), npv_r(hi)
    if f_lo*f_hi>0: return None
    for _ in range(80):
        mid=(lo+hi)/2; f_mid=npv_r(mid)
        if f_lo*f_mid<=0: hi, f_hi = mid, f_mid
        else: lo, f_lo = mid, f_mid
    return (lo+hi)/2

def irr_from_df(df):
    return irr_from_series(df.sort_values("Jahr")["Netto Cashflow"].values.astype(float))

def eur2(v):
    try:
        s=f"{float(v):,.2f}"; return s.replace(',', 'X').replace('.', ',').replace('X', '.')
    except: return "n/a"

def build_scenario(label,kWp,specific_yield_kwh_per_kwp,self_consumption_share,grid_share_override,
    grundversorgung_ct_per_kwh,eeg_feed_in_ct_per_kwh,dm_fee_ct_per_kwh,internal_price_ct_per_kwh,
    mieterstrom_price_cap_factor,mieterstrom_premium_ct_per_kwh,capex_eur,opex_fixed_eur,
    lifetime_years,degradation_pct_per_year,inflation_pct,price_growth_pct,discount_rate_pct,
    is_mieterstrom,battery_shift_share_pp,storage_lcos_eur_per_kwh,index_eeg_price=False,index_ms_premium=False,
    ms_full_supply=False,total_ms_consumption_kwh=None,ne_count=1,cons_per_ne_kwh=2000,procurement_ct_per_kwh=28.0):
    annual_production_kwh = kWp*specific_yield_kwh_per_kwp
    deg=degradation_pct_per_year/100.0; infl=inflation_pct/100.0; price_growth=price_growth_pct/100.0
    sc_share = np.clip(self_consumption_share/100.0,0,1)
    if grid_share_override is not None:
        grid_share=grid_share_override/100.0; sc_share=1-grid_share
    else:
        grid_share=1-sc_share
    eeg_price=eeg_feed_in_ct_per_kwh/100.0; dm_fee=dm_fee_ct_per_kwh/100.0
    internal_price=internal_price_ct_per_kwh/100.0; grund=grundversorgung_ct_per_kwh/100.0
    cap_factor=mieterstrom_price_cap_factor; premium=mieterstrom_premium_ct_per_kwh/100.0
    procurement=procurement_ct_per_kwh/100.0; export_price_base=max(eeg_price-dm_fee,0.0)
    rows=[]
    for year in range(0,lifetime_years+1):
        prod=0.0 if year==0 else annual_production_kwh*((1-deg)**(year-1))
        sc_kwh=prod*sc_share; grid_kwh=prod*(1-sc_share)
        gs_y=grund*((1+price_growth)**max(0,year-1)); cap_y=cap_factor*gs_y
        base_int=internal_price*((1+price_growth)**max(0,year-1))
        internal_y = min(base_int, cap_y) if is_mieterstrom else base_int
        export_y = export_price_base*((1+price_growth)**max(0,year-1)) if index_eeg_price else export_price_base
        prem_y = premium*((1+infl)**max(0,year-1)) if index_ms_premium else premium
        internal_rev = sc_kwh*internal_y; export_rev=grid_kwh*export_y; premium_rev=(sc_kwh*prem_y) if is_mieterstrom else 0.0
        nonpv_rev=0.0; nonpv_cost=0.0
        if is_mieterstrom and ms_full_supply and year>0:
            total_cons = total_ms_consumption_kwh if total_ms_consumption_kwh is not None else ne_count*cons_per_ne_kwh
            comp_kwh=max(total_cons-sc_kwh,0.0); procurement_y=procurement*((1+price_growth)**max(0,year-1))
            nonpv_rev=comp_kwh*internal_y; nonpv_cost=comp_kwh*procurement_y
        total_rev=internal_rev+export_rev+premium_rev+nonpv_rev
        opex_y=0.0 if year==0 else opex_fixed_eur*((1+infl)**max(0,year-1))
        if year>0 and battery_shift_share_pp>0 and storage_lcos_eur_per_kwh>0:
            shifted_kwh=prod*(battery_shift_share_pp/100.0); opex_y += shifted_kwh*storage_lcos_eur_per_kwh
        capex_y=capex_eur if year==0 else 0.0
        net_cf = total_rev - opex_y - nonpv_cost - capex_y
        rows.append({"Szenario":label,"Jahr":year,"Produktion [kWh]":prod,"EV [kWh]":sc_kwh,"Einspeisung [kWh]":grid_kwh,
                     "Erlös intern [€]":internal_rev+nonpv_rev,"Einspeiseerlös [€]":export_rev,"Mieterstromzuschlag [€]":premium_rev,
                     "Beschaffung nicht‑PV [€]":nonpv_cost,"OPEX [€]":opex_y,"CAPEX [€]":capex_y,"Umsatz gesamt [€]":total_rev,"Netto Cashflow":net_cf})
    df=pd.DataFrame(rows)
    irr=irr_from_df(df)
    # NPV & Payback for display
    disc = discount_rate_pct/100.0
    npv=0.0; cum=0.0; pb=None
    for _,r in df.iterrows():
        y=int(r["Jahr"]); cf=float(r["Netto Cashflow"])
        npv += cf/((1+disc)**y); cum+=cf
        if pb is None and cum>=0 and y>0: pb=y
    return df, irr, npv, pb

# ---------------- Sidebar: Kundendaten ----------------
st.sidebar.title("Kundendaten & Projekt")
with st.sidebar.expander("Stammdaten", expanded=True):
    customer_name = st.text_input("Kundenname", value="")
    customer_email = st.text_input("E‑Mail", value="")
    customer_phone = st.text_input("Telefon", value="")
    customer_address = st.text_area("Kundenadresse", value="", height=70)
    property_address = st.text_area("Adresse der Liegenschaft", value="", height=70)
    project_number = st.text_input("Projektnummer", value="")

# --------------- Sidebar: Parameter -------------------
st.sidebar.title("Anlage & Parameter")
with st.sidebar.expander("Projekt & Anlage", expanded=True):
    n_units = st.number_input("Anzahl Nutzeinheiten (NE)", min_value=1, value=30, step=1)
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
    storage_lcos = st.number_input("LCOS Speicher [€/kWh]", min_value=0.0, value=0.00, step=0.01, format="%.2f")

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

with st.sidebar.expander("Kosten – Detailliert", expanded=True):
    colA, colB = st.columns(2)
    with colA:
        st.markdown("**GGV – Einmal**")
        zpl_ne = st.number_input("ZPL je NE [€]", min_value=0.0, value=700.0, step=50.0)
        subm_ne = st.number_input("Submeter je NE [€]", min_value=0.0, value=180.0, step=10.0)
        imsys_ne = st.number_input("iMSys‑Upgrade je NE [€]", min_value=0.0, value=350.0, step=10.0)
        smgw_cent = st.number_input("SMGw zentral [€]", min_value=0.0, value=600.0, step=10.0)
        it_setup_ggv = st.number_input("IT/Abrechnung Setup GGV [€]", min_value=0.0, value=4000.0, step=100.0)
        legal_once = st.number_input("Recht/Reg Setup [€]", min_value=0.0, value=2500.0, step=100.0)
        proj_mk = st.number_input("Projektierung Messkonzept [€]", min_value=0.0, value=3000.0, step=100.0)
        gen_meter = st.number_input("Erzeugungszähler (Einbau) [€]", min_value=0.0, value=250.0, step=10.0)
        pv_capex_ggv = st.number_input("PV‑Anlage [€]", min_value=0.0, value=41103.77, step=100.0, format="%.2f")
    with colB:
        st.markdown("**Mieterstrom – Einmal**")
        zpl_ne_ms = st.number_input("ZPL je NE [€] (MS)", min_value=0.0, value=700.0, step=50.0)
        subm_ne_ms = st.number_input("Submeter je NE [€] (MS)", min_value=0.0, value=180.0, step=10.0)
        imsys_ne_ms = st.number_input("iMSys‑Upgrade je NE [€] (MS)", min_value=0.0, value=350.0, step=10.0)
        smgw_cent_ms = st.number_input("SMGw zentral [€] (MS)", min_value=0.0, value=600.0, step=10.0)
        it_setup_ms = st.number_input("IT/Abrechnung Setup Mieterstrom [€]", min_value=0.0, value=5600.0, step=100.0)
        legal_once_ms = st.number_input("Recht/Reg Setup [€] (MS)", min_value=0.0, value=2500.0, step=100.0)
        proj_mk_ms = st.number_input("Projektierung Messkonzept [€] (MS)", min_value=0.0, value=3000.0, step=100.0)
        gen_meter_ms = st.number_input("Erzeugungszähler (Einbau) [€] (MS)", min_value=0.0, value=250.0, step=10.0)
        pv_capex_ms = st.number_input("PV‑Anlage [€] (MS)", min_value=0.0, value=53766.61, step=100.0, format="%.2f")

    colC, colD = st.columns(2)
    with colC:
        st.markdown("**GGV – laufend**")
        msb_gen = st.number_input("MSB Erzeugungszähler [€/a]", min_value=0.0, value=120.0, step=10.0)
        smgw_gate = st.number_input("Gatewaybetrieb (zentral) [€/a]", min_value=0.0, value=120.0, step=10.0)
        it_saas_ggv = st.number_input("IT/SaaS Abrechnung GGV [€/a]", min_value=0.0, value=1800.0, step=50.0)
        dv_fix = st.number_input("Direktvermarktung fix [€/a]", min_value=0.0, value=0.0, step=50.0)
        opex_other = st.number_input("Weitere OPEX [€/a]", min_value=0.0, value=816.0, step=50.0)
    with colD:
        st.markdown("**Mieterstrom – laufend**")
        msb_gen_ms = st.number_input("MSB Erzeugungszähler [€/a] (MS)", min_value=0.0, value=120.0, step=10.0)
        smgw_gate_ms = st.number_input("Gatewaybetrieb (zentral) [€/a] (MS)", min_value=0.0, value=120.0, step=10.0)
        it_saas_ms = st.number_input("IT/SaaS Abrechnung Mieterstrom [€/a]", min_value=0.0, value=1524.0, step=50.0)
        dv_fix_ms = st.number_input("Direktvermarktung fix [€/a] (MS)", min_value=0.0, value=0.0, step=50.0)
        opex_other_ms = st.number_input("Weitere OPEX [€/a] (MS)", min_value=0.0, value=0.0, step=50.0)

with st.sidebar.expander("Mieterstrom – Vollversorgung & Teilnehmer", expanded=False):
    ms_full_supply = st.checkbox("Vollversorgung modellieren (Restbezug einkaufen)", value=False)
    use_detailed_participants = st.checkbox("Teilnehmer einzeln erfassen (Verbrauch je Person)", value=False)
    if not use_detailed_participants:
        cons_per_ne = st.number_input("Jahresverbrauch je NE [kWh] (global)", min_value=0, value=2000, step=100)
        total_ms_consumption_kwh = None
    else:
        n_part = st.number_input("Anzahl Mieterstrom‑Teilnehmer", min_value=1, value=5, step=1, key="n_part")
        for i in range(int(n_part)):
            col1, col2 = st.columns([2,1])
            with col1:
                st.text_input(f"Teilnehmer {i+1} – Name", value=f"Teilnehmer {i+1}", key=f"pname_{i}")
            with col2:
                st.number_input(f"Verbrauch {i+1} [kWh/a]", min_value=0, value=2000, step=100, key=f"pcons_{i}")
        total_ms_consumption_kwh = sum(st.session_state.get(f"pcons_{i}",0) for i in range(int(n_part)))
    procurement_ct = st.number_input("Beschaffungspreis Restbezug [ct/kWh]", min_value=0.0, value=28.0, step=0.1)
    index_eeg = st.checkbox("EEG-Einspeisevergütung indexieren", value=False)
    index_premium = st.checkbox("Mieterstromzuschlag indexieren", value=False)

# --------- Aggregation CAPEX/OPEX ----------
mieterstrom_cap_factor=0.9
sc_share = np.clip(sc_share_base + (delta_ev_pp if battery_enabled else 0) + sens_ev, 0, 100)
ggv_price_ct = max(0.0, ggv_price_ct_base + sens_pint_ct)
ms_price_ct = max(0.0, ms_price_ct_base + sens_pint_ct)
capex_ggv = (n_units*(zpl_ne + subm_ne + (imsys_ne if sec42b else 0.0)) + (smgw_cent if sec42b else 0.0) + it_setup_ggv + legal_once + proj_mk + gen_meter + pv_capex_ggv)
capex_ms  = (n_units*(zpl_ne_ms + subm_ne_ms + (imsys_ne_ms if sec42b else 0.0)) + (smgw_cent_ms if sec42b else 0.0) + it_setup_ms + legal_once_ms + proj_mk_ms + gen_meter_ms + pv_capex_ms)
opex_ggv = (msb_gen + (smgw_gate if sec42b else 0.0) + it_saas_ggv + (dv_fix if dv_required else 0.0) + opex_other) * (1 + sens_opex_pct/100.0)
opex_ms  = (msb_gen_ms + (smgw_gate_ms if sec42b else 0.0) + it_saas_ms + (dv_fix_ms if dv_required else 0.0) + opex_other_ms) * (1 + sens_opex_pct/100.0)

# --------- Build scenarios ----------
grid_override = (grid_share_override if use_override else None)
df_ggv, irr_ggv, npv_ggv, pb_ggv = build_scenario("GGV", kWp, specific_yield, sc_share, grid_override, grundversorgung_ct, eeg_feed_ct,
    (dm_fee_ct if dv_required else 0.0), ggv_price_ct, mieterstrom_cap_factor, 0.0, capex_ggv, opex_ggv, 30, 0.5,
    (global_infl if use_global_infl else inflation), (global_infl if use_global_infl else price_growth), discount, False,
    (delta_ev_pp if battery_enabled else 0), storage_lcos, False, False, False)

df_ms, irr_ms, npv_ms, pb_ms = build_scenario("Mieterstrom", kWp, specific_yield, sc_share, grid_override, grundversorgung_ct, eeg_feed_ct,
    (dm_fee_ct if dv_required else 0.0), ms_price_ct, mieterstrom_cap_factor, mieterstrom_premium_ct, capex_ms, opex_ms, 30, 0.5,
    (global_infl if use_global_infl else inflation), (global_infl if use_global_infl else price_growth), discount, True,
    (delta_ev_pp if battery_enabled else 0), storage_lcos, index_eeg, index_premium, ms_full_supply,
    (total_ms_consumption_kwh if use_detailed_participants else None), n_units, (0 if use_detailed_participants else cons_per_ne if 'cons_per_ne' in locals() else 2000), procurement_ct)

df_all = pd.concat([df_ggv, df_ms], ignore_index=True)

# --------- NEW: Kosten-Kacheln oben ----------
st.subheader("Kostenübersicht – GGV vs. Mieterstrom")
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"<div class='q-card'><div style='font-weight:700'>Einmalkosten GGV (LG)</div><div style='font-size:22px'>{eur2(capex_ggv)}</div></div>", unsafe_allow_html=True)
with k2:
    st.markdown(f"<div class='q-card'><div style='font-weight:700'>Einmalkosten MS (LG)</div><div style='font-size:22px'>{eur2(capex_ms)}</div></div>", unsafe_allow_html=True)
with k3:
    st.markdown(f"<div class='q-card'><div style='font-weight:700'>Laufende Kosten/a GGV (LG)</div><div style='font-size:22px'>{eur2(opex_ggv)}</div></div>", unsafe_allow_html=True)
with k4:
    st.markdown(f"<div class='q-card'><div style='font-weight:700'>Laufende Kosten/a MS (LG)</div><div style='font-size:22px'>{eur2(opex_ms)}</div></div>", unsafe_allow_html=True)

# --------- KPIs ----------
st.subheader("Übersicht – KPIs (NE & LG)")
cap_now = 0.9*grundversorgung_ct
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"<div class='q-card' title='Mieterstrom‑Cap: max. 90% des örtlichen Grundversorgungstarifs.'>"
                f"<div style='font-weight:600;'>MS‑Preisdeckel [ct/kWh]</div>"
                f"<div style='font-size:22px;'>{cap_now:.1f}</div></div>", unsafe_allow_html=True)
with c2:
    st.markdown(f"<div class='q-card'><div style='font-weight:600;'>NPV GGV [€] (NE)</div><div style='font-size:22px;'>{eur2(npv_ggv/max(n_units,1))}</div></div>", unsafe_allow_html=True)
with c3:
    st.markdown(f"<div class='q-card'><div style='font-weight:600;'>NPV MS [€] (NE)</div><div style='font-size:22px;'>{eur2(npv_ms/max(n_units,1))}</div></div>", unsafe_allow_html=True)
with c4:
    st.markdown(f"<div class='q-card'><div style='font-weight:600;'>Payback: GGV / MS</div><div style='font-size:22px;'>{('n/a' if pb_ggv is None else str(pb_ggv)+' a')} / {('n/a' if pb_ms is None else str(pb_ms)+' a')}</div></div>", unsafe_allow_html=True)

d1, d2, d3, d4 = st.columns(4)
with d1:
    st.markdown(f"<div class='q-card-accent'><div style='font-weight:700;color:{Q_DARK};'>MS‑Preisdeckel [ct/kWh]</div><div style='font-size:22px;font-weight:700;color:{Q_DARK};'>{cap_now:.1f}</div></div>", unsafe_allow_html=True)
with d2:
    st.markdown(f"<div class='q-card-accent'><div style='font-weight:700;color:{Q_DARK};'>NPV GGV [€] (LG)</div><div style='font-size:22px;font-weight:700;color:{Q_DARK};'>{eur2(npv_ggv)}</div></div>", unsafe_allow_html=True)
with d3:
    st.markdown(f"<div class='q-card-accent'><div style='font-weight:700;color:{Q_DARK};'>NPV MS [€] (LG)</div><div style='font-size:22px;font-weight:700;color:{Q_DARK};'>{eur2(npv_ms)}</div></div>", unsafe_allow_html=True)
with d4:
    st.markdown(f"<div class='q-card-accent'><div style='font-weight:700;color:{Q_DARK};'>Payback: GGV / MS</div><div style='font-size:22px;font-weight:700;color:{Q_DARK};'>{('n/a' if pb_ggv is None else str(pb_ggv)+' a')} / {('n/a' if pb_ms is None else str(pb_ms)+' a')}</div></div>", unsafe_allow_html=True)

# --------- Wirtschaftlichkeit ---------
st.markdown("### Wirtschaftlichkeit / Rendite (Projekt)")
years = st.slider("Analysehorizont [Jahre]", min_value=2, max_value=30, value=20, step=1)
def roi_over_horizon(df, years):
    capex=float(df.loc[df["Jahr"]==0,"CAPEX [€]"].sum())
    cum_net=float(df[df["Jahr"]<=years]["Netto Cashflow"].sum())
    return (cum_net/capex) if capex>0 else None
roi_ggv = roi_over_horizon(df_ggv, years)
roi_ms = roi_over_horizon(df_ms, years)
colr1, colr2, colr3, colr4 = st.columns(4)
colr1.metric(f"Projekt‑ROI GGV bis Jahr {years}", f"{(roi_ggv*100):.1f}%")
colr2.metric(f"Projekt‑ROI MS bis Jahr {years}", f"{(roi_ms*100):.1f}%")
colr3.metric("Projekt‑IRR GGV", f"{(irr_ggv*100):.2f}%" if irr_ggv is not None else "n/a")
colr4.metric("Projekt‑IRR MS", f"{(irr_ms*100):.2f}%" if irr_ms is not None else "n/a")

# --------- Tabs ----------
tab1, tab2, tab3, tab4 = st.tabs(["Cashflows", "Energieflüsse", "Jahreswerte", "Kosten (Detail)"])
with tab1:
    df_plot=df_all[df_all["Jahr"]>0].copy()
    fig_cf=px.line(df_plot, x="Jahr", y="Netto Cashflow", color="Szenario", title="Jährlicher Netto-Cashflow")
    fig_cf.update_traces(hovertemplate="Jahr=%{x}<br>Netto‑Cashflow=%{y:,.2f} €<extra></extra>")
    st.plotly_chart(fig_cf, use_container_width=True)
    df_cum=df_plot.copy(); df_cum["Kumuliert [€]"]=df_cum.groupby("Szenario")["Netto Cashflow"].cumsum()
    fig_cum=px.line(df_cum, x="Jahr", y="Kumuliert [€]", color="Szenario", title="Kumulierter Cashflow")
    fig_cum.update_traces(hovertemplate="Jahr=%{x}<br>Kumuliert=%{y:,.2f} €<extra></extra>")
    st.plotly_chart(fig_cum, use_container_width=True)
with tab2:
    df_energy=df_all[df_all["Jahr"]>0].copy()
    df_energy=df_energy.melt(id_vars=["Szenario","Jahr"], value_vars=["EV [kWh]","Einspeisung [kWh]"], var_name="Art", value_name="kWh")
    fig_e=px.area(df_energy, x="Jahr", y="kWh", color="Art", facet_col="Szenario", facet_col_wrap=2, title="Energieflüsse EV vs. Einspeisung")
    fig_e.update_traces(hovertemplate="Jahr=%{x}<br>%{fullData.name}=%{y:,.2f} kWh<extra></extra>")
    st.plotly_chart(fig_e, use_container_width=True)
with tab3:
    st.dataframe(df_all.style.format({
        "Produktion [kWh]":"{:,.0f}","EV [kWh]":"{:,.0f}","Einspeisung [kWh]":"{:,.0f}",
        "Erlös intern [€]":"{:,.2f}","Einspeiseerlös [€]":"{:,.2f}","Mieterstromzuschlag [€]":"{:,.2f}",
        "Beschaffung nicht‑PV [€]":"{:,.2f}","OPEX [€]":"{:,.2f}","CAPEX [€]":"{:,.2f}","Umsatz gesamt [€]":"{:,.2f}","Netto Cashflow":"{:,.2f}"
    }), use_container_width=True)
with tab4:
    st.write("**Einmalkosten**")
    df_once=pd.DataFrame({
        "Kostenposition":["ZPL je NE","Submeter je NE","iMSys‑Upgrade je NE","SMGw zentral","IT‑Setup","Recht/Reg","Projektierung MK","Erzeugungszähler","PV‑Anlage"],
        "GGV [€]":[zpl_ne,subm_ne,(imsys_ne if sec42b else 0.0),(smgw_cent if sec42b else 0.0),it_setup_ggv,legal_once,proj_mk,gen_meter,pv_capex_ggv],
        "Mieterstrom [€]":[zpl_ne_ms,subm_ne_ms,(imsys_ne_ms if sec42b else 0.0),(smgw_cent_ms if sec42b else 0.0),it_setup_ms,legal_once_ms,proj_mk_ms,gen_meter_ms,pv_capex_ms],
    })
    st.dataframe(df_once.style.format({"GGV [€]":"{:,.2f}","Mieterstrom [€]":"{:,.2f}"}), use_container_width=True)

# ---------- PDF builder ----------
def draw_header(c, title):
    c.setFillColor(colors.HexColor(Q_ORANGE)); c.rect(0, A4[1]-18*mm, A4[0], 18*mm, fill=1, stroke=0)
    c.setFillColor(colors.white); c.setFont("Helvetica-Bold", 12)
    c.drawString(15*mm, A4[1]-11*mm, "Qrauts AG – Szenariorechner")
    c.setFont("Helvetica", 11); c.drawRightString(A4[0]-15*mm, A4[1]-11*mm, title)

def fig_cashflow_png(df_plot):
    fig, ax = plt.subplots(figsize=(6,3))
    for scen, d in df_plot.groupby("Szenario"):
        ax.plot(d["Jahr"], d["Netto Cashflow"], label=scen)
    ax.set_title("Netto‑Cashflow pro Jahr"); ax.set_xlabel("Jahr"); ax.set_ylabel("€")
    ax.grid(True, alpha=0.3); ax.legend()
    buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png", dpi=200); plt.close(fig); buf.seek(0); return buf

def fig_cum_png(df_plot):
    fig, ax = plt.subplots(figsize=(6,3))
    d=df_plot.copy(); d["Kum"]=d.groupby("Szenario")["Netto Cashflow"].cumsum()
    for scen, s in d.groupby("Szenario"):
        ax.plot(s["Jahr"], s["Kum"], label=scen)
    ax.set_title("Kumulierter Cashflow"); ax.set_xlabel("Jahr"); ax.set_ylabel("€")
    ax.grid(True, alpha=0.3); ax.legend()
    buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png", dpi=200); plt.close(fig); buf.seek(0); return buf

def build_pdf():
    buf = io.BytesIO(); c = canvas.Canvas(buf, pagesize=A4)
    # Cover
    draw_header(c, "Projekt-Cover")
    c.setFont("Helvetica-Bold", 18); c.setFillColor(colors.black)
    c.drawString(20*mm, A4[1]-30*mm, "GGV vs. Mieterstrom – Projektbericht")
    c.setFont("Helvetica", 11); y=A4[1]-45*mm
    c.drawString(20*mm, y, f"Kunde: {customer_name}"); y-=6*mm
    c.drawString(20*mm, y, f"E‑Mail: {customer_email}"); y-=6*mm
    c.drawString(20*mm, y, f"Telefon: {customer_phone}"); y-=8*mm
    c.drawString(20*mm, y, "Kundenadresse:"); y-=6*mm
    for line in (customer_address or "").splitlines():
        c.drawString(25*mm, y, line); y-=5*mm
    y-=4*mm; c.drawString(20*mm, y, "Adresse der Liegenschaft:"); y-=6*mm
    for line in (property_address or "").splitlines():
        c.drawString(25*mm, y, line); y-=5*mm
    y-=6*mm; c.drawString(20*mm, y, f"Projektnummer: {project_number}")
    c.showPage()
    # Parameters
    draw_header(c, "Parameter & Annahmen")
    c.setFont("Helvetica-Bold", 14); c.drawString(20*mm, A4[1]-30*mm, "Wesentliche Parameter")
    c.setFont("Helvetica", 11); y=A4[1]-40*mm
    def line(txt):
        nonlocal y; c.drawString(20*mm, y, txt); y-=6*mm
    line(f"NE (Anzahl): {n_units}")
    line(f"Leistung [kWp]: {kWp}")
    line(f"Spez. Ertrag [kWh/kWp·a]: {specific_yield}")
    line(f"Grundversorgung [ct/kWh]: {grundversorgung_ct}")
    line(f"Preis GGV [ct/kWh]: {ggv_price_ct_base}")
    line(f"Preis MS [ct/kWh]: {ms_price_ct_base}")
    line(f"EEG-Vergütung [ct/kWh]: {eeg_feed_ct}; MS-Zuschlag [ct/kWh]: {mieterstrom_premium_ct}")
    line(f"EV-Anteil Basis [%]: {sc_share_base}; Einspeise-Override [%]: {grid_share_override if grid_share_override is not None else '-'}")
    line(f"Inflation/Preiswachstum [%/a]: {(global_infl if use_global_infl else inflation)} / {(global_infl if use_global_infl else price_growth)}")
    line(f"Diskontsatz [%/a]: {discount}; DV: {'Ja' if dv_required else 'Nein'}; §42b: {'Ja' if sec42b else 'Nein'}")
    if use_detailed_participants:
        line("Teilnehmer (kWh/a):")
        for i in range(int(st.session_state.get('n_part',0))):
            name = st.session_state.get(f"pname_{i}", f'Teilnehmer {i+1}')
            kwh = st.session_state.get(f"pcons_{i}", 0)
            line(f"  - {name}: {kwh}")
    c.showPage()
    # NEW: Cost summary page
    draw_header(c, "Kostenübersicht")
    c.setFont("Helvetica-Bold", 14); c.drawString(20*mm, A4[1]-30*mm, "Einmalkosten & Laufende Kosten")
    c.setFont("Helvetica", 12)
    c.drawString(20*mm, A4[1]-40*mm, "Liegenschaft gesamt")
    c.setFont("Helvetica", 11)
    y=A4[1]-48*mm
    c.drawString(20*mm, y, f"Einmalkosten GGV: {eur2(capex_ggv)} €"); y-=6*mm
    c.drawString(20*mm, y, f"Einmalkosten Mieterstrom: {eur2(capex_ms)} €"); y-=10*mm
    c.drawString(20*mm, y, f"Laufende Kosten/a GGV: {eur2(opex_ggv)} €"); y-=6*mm
    c.drawString(20*mm, y, f"Laufende Kosten/a Mieterstrom: {eur2(opex_ms)} €"); y-=8*mm
    c.setFont("Helvetica-Oblique", 10); c.setFillColor(colors.gray)
    c.drawString(20*mm, y, "Hinweis: Laufende Kosten enthalten MSB, Gateway, IT/SaaS, DV (falls gesetzt) und weitere OPEX.")
    c.setFillColor(colors.black)
    c.showPage()
    # KPIs + Charts
    draw_header(c, "KPIs & Charts")
    c.setFont("Helvetica", 11)
    cap_now = 0.9*grundversorgung_ct
    y=A4[1]-30*mm
    for k,v in [
        ("MS-Preisdeckel [ct/kWh]", f"{cap_now:.1f}"),
        ("NPV GGV [€] (NE/LG)", f"{eur2(npv_ggv/max(n_units,1))} / {eur2(npv_ggv)}"),
        ("NPV MS [€] (NE/LG)", f"{eur2(npv_ms/max(n_units,1))} / {eur2(npv_ms)}"),
        ("Payback GGV / MS [a]", f"{('n/a' if pb_ggv is None else pb_ggv)} / {('n/a' if pb_ms is None else pb_ms)}"),
        ("Projekt‑IRR GGV / MS [%]", f"{(irr_ggv*100 if irr_ggv is not None else float('nan')):.2f} / {(irr_ms*100 if irr_ms is not None else float('nan')):.2f}")
    ]:
        c.drawString(20*mm, y, f"{k}: {v}"); y-=6*mm
    df_plot=df_all[df_all["Jahr"]>0].copy()
    img1=ImageReader(fig_cashflow_png(df_plot)); img2=ImageReader(fig_cum_png(df_plot))
    c.drawImage(img1, 20*mm, 40*mm, width=80*mm, height=45*mm, preserveAspectRatio=True, mask='auto')
    c.drawImage(img2, 110*mm, 40*mm, width=80*mm, height=45*mm, preserveAspectRatio=True, mask='auto')
    c.showPage()
    # Contact
    draw_header(c, "Kontakt")
    c.setFont("Helvetica-Bold", 16); c.drawString(20*mm, A4[1]-30*mm, "Ihr Ansprechpartner")
    c.setFont("Helvetica", 12)
    c.drawString(20*mm, A4[1]-40*mm, "Kurt‑M. Rosenthal – Vorstand Vertrieb | CSO")
    c.drawString(20*mm, A4[1]-48*mm, "E‑Mail: krosenthal@qrauts.de")
    c.drawString(20*mm, A4[1]-56*mm, "Mobil: +49 (0)151 581 481 62")
    c.drawString(20*mm, A4[1]-64*mm, "Web: www.qrauts.de")
    img_path=os.path.join(os.path.dirname(__file__),"assets","KurtRosenthal.png")
    if os.path.exists(img_path):
        c.drawImage(ImageReader(img_path), A4[0]-80*mm, A4[1]-120*mm, width=60*mm, height=60*mm, mask='auto')
    c.setFont("Helvetica", 10); c.setFillColor(colors.gray)
    c.drawString(20*mm, 20*mm, "Qrauts AG | Oltmannstr. 34 | 79100 Freiburg")
    c.showPage()
    c.save(); buf.seek(0); return buf

st.markdown("---")
colA, colB = st.columns([1,3])
with colA:
    if st.button("📄 Projekt‑Report (PDF) erstellen", type="primary"):
        pdf_bytes = build_pdf().getvalue()
        st.download_button("Download PDF", data=pdf_bytes, file_name="Qrauts_Projektbericht.pdf", mime="application/pdf")
with colB:
    st.caption("Der PDF‑Report enthält: Cover mit Kundendaten, Parameterübersicht, **Kostenübersicht**, KPIs/Charts und CSO‑Kontaktseite.")

st.download_button(
    "📤 Export: Jahreswerte (CSV) – Projekt",
    data=df_all.to_csv(index=False).encode("utf-8"),
    file_name="szenario_jahreswerte_projekt.csv",
    mime="text/csv"
)

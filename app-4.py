# app.py — customer-only version (no PDF)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="GGV vs. Mieterstrom – Szenariorechner", layout="wide")

# =============================
# Kundendaten (oben links)
# =============================
st.markdown("### Kundendaten")
c_left, c_right = st.columns([2,3])
with c_left:
    customer_name = st.text_input("Kundenname", value="")
    customer_email = st.text_input("E‑Mail", value="")
    customer_phone = st.text_input("Telefon", value="")
    customer_address = st.text_area("Kundenadresse", value="", height=70)
    property_address = st.text_area("Adresse der Liegenschaft", value="", height=70)
    project_number = st.text_input("Projektnummer", value="")
with c_right:
    st.caption("Die eingegebenen Kundendaten werden nur in der App angezeigt (kein PDF‑Export).")

# =============================
# Helferfunktionen
# =============================
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

def npv_and_payback(df, discount_rate):
    npv=0.0; cum=0.0; pb=None
    for _,row in df.sort_values("Jahr").iterrows():
        y=int(row["Jahr"]); cf=float(row["Netto Cashflow"])
        npv += cf/((1+discount_rate)**y)
        cum += cf
        if pb is None and cum>=0 and y>0:
            pb=y
    return npv, pb

def eur2(v):
    try:
        s=f"{float(v):,.2f}"
        return s.replace(",", "X").replace(".", ",").replace("X", ".")
    except:
        return "n/a"

def build_scenario(
    label, kWp, specific_yield_kwh_per_kwp, self_consumption_share, grid_share_override,
    grundversorgung_ct_per_kwh, eeg_feed_in_ct_per_kwh, dm_fee_ct_per_kwh, internal_price_ct_per_kwh,
    mieterstrom_price_cap_factor, mieterstrom_premium_ct_per_kwh, capex_eur, opex_fixed_eur,
    lifetime_years, degradation_pct_per_year, inflation_pct, price_growth_pct, discount_rate_pct,
    is_mieterstrom, battery_shift_share_pp, storage_lcos_eur_per_kwh, index_eeg_price=False,
    index_ms_premium=False, ms_full_supply=False, ne_count=1, cons_per_ne_kwh=2000, procurement_ct_per_kwh=28.0
):
    # Produktion
    annual_production_kwh = kWp * specific_yield_kwh_per_kwp
    deg = degradation_pct_per_year / 100.0
    infl = inflation_pct / 100.0
    price_growth = price_growth_pct / 100.0

    sc_share = np.clip(self_consumption_share/100.0, 0, 1)
    if grid_share_override is not None:
        grid_share = grid_share_override/100.0
        sc_share = 1 - grid_share
    else:
        grid_share = 1 - sc_share

    eeg_price = eeg_feed_in_ct_per_kwh / 100.0
    dm_fee = dm_fee_ct_per_kwh / 100.0
    internal_price = internal_price_ct_per_kwh / 100.0
    grund = grundversorgung_ct_per_kwh / 100.0
    cap_factor = mieterstrom_price_cap_factor
    premium = mieterstrom_premium_ct_per_kwh / 100.0
    procurement = procurement_ct_per_kwh / 100.0

    export_price_base = max(eeg_price - dm_fee, 0.0)

    rows = []
    for year in range(0, lifetime_years+1):
        prod = 0.0 if year == 0 else annual_production_kwh * ((1 - deg) ** (year-1))
        sc_kwh = prod * sc_share
        grid_kwh = prod * grid_share

        gs_y = grund * ((1 + price_growth) ** max(0, year-1))
        cap_y = cap_factor * gs_y

        internal_y_base = internal_price * ((1 + price_growth) ** max(0, year-1))
        internal_y = min(internal_y_base, cap_y) if is_mieterstrom else internal_y_base

        export_y = export_price_base * ((1 + price_growth) ** max(0, year-1)) if index_eeg_price else export_price_base
        prem_y = premium * ((1 + infl) ** max(0, year-1)) if index_ms_premium else premium

        internal_rev = sc_kwh * internal_y
        export_rev = grid_kwh * export_y
        premium_rev = (sc_kwh * prem_y) if is_mieterstrom else 0.0

        nonpv_rev = 0.0; nonpv_cost = 0.0
        if is_mieterstrom and ms_full_supply and year>0:
            total_cons = ne_count * cons_per_ne_kwh
            comp_kwh = max(total_cons - sc_kwh, 0.0)
            procurement_y = procurement * ((1 + price_growth) ** max(0, year-1))
            nonpv_rev = comp_kwh * internal_y
            nonpv_cost = comp_kwh * procurement_y

        total_rev = internal_rev + export_rev + premium_rev + nonpv_rev

        opex_y = 0.0 if year==0 else opex_fixed_eur * ((1 + infl) ** max(0, year-1))
        if year>0 and battery_shift_share_pp>0 and storage_lcos_eur_per_kwh>0:
            shifted_kwh = prod * (battery_shift_share_pp/100.0)
            opex_y += shifted_kwh * storage_lcos_eur_per_kwh

        capex_y = capex_eur if year == 0 else 0.0
        net_cf = total_rev - opex_y - nonpv_cost - capex_y

        rows.append({
            "Szenario": label, "Jahr": year,
            "Produktion [kWh]": prod, "EV [kWh]": sc_kwh, "Einspeisung [kWh]": grid_kwh,
            "Erlös intern [€]": internal_rev + nonpv_rev, "Einspeiseerlös [€]": export_rev,
            "Mieterstromzuschlag [€]": premium_rev, "Beschaffung nicht‑PV [€]": nonpv_cost,
            "OPEX [€]": opex_y, "CAPEX [€]": capex_y, "Umsatz gesamt [€]": total_rev, "Netto Cashflow": net_cf
        })
    df = pd.DataFrame(rows)
    irr = irr_from_df(df)
    npv, pb = npv_and_payback(df, discount_rate_pct/100.0)
    return df, irr, npv, pb

# =============================
# Sidebar Eingaben (kompakt)
# =============================
st.sidebar.title("Eingaben – Anlage & Preise")
with st.sidebar.expander("Projekt & Anlage", expanded=True):
    n_units = st.number_input("Anzahl Nutzeinheiten (NE)", min_value=1, value=30, step=1)
    kWp = st.number_input("Anlagengröße [kWp]", min_value=1.0, value=99.0, step=1.0)
    specific_yield = st.number_input("Spezifischer Ertrag [kWh/kWp·a]", min_value=400.0, value=600.0, step=10.0)
with st.sidebar.expander("Regulatorik", expanded=True):
    sec42b = st.checkbox("§42b EnWG aktiv (iMSys je NE, 15‑min)", value=True)
    dv_required = st.checkbox("Direktvermarktung aktiv (typ. >100 kWp)", value=False)
with st.sidebar.expander("Preise & Zuschläge", expanded=True):
    grundversorgung_ct = st.number_input("Grundversorgung [ct/kWh]", min_value=10.0, value=40.0, step=0.1)
    ggv_price_ct = st.number_input("Interner Preis GGV [ct/kWh]", min_value=0.0, value=27.0, step=0.1)
    ms_price_ct  = st.number_input("Endkundenpreis Mieterstrom [ct/kWh] (≤90% Grundversorgung)", min_value=0.0, value=29.0, step=0.1)
    eeg_feed_ct = st.number_input("EEG-Vergütung [ct/kWh]", min_value=0.0, value=7.0, step=0.1)
    dm_fee_ct = st.number_input("Direktvermarktung [ct/kWh] (bei DV)", min_value=0.0, value=0.4, step=0.1)
    ms_premium_ct = st.number_input("Mieterstromzuschlag [ct/kWh]", min_value=0.0, value=3.0, step=0.1)
with st.sidebar.expander("EV-Anteil & Batterie", expanded=True):
    sc_share_base = st.slider("Eigenverbrauchsanteil Basis [%]", 0, 100, 35)
    use_override = st.checkbox("Einspeiseanteil-Override", value=True)
    grid_share_override = st.slider("Override Einspeiseanteil [%]", 0, 100, 65) if use_override else None
    battery_enabled = st.checkbox("Batterie/Optimierung aktiv (ΔEV)", value=False)
    delta_ev_pp = st.slider("ΔEV durch Batterie [%‑Pkte]", 0, 60, 10) if battery_enabled else 0
    storage_lcos = st.number_input("LCOS Speicher [€/kWh]", min_value=0.0, value=0.00, step=0.01, format="%.2f")
with st.sidebar.expander("Inflation & Diskontierung", expanded=True):
    global_infl = st.number_input("Inflation [%/a]", min_value=0.0, value=2.0, step=0.1)
    price_growth = st.number_input("Preiswachstum [%/a]", min_value=0.0, value=2.0, step=0.1)
    discount = st.number_input("Diskontsatz (NPV) [%/a]", min_value=0.0, value=6.0, step=0.1)
with st.sidebar.expander("Kosten – Detailliert (CAPEX/OPEX)", expanded=True):
    colA, colB = st.columns(2)
    with colA:
        zpl_ne = st.number_input("ZPL je NE [€]", min_value=0.0, value=700.0, step=50.0)
        subm_ne = st.number_input("Submeter je NE [€]", min_value=0.0, value=180.0, step=10.0)
        imsys_ne = st.number_input("iMSys‑Upgrade je NE [€]", min_value=0.0, value=350.0, step=10.0)
        smgw_cent = st.number_input("SMGw zentral [€]", min_value=0.0, value=600.0, step=10.0)
        it_setup = st.number_input("IT/Abrechnung Setup [€]", min_value=0.0, value=4000.0, step=100.0)
        legal_once = st.number_input("Recht/Reg Setup [€]", min_value=0.0, value=2500.0, step=100.0)
        proj_mk = st.number_input("Projektierung Messkonzept [€]", min_value=0.0, value=3000.0, step=100.0)
        gen_meter = st.number_input("Erzeugungszähler (Einbau) [€]", min_value=0.0, value=250.0, step=10.0)
        pv_capex = st.number_input("PV‑Anlage [€]", min_value=0.0, value=45000.0, step=100.0, format="%.2f")
    with colB:
        msb_gen = st.number_input("MSB Erzeugungszähler [€/a]", min_value=0.0, value=120.0, step=10.0)
        smgw_gate = st.number_input("Gatewaybetrieb (zentral) [€/a]", min_value=0.0, value=120.0, step=10.0)
        it_saas = st.number_input("IT/SaaS Abrechnung [€/a]", min_value=0.0, value=1600.0, step=50.0)
        dv_fix = st.number_input("Direktvermarktung fix [€/a]", min_value=0.0, value=0.0, step=50.0)
        opex_other = st.number_input("Weitere OPEX [€/a]", min_value=0.0, value=800.0, step=50.0)

# CAPEX/OPEX Totale
capex_total = (n_units*(zpl_ne+subm_ne+(imsys_ne if sec42b else 0.0)) + (smgw_cent if sec42b else 0.0)
               + it_setup + legal_once + proj_mk + gen_meter + pv_capex)
opex_total = (msb_gen + (smgw_gate if sec42b else 0.0) + it_saas + (dv_fix if dv_required else 0.0) + opex_other)

# Szenarien bauen
sc_share = np.clip(sc_share_base + (delta_ev_pp if battery_enabled else 0), 0, 100)
grid_override = grid_share_override if use_override else None

df_ggv, irr_ggv, npv_ggv, pb_ggv = build_scenario(
    "GGV", kWp, specific_yield, sc_share, grid_override,
    grundversorgung_ct, eeg_feed_ct, (dm_fee_ct if dv_required else 0.0), ggv_price_ct,
    mieterstrom_price_cap_factor=0.9, mieterstrom_premium_ct_per_kwh=0.0,
    capex_eur=capex_total, opex_fixed_eur=opex_total, lifetime_years=30, degradation_pct_per_year=0.5,
    inflation_pct=global_infl, price_growth_pct=price_growth, discount_rate_pct=discount,
    is_mieterstrom=False, battery_shift_share_pp=(delta_ev_pp if battery_enabled else 0),
    storage_lcos_eur_per_kwh=0.0, index_eeg_price=False, index_ms_premium=False,
    ms_full_supply=False, ne_count=n_units, cons_per_ne_kwh=2000, procurement_ct_per_kwh=28.0
)

df_ms, irr_ms, npv_ms, pb_ms = build_scenario(
    "Mieterstrom", kWp, specific_yield, sc_share, grid_override,
    grundversorgung_ct, eeg_feed_ct, (dm_fee_ct if dv_required else 0.0), ms_price_ct,
    mieterstrom_price_cap_factor=0.9, mieterstrom_premium_ct_per_kwh=ms_premium_ct,
    capex_eur=capex_total, opex_fixed_eur=opex_total, lifetime_years=30, degradation_pct_per_year=0.5,
    inflation_pct=global_infl, price_growth_pct=price_growth, discount_rate_pct=discount,
    is_mieterstrom=True, battery_shift_share_pp=(delta_ev_pp if battery_enabled else 0),
    storage_lcos_eur_per_kwh=0.0, index_eeg_price=False, index_ms_premium=True,
    ms_full_supply=False, ne_count=n_units, cons_per_ne_kwh=2000, procurement_ct_per_kwh=28.0
)

df_all = pd.concat([df_ggv, df_ms], ignore_index=True)

# =============================
# KPIs & Charts
# =============================
st.subheader("KPIs – NE & LG")
cap_now = 0.9*grundversorgung_ct
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("MS‑Preisdeckel [ct/kWh]", f"{cap_now:.1f}")
with col2:
    st.metric("NPV GGV (LG) [€]", eur2(npv_ggv))
with col3:
    st.metric("NPV MS (LG) [€]", eur2(npv_ms))
with col4:
    st.metric("Payback GGV / MS [a]", f"{('n/a' if pb_ggv is None else pb_ggv)} / {('n/a' if pb_ms is None else pb_ms)}")

tab1, tab2 = st.tabs(["Cashflows", "Energieflüsse"])
with tab1:
    df_plot = df_all[df_all["Jahr"]>0].copy()
    fig_cf = px.line(df_plot, x="Jahr", y="Netto Cashflow", color="Szenario", title="Jährlicher Netto‑Cashflow")
    fig_cf.update_traces(hovertemplate="Jahr=%{x}<br>Netto‑CF=%{y:,.2f} €<extra></extra>")
    st.plotly_chart(fig_cf, use_container_width=True)
    df_plot["Kumuliert [€]"] = df_plot.groupby("Szenario")["Netto Cashflow"].cumsum()
    fig_cum = px.line(df_plot, x="Jahr", y="Kumuliert [€]", color="Szenario", title="Kumulierter Cashflow")
    fig_cum.update_traces(hovertemplate="Jahr=%{x}<br>Kumuliert=%{y:,.2f} €<extra></extra>")
    st.plotly_chart(fig_cum, use_container_width=True)
with tab2:
    df_energy = df_all[df_all["Jahr"]>0].copy()
    df_energy = df_energy.melt(id_vars=["Szenario","Jahr"], value_vars=["EV [kWh]","Einspeisung [kWh]"], var_name="Art", value_name="kWh")
    fig_e = px.area(df_energy, x="Jahr", y="kWh", color="Art", facet_col="Szenario", facet_col_wrap=2, title="Energieflüsse")
    fig_e.update_traces(hovertemplate="Jahr=%{x}<br>%{fullData.name}=%{y:,.2f} kWh<extra></extra>")
    st.plotly_chart(fig_e, use_container_width=True)

# CSV Export
st.download_button(
    "📤 Export: Jahreswerte (CSV)",
    data=df_all.to_csv(index=False).encode("utf-8"),
    file_name="szenario_jahreswerte.csv",
    mime="text/csv"
)

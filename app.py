# app.py (v8 – Finanzierung & Eigenkapitalrendite)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="GGV vs. Mieterstrom – Szenariorechner (v8)", layout="wide")

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

def irr_from_series(cf):
    # Bisection IRR between -99% and +200%
    lo, hi = -0.99, 2.0
    def npv_r(r):
        return np.sum(cf / ((1+r)**np.arange(len(cf))))
    f_lo, f_hi = npv_r(lo), npv_r(hi)
    if f_lo * f_hi > 0:
        return None
    for _ in range(80):
        mid = (lo + hi) / 2
        f_mid = npv_r(mid)
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return (lo + hi) / 2

def irr_from_df(df):
    return irr_from_series(df.sort_values("Jahr")["Netto Cashflow"].values.astype(float))

def eur2(value, german=True):
    if value is None:
        return "n/a"
    s = f"{float(value):,.2f}"
    if german:
        s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    return s

def debt_schedule_annuity(principal, rate_pct, tenor_years, grace_years, horizon_years):
    """Return yearly schedule DataFrame with columns: Jahr, Zins, Tilgung, Schuldendienst, Restschuld"""
    r = rate_pct/100.0
    rows = []
    outstanding = principal
    # Years start at 1
    # Grace: interest-only
    for year in range(1, grace_years+1):
        if year > horizon_years: break
        interest = outstanding * r
        principal_pay = 0.0
        debt_service = interest + principal_pay
        rows.append({"Jahr": year, "Zins": interest, "Tilgung": principal_pay, "Schuldendienst": debt_service, "Restschuld": outstanding})
    # Amortization years
    n_amort = max(tenor_years - grace_years, 0)
    annuity = 0.0
    if n_amort > 0:
        if r > 0:
            annuity = outstanding * (r*(1+r)**n_amort) / ((1+r)**n_amort - 1)
        else:
            annuity = outstanding / n_amort
        for j in range(1, n_amort+1):
            year = grace_years + j
            if year > horizon_years: break
            interest = outstanding * r
            principal_pay = annuity - interest
            # numerical guard
            if principal_pay < 0:
                principal_pay = 0.0
            outstanding = max(outstanding - principal_pay, 0.0)
            debt_service = interest + principal_pay
            rows.append({"Jahr": year, "Zins": interest, "Tilgung": principal_pay, "Schuldendienst": debt_service, "Restschuld": outstanding})
    # Fill remaining horizon with zeros
    for year in range(len(rows)+1, horizon_years+1):
        rows.append({"Jahr": year, "Zins": 0.0, "Tilgung": 0.0, "Schuldendienst": 0.0, "Restschuld": outstanding})
    return pd.DataFrame(rows[:horizon_years])

def equity_metrics(project_df, capex, eq_ratio_pct, rate_pct, tenor_years, grace_years, horizon_years, eq_discount_pct):
    """Compute equity cash flows & metrics"""
    debt = capex * (1 - eq_ratio_pct/100.0)
    equity = capex * (eq_ratio_pct/100.0)
    sched = debt_schedule_annuity(debt, rate_pct, tenor_years, grace_years, horizon_years)
    # Project cash flows by year
    cf = project_df.sort_values("Jahr")[["Jahr","Netto Cashflow"]].copy()
    # Equity CF: E0 = -equity; Et = ProjectCF_t - DebtService_t; also add debt draw at t0 to reconcile, i.e. E0 = P0 + debt
    eq_cf = np.zeros(horizon_years+1, dtype=float)
    # Year 0
    p0 = float(cf.loc[cf["Jahr"]==0, "Netto Cashflow"].sum())
    eq_cf[0] = p0 + debt  # equals -equity
    # Years 1..horizon
    for year in range(1, horizon_years+1):
        p_t = float(cf.loc[cf["Jahr"]==year, "Netto Cashflow"].sum())
        ds_t = float(sched.loc[sched["Jahr"]==year, "Schuldendienst"].sum()) if year <= len(sched) else 0.0
        eq_cf[year] = p_t - ds_t
    # Metrics
    eq_npv = 0.0
    r_e = eq_discount_pct/100.0
    for t in range(0, horizon_years+1):
        eq_npv += eq_cf[t] / ((1+r_e)**t)
    irr = irr_from_series(eq_cf)
    cum = 0.0
    payback = None
    for t in range(0, horizon_years+1):
        cum += eq_cf[t]
        if payback is None and cum >= 0 and t>0:
            payback = t
    # DSCR
    dscr_list = []
    for year in range(1, horizon_years+1):
        cfads = float(cf.loc[cf["Jahr"]==year, "Netto Cashflow"].sum())
        ds_t = float(sched.loc[sched["Jahr"]==year, "Schuldendienst"].sum())
        if ds_t > 0:
            dscr_list.append({"Jahr": year, "DSCR": cfads/ds_t})
    dscr_df = pd.DataFrame(dscr_list) if dscr_list else pd.DataFrame(columns=["Jahr","DSCR"])
    min_dscr = None
    min_dscr_year = None
    if not dscr_df.empty:
        idxmin = dscr_df["DSCR"].idxmin()
        min_dscr = float(dscr_df.loc[idxmin, "DSCR"])
        min_dscr_year = int(dscr_df.loc[idxmin, "Jahr"])
    # Assemble DataFrame for display
    eq_df = pd.DataFrame({
        "Jahr": list(range(0, horizon_years+1)),
        "EK-CF [€]": eq_cf
    })
    return {
        "eq_df": eq_df,
        "schedule": sched,
        "eq_npv": eq_npv,
        "eq_irr": irr,
        "eq_payback": payback,
        "min_dscr": min_dscr,
        "min_dscr_year": min_dscr_year,
    }

def build_scenario(
    label,
    kWp,
    specific_yield_kwh_per_kwp,
    self_consumption_share,     # % nach Sensitivität & Batterie
    grid_share_override,        # % or None
    grundversorgung_ct_per_kwh,
    eeg_feed_in_ct_per_kwh,
    dm_fee_ct_per_kwh,
    internal_price_ct_per_kwh,
    mieterstrom_price_cap_factor,   # z.B. 0.9
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
    index_eeg_price=False,
    index_ms_premium=False,
    ms_full_supply=False,
    ne_count=1,
    cons_per_ne_kwh=2000,
    procurement_ct_per_kwh=28.0,
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
    grundversorgung_eur = grundversorgung_ct_per_kwh / 100.0
    mieterstrom_cap_factor = mieterstrom_price_cap_factor
    mieterstrom_premium_eur = mieterstrom_premium_ct_per_kwh / 100.0
    procurement_eur = procurement_ct_per_kwh / 100.0

    export_price_base = max(eeg_price_eur - dm_fee_eur, 0.0)

    rows = []
    for year in range(0, lifetime_years+1):
        if year == 0:
            prod = 0.0
        else:
            prod = annual_production_kwh * ((1 - deg) ** (year-1))

        sc_kwh = prod * sc_share
        grid_kwh = prod * grid_share

        # Preise/Indizes
        gs_y = grundversorgung_eur * ((1 + price_growth) ** max(0, year-1))
        cap_y = mieterstrom_cap_factor * gs_y

        internal_price_y_base = internal_price_eur * ((1 + price_growth) ** max(0, year-1))
        if is_mieterstrom:
            internal_price_y = min(internal_price_y_base, cap_y)
        else:
            internal_price_y = internal_price_y_base

        export_price_y = export_price_base * ((1 + price_growth) ** max(0, year-1)) if index_eeg_price else export_price_base
        premium_y = mieterstrom_premium_eur * ((1 + infl) ** max(0, year-1)) if index_ms_premium else mieterstrom_premium_eur

        # Erlöse PV
        internal_rev = sc_kwh * internal_price_y
        export_rev = grid_kwh * export_price_y
        premium_rev = (sc_kwh * premium_y) if is_mieterstrom else 0.0

        nonpv_rev = 0.0
        nonpv_cost = 0.0
        if is_mieterstrom and ms_full_supply and year>0:
            total_cons = ne_count * cons_per_ne_kwh
            comp_kwh = max(total_cons - sc_kwh, 0.0)
            procurement_y = procurement_eur * ((1 + price_growth) ** max(0, year-1))
            nonpv_rev = comp_kwh * internal_price_y  # Verkauf
            nonpv_cost = comp_kwh * procurement_y    # Beschaffung

        total_rev = internal_rev + export_rev + premium_rev + nonpv_rev

        # OPEX fix (ab Jahr 1, inflationsindexiert)
        opex_y = 0.0 if year == 0 else opex_fixed_eur * ((1 + infl) ** max(0, year-1))

        # Speicher-Betriebskosten via LCOS auf den ΔEV-Anteil (nur wenn year>0)
        if year > 0 and battery_shift_share_pp > 0 and storage_lcos_eur_per_kwh > 0:
            shifted_kwh = prod * (battery_shift_share_pp/100.0)
            opex_y += shifted_kwh * storage_lcos_eur_per_kwh

        capex_y = capex_eur if year == 0 else 0.0

        net_cf = total_rev - opex_y - nonpv_cost - capex_y

        rows.append({
            "Szenario": label,
            "Jahr": year,
            "Produktion [kWh]": prod,
            "EV [kWh]": sc_kwh,
            "Einspeisung [kWh]": grid_kwh,
            "Erlös intern [€]": internal_rev + nonpv_rev,
            "Einspeiseerlös [€]": export_rev,
            "Mieterstromzuschlag [€]": premium_rev,
            "Beschaffung nicht‑PV [€]": nonpv_cost,
            "OPEX [€]": opex_y,
            "CAPEX [€]": capex_y,
            "Umsatz gesamt [€]": total_rev,
            "Netto Cashflow": net_cf,
        })

    df = pd.DataFrame(rows)
    npv, payback = cashflow_summary(df, discount_rate_pct/100.0)
    irr = irr_from_df(df)
    return df, npv, payback, irr

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.title("Eingaben – Anlage & Preise")

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
    st.caption("Wird automatisch zu CAPEX/OPEX aggregiert.")
    # Einmal
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

    # Laufend Eigentümer
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

with st.sidebar.expander("Mieterstrom – Vollversorgung & Index-Toggles", expanded=False):
    ms_full_supply = st.checkbox("Vollversorgung modellieren (Restbezug einkaufen)", value=False)
    cons_per_ne = st.number_input("Jahresverbrauch je NE [kWh]", min_value=0, value=2000, step=100)
    procurement_ct = st.number_input("Beschaffungspreis Restbezug [ct/kWh]", min_value=0.0, value=28.0, step=0.1)
    index_eeg = st.checkbox("EEG-Einspeisevergütung jährlich indexieren (Export)", value=False)
    index_premium = st.checkbox("Mieterstromzuschlag jährlich indexieren", value=False)

with st.sidebar.expander("Finanzierung (für EK-Rendite)", expanded=True):
    eq_ratio = st.number_input("Eigenkapitalquote [% vom CAPEX]", min_value=0.0, max_value=100.0, value=30.0, step=1.0)
    fk_rate = st.number_input("Fremdkapitalzins p.a. [%]", min_value=0.0, value=4.5, step=0.1)
    fk_tenor = st.number_input("Fremdkapital-Laufzeit [Jahre]", min_value=1, value=15, step=1)
    fk_grace = st.number_input("Tilgungsfreie Jahre (Zins-only)", min_value=0, value=1, step=1)
    eq_discount = st.number_input("EK-Diskontsatz (NPV) [%/a]", min_value=0.0, value=8.0, step=0.1)

# -----------------------------
# Derive CAPEX/OPEX totals
# -----------------------------
mieterstrom_cap_factor = 0.9  # 90% Deckel
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
opex_ggv = (msb_gen + (smgw_gate if sec42b else 0.0) + it_saas_ggv + (dv_fix if dv_required else 0.0) + opex_other) * (1 + sens_opex_pct/100.0)
opex_ms  = (msb_gen_ms + (smgw_gate_ms if sec42b else 0.0) + it_saas_ms + (dv_fix_ms if dv_required else 0.0) + opex_other_ms) * (1 + sens_opex_pct/100.0)

# -----------------------------
# Build scenarios (Projekt-CF)
# -----------------------------
df_ggv, npv_ggv, pb_ggv, irr_ggv = build_scenario(
    label="GGV",
    kWp=kWp,
    specific_yield_kwh_per_kwp=specific_yield,
    self_consumption_share=sc_share,
    grid_share_override=(grid_share_override if use_override else None),
    grundversorgung_ct_per_kwh=grundversorgung_ct,
    eeg_feed_in_ct_per_kwh=eeg_feed_ct,
    dm_fee_ct_per_kwh=(dm_fee_ct if dv_required else 0.0),
    internal_price_ct_per_kwh=ggv_price_ct,
    mieterstrom_price_cap_factor=mieterstrom_cap_factor,
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
    storage_lcos_eur_per_kwh=storage_lcos,
    index_eeg_price=False,      # EEG fix für GGV-Export
    index_ms_premium=False,
    ms_full_supply=False,
    ne_count=n_units,
)

df_ms, npv_ms, pb_ms, irr_ms = build_scenario(
    label="Mieterstrom",
    kWp=kWp,
    specific_yield_kwh_per_kwp=specific_yield,
    self_consumption_share=sc_share,
    grid_share_override=(grid_share_override if use_override else None),
    grundversorgung_ct_per_kwh=grundversorgung_ct,
    eeg_feed_in_ct_per_kwh=eeg_feed_ct,
    dm_fee_ct_per_kwh=(dm_fee_ct if dv_required else 0.0),
    internal_price_ct_per_kwh=ms_price_ct,
    mieterstrom_price_cap_factor=mieterstrom_cap_factor,
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
    storage_lcos_eur_per_kwh=storage_lcos,
    index_eeg_price=index_eeg,
    index_ms_premium=index_premium,
    ms_full_supply=ms_full_supply,
    ne_count=n_units,
    cons_per_ne_kwh=cons_per_ne,
    procurement_ct_per_kwh=procurement_ct,
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
cap_now = 0.9*grundversorgung_ct
with c1:
    st.markdown(f"<div style='border:1px solid #e5e7eb; padding:10px; border-radius:12px; text-align:center;'>"
                f"<div style='font-weight:600;'>Mieterstrom‑Preisdeckel [ct/kWh]</div>"
                f"<div style='font-size:22px;'>{cap_now:.1f}</div></div>", unsafe_allow_html=True)
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
                f"<div style='font-size:22px; font-weight:700; color:#1f4acc;'>{cap_now:.1f}</div></div>", unsafe_allow_html=True)
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
# ROI & IRR (Projekt) + EK-Rendite
# -----------------------------
st.markdown("### Wirtschaftlichkeit / Rendite – Projekt vs. Eigenkapital")
years = st.slider("Analysehorizont [Jahre]", min_value=2, max_value=30, value=20, step=1)
def roi_over_horizon(df, years):
    capex = float(df.loc[df["Jahr"]==0, "CAPEX [€]"].sum())
    cum_net = float(df[df["Jahr"]<=years]["Netto Cashflow"].sum())
    if capex <= 0:
        return None
    return cum_net / capex

# Projekt-ROI/IRR
roi_ggv = roi_over_horizon(df_ggv, years)
roi_ms = roi_over_horizon(df_ms, years)
irr_prj_ggv = irr_ggv
irr_prj_ms = irr_ms

# EK-Metriken
ek_ggv = equity_metrics(df_ggv, capex_ggv, eq_ratio, fk_rate, fk_tenor, fk_grace, years, eq_discount)
ek_ms  = equity_metrics(df_ms,  capex_ms,  eq_ratio, fk_rate, fk_tenor, fk_grace, years, eq_discount)

colA, colB, colC, colD = st.columns(4)
colA.metric(f"Projekt-ROI GGV (bis Jahr {years})", f"{(roi_ggv*100):.1f}%")
colB.metric(f"Projekt-ROI MS (bis Jahr {years})", f"{(roi_ms*100):.1f}%")
colC.metric("Projekt-IRR GGV", f"{(irr_prj_ggv*100):.2f}%" if irr_prj_ggv is not None else "n/a")
colD.metric("Projekt-IRR MS", f"{(irr_prj_ms*100):.2f}%" if irr_prj_ms is not None else "n/a")

colE, colF, colG, colH = st.columns(4)
colE.metric("**EK‑IRR GGV**", f"{(ek_ggv['eq_irr']*100):.2f}%" if ek_ggv['eq_irr'] is not None else "n/a")
colF.metric("**EK‑IRR MS**", f"{(ek_ms['eq_irr']*100):.2f}%" if ek_ms['eq_irr'] is not None else "n/a")
colG.metric("**EK‑NPV GGV**", f"{eur2(ek_ggv['eq_npv'])}")
colH.metric("**EK‑NPV MS**", f"{eur2(ek_ms['eq_npv'])}")

colI, colJ = st.columns(2)
colI.metric("**EK‑Payback GGV**", f"{ek_ggv['eq_payback']} a" if ek_ggv['eq_payback'] is not None else "n/a")
colJ.metric("**EK‑Payback MS**", f"{ek_ms['eq_payback']} a" if ek_ms['eq_payback'] is not None else "n/a")

colK, colL = st.columns(2)
colK.metric("Min. DSCR GGV", f"{ek_ggv['min_dscr']:.2f}" if ek_ggv['min_dscr'] is not None else "n/a", help=f"Jahr {ek_ggv['min_dscr_year']}" if ek_ggv['min_dscr_year'] else None)
colL.metric("Min. DSCR MS", f"{ek_ms['min_dscr']:.2f}" if ek_ms['min_dscr'] is not None else "n/a", help=f"Jahr {ek_ms['min_dscr_year']}" if ek_ms['min_dscr_year'] else None)

# Charts
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Cashflows (Projekt)", "Energieflüsse", "Jahreswerte", "Kosten (Detail)", "Finanzierung/EK"])

with tab1:
    df_plot = df_all[df_all["Jahr"]>0].copy()
    fig_cf = px.line(df_plot, x="Jahr", y="Netto Cashflow", color="Szenario", title="Jährlicher Netto-Cashflow (Projekt)")
    st.plotly_chart(fig_cf, use_container_width=True)

    df_cum = df_plot.copy()
    df_cum["Kumuliert [€]"] = df_cum.groupby("Szenario")["Netto Cashflow"].cumsum()
    fig_cum = px.line(df_cum, x="Jahr", y="Kumuliert [€]", color="Szenario", title="Kumulierter Cashflow (Projekt)")
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
        "Beschaffung nicht‑PV [€]":"{:,.0f}",
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

with tab5:
    st.markdown("### Eigenkapital-Cashflows & Schuldendienst")
    st.write("**GGV – EK‑CF und Schuldendienst**")
    st.dataframe(ek_ggv["eq_df"].merge(ek_ggv["schedule"], how="left", on="Jahr").fillna(0.0).style.format({
        "EK-CF [€]":"{:,.0f}","Zins":"{:,.0f}","Tilgung":"{:,.0f}","Schuldendienst":"{:,.0f}","Restschuld":"{:,.0f}"
    }), use_container_width=True)
    st.write("**Mieterstrom – EK‑CF und Schuldendienst**")
    st.dataframe(ek_ms["eq_df"].merge(ek_ms["schedule"], how="left", on="Jahr").fillna(0.0).style.format({
        "EK-CF [€]":"{:,.0f}","Zins":"{:,.0f}","Tilgung":"{:,.0f}","Schuldendienst":"{:,.0f}","Restschuld":"{:,.0f}"
    }), use_container_width=True)

# -----------------------------
# Export
# -----------------------------
st.download_button(
    "📤 Export: Jahreswerte (CSV) – Projekt",
    data=df_all.to_csv(index=False).encode("utf-8"),
    file_name="szenario_jahreswerte_projekt.csv",
    mime="text/csv"
)

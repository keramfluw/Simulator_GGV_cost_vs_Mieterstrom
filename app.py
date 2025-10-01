# app.py (v4 – NE/LG wording & € with 2 decimals, DE formatting)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="GGV vs. Mieterstrom – Szenariorechner (v4)", layout="wide")

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

def build_scenario(
    label,
    kWp,
    specific_yield_kwh_per_kwp,
    self_consumption_share,
    grid_share_override,
    grid_price_ct_per_kwh,
    eeg_feed_in_ct_per_kwh,
    direct_marketing_fee_ct_per_kwh,
    internal_price_ct_per_kwh,
    mieterstrom_price_cap_ct_per_kwh,
    mieterstrom_premium_ct_per_kwh,
    capex_eur,
    opex_pct_of_capex,
    opex_fixed_eur,
    lifetime_years,
    degradation_pct_per_year,
    inflation_pct,
    energy_price_growth_pct,
    discount_rate_pct,
    is_mieterstrom,
    battery_note
):
    annual_production_kwh = kWp * specific_yield_kwh_per_kwp
    deg = degradation_pct_per_year / 100.0
    infl = inflation_pct / 100.0
    price_growth = energy_price_growth_pct / 100.0
    disc = discount_rate_pct / 100.0

    sc_share = np.clip(self_consumption_share/100.0, 0, 1)
    if grid_share_override is not None:
        grid_share = grid_share_override/100.0
        sc_share = 1 - grid_share
    else:
        grid_share = 1 - sc_share

    eeg_price_eur = eeg_feed_in_ct_per_kwh / 100.0
    dm_fee_eur = direct_marketing_fee_ct_per_kwh / 100.0
    internal_price_eur = internal_price_ct_per_kwh / 100.0
    mieterstrom_cap_eur = mieterstrom_price_cap_ct_per_kwh / 100.0
    mieterstrom_premium_eur = mieterstrom_premium_ct_per_kwh / 100.0

    if is_mieterstrom:
        internal_price_eur = min(internal_price_eur, mieterstrom_cap_eur)

    export_price_eur = max(eeg_price_eur - dm_fee_eur, 0.0)

    rows = []
    for year in range(0, lifetime_years+1):
        if year == 0:
            prod = 0.0
        else:
            prod = annual_production_kwh * ((1 - deg) ** (year-1))

        sc_kwh = prod * sc_share
        grid_kwh = prod * grid_share

        internal_price_y = internal_price_eur * ((1 + price_growth) ** max(0, year-1))
        export_price_y = export_price_eur * ((1 + price_growth) ** max(0, year-1))
        premium_y = mieterstrom_premium_eur * ((1 + infl) ** max(0, year-1))

        internal_rev = sc_kwh * internal_price_y
        export_rev = grid_kwh * export_price_y
        premium_rev = (sc_kwh * premium_y) if is_mieterstrom else 0.0
        total_rev = internal_rev + export_rev + premium_rev

        opex = (capex_eur * (opex_pct_of_capex/100.0)) + opex_fixed_eur
        opex_y = opex * ((1 + infl) ** max(0, year-1))

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
            "Annahme Batterie": battery_note,
        })

    df = pd.DataFrame(rows)
    npv, payback = cashflow_summary(df, disc)
    return df, npv, payback

def roi_over_horizon(df, years):
    capex = float(df.loc[df["Jahr"]==0, "CAPEX [€]"].sum())
    cum_net = float(df[df["Jahr"]<=years]["Netto Cashflow"].sum())
    if capex <= 0:
        return None
    return cum_net / capex

def eur2(value, german=True):
    """Format value to Euro with two decimals. German style if german=True."""
    if value is None:
        return "n/a"
    s = f"{value:,.2f}"
    if german:
        s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    return s

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.title("Eingaben – Anlage & Preise")

with st.sidebar.expander("Projekt & Anlage", expanded=True):
    n_units = st.number_input("Anzahl Nutzeinheiten (NE)", min_value=1, value=30, step=1)
    kWp = st.number_input("Anlagengröße [kWp]", min_value=1.0, value=99.0, step=1.0)
    specific_yield = st.number_input("Spezifischer Ertrag [kWh/kWp*a]", min_value=600.0, value=1000.0, step=10.0)
    sc_share = st.slider("Eigenverbrauchsanteil [%] (wenn keine Batterie-Optimierung)", 0, 100, 35)
    grid_share_override = st.slider("Optional: Einspeiseanteil [%] überschreiben", 0, 100, 65)
    use_override = st.checkbox("Einspeiseanteil-Override verwenden", value=True)

with st.sidebar.expander("Preise & Vergütungen", expanded=True):
    grundversorgung_ct = st.number_input("Örtlicher Grundversorgungstarif [ct/kWh] (für Mieterstrom-Deckel)", min_value=10.0, value=40.0, step=0.5)
    ggv_price_ct = st.number_input("Interner Abgabepreis GGV [ct/kWh]", min_value=0.0, value=32.0, step=0.5)
    mieterstrom_price_ct = st.number_input("Geplanter Endkundenpreis Mieterstrom [ct/kWh] (<= 90% Grundversorgung)", min_value=0.0, value=34.0, step=0.5)
    eeg_feed_ct = st.number_input("EEG-Einspeisevergütung [ct/kWh]", min_value=0.0, value=7.0, step=0.1)
    dm_fee_ct = st.number_input("Direktvermarktungsgebühr [ct/kWh] (typisch >100 kWp)", min_value=0.0, value=0.4, step=0.1)
    mieterstrom_premium_ct = st.number_input("Mieterstromzuschlag [ct/kWh] (für EV-Mengen)", min_value=0.0, value=3.0, step=0.1)

with st.sidebar.expander("Kosten, Laufzeit & Finanzen", expanded=True):
    capex = st.number_input("CAPEX gesamt [€]", min_value=1000.0, value=120000.0, step=1000.0)
    opex_pct = st.number_input("OPEX [% von CAPEX/Jahr]", min_value=0.0, value=1.5, step=0.1)
    opex_fixed = st.number_input("OPEX fix [€/Jahr]", min_value=0.0, value=1500.0, step=100.0)
    lifetime = st.number_input("Laufzeit [Jahre]", min_value=5, value=20, step=1)
    degradation = st.number_input("Moduldegradation [%/a]", min_value=0.0, value=0.5, step=0.1)
    discount = st.number_input("Diskontsatz [%/a] (NPV)", min_value=0.0, value=6.0, step=0.1)

with st.sidebar.expander("Inflation & Wachstumsannahmen", expanded=True):
    global_infl = st.number_input("Globale Inflationsrate [%/a] (Optional-Override)", min_value=0.0, value=2.0, step=0.1)
    use_global_infl = st.checkbox("Globale Inflation für Kosten UND Preise verwenden", value=True)
    if use_global_infl:
        inflation = global_infl
        price_growth = global_infl
    else:
        inflation = st.number_input("Inflation Kosten [%/a]", min_value=0.0, value=2.0, step=0.1)
        price_growth = st.number_input("Preiswachstum Erlöse [%/a]", min_value=0.0, value=2.0, step=0.1)

with st.sidebar.expander("Batterie / EV-Optimierung", expanded=False):
    battery_enabled = st.checkbox("Batterie/Optimierung wirkt – erhöht EV-Anteil um Δ", value=False)
    delta_ev = st.slider("Zusätzlicher EV durch Batterie [%punkte]", 0, 60, 10)
    battery_note = "mit Speicher/Optimierung" if battery_enabled else "ohne Speicher"
    if battery_enabled:
        sc_share = min(100, sc_share + delta_ev)
        if use_override:
            grid_share_override = max(0, grid_share_override - delta_ev)

mieterstrom_cap = 0.9 * grundversorgung_ct

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
    direct_marketing_fee_ct_per_kwh=dm_fee_ct,
    internal_price_ct_per_kwh=ggv_price_ct,
    mieterstrom_price_cap_ct_per_kwh=mieterstrom_cap,
    mieterstrom_premium_ct_per_kwh=0.0,
    capex_eur=capex,
    opex_pct_of_capex=opex_pct,
    opex_fixed_eur=opex_fixed,
    lifetime_years=lifetime,
    degradation_pct_per_year=degradation,
    inflation_pct=inflation,
    energy_price_growth_pct=price_growth,
    discount_rate_pct=discount,
    is_mieterstrom=False,
    battery_note=battery_note
)

df_ms, npv_ms, pb_ms = build_scenario(
    label="Mieterstrom",
    kWp=kWp,
    specific_yield_kwh_per_kwp=specific_yield,
    self_consumption_share=sc_share,
    grid_share_override=(grid_share_override if use_override else None),
    grid_price_ct_per_kwh=grundversorgung_ct,
    eeg_feed_in_ct_per_kwh=eeg_feed_ct,
    direct_marketing_fee_ct_per_kwh=dm_fee_ct,
    internal_price_ct_per_kwh=mieterstrom_price_ct,
    mieterstrom_price_cap_ct_per_kwh=mieterstrom_cap,
    mieterstrom_premium_ct_per_kwh=mieterstrom_premium_ct,
    capex_eur=capex,
    opex_pct_of_capex=opex_pct,
    opex_fixed_eur=opex_fixed,
    lifetime_years=lifetime,
    degradation_pct_per_year=degradation,
    inflation_pct=inflation,
    energy_price_growth_pct=price_growth,
    discount_rate_pct=discount,
    is_mieterstrom=True,
    battery_note=battery_note
)

# -----------------------------
# KPIs – NE (Nutzeinheit) & LG (Liegenschaft)
# -----------------------------
ne_npv_ggv = npv_ggv / n_units
ne_npv_ms = npv_ms / n_units
pb1 = "n/a" if pb_ggv is None else f"{pb_ggv} a"
pb2 = "n/a" if pb_ms is None else f"{pb_ms} a"

st.subheader("Übersicht – KPIs")

# NE row
st.markdown("**NE – Nutzeinheit (pro Einheit)**")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"<div style='border:1px solid #e5e7eb; padding:10px; border-radius:12px; text-align:center;'>"
                f"<div style='font-weight:600;'>Mieterstrom‑Preisdeckel [ct/kWh]</div>"
                f"<div style='font-size:22px;'>{mieterstrom_cap:.1f}</div></div>", unsafe_allow_html=True)
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
                f"<div style='font-size:22px; font-weight:700; color:#1f4acc;'>{mieterstrom_cap:.1f}</div></div>", unsafe_allow_html=True)
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
# Rendite (2–30 Jahre)
# -----------------------------
st.markdown("### Wirtschaftlichkeit / Rendite Eigentümer über Jahre")
years = st.slider("Analysehorizont [Jahre]", min_value=2, max_value=30, value=10, step=1)
roi_ggv = roi_over_horizon(df_ggv, years)
roi_ms = roi_over_horizon(df_ms, years)

r1, r2 = st.columns(2)
r1.metric(f"ROI GGV bis Jahr {years}", f"{(roi_ggv*100):.1f}%")
r2.metric(f"ROI Mieterstrom bis Jahr {years}", f"{(roi_ms*100):.1f}%")

years_range = list(range(2, min(int(lifetime), 30)+1))
roi_curve = pd.DataFrame({
    "Jahr": years_range,
    "ROI GGV [%]": [roi_over_horizon(df_ggv, y)*100 for y in years_range],
    "ROI Mieterstrom [%]": [roi_over_horizon(df_ms, y)*100 for y in years_range],
})
fig_roi = px.line(roi_curve.melt(id_vars="Jahr", var_name="Szenario", value_name="ROI [%]"),
                  x="Jahr", y="ROI [%]", color="Szenario", title="ROI-Verlauf (Eigentümer)")
st.plotly_chart(fig_roi, use_container_width=True)

st.caption("Hinweis: NE = Nutzeinheit (pro Einheit). LG = Liegenschaft (Summe aller NE). Euro‑Werte mit zwei Dezimalstellen in deutscher Darstellung.")

# -----------------------------
# Charts
# -----------------------------
tab1, tab2, tab3 = st.tabs(["Cashflows", "Energieflüsse", "Jahreswerte"])

with tab1:
    df_plot = pd.concat([df_ggv[df_ggv["Jahr"]>0], df_ms[df_ms["Jahr"]>0]], ignore_index=True)
    fig_cf = px.line(df_plot, x="Jahr", y="Netto Cashflow", color="Szenario", title="Jährlicher Netto-Cashflow")
    st.plotly_chart(fig_cf, use_container_width=True)

    df_cum = df_plot.copy()
    df_cum["CumCF"] = df_cum.groupby("Szenario")["Netto Cashflow"].cumsum()
    fig_cum = px.line(df_cum, x="Jahr", y="CumCF", color="Szenario", title="Kumulierter Cashflow")
    st.plotly_chart(fig_cum, use_container_width=True)

with tab2:
    df_energy = pd.concat([df_ggv[df_ggv["Jahr"]>0], df_ms[df_ms["Jahr"]>0]], ignore_index=True)
    df_energy = df_energy.melt(id_vars=["Szenario","Jahr"], value_vars=["EV [kWh]","Einspeisung [kWh]"], var_name="Art", value_name="kWh")
    fig_e = px.area(df_energy, x="Jahr", y="kWh", color="Art", facet_col="Szenario", facet_col_wrap=2, title="Energieflüsse EV vs. Einspeisung")
    st.plotly_chart(fig_e, use_container_width=True)

with tab3:
    st.dataframe(pd.concat([df_ggv, df_ms], ignore_index=True).style.format({
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

st.download_button(
    "📤 Export: Jahreswerte (CSV)",
    data=pd.concat([df_ggv, df_ms], ignore_index=True).to_csv(index=False).encode("utf-8"),
    file_name="szenario_jahreswerte.csv",
    mime="text/csv"
)

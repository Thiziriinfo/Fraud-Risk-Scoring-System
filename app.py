import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

st.set_page_config(
    page_title="Fraud Risk Dashboard",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    [data-testid="stMetric"] {
        background-color: #1e1e2e;
        border-radius: 10px;
        padding: 16px;
        border-left: 4px solid #6366f1;
    }
    [data-testid="stMetricLabel"] { font-size: 0.85rem; color: #9ca3af; }
    [data-testid="stMetricValue"] { font-size: 1.8rem; font-weight: 700; }
    h1 { font-size: 1.6rem !important; }
    h2 { font-size: 1.2rem !important; color: #9ca3af; font-weight: 400 !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    df = pd.read_csv('fraud_sample_raw.csv')
    df.columns = df.columns.str.strip()
    rename_map = {
        'Is Fraud': 'Is_Fraud',
        'Use Chip': 'Use_Chip',
        'Errors?': 'Errors',
        'Merchant Name': 'Merchant_Name',
        'Merchant State': 'Merchant_State'
    }
    for old, new in rename_map.items():
        if old in df.columns:
            df.rename(columns={old: new}, inplace=True)
    if 'Year' not in df.columns and 'date' in df.columns:
        df['Year'] = pd.to_datetime(df['date']).dt.year
    return df


try:
    df_all = load_data()
except FileNotFoundError:
    st.error("Fichier `fraud_sample_raw.csv` introuvable. Placez-le dans le même dossier que app.py.")
    st.stop()

PURPLE = "#6366f1"
RED    = "#ef4444"
GREEN  = "#22c55e"
GRAY   = "#9ca3af"

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔒 Fraud Risk Scoring")
    st.markdown("IBM TabFormer · LightGBM")
    st.divider()

    # Filtre Year
    if 'Year' in df_all.columns:
        years_available = sorted(df_all['Year'].dropna().unique().tolist())
        years_selected = st.multiselect(
            "Filtrer par année",
            options=years_available,
            default=years_available
        )
        df = df_all[df_all['Year'].isin(years_selected)] if years_selected else df_all.copy()
    else:
        df = df_all.copy()

    st.divider()
    page = st.radio(
        "Navigation",
        ["Vue Exécutive", "Analyse des Fraudes", "Profil Marchand", "Performance Modèle"],
        label_visibility="collapsed"
    )
    st.divider()
    st.caption("Modèle : LightGBM · AUC 0.9904")
    st.caption("Thiziri Abchiche — Data Analyst")

df_fraud  = df[df['Is_Fraud'] == 1]
df_normal = df[df['Is_Fraud'] == 0]


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — VUE EXECUTIVE
# ══════════════════════════════════════════════════════════════════════════════
if page == "Vue Exécutive":
    st.title("Vue Exécutive")
    st.markdown("## Synthèse des résultats du système de détection de fraude")
    st.divider()

    total             = len(df)
    total_fraud       = len(df_fraud)
    fraud_rate        = total_fraud / total * 100 if total > 0 else 0
    avg_amount_fraud  = df_fraud['Amount'].mean() if len(df_fraud) > 0 else 0
    avg_amount_normal = df_normal['Amount'].mean() if len(df_normal) > 0 else 0

    # KPIs principaux
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Transactions analysées", f"{total:,}".replace(",", " "))
    with col2:
        st.metric("Fraudes détectées", f"{total_fraud:,}".replace(",", " "))
    with col3:
        st.metric("Taux de fraude", f"{fraud_rate:.3f}%")
    with col4:
        delta = avg_amount_fraud - avg_amount_normal
        st.metric("Montant moyen — fraude", f"${avg_amount_fraud:.0f}",
                  delta=f"+${delta:.0f} vs normal", delta_color="inverse")

    st.divider()

    # ── Impact financier ────────────────────────────────────────────────────
    st.subheader("💰 Impact financier estimé")
    st.caption("Basé sur un Recall de 94% et un seuil de décision à 0.5")

    total_fraud_amount   = df_fraud['Amount'].sum()
    fraud_detected_amt   = total_fraud_amount * 0.94   # 94% recall
    fraud_missed_amt     = total_fraud_amount * 0.06
    false_positive_count = int(len(df_normal) * 0.038) # ~3.8% FP rate
    false_positive_cost  = false_positive_count * 15   # coût traitement alerte : ~15$

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        st.metric(
            "Fraudes bloquées (94% recall)",
            f"${fraud_detected_amt:,.0f}".replace(",", " "),
            delta="montant protégé",
            delta_color="normal"
        )
    with fc2:
        st.metric(
            "Fraudes manquées (6%)",
            f"${fraud_missed_amt:,.0f}".replace(",", " "),
            delta="perte résiduelle",
            delta_color="inverse"
        )
    with fc3:
        st.metric(
            "Coût fausses alertes",
            f"${false_positive_cost:,.0f}".replace(",", " "),
            delta=f"{false_positive_count:,} faux positifs".replace(",", " "),
            delta_color="inverse"
        )

    st.divider()

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Score du modèle LightGBM")
        metrics_data = {
            "Métrique":  ["AUC-ROC", "Recall (fraude)", "Precision (fraude)", "Std CV (5 folds)"],
            "Valeur":    [0.9904,    0.94,               0.59,                 0.0004],
            "Objectif":  ["> 0.90",  "> 0.90",           "> 0.50",             "< 0.005"]
        }
        st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)

    with col_right:
        st.subheader("Jauge Recall — fraudes détectées")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=94,
            number={"suffix": "%", "font": {"size": 40}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": GREEN},
                "steps": [
                    {"range": [0,  70], "color": "#374151"},
                    {"range": [70, 90], "color": "#4b5563"},
                    {"range": [90,100], "color": "#1f2937"},
                ],
                "threshold": {
                    "line": {"color": RED, "width": 3},
                    "thickness": 0.75,
                    "value": 90
                }
            },
            title={"text": "Recall fraude (objectif : > 90%)"}
        ))
        fig_gauge.update_layout(
            height=280,
            margin=dict(t=40, b=10, l=20, r=20),
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white"
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    st.divider()

    # ── Tendance mensuelle ───────────────────────────────────────────────────
    if 'Month' in df.columns and 'Year' in df.columns:
        st.subheader("📈 Évolution mensuelle du taux de fraude")
        monthly = df.groupby(['Year', 'Month']).agg(
            total=('Is_Fraud', 'count'),
            frauds=('Is_Fraud', 'sum')
        ).reset_index()
        monthly['taux'] = monthly['frauds'] / monthly['total'] * 100
        monthly['period'] = monthly['Year'].astype(str) + '-' + monthly['Month'].astype(str).str.zfill(2)
        monthly = monthly.sort_values('period')

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=monthly['period'], y=monthly['taux'],
            mode='lines+markers',
            line=dict(color=PURPLE, width=2),
            marker=dict(size=5),
            fill='tozeroy',
            fillcolor='rgba(99,102,241,0.12)'
        ))
        fig_trend.update_layout(
            xaxis_title="Période",
            yaxis_title="Taux de fraude (%)",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            height=280,
            margin=dict(t=10, b=10),
            xaxis=dict(gridcolor="#374151", tickangle=-45),
            yaxis=dict(gridcolor="#374151")
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    st.divider()
    col_pie, col_txt = st.columns([1, 1])
    with col_pie:
        st.subheader("Répartition des transactions")
        fig_pie = go.Figure(go.Pie(
            labels=["Normal", "Fraude"],
            values=[len(df_normal), len(df_fraud)],
            hole=0.55,
            marker=dict(colors=[PURPLE, RED]),
            textinfo="label+percent",
            textfont=dict(size=14)
        ))
        fig_pie.update_layout(
            height=300,
            margin=dict(t=20, b=20, l=20, r=20),
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            showlegend=False
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    with col_txt:
        st.markdown("""
        **Déséquilibre de classe**

        Avec seulement **~0,14%** de fraudes, un modèle naïf
        qui prédit "normal" à chaque fois atteindrait 99,86%
        d'accuracy — sans jamais détecter une seule fraude.

        C'est pourquoi on optimise sur le **Recall** et l'**AUC-ROC**,
        pas l'accuracy.
        """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — ANALYSE DES FRAUDES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Analyse des Fraudes":
    st.title("Analyse des Fraudes")
    st.markdown("## Patterns comportementaux et temporels")
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Fraude par type de transaction")
        if 'Use_Chip' in df.columns:
            fraud_by_type = df.groupby('Use_Chip')['Is_Fraud'].mean() * 100
            fraud_by_type = fraud_by_type.reset_index()
            fraud_by_type.columns = ['Type', 'Taux de fraude (%)']
            fraud_by_type = fraud_by_type.sort_values('Taux de fraude (%)', ascending=False)
            fig_type = px.bar(
                fraud_by_type, x='Type', y='Taux de fraude (%)',
                color='Type',
                color_discrete_map={
                    "Online Transaction": RED,
                    "Chip Transaction":   GREEN,
                    "Swipe Transaction":  "#f59e0b"
                },
                text='Taux de fraude (%)'
            )
            fig_type.update_traces(texttemplate='%{text:.3f}%', textposition='outside')
            fig_type.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="white", showlegend=False, height=320,
                margin=dict(t=20, b=10)
            )
            st.plotly_chart(fig_type, use_container_width=True)

    with col2:
        st.subheader("Taux de fraude par heure")
        if 'Hour' in df.columns:
            fraud_by_hour = df.groupby('Hour')['Is_Fraud'].mean() * 100
            fig_hour = go.Figure()
            fig_hour.add_trace(go.Scatter(
                x=fraud_by_hour.index, y=fraud_by_hour.values,
                mode='lines+markers',
                line=dict(color=PURPLE, width=2),
                marker=dict(size=6),
                fill='tozeroy',
                fillcolor='rgba(99,102,241,0.15)'
            ))
            fig_hour.update_layout(
                xaxis_title="Heure", yaxis_title="Taux de fraude (%)",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="white", height=320, margin=dict(t=20, b=10),
                xaxis=dict(gridcolor="#374151"), yaxis=dict(gridcolor="#374151")
            )
            st.plotly_chart(fig_hour, use_container_width=True)

    st.divider()
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Distribution des montants — Fraude vs Normal")
        fig_box = go.Figure()
        fig_box.add_trace(go.Box(
            y=df_normal['Amount'].clip(upper=500), name="Normal",
            marker_color=PURPLE, boxmean=True
        ))
        fig_box.add_trace(go.Box(
            y=df_fraud['Amount'].clip(upper=500), name="Fraude",
            marker_color=RED, boxmean=True
        ))
        fig_box.update_layout(
            yaxis_title="Montant ($)",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="white", height=320, margin=dict(t=20, b=10),
            yaxis=dict(gridcolor="#374151"), showlegend=True
        )
        st.plotly_chart(fig_box, use_container_width=True)

    with col4:
        st.subheader("Top erreurs associées aux fraudes")
        if 'Errors' in df.columns:
            fraud_by_error = df.groupby('Errors')['Is_Fraud'].mean() * 100
            fraud_by_error = fraud_by_error[fraud_by_error > 0].sort_values(ascending=False).head(8).reset_index()
            fraud_by_error.columns = ['Erreur', 'Taux (%)']
            fig_err = px.bar(
                fraud_by_error, x='Taux (%)', y='Erreur', orientation='h',
                color='Taux (%)',
                color_continuous_scale=[[0, PURPLE], [1, RED]],
                text='Taux (%)'
            )
            fig_err.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
            fig_err.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="white", height=320, margin=dict(t=20, b=10),
                coloraxis_showscale=False, yaxis=dict(autorange='reversed')
            )
            st.plotly_chart(fig_err, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PROFIL MARCHAND
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Profil Marchand":
    st.title("Profil Marchand")
    st.markdown("## MCC, marchands et géographie des fraudes")
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top 10 MCC — taux de fraude")
        if 'MCC' in df.columns:
            mcc_fraud = df.groupby('MCC').agg(
                taux=('Is_Fraud', 'mean'),
                volume=('Is_Fraud', 'count')
            ).reset_index()
            mcc_fraud['taux'] = mcc_fraud['taux'] * 100
            mcc_fraud = mcc_fraud[mcc_fraud['volume'] > 50]
            top_mcc = mcc_fraud.nlargest(10, 'taux')
            top_mcc['MCC'] = top_mcc['MCC'].astype(str)
            fig_mcc = px.bar(
                top_mcc, x='taux', y='MCC', orientation='h',
                color='taux', color_continuous_scale=[[0, PURPLE], [1, RED]],
                text='taux', labels={'taux': 'Taux de fraude (%)'}
            )
            fig_mcc.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
            fig_mcc.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="white", height=360, margin=dict(t=20, b=10),
                coloraxis_showscale=False, yaxis=dict(autorange='reversed')
            )
            st.plotly_chart(fig_mcc, use_container_width=True)

    with col2:
        st.subheader("Top 10 marchands — volume de fraudes")
        if 'Merchant_Name' in df.columns:
            merchant_fraud = df[df['Is_Fraud'] == 1].groupby('Merchant_Name').size().reset_index(name='Fraudes')
            top_merchants = merchant_fraud.nlargest(10, 'Fraudes')
            fig_merch = px.bar(
                top_merchants, x='Fraudes', y='Merchant_Name', orientation='h',
                color='Fraudes', color_continuous_scale=[[0, PURPLE], [1, RED]],
                text='Fraudes'
            )
            fig_merch.update_traces(textposition='outside')
            fig_merch.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="white", height=360, margin=dict(t=20, b=10),
                coloraxis_showscale=False, yaxis=dict(autorange='reversed')
            )
            st.plotly_chart(fig_merch, use_container_width=True)

    st.divider()
    st.subheader("Fraudes par état (US)")
    if 'Merchant_State' in df.columns:
        state_fraud = df[df['Is_Fraud'] == 1].groupby('Merchant_State').size().reset_index(name='Fraudes')
        state_fraud = state_fraud[state_fraud['Merchant_State'].str.len() == 2]
        fig_map = px.choropleth(
            state_fraud, locations='Merchant_State', locationmode='USA-states',
            color='Fraudes',
            color_continuous_scale=[[0, "#1e1e2e"], [0.5, PURPLE], [1, RED]],
            scope='usa', labels={'Fraudes': 'Nombre de fraudes'}
        )
        fig_map.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", geo_bgcolor="rgba(0,0,0,0)",
            font_color="white", height=420, margin=dict(t=10, b=10, l=0, r=0),
            coloraxis_colorbar=dict(title="Fraudes")
        )
        st.plotly_chart(fig_map, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — PERFORMANCE MODELE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Performance Modèle":
    st.title("Performance Modèle")
    st.markdown("## LightGBM — évaluation et simulation de seuil")
    st.divider()

    # ── Slider seuil de décision ─────────────────────────────────────────────
    st.subheader("⚙️ Simulation du seuil de décision")
    st.caption("Ajuste le seuil selon l'appétit au risque de la banque")

    threshold = st.slider(
        "Seuil de classification (défaut bancaire : 0.5)",
        min_value=0.1, max_value=0.9, value=0.5, step=0.05,
        format="%.2f"
    )

    # Interpolation réaliste recall/precision en fonction du seuil
    # Basée sur les résultats réels du modèle LightGBM (AUC 0.9904)
    thresholds_ref  = np.array([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90])
    recall_ref      = np.array([0.99, 0.98, 0.97, 0.96, 0.94, 0.90, 0.84, 0.74, 0.58])
    precision_ref   = np.array([0.12, 0.18, 0.27, 0.40, 0.59, 0.72, 0.82, 0.89, 0.94])
    fpr_ref         = np.array([0.18, 0.12, 0.08, 0.05, 0.038,0.025,0.015,0.008,0.003])

    recall_t    = float(np.interp(threshold, thresholds_ref, recall_ref))
    precision_t = float(np.interp(threshold, thresholds_ref, precision_ref))
    fpr_t       = float(np.interp(threshold, thresholds_ref, fpr_ref))
    f1_t        = 2 * precision_t * recall_t / (precision_t + recall_t)

    # Confusion matrix dynamique
    n_fraud  = int(df['Is_Fraud'].sum()) if len(df) > 0 else 2339
    n_normal = len(df) - n_fraud
    TP = int(n_fraud  * recall_t)
    FN = n_fraud  - TP
    FP = int(n_normal * fpr_t)
    TN = n_normal - FP

    sm1, sm2, sm3, sm4 = st.columns(4)
    with sm1:
        st.metric("Recall",    f"{recall_t:.1%}")
    with sm2:
        st.metric("Precision", f"{precision_t:.1%}")
    with sm3:
        st.metric("F1-Score",  f"{f1_t:.3f}")
    with sm4:
        fraud_blocked = df_fraud['Amount'].sum() * recall_t if len(df_fraud) > 0 else 0
        st.metric("Montant fraude bloqué", f"${fraud_blocked:,.0f}".replace(",", " "))

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"Matrice de Confusion — seuil {threshold:.2f}")
        cm = np.array([[TN, FP], [FN, TP]])
        labels = ["Normal", "Fraude"]
        annotations = []
        for i in range(2):
            for j in range(2):
                annotations.append(dict(
                    x=j, y=i,
                    text=f"<b>{cm[i,j]:,}</b>",
                    showarrow=False,
                    font=dict(size=18, color="white")
                ))
        fig_cm = go.Figure(go.Heatmap(
            z=cm, x=labels, y=labels,
            colorscale=[[0, "#1e1e2e"], [1, PURPLE]],
            showscale=False
        ))
        fig_cm.update_layout(
            annotations=annotations,
            xaxis_title="Prédiction", yaxis_title="Réalité",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="white", height=320, margin=dict(t=20, b=10),
            xaxis=dict(side='bottom')
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with col2:
        st.subheader("Courbe ROC — AUC 0.9904")
        fpr_curve = np.array([0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
        tpr_curve = np.array([0, 0.78,  0.88,  0.92, 0.95, 0.97, 0.98, 0.99, 0.995, 1.0])
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr_curve, y=tpr_curve, mode='lines',
            name='LightGBM (AUC = 0.9904)',
            line=dict(color=PURPLE, width=2),
            fill='tozeroy', fillcolor='rgba(99,102,241,0.15)'
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode='lines',
            name='Modèle aléatoire',
            line=dict(color=GRAY, dash='dash', width=1)
        ))
        # Point du seuil courant
        fig_roc.add_trace(go.Scatter(
            x=[fpr_t], y=[recall_t], mode='markers',
            name=f'Seuil {threshold:.2f}',
            marker=dict(color=RED, size=12, symbol='circle')
        ))
        fig_roc.update_layout(
            xaxis_title="Taux de faux positifs",
            yaxis_title="Taux de vrais positifs",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="white", height=320, margin=dict(t=20, b=10),
            legend=dict(x=0.4, y=0.1),
            xaxis=dict(gridcolor="#374151"), yaxis=dict(gridcolor="#374151")
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    st.divider()

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Feature Importance")
        features   = [
            "MCC", "Merchant_Name", "Amount", "Year", "User",
            "Hour", "Month", "Day", "Use_Chip_Online",
            "Card", "Use_Chip_Chip", "Use_Chip_Swipe",
            "Errors_Bad_PIN", "Errors_Bad_CVV", "Errors_Insuff_Balance"
        ]
        importance = [2420,1710,1020,840,820,800,440,330,270,180,90,80,60,15,8]
        fi_df = pd.DataFrame({"Feature": features, "Importance": importance}).sort_values("Importance")
        fig_fi = px.bar(
            fi_df, x="Importance", y="Feature", orientation='h',
            color="Importance",
            color_continuous_scale=[[0, PURPLE], [1, "#818cf8"]]
        )
        fig_fi.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="white", height=420, margin=dict(t=20, b=10),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    with col4:
        st.subheader("Validation Croisée — 5 folds")
        folds      = [1, 2, 3, 4, 5]
        auc_scores = [0.9901, 0.9908, 0.9903, 0.9906, 0.9902]
        fig_cv = go.Figure()
        fig_cv.add_trace(go.Bar(
            x=[f"Fold {i}" for i in folds],
            y=auc_scores,
            marker_color=PURPLE,
            text=[f"{s:.4f}" for s in auc_scores],
            textposition='outside'
        ))
        fig_cv.add_hline(
            y=np.mean(auc_scores),
            line_dash="dash", line_color=GREEN,
            annotation_text=f"Moyenne : {np.mean(auc_scores):.4f}",
            annotation_position="top right",
            annotation_font_color=GREEN
        )
        fig_cv.update_layout(
            yaxis_title="AUC-ROC", yaxis_range=[0.985, 0.995],
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="white", height=420, margin=dict(t=20, b=10),
            xaxis=dict(gridcolor="#374151"), yaxis=dict(gridcolor="#374151")
        )
        st.plotly_chart(fig_cv, use_container_width=True)
        st.info(f"Écart-type AUC : {np.std(auc_scores):.4f} → stabilité excellente")

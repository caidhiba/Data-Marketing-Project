import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. PRÉPARATION DU DATASET DES RÉSULTATS ---
# On récupère les prédictions XGBoost sur le jeu de test
preds_xgb = predictions["3. XGBoost"]

# On crée un tableau récapitulatif pour les clients du jeu de test
df_test_results = X_test.copy()
df_test_results['customer_id'] = df_final.loc[X_test.index, 'customer_id']
df_test_results['CLV_Reelle'] = y_test
df_test_results['CLV_Predite'] = preds_xgb

# --- 2. CRÉATION DES DÉCILES DE CLV PRÉDITE ---
# On divise en 10 parts égales (1 = Pire 10%, 10 = Top 10%)
# rank(method='first') gère les égalités (ex: plusieurs clients prédits à 0€)
df_test_results['Decile_CLV'] = pd.qcut(df_test_results['CLV_Predite'].rank(method='first'), 10, labels=range(1, 11))


# --- 3. CARACTÉRISATION DES SEGMENTS (Lien avec RFM) ---
profil_deciles = df_test_results.groupby('Decile_CLV').agg(
    Nombre_Clients=('customer_id', 'count'),
    CLV_Predite_Moy=('CLV_Predite', 'mean'),
    CLV_Reelle_Moy=('CLV_Reelle', 'mean'),
    Recence_Moy=('recency', 'mean'),
    Frequence_Moy=('frequency', 'mean'),
    Montant_Histo_Moy=('monetary', 'mean')
).round(1)

print("--- PROFIL DES DÉCILES DE CLV PRÉDITE (10 = Top VIP) ---")
print(profil_deciles[['CLV_Predite_Moy', 'Recence_Moy', 'Frequence_Moy', 'Montant_Histo_Moy']])


# --- 4. CALCUL DU ROI SUR LE TOP 10% (Challenge CMO) ---
top_10_percent = df_test_results[df_test_results['Decile_CLV'] == 10]

# Hypothèses de la campagne
cout_par_client = 2.0  # 2€
lift_conversion = 0.15 # 15%

# Calculs financiers
nb_cibles = len(top_10_percent)
cout_campagne = nb_cibles * cout_par_client
valeur_totale_predite = top_10_percent['CLV_Predite'].sum()

# Le gain incrémental est de 15% sur la valeur totale que ces clients vont générer
gain_incremental = valeur_totale_predite * lift_conversion

# Calcul du ROI
profit_net = gain_incremental - cout_campagne
roi_pourcentage = (profit_net / cout_campagne) * 100

print("\n--- 💰 SIMULATION ROI : CAMPAGNE DE RÉTENTION TOP 10% ---")
print(f"Volume ciblé        : {nb_cibles} clients")
print(f"Coût de la campagne : {cout_campagne:,.2f} €")
print(f"Valeur sécurisée    : {gain_incremental:,.2f} € (Lift de 15% sur la CLV totale)")
print(f"Profit Net          : {profit_net:,.2f} €")
print(f"ROI de l'opération  : {roi_pourcentage:,.0f} %")


# --- 5. VISUALISATION : VALEUR PAR DÉCILE ---
plt.figure(figsize=(10, 6))

# Création d'une palette allant du bleu ciel (1) au violet foncé (10)
colors = sns.color_palette("light:purple", 10)
# On force le décile 10 en rose vif pour le faire ressortir !
colors[9] = 'hotpink' 

sns.barplot(
    x=profil_deciles.index, 
    y=profil_deciles['CLV_Predite_Moy'], 
    palette=colors,
    edgecolor='purple'
)

plt.title("CLV Moyenne Prédite par Décile\n(Le Top 10% concentre la majorité de la valeur)", 
          fontsize=15, color='purple', fontweight='bold', pad=20)
plt.xlabel("Déciles de CLV Prédite (10 = Les Champions)", fontsize=12, color='purple')
plt.ylabel("CLV Moyenne Prédite à 12 mois (€)", fontsize=12, color='purple')

ax = plt.gca()
ax.tick_params(colors='purple')
for spine in ax.spines.values():
    spine.set_color('purple')
plt.grid(True, axis='y', linestyle='--', alpha=0.3, color='purple')

plt.tight_layout()
plt.savefig('deciles_clv_roi_colors.png', bbox_inches='tight', dpi=150)
plt.close()

print("\nLe graphique de la répartition par décile a été sauvegardé sous 'deciles_clv_roi_colors.png'.")
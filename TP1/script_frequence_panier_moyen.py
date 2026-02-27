import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- 1. CHARGEMENT ET NETTOYAGE ---
df = pd.read_csv('customers.csv')
df = df.drop_duplicates()
df = df[df['country'] != 'Unspecified']

cols_to_float = ['n_orders', 'total_spent', 'avg_basket', 'recency_days', 'tenure_days']
for col in cols_to_float:
    df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)

df['first_purchase'] = pd.to_datetime(df['first_purchase'], errors='coerce')
df['last_purchase'] = pd.to_datetime(df['last_purchase'], errors='coerce')

cond_dates = (df['first_purchase'] <= df['last_purchase']) | df['first_purchase'].isna()
cond_recence = (df['recency_days'] <= df['tenure_days']) | df['recency_days'].isna()
cond_montants = (df['total_spent'] >= df['avg_basket']) | df['total_spent'].isna()

# Récupération des données propres (en enlevant les valeurs manquantes sur nos 2 colonnes cibles)
df_clean = df[cond_dates & cond_recence & cond_montants].dropna(subset=['n_orders', 'avg_basket']).copy()

# On filtre les valeurs extrêmes (au-delà du 95e percentile) pour un graphique lisible
q_orders = df_clean['n_orders'].quantile(0.95)
q_basket = df_clean['avg_basket'].quantile(0.95)
df_plot = df_clean[(df_clean['n_orders'] <= q_orders) & (df_clean['avg_basket'] <= q_basket)]

# Calcul du coefficient de corrélation
correlation = df_clean['n_orders'].corr(df_clean['avg_basket'])


# --- 2. CRÉATION DU GRAPHIQUE ---
plt.figure(figsize=(10, 6))

# Nuage de points en BLEU CIEL
plt.scatter(
    df_plot['n_orders'], df_plot['avg_basket'], 
    alpha=0.4, color='skyblue', edgecolors='none', s=20
)

# Ligne de tendance en ROSE
z = np.polyfit(df_plot['n_orders'], df_plot['avg_basket'], 1)
p = np.poly1d(z)
plt.plot(
    df_plot['n_orders'], p(df_plot['n_orders']), 
    color='hotpink', linestyle='--', linewidth=2.5, label="Tendance"
)

# Textes et Labels en VIOLET
plt.title(
    f"Fréquence vs Panier Moyen\nCoefficient de corrélation : {correlation:.2f}", 
    fontsize=14, pad=15, color='purple', fontweight='bold'
)
plt.xlabel("Fréquence (Nombre de commandes)", fontsize=12, color='purple')
plt.ylabel("Panier Moyen (€)", fontsize=12, color='purple')

# Grille et contours en VIOLET clair
plt.grid(True, linestyle='--', alpha=0.3, color='purple')

ax = plt.gca()
ax.tick_params(colors='purple')
for spine in ax.spines.values():
    spine.set_color('purple')

# Légende
plt.legend(loc='upper right', frameon=True, labelcolor='purple')

# Sauvegarde de l'image
plt.savefig('frequency_vs_basket_colors.png', bbox_inches='tight', dpi=150)
plt.close()

print(f"Graphique sauvegardé avec succès. Corrélation calculée : {correlation:.4f}")

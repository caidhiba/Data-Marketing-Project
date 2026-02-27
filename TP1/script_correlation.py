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

df_clean = df[cond_dates & cond_recence & cond_montants].dropna(subset=['recency_days', 'total_spent']).copy()

# On ne garde que les 95% des valeurs normales pour la lisibilité
q_spent = df_clean['total_spent'].quantile(0.95)
df_plot = df_clean[df_clean['total_spent'] <= q_spent]

# Calcul de la corrélation
correlation = df_clean['recency_days'].corr(df_clean['total_spent'])

# --- 2. CRÉATION DU GRAPHIQUE ---
plt.figure(figsize=(10, 6))

# Nuage de points en BLEU CIEL
plt.scatter(
    df_plot['recency_days'], df_plot['total_spent'], 
    alpha=0.4, color='skyblue', edgecolors='none', s=20
)

# Ligne de tendance en ROSE
z = np.polyfit(df_plot['recency_days'], df_plot['total_spent'], 1)
p = np.poly1d(z)
plt.plot(
    df_plot['recency_days'], p(df_plot['recency_days']), 
    color='hotpink', linestyle='--', linewidth=2.5, label="Tendance"
)

# Titres et labels en VIOLET
plt.title(
    f"Corrélation entre la Récence et le Montant Dépensé\nCoefficient : {correlation:.2f}", 
    fontsize=14, pad=15, color='purple', fontweight='bold'
)
plt.xlabel("Récence (Jours depuis le dernier achat)", fontsize=12, color='purple')
plt.ylabel("Montant Total Dépensé (€)", fontsize=12, color='purple')

# Grille et axes en VIOLET léger
plt.grid(True, linestyle='--', alpha=0.3, color='purple')

ax = plt.gca() # Récupère l'axe actuel
ax.tick_params(colors='purple') # Chiffres de l'axe X et Y
for spine in ax.spines.values():
    spine.set_color('purple') # Cadre autour du graphe

# Légende
plt.legend(loc='upper right', frameon=True, labelcolor='purple')

# Sauvegarde de l'image
plt.savefig('correlation_recency_spent_colors.png', bbox_inches='tight', dpi=150)
plt.close()

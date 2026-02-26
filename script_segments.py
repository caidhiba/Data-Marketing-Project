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

df_clean = df[cond_dates & cond_recence & cond_montants].dropna(subset=['recency_days', 'n_orders', 'total_spent']).copy()

# On ne garde que les 95% des valeurs normales pour éviter l'écrasement visuel
q_recency = df_clean['recency_days'].quantile(0.95)
q_orders = df_clean['n_orders'].quantile(0.95)
q_spent = df_clean['total_spent'].quantile(0.95)

df_plot = df_clean[
    (df_clean['recency_days'] <= q_recency) & 
    (df_clean['n_orders'] <= q_orders) & 
    (df_clean['total_spent'] <= q_spent)
]

# --- 2. CRÉATION DES GRAPHIQUES ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Recherche visuelle de segments naturels (Récence, Fréquence, Montant)", 
             fontsize=16, color='purple', fontweight='bold', y=1.05)

# Fonction pour appliquer vos couleurs à tous les sous-graphiques
def style_ax(ax):
    ax.tick_params(colors='purple')
    for spine in ax.spines.values():
        spine.set_color('purple')
    ax.grid(True, linestyle='--', alpha=0.3, color='purple')

# Graphique 1: Récence vs Fréquence
axes[0].scatter(df_plot['recency_days'], df_plot['n_orders'], alpha=0.3, color='skyblue', s=10)
axes[0].set_title("Récence vs Fréquence", color='purple')
axes[0].set_xlabel("Récence (Jours)", color='purple')
axes[0].set_ylabel("Fréquence (Commandes)", color='purple')
style_ax(axes[0])

# Graphique 2: Récence vs Montant
axes[1].scatter(df_plot['recency_days'], df_plot['total_spent'], alpha=0.3, color='skyblue', s=10)
axes[1].set_title("Récence vs Montant Total", color='purple')
axes[1].set_xlabel("Récence (Jours)", color='purple')
axes[1].set_ylabel("Montant Dépensé (€)", color='purple')
style_ax(axes[1])

# Graphique 3: Fréquence vs Montant
axes[2].scatter(df_plot['n_orders'], df_plot['total_spent'], alpha=0.3, color='skyblue', s=10)
axes[2].set_title("Fréquence vs Montant Total", color='purple')
axes[2].set_xlabel("Fréquence (Commandes)", color='purple')
axes[2].set_ylabel("Montant Dépensé (€)", color='purple')
style_ax(axes[2])

# Ajustement des espacements et sauvegarde
plt.tight_layout()
plt.savefig('segments_naturels.png', bbox_inches='tight', dpi=150)
plt.close()

print("Graphique croisé généré sous 'segments_naturels.png'.")
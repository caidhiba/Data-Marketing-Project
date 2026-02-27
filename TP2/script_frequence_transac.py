import pandas as pd
import matplotlib.pyplot as plt

# --- 1. CHARGEMENT ET PRÉPARATION ---
df = pd.read_csv('customers.csv')

# On s'assure que la colonne est bien numérique
df['n_orders'] = pd.to_numeric(df['n_orders'], errors='coerce')
df = df.dropna(subset=['n_orders']).copy()

# Pour que le graphique soit visuellement lisible, on limite l'affichage
# aux 95% des clients (on évite que les extrêmes n'écrasent l'histogramme)
q_orders = df['n_orders'].quantile(0.95)
df_plot = df[df['n_orders'] <= q_orders]


# --- 2. CRÉATION DU GRAPHIQUE (Histogramme) ---
plt.figure(figsize=(10, 6))

# Histogramme (barres) en BLEU CIEL avec des contours en VIOLET
plt.hist(
    df_plot['n_orders'], bins=40, 
    color='skyblue', edgecolor='purple', alpha=0.8
)

# Ajout des lignes pour la moyenne et la médiane en ROSE
moyenne_freq = df_plot['n_orders'].mean()
mediane_freq = df_plot['n_orders'].median()

plt.axvline(moyenne_freq, color='hotpink', linestyle='dashed', linewidth=2.5, 
            label=f'Moyenne ({moyenne_freq:.1f} cmds)')
plt.axvline(mediane_freq, color='hotpink', linestyle='dotted', linewidth=2.5, 
            label=f'Médiane ({mediane_freq:.1f} cmds)')

# Titres et labels en VIOLET
plt.title(
    "Distribution de la Fréquence (Sous le 95e percentile)", 
    fontsize=15, pad=15, color='purple', fontweight='bold'
)
plt.xlabel("Fréquence (Nombre de commandes / n_orders)", fontsize=12, color='purple')
plt.ylabel("Nombre de Clients", fontsize=12, color='purple')

# Grille et axes en VIOLET léger
plt.grid(True, linestyle='--', alpha=0.3, color='purple')

ax = plt.gca()
ax.tick_params(colors='purple')
for spine in ax.spines.values():
    spine.set_color('purple')

# Légende
plt.legend(loc='upper right', frameon=True, labelcolor='purple')

# Sauvegarde de l'image
plt.savefig('distribution_frequence_colors.png', bbox_inches='tight', dpi=150)
plt.close()

print("Graphique sauvegardé sous 'distribution_frequence_colors.png'.")

import pandas as pd
import matplotlib.pyplot as plt

# --- 1. CHARGEMENT ET PRÉPARATION ---
df = pd.read_csv('customers.csv')

# On s'assure que la colonne est bien numérique
df['avg_basket'] = pd.to_numeric(df['avg_basket'], errors='coerce')
df = df.dropna(subset=['avg_basket']).copy()

# Pour que le graphique soit visuellement lisible, on limite l'affichage
# aux 95% des clients (on évite que les très gros paniers n'écrasent l'histogramme)
q_basket = df['avg_basket'].quantile(0.95)
df_plot = df[df['avg_basket'] <= q_basket]


# --- 2. CRÉATION DU GRAPHIQUE (Histogramme) ---
plt.figure(figsize=(10, 6))

# Histogramme (barres) en BLEU CIEL avec des contours en VIOLET
plt.hist(
    df_plot['avg_basket'], bins=50, 
    color='skyblue', edgecolor='purple', alpha=0.8
)

# Ajout des lignes pour la moyenne et la médiane en ROSE
moyenne_panier = df_plot['avg_basket'].mean()
mediane_panier = df_plot['avg_basket'].median()

plt.axvline(moyenne_panier, color='hotpink', linestyle='dashed', linewidth=2.5, 
            label=f'Moyenne ({moyenne_panier:.1f} €)')
plt.axvline(mediane_panier, color='hotpink', linestyle='dotted', linewidth=2.5, 
            label=f'Médiane ({mediane_panier:.1f} €)')

# Titres et labels en VIOLET
plt.title(
    "Distribution du Panier Moyen (Sous le 95e percentile)", 
    fontsize=15, pad=15, color='purple', fontweight='bold'
)
plt.xlabel("Panier Moyen (€)", fontsize=12, color='purple')
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
plt.savefig('distribution_panier_moyen_colors.png', bbox_inches='tight', dpi=150)
plt.close()

print("Graphique sauvegardé sous 'distribution_panier_moyen_colors.png'.")
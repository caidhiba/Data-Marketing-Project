import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# --- 1. CHARGEMENT ET PRÉPARATION ---
df = pd.read_csv('customers.csv')

# On sélectionne uniquement les colonnes numériques pertinentes pour la corrélation
colonnes_cibles = ['recency_days', 'n_orders', 'total_spent', 'avg_basket', 'tenure_days']
for col in colonnes_cibles:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# On supprime les valeurs nulles pour ne pas fausser le calcul
df_clean = df.dropna(subset=colonnes_cibles).copy()

# On renomme les colonnes pour que l'affichage sur la Heatmap soit propre et en français
df_clean.rename(columns={
    'recency_days': 'Récence',
    'n_orders': 'Fréquence',
    'total_spent': 'Montant Total',
    'avg_basket': 'Panier Moyen',
    'tenure_days': 'Ancienneté'
}, inplace=True)


# --- 2. CALCUL DE LA MATRICE DE CORRÉLATION ---
corr = df_clean[['Récence', 'Fréquence', 'Montant Total', 'Panier Moyen', 'Ancienneté']].corr()


# --- 3. CRÉATION DU GRAPHIQUE (Heatmap) ---
plt.figure(figsize=(10, 8))

# Création de votre palette de couleurs sur-mesure : Bleu Ciel -> Blanc -> Rose -> Violet
colors = ["skyblue", "white", "hotpink", "purple"]
cmap_perso = LinearSegmentedColormap.from_list("custom_cmap", colors)

# Paramétrage et tracé de la Heatmap via Seaborn
ax = sns.heatmap(
    corr, 
    annot=True,              # Affiche les valeurs chiffrées dans les cases
    fmt=".2f",               # Format : 2 chiffres après la virgule
    cmap=cmap_perso,         # Application de votre palette
    vmin=-1, vmax=1,         # Échelle stricte de corrélation (de -1 à 1)
    linewidths=1,            # Épaisseur des bordures de cases
    linecolor='purple',      # Couleur des bordures de cases
    cbar_kws={'label': 'Niveau de Corrélation'} # Légende de la barre latérale
)

# --- 4. HABILLAGE VIOLET ---
plt.title(
    "Matrice de Corrélation des Comportements Clients", 
    fontsize=16, color='purple', fontweight='bold', pad=20
)

# Textes des axes (noms des variables)
plt.xticks(color='purple', rotation=45, ha='right', fontsize=11)
plt.yticks(color='purple', rotation=0, fontsize=11)

# Changement de couleur de la barre de légende latérale (Colorbar)
cbar = ax.collections[0].colorbar
cbar.ax.yaxis.set_tick_params(colors='purple')
cbar.set_label('Niveau de Corrélation', color='purple', size=12, labelpad=15)

# Sauvegarde de l'image
plt.savefig('heatmap_correlation_colors.png', bbox_inches='tight', dpi=150)
plt.close()

print("La Heatmap de corrélation a été générée avec succès !")
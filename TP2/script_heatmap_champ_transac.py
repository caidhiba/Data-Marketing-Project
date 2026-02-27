import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# --- 1. CHARGEMENT ET PRÉPARATION DES DONNÉES ---
# On recharge le fichier depuis le début pour s'assurer que df_clean existe
df_clean = pd.read_csv('customers.csv')

# Sécurité : Si vos colonnes sont encore en anglais dans le CSV, on les renomme en français
colonnes_a_renommer = {
    'recency_days': 'Récence',
    'n_orders': 'Fréquence',
    'total_spent': 'Montant Total'
}
df_clean = df_clean.rename(columns=colonnes_a_renommer)

# On s'assure que les 3 colonnes nécessaires sont bien numériques et sans valeurs vides
for col in ['Récence', 'Fréquence', 'Montant Total']:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

df_clean = df_clean.dropna(subset=['Récence', 'Fréquence', 'Montant Total'])


# --- 2. CALCUL DES SCORES RFM ---
# Récence : 5 est le meilleur (le plus récent, donc le plus petit nombre de jours)
df_clean['R_Score'] = pd.qcut(df_clean['Récence'].rank(method='first'), 5, labels=[5, 4, 3, 2, 1])

# Fréquence : 5 est le meilleur (le plus fidèle, donc le plus grand nombre de commandes)
df_clean['F_Score'] = pd.qcut(df_clean['Fréquence'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])


# --- 3. PRÉPARATION DE LA MATRICE (TABLEAU CROISÉ) ---
heatmap_data = df_clean.pivot_table(
    index='R_Score', 
    columns='F_Score', 
    values='Montant Total', 
    aggfunc='mean'
)

# Inversion de l'axe Y : on veut que le score Récence = 5 soit tout en haut
heatmap_data = heatmap_data.sort_index(ascending=False)


# --- 4. CRÉATION DE LA PALETTE (Bleu Ciel -> Rose -> Violet) ---
colors = ["skyblue", "hotpink", "purple"]
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)


# --- 5. DESSIN ET HABILLAGE DE LA HEATMAP ---
plt.figure(figsize=(10, 8))

ax = sns.heatmap(
    heatmap_data, 
    annot=True,          # Affiche les valeurs dans les cases
    fmt=".0f",           # Format sans décimales
    cmap=custom_cmap,    # Utilisation de la palette sur-mesure
    linewidths=.5,       # Sépare légèrement les cases avec des lignes
    cbar_kws={'label': 'Montant Moyen (€)'}
)

# Personnalisation des textes en VIOLET
plt.title(
    "Matrice RFM : Montant Moyen par Récence et Fréquence", 
    fontsize=15, pad=15, color='purple', fontweight='bold'
)
plt.ylabel("Score Récence (5 = Très Récent)", fontsize=12, color='purple')
plt.xlabel("Score Fréquence (5 = Très Fidèle)", fontsize=12, color='purple')

# Ajustement des couleurs des axes
ax = plt.gca()
ax.tick_params(colors='purple')
for spine in ax.spines.values():
    spine.set_color('purple')

# Ajustement de la couleur de la barre de légende latérale
cbar = ax.collections[0].colorbar
cbar.ax.yaxis.set_tick_params(colors='purple')
cbar.set_label('Montant Moyen (€)', color='purple', size=12, labelpad=15)

# Sauvegarde de l'image
plt.savefig('rfm_heatmap_montant_colors.png', bbox_inches='tight', dpi=150)
plt.close()

print("La matrice Heatmap RFM a été générée et sauvegardée avec succès !")

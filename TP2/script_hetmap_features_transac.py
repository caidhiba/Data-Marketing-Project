import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from matplotlib.colors import LinearSegmentedColormap

# --- 1. CHARGEMENT ET CLUSTERING ---
df = pd.read_csv('customers.csv')

# Renommage en français et nettoyage
cols_rfm = {
    'recency_days': 'Récence',
    'n_orders': 'Fréquence',
    'total_spent': 'Montant Total'
}
df_clean = df.rename(columns=cols_rfm).dropna(subset=['Récence', 'Fréquence', 'Montant Total'])

# Recalcul des clusters (si vous ne les aviez pas sauvegardés)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df_clean[['Récence', 'Fréquence', 'Montant Total']])
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df_clean['Cluster'] = kmeans.fit_predict(data_scaled)

# Tri des clusters du plus petit montant (0) au plus grand (3)
cluster_order = df_clean.groupby('Cluster')['Montant Total'].mean().sort_values().index
mapping = {ancien: nouveau for nouveau, ancien in enumerate(cluster_order)}
df_clean['Cluster'] = df_clean['Cluster'].map(mapping)

noms_clusters = {0: 'C0 (Occasionnels)', 1: 'C1 (Réguliers)', 2: 'C2 (Fidèles)', 3: 'C3 (Champions)'}
df_clean['Nom_Cluster'] = df_clean['Cluster'].map(noms_clusters)


# --- 2. PRÉPARATION DES MOYENNES POUR LA HEATMAP ---
# A. On calcule le vrai chiffre moyen pour les textes (ex: 4500 €)
df_moyennes = df_clean.groupby('Nom_Cluster')[['Récence', 'Fréquence', 'Montant Total']].mean()

# B. On "met à l'échelle" (0 à 1) chaque colonne individuellement pour l'intensité des couleurs
scaler_minmax = MinMaxScaler()
df_couleurs = pd.DataFrame(
    scaler_minmax.fit_transform(df_moyennes), 
    columns=df_moyennes.columns, 
    index=df_moyennes.index
)


# --- 3. DESSIN DE LA HEATMAP ---
plt.figure(figsize=(10, 6))

# Notre palette (Bleu ciel = plus bas de la colonne, Violet = plus haut de la colonne)
colors = ["skyblue", "white", "hotpink", "purple"]
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

ax = sns.heatmap(
    df_couleurs,            # Les données utilisées pour colorier la case
    annot=df_moyennes,      # Les données utilisées pour ECRIRE le texte dans la case
    fmt=".1f",              # Affichage avec 1 décimale
    cmap=custom_cmap, 
    linewidths=1,           # Séparation des cases
    linecolor='purple',
    cbar=False              # On masque la légende latérale (qui irait de 0 à 1 et serait confuse)
)


# --- 4. HABILLAGE VIOLET ---
plt.title("Moyenne des Features RFM par Segment (Cluster)", 
          color='purple', fontsize=16, fontweight='bold', pad=20)
plt.ylabel("Segments (Clusters)", color='purple', fontsize=13)

# Axe des abscisses (Features) mis à plat (rotation=0) pour la lisibilité
plt.xticks(rotation=0, color='purple', fontsize=11)
plt.yticks(color='purple', fontsize=11)

# Coloration des bordures
for spine in ax.spines.values(): 
    spine.set_color('purple')

# Sauvegarde de l'image
plt.tight_layout()
plt.savefig('heatmap_features_clusters_colors.png', bbox_inches='tight', dpi=150)
plt.close()

print("La heatmap des moyennes RFM a été sauvegardée avec succès !")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# --- 1. CHARGEMENT ET PRÉPARATION ---
df = pd.read_csv('customers.csv')

colonnes_a_renommer = {
    'recency_days': 'Récence',
    'n_orders': 'Fréquence',
    'total_spent': 'Montant Total'
}
df_clean = df.rename(columns=colonnes_a_renommer)

# On s'assure que les colonnes sont numériques et on retire les vides
cols_rfm = ['Récence', 'Fréquence', 'Montant Total']
for col in cols_rfm:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

df_clean = df_clean.dropna(subset=cols_rfm).copy()


# --- 2. CALCUL DES CLUSTERS (K-MEANS) ---
# Standardisation des données (obligatoire)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df_clean[cols_rfm])

# Création des 4 clusters
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df_clean['Cluster'] = kmeans.fit_predict(data_scaled)

# Tri des clusters du plus faible (0) au meilleur (3) basé sur le montant total moyen
cluster_order = df_clean.groupby('Cluster')['Montant Total'].mean().sort_values().index
mapping = {ancien_id: nouvel_id for nouvel_id, ancien_id in enumerate(cluster_order)}
df_clean['Cluster'] = df_clean['Cluster'].map(mapping)

noms_clusters = {0: 'C0 (Occasionnels)', 1: 'C1 (Réguliers)', 2: 'C2 (Fidèles)', 3: 'C3 (Champions)'}
df_clean['Nom_Cluster'] = df_clean['Cluster'].map(noms_clusters)
order_clusters = [noms_clusters[i] for i in range(4)]


# --- 3. CRÉATION DU NUAGE DE POINTS (SCATTER PLOT) ---
plt.figure(figsize=(12, 8))

# Votre palette de couleurs par cluster
palette_clusters = ['skyblue', 'hotpink', 'purple', '#FFD700']

# Tracé du graphique
sns.scatterplot(
    x='Récence', 
    y='Montant Total', 
    hue='Nom_Cluster', 
    hue_order=order_clusters,
    palette=palette_clusters, 
    data=df_clean, 
    alpha=0.6,          # Transparence 
    edgecolor='w',      # Bordure blanche
    s=40                # Taille des points
)

# --- 4. PERSONNALISATION EN VIOLET ---
plt.title("Relation entre la Récence et le Montant Dépensé (par Segment)", 
          color='purple', fontsize=16, fontweight='bold', pad=20)
plt.ylabel("Montant Total Dépensé (€)", color='purple', fontsize=13)

# 💡 INVERSION DE L'AXE X (Meilleure pratique marketing RFM)
plt.gca().invert_xaxis()
plt.xlabel("Récence (Jours depuis le dernier achat) ➔ Plus récent", color='purple', fontsize=13)

# Habillage des axes, ticks et grille
ax = plt.gca()
ax.tick_params(colors='purple')
for spine in ax.spines.values(): 
    spine.set_color('purple')
ax.grid(True, linestyle='--', alpha=0.3, color='purple')

# Légende
plt.legend(
    title='Segments (Clusters)', 
    bbox_to_anchor=(1.02, 1), # Place la légende à l'extérieur
    loc='upper left', 
    labelcolor='purple', 
    title_fontsize='11'
)

# Ajustement des marges et Sauvegarde
plt.tight_layout()
plt.savefig('scatter_recence_montant_colors.png', bbox_inches='tight', dpi=150)
plt.close()

print("Succès ! Les clusters ont été calculés et le Scatter Plot a été sauvegardé sous 'scatter_recence_montant_colors.png'.")
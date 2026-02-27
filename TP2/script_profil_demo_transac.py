import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# --- 1. CHARGEMENT ET PRÉPARATION DES DONNÉES ---
df = pd.read_csv('customers.csv')

# Uniformisation du nom des colonnes (si vous les avez conservées en anglais)
colonnes_a_renommer = {
    'country': 'Pays',
    'tenure_days': 'Ancienneté',
    'recency_days': 'Récence',
    'n_orders': 'Fréquence',
    'total_spent': 'Montant Total'
}
df_clean = df.rename(columns=colonnes_a_renommer)

# S'assurer que les colonnes nécessaires sont présentes
cols_rfm = ['Récence', 'Fréquence', 'Montant Total']
for col in cols_rfm + ['Ancienneté']:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

# Supprimer les lignes vides pour éviter que le calcul ne plante
df_clean = df_clean.dropna(subset=cols_rfm + ['Ancienneté', 'Pays']).copy()


# --- 2. CRÉATION DES CLUSTERS (Si non existants) ---
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df_clean[cols_rfm])

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df_clean['Cluster'] = kmeans.fit_predict(data_scaled)

# Tri des clusters par Montant Moyen pour leur donner un nom logique (0 = Faible, 3 = Champion)
cluster_order = df_clean.groupby('Cluster')['Montant Total'].mean().sort_values().index
mapping = {ancien_id: nouvel_id for nouvel_id, ancien_id in enumerate(cluster_order)}
df_clean['Cluster'] = df_clean['Cluster'].map(mapping)

noms_clusters = {0: 'C0 (Occasionnels)', 1: 'C1 (Réguliers)', 2: 'C2 (Fidèles)', 3: 'C3 (Champions)'}
df_clean['Nom_Cluster'] = df_clean['Cluster'].map(noms_clusters)
order_clusters = [noms_clusters[i] for i in range(4)]


# --- 3. CRÉATION DES GRAPHIQUES ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Votre palette de couleurs de segments
palette_clusters = ['skyblue', 'hotpink', 'purple', '#FFD700']

# -- Graphique 1 : Ancienneté par Cluster (Boxplot) --
sns.boxplot(
    x='Nom_Cluster', y='Ancienneté', data=df_clean, 
    palette=palette_clusters, ax=ax1, order=order_clusters, width=0.6,
    boxprops=dict(alpha=0.8, edgecolor='purple'),
    medianprops=dict(color='white', linewidth=2),
    flierprops=dict(marker='o', color='purple', markersize=3, alpha=0.3)
)
ax1.set_title("1. Ancienneté (Tenure) selon les Clusters", color='purple', fontsize=14, fontweight='bold')
ax1.set_xlabel("Segments (Clusters)", color='purple')
ax1.set_ylabel("Ancienneté (Jours depuis le 1er achat)", color='purple')


# -- Graphique 2 : Répartition Géographique (Pays) --
# Si vous avez trop de pays, on regroupe les petits pays dans "Autres" pour la lisibilité
top_pays = df_clean['Pays'].value_counts().nlargest(4).index # Les 4 pays principaux
df_clean['Pays_Abrege'] = df_clean['Pays'].apply(lambda x: x if x in top_pays else 'Autres')

cross_tab = pd.crosstab(df_clean['Nom_Cluster'], df_clean['Pays_Abrege'], normalize='index') * 100

# Couleurs pour les Pays : On décline votre charte
colors_pays = ['skyblue', 'hotpink', '#FFD700', 'purple', 'lightgreen']

# Tracer le bar chart 100% empilé
cross_tab.plot(kind='bar', stacked=True, ax=ax2, color=colors_pays[:len(cross_tab.columns)], edgecolor='white', alpha=0.9)

ax2.set_title("2. Répartition Géographique (Pays) par Cluster", color='purple', fontsize=14, fontweight='bold')
ax2.set_xlabel("Segments (Clusters)", color='purple')
ax2.set_ylabel("Pourcentage de clients (%)", color='purple')
ax2.legend(title='Pays', bbox_to_anchor=(1.05, 1), loc='upper left', labelcolor='purple')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)


# -- Habillage global (Textes et axes en VIOLET) --
for ax in [ax1, ax2]:
    ax.grid(True, linestyle='--', alpha=0.3, color='purple', axis='y')
    ax.tick_params(colors='purple')
    for spine in ax.spines.values(): 
        spine.set_color('purple')

plt.suptitle("Profil Démographique des Segments (Ancienneté & Pays)", color='purple', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()

# Sauvegarde de l'image
plt.savefig('profil_demographique_colors.png', bbox_inches='tight', dpi=150)
plt.close()

print("Le tableau de bord démographique (Ancienneté et Pays) a été sauvegardé avec succès !")

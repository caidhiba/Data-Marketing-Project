import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Vérification pour UMAP, sinon on utilise t-SNE en remplaçant
try:
    import umap.umap_ as umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    from sklearn.manifold import TSNE

# --- 1. CHARGEMENT ET PRÉPARATION ---
df = pd.read_csv('customers.csv')

# On s'assure que les colonnes existent et sont numériques
cols_rfm = ['recency_days', 'n_orders', 'total_spent']
for col in cols_rfm:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# On supprime les lignes avec des valeurs manquantes
df_clean = df.dropna(subset=cols_rfm).copy()


# --- 2. CLUSTERING K-MEANS (Création de la colonne 'Cluster') ---
# Étape 2.1 : Standardiser les données (OBLIGATOIRE avant un K-Means ou une PCA)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df_clean[cols_rfm])

# Étape 2.2 : Application de l'algorithme K-Means (On choisit 4 clusters)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df_clean['Cluster'] = kmeans.fit_predict(data_scaled)


# --- 3. RÉDUCTION DE DIMENSION (PCA & UMAP) ---
# 3.1 PCA (Rapide, pour la structure globale)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(data_scaled)
df_clean['PCA1'] = pca_result[:, 0]
df_clean['PCA2'] = pca_result[:, 1]

# 3.2 UMAP ou t-SNE (Pour la belle visualisation des îlots)
if HAS_UMAP:
    print("Calcul UMAP en cours (cela peut prendre quelques secondes)...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    umap_result = reducer.fit_transform(data_scaled)
    df_clean['UMAP1'] = umap_result[:, 0]
    df_clean['UMAP2'] = umap_result[:, 1]
    dim1, dim2, title2 = 'UMAP1', 'UMAP2', "2. UMAP (Structure Locale)"
else:
    print("UMAP non détecté. Calcul t-SNE en cours (cela peut prendre un peu de temps)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_result = tsne.fit_transform(data_scaled)
    df_clean['TSNE1'] = tsne_result[:, 0]
    df_clean['TSNE2'] = tsne_result[:, 1]
    dim1, dim2, title2 = 'TSNE1', 'TSNE2', "2. t-SNE (Substitut à UMAP)"


# --- 4. CRÉATION DES GRAPHIQUES ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# La palette de couleurs pour 4 clusters : Bleu, Rose, Violet, et Or
palette_clusters = ['skyblue', 'hotpink', 'purple', '#FFD700']

# Graphique 1 : PCA
sns.scatterplot(
    x='PCA1', y='PCA2', hue='Cluster', 
    palette=palette_clusters, 
    data=df_clean, alpha=0.7, edgecolor='w', s=50, ax=ax1
)
ax1.set_title("1. PCA (Aperçu Rapide - Structure Globale)", color='purple', fontsize=14, fontweight='bold')
ax1.set_xlabel("Composante Principale 1", color='purple')
ax1.set_ylabel("Composante Principale 2", color='purple')

# Graphique 2 : UMAP (ou t-SNE)
sns.scatterplot(
    x=dim1, y=dim2, hue='Cluster', 
    palette=palette_clusters, 
    data=df_clean, alpha=0.7, edgecolor='w', s=50, ax=ax2
)
ax2.set_title(title2, color='purple', fontsize=14, fontweight='bold')
ax2.set_xlabel(f"Dimension {dim1}", color='purple')
ax2.set_ylabel(f"Dimension {dim2}", color='purple')

# Habillage global (Textes et axes en VIOLET)
for ax in [ax1, ax2]:
    ax.tick_params(colors='purple')
    for spine in ax.spines.values(): 
        spine.set_color('purple')
    ax.legend(title='Cluster', title_fontsize='11')

plt.suptitle("Projection 2D des Clusters RFM (K-Means)", color='purple', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()

# Sauvegarde de l'image
plt.savefig('pca_umap_clusters_colors.png', bbox_inches='tight', dpi=150)
plt.close()

print("Succès ! Les clusters ont été créés et le graphique a été sauvegardé sous 'pca_umap_clusters_colors.png'.")

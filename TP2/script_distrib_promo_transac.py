import pandas as pd
import matplotlib.pyplot as plt

# --- 1. CHARGEMENT ET PRÉPARATION ---
# Chargez votre fichier complet de transactions
df_trans = pd.read_csv('transactions.csv') # Modifiez le nom du fichier si nécessaire

# On s'assure que les dates et les identifiants sont au bon format
df_trans['customer_id'] = pd.to_numeric(df_trans['customer_id'], errors='coerce')
df_trans['invoice_date'] = pd.to_datetime(df_trans['invoice_date'], errors='coerce')
df_trans = df_trans.dropna(subset=['customer_id', 'invoice_date']).copy()


# --- 2. DÉFINITION DES PROMOTIONS ET CALCUL ---
# ÉTAPE CLÉ : On définit ici ce qu'est un achat en promotion. 
# Exemple : Achats effectués en Novembre (11) ou Décembre (12)
df_trans['is_promo'] = df_trans['invoice_date'].dt.month.isin([11, 12])

# On compte le nombre total d'articles achetés par client
achats_totaux = df_trans.groupby('customer_id')['invoice_id'].count().reset_index()
achats_totaux.rename(columns={'invoice_id': 'total_items'}, inplace=True)

# On compte le nombre d'articles achetés EN PROMOTION par client
achats_promo = df_trans[df_trans['is_promo']].groupby('customer_id')['invoice_id'].count().reset_index()
achats_promo.rename(columns={'invoice_id': 'promo_items'}, inplace=True)

# On fusionne les deux tableaux pour calculer le pourcentage
df_proportion = pd.merge(achats_totaux, achats_promo, on='customer_id', how='left')
df_proportion['promo_items'] = df_proportion['promo_items'].fillna(0) # Ceux qui n'ont rien acheté en promo valent 0

# Calcul du pourcentage d'achats en promotion pour chaque client
df_proportion['promo_percentage'] = (df_proportion['promo_items'] / df_proportion['total_items']) * 100


# --- 3. CRÉATION DU GRAPHIQUE (Histogramme) ---
plt.figure(figsize=(10, 6))

# Histogramme en BLEU CIEL avec des contours en VIOLET
plt.hist(
    df_proportion['promo_percentage'], bins=20, 
    color='skyblue', edgecolor='purple', alpha=0.8
)

# Moyenne et Médiane en ROSE
moyenne_prop = df_proportion['promo_percentage'].mean()
mediane_prop = df_proportion['promo_percentage'].median()

plt.axvline(moyenne_prop, color='hotpink', linestyle='dashed', linewidth=2.5, 
            label=f'Moyenne ({moyenne_prop:.1f} %)')
plt.axvline(mediane_prop, color='hotpink', linestyle='dotted', linewidth=2.5, 
            label=f'Médiane ({mediane_prop:.1f} %)')

# Titres et labels en VIOLET
plt.title(
    "Proportion des achats réalisés en période de promotion", 
    fontsize=15, pad=15, color='purple', fontweight='bold'
)
plt.xlabel("Part des achats en promotion (%)", fontsize=12, color='purple')
plt.ylabel("Nombre de Clients", fontsize=12, color='purple')

# Grille et axes en VIOLET clair
plt.grid(True, linestyle='--', alpha=0.3, color='purple')

ax = plt.gca()
ax.tick_params(colors='purple')
for spine in ax.spines.values():
    spine.set_color('purple')

# Légende
plt.legend(loc='upper right', frameon=True, labelcolor='purple')

# Sauvegarde de l'image
plt.savefig('distribution_promotions_colors.png', bbox_inches='tight', dpi=150)
plt.close()

print("Le graphique a été généré sous le nom 'distribution_promotions_colors.png'.")

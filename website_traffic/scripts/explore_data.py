"""
Script d'exploration du dataset Website Traffic
Exécute : python scripts/explore_data.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Créer le dossier dataset s'il n'existe pas
os.makedirs('dataset', exist_ok=True)

print("="*60)
print("ANALYSE DU DATASET : WEBSITE TRAFFIC")
print("="*60)

# 1. Charger les données (assure-toi que le fichier est au bon endroit)
try:
    df = pd.read_csv('dataset/website_wata.csv')
    print("✅ Dataset chargé avec succès!")
except FileNotFoundError:
    print("❌ Fichier non trouvé. Place 'website_traffic.csv' dans le dossier 'dataset/'")
    exit()

# 2. Informations générales
print(f"\n📊 TAILLE DU DATASET : {df.shape[0]} lignes, {df.shape[1]} colonnes")

print("\n📋 COLONNES DISPONIBLES :")
for i, col in enumerate(df.columns, 1):
    dtype = df[col].dtype
    print(f"  {i:2d}. {col:20} ({dtype}) - Exemple: {df[col].iloc[0]}")

print("\n🔍 APERÇU DES DONNÉES (5 premières lignes) :")
print(df.head())

print("\n📈 STATISTIQUES DESCRIPTIVES :")
print(df.describe())

print("\n❓ VALEURS MANQUANTES :")
missing = df.isnull().sum()
for col in df.columns:
    if missing[col] > 0:
        print(f"  {col}: {missing[col]} valeurs manquantes ({missing[col]/len(df)*100:.1f}%)")
    else:
        print(f"  {col}: Aucune valeur manquante ✅")

print("\n🎯 VALEURS UNIQUES PAR COLONNE :")
for col in df.columns:
    unique_count = df[col].nunique()
    if unique_count < 10:  # Si peu de valeurs uniques, les afficher
        print(f"  {col}: {unique_count} valeurs → {df[col].unique()}")
    else:
        print(f"  {col}: {unique_count} valeurs uniques")

print("\n🌐 TRAFFIC SOURCE - DISTRIBUTION :")
if 'Traffic Source' in df.columns:
    source_counts = df['Traffic Source'].value_counts()
    print(source_counts)

# 3. Corrélations (si toutes les colonnes sont numériques)
print("\n📊 MATRICE DE CORRÉLATION (colonnes numériques) :")
numeric_df = df.select_dtypes(include=[np.number])
if not numeric_df.empty:
    corr_matrix = numeric_df.corr()
    print(corr_matrix)
    
    # Visualisation (optionnel)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Matrice de corrélation')
    plt.tight_layout()
    plt.savefig('dataset/correlation_heatmap.png', dpi=100)
    print("\n✅ Heatmap sauvegardée : 'dataset/correlation_heatmap.png'")

# 4. Sauvegarder un échantillon pour référence
sample_df = df.head(50)
sample_df.to_csv('dataset/sample_data.csv', index=False)
print("\n💾 Échantillon sauvegardé : 'dataset/sample_data.csv' (50 premières lignes)")

print("\n" + "="*60)
print("ANALYSE TERMINÉE - RECOMMANDATIONS :")
print("="*60)

print("\n🎯 CHOIX POSSIBLES POUR LA PRÉDICTION :")
print("1. Conversion Rate (recommandé) - Métrique business importante")
print("2. Bounce Rate - Comprendre l'engagement")
print("3. Page Views - Mesurer l'intérêt")
print("4. Session Duration - Temps d'engagement")

print("\n🔧 VARIABLES D'ENTRÉE (features) potentielles :")
print("  - Traffic Source (à encoder)")
print("  - Previous Visits")
print("  - Time on Page")
print("  - Page Views")
print("  - Session Duration")

print("\n⚠️  ACTIONS REQUISES :")
print("  1. Convertir 'Traffic Source' en variables numériques (Label Encoding)")
print("  2. Normaliser les données si nécessaire")
print("  3. Diviser en train/test (80%/20%)")
print("  4. Choisir un modèle : RandomForestRegressor ou XGBoost")

print("\n🚀 Prochaine étape : Exécuter 'python scripts/train_model.py'")
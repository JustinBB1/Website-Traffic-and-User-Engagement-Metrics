"""
Script d'entraînement du modèle pour prédire la Conversion Rate
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

print("="*60)
print("ENTRAÎNEMENT DU MODÈLE DE PRÉDICTION")
print("="*60)

# 1. Charger les données
df = pd.read_csv('dataset/website_wata.csv')
print(f"✅ Données chargées : {df.shape[0]} lignes")

# 2. DÉFINITION DES VARIABLES
# À MODIFIER SI TU VEUX PRÉDIRE AUTRE CHOSE
TARGET = 'Conversion Rate'  # Variable à prédire
FEATURES = ['Page Views', 'Session Duration', 'Bounce Rate', 
            'Traffic Source', 'Time on Page', 'Previous Visits']

print(f"\n🎯 Variable cible : {TARGET}")
print(f"📊 Variables d'entrée : {FEATURES}")

# 3. Préparation des données
# Encoder la variable catégorielle 'Traffic Source'
print("\n🔧 Encodage des variables catégorielles...")
le = LabelEncoder()
df['Traffic Source_encoded'] = le.fit_transform(df['Traffic Source'])

# Sauvegarder l'encodeur pour plus tard
os.makedirs('models/encoders', exist_ok=True)
joblib.dump(le, 'models/encoders/traffic_source_encoder.pkl')
print("✅ Encodeur sauvegardé : 'models/encoders/traffic_source_encoder.pkl'")

# Remplacer la colonne originale par la version encodée
FEATURES_ENCODED = ['Page Views', 'Session Duration', 'Bounce Rate',
                    'Traffic Source_encoded', 'Time on Page', 'Previous Visits']

# 4. Séparer features (X) et target (y)
X = df[FEATURES_ENCODED]
y = df[TARGET]

print(f"\n📐 Dimensions :")
print(f"  X (features) : {X.shape}")
print(f"  y (target)   : {y.shape}")

# 5. Diviser en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n🔀 Division train/test :")
print(f"  Train : {X_train.shape[0]} échantillons")
print(f"  Test  : {X_test.shape[0]} échantillons")

# 6. Entraîner le modèle
print("\n🤖 Entraînement du modèle RandomForest...")
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1  # Utiliser tous les cœurs CPU
)
model.fit(X_train, y_train)
print("✅ Modèle entraîné !")

# 7. Évaluation
print("\n📈 ÉVALUATION DU MODÈLE :")
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"  MAE  : {mae:.4f}")
print(f"  MSE  : {mse:.4f}")
print(f"  RMSE : {rmse:.4f}")
print(f"  R²   : {r2:.4f}")

# 8. Importance des features
print("\n🏆 IMPORTANCE DES VARIABLES :")
feature_importance = pd.DataFrame({
    'feature': FEATURES_ENCODED,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.to_string(index=False))

# 9. Sauvegarder le modèle
os.makedirs('models', exist_ok=True)
model_path = 'models/traffic_model.pkl'
joblib.dump(model, model_path)
print(f"\n💾 Modèle sauvegardé : '{model_path}'")

# 10. Sauvegarder les métadonnées
metadata = {
    'target': TARGET,
    'features': FEATURES,
    'features_encoded': FEATURES_ENCODED,
    'performance': {
        'mae': float(mae),
        'mse': float(mse),
        'rmse': float(rmse),
        'r2': float(r2)
    }
}

import json
with open('models/model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"💾 Métadonnées sauvegardées : 'models/model_metadata.json'")

print("\n" + "="*60)
print("ENTRAÎNEMENT TERMINÉ !")
print("="*60)
print("\n🚀 Prochaine étape : Intégrer le modèle dans Django")
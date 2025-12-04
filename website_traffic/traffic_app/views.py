from django.shortcuts import render, redirect
from django.http import HttpResponse
import joblib
import numpy as np
import os
import json

# Charger le modèle et les encodeurs une fois au démarrage
MODEL_PATH = 'models/traffic_model.pkl'
ENCODER_PATH = 'models/encoders/traffic_source_encoder.pkl'
METADATA_PATH = 'models/model_metadata.json'

try:
    model = joblib.load(MODEL_PATH)
    traffic_encoder = joblib.load(ENCODER_PATH)
    
    # Charger les métadonnées
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)
    
    MODEL_LOADED = True
    print("✅ Modèle ML chargé avec succès!")
except Exception as e:
    MODEL_LOADED = False
    print(f"❌ Erreur chargement modèle: {e}")
    model = None
    traffic_encoder = None
    metadata = {}

def home(request):
    """Page d'accueil"""
    context = {
        'model_loaded': MODEL_LOADED,
        'features': metadata.get('features', []),
        'target': metadata.get('target', 'Conversion Rate'),
    }
    return render(request, 'index.html', context)

def predict(request):
    """Page de prédiction - Formulaire"""
    if not MODEL_LOADED:
        return render(request, 'error.html', {
            'message': 'Le modèle ML n\'est pas chargé. Contactez l\'administrateur.'
        })
    
    # Options pour le formulaire
    context = {
        'traffic_sources': traffic_encoder.classes_.tolist() if traffic_encoder else [],
        'target': metadata.get('target', 'Conversion Rate'),
    }
    return render(request, 'predict.html', context)

def result(request):
    """Traitement de la prédiction et affichage des résultats"""
    if request.method != 'POST':
        return redirect('predict')
    
    if not MODEL_LOADED:
        return render(request, 'error.html', {
            'message': 'Le modèle ML n\'est pas disponible.'
        })
    
    try:
        # Récupérer les données du formulaire
        page_views = float(request.POST.get('page_views', 0))
        session_duration = float(request.POST.get('session_duration', 0))
        bounce_rate = float(request.POST.get('bounce_rate', 0))
        traffic_source = request.POST.get('traffic_source', 'Organic')
        time_on_page = float(request.POST.get('time_on_page', 0))
        previous_visits = float(request.POST.get('previous_visits', 0))
        
        # Encoder la source de trafic
        traffic_source_encoded = traffic_encoder.transform([traffic_source])[0]
        
        # Préparer les features pour le modèle
        features = np.array([[
            page_views,
            session_duration,
            bounce_rate,
            traffic_source_encoded,
            time_on_page,
            previous_visits
        ]])
        
        # Faire la prédiction
        prediction = model.predict(features)[0]
        
        # Générer des recommandations basées sur les inputs
        recommendations = generate_recommendations(
            page_views, session_duration, bounce_rate, 
            traffic_source, time_on_page, previous_visits, prediction
        )
        
        # Préparer le contexte pour le template
        context = {
            'prediction': round(prediction, 2),
            'inputs': {
                'page_views': page_views,
                'session_duration': session_duration,
                'bounce_rate': bounce_rate,
                'traffic_source': traffic_source,
                'time_on_page': time_on_page,
                'previous_visits': previous_visits,
            },
            'recommendations': recommendations,
            'target': metadata.get('target', 'Conversion Rate'),
        }
        
        return render(request, 'result.html', context)
        
    except Exception as e:
        return render(request, 'error.html', {
            'message': f'Erreur lors de la prédiction: {str(e)}'
        })

def about(request):
    """Page À propos"""
    performance = metadata.get('performance', {})
    context = {
        'model_loaded': MODEL_LOADED,
        'performance': performance,
        'features': metadata.get('features', []),
        'features_count': len(metadata.get('features', [])),
        'target': metadata.get('target', 'Conversion Rate'),
    }
    return render(request, 'about.html', context)

def generate_recommendations(page_views, session_duration, bounce_rate, 
                            traffic_source, time_on_page, previous_visits, prediction):
    """Génère des recommandations basées sur les inputs et la prédiction"""
    recommendations = []
    
    # Recommandations basées sur le bounce rate
    if bounce_rate > 50:
        recommendations.append({
            'icon': 'fa-exclamation-triangle',
            'color': 'warning',
            'text': f'Votre taux de rebond ({bounce_rate}%) est élevé. Améliorez la pertinence du contenu.'
        })
    elif bounce_rate < 20:
        recommendations.append({
            'icon': 'fa-check-circle',
            'color': 'success',
            'text': f'Excellent taux de rebond ({bounce_rate}%) ! Vos visiteurs sont engagés.'
        })
    
    # Recommandations basées sur la durée de session
    if session_duration < 2:
        recommendations.append({
            'icon': 'fa-clock',
            'color': 'warning',
            'text': 'La durée de session est courte. Enrichissez votre contenu.'
        })
    
    # Recommandations basées sur les pages vues
    if page_views < 3:
        recommendations.append({
            'icon': 'fa-sitemap',
            'color': 'info',
            'text': 'Augmentez la navigation interne pour plus de pages vues.'
        })
    
    # Recommandations basées sur la source de trafic
    if traffic_source == 'Paid':
        recommendations.append({
            'icon': 'fa-ad',
            'color': 'info',
            'text': 'Trafic payant détecté. Surveillez votre ROI.'
        })
    
    # Recommandation basée sur la prédiction
    if prediction < 3:
        recommendations.append({
            'icon': 'fa-chart-line',
            'color': 'danger',
            'text': f'Taux de conversion prédit bas ({prediction}%). Optimisez vos call-to-action.'
        })
    elif prediction > 8:
        recommendations.append({
            'icon': 'fa-trophy',
            'color': 'success',
            'text': f'Excellent taux de conversion prédit ({prediction}%) ! Continuez ainsi.'
        })
    
    # Recommandation générale
    if not recommendations:
        recommendations.append({
            'icon': 'fa-thumbs-up',
            'color': 'info',
            'text': 'Vos métriques sont bonnes. Concentrez-vous sur l\'optimisation SEO.'
        })
    
    return recommendations
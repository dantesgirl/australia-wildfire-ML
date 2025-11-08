import pickle 
import pandas as pd
import numpy as np
from datetime import datetime
import os

def load_model():
    model_path = 'fire_model.pkl'
    if not os.path.exists(model_path):
        print(f"Modelo não encontrado no {model_path}")
        print("Execute o model_training.py primeiro")
        exit()
    
    print(f"Carregando modelo de: {model_path}")
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print("Modelo carregado.")
        return model
    except Exception as e:
        print(f"Erro ao carregar modelo: {e}")
        exit()

rf_intensity = load_model()

def predict_fire(latitude, longitude, month, day, year=2026, brightness=None, confidence=None, time_of_day='D'):
    if brightness is None:
        if month in [12, 1, 2]:
            brightness = 340.0
        elif month in [6, 7, 8]:
            brightness = 315.0
        else: 
            brightness = 328.0
    if confidence is None:
        confidence = 80
    
    date = datetime(year, month, day)
    day_of_year = date.timetuple().tm_yday
    week_of_year = date.isocalendar()[1]
    quarter = (month - 1) // 3 + 1

    if month in [12, 1, 2]:
        season = 3
        season_name = "Verão"
        risk_level = "Muito Alto"
    elif month in [9, 10, 11]:
        season = 2
        season_name = "Primavera"
        risk_level = "Alto"
    elif month in [3, 4, 5]:
        season = 1
        season_name = "Outono"
        risk_level = "Médio"
    else:
        season = 0
        season_name = "Inverno"
        risk_level = "Baixo"

    daynight_encoded = 0 if time_of_day == 'D' else 1
    satellite_encoded = 0
    bright_t31 = brightness - 25.0

    input_data = pd.DataFrame({
        'latitude': [latitude],
        'longitude': [longitude],
        'brightness': [brightness],
        'bright_t31': [bright_t31],
        'confidence': [confidence],
        'scan': [1.5],
        'track': [1.2],
        'month': [month],
        'day_of_year': [day_of_year],
        'season': [season],
        'quarter': [quarter],
        'week_of_year': [week_of_year],
        'daynight_encoded': [daynight_encoded],
        'satellite_encoded': [satellite_encoded]
    })

    prediction = rf_intensity.predict(input_data)[0]
    proba = rf_intensity.predict_proba(input_data)[0]

    intensity = "Alto" if prediction == 1 else "Baixo"
    confidence_score = proba[prediction] * 100

    result = {
        'date': f"{year}-{month:02d}-{day:02d}",
        'location': f"({latitude:.2f}, {longitude:.2f})",
        'season': season_name,
        'fire_risk': risk_level,
        'predicted_intensity': intensity,
        'confidence_score': f"{confidence_score:.1f}%",
        'brightness': brightness,
        'confidence': confidence,
        'time_of_day': 'Dia' if time_of_day == 'D' else 'Noite'
    }

    return result

def predict_monthly_risk(latitude, longitude, year=2026):
    months = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
    results = []

    for month_num in range (1, 13):
        result = predict_fire(latitude, longitude, month_num, 15, year)
        results.append({
            'Mês': months[month_num - 1],
            'Estação': result['season'],
            'Risco': result['fire_risk'],
            'Intensidade': result['predicted_intensity'],
            'Confiança': result['confidence_score']
        })

    return pd.DataFrame(results)

def find_high_risk_periods(latitude, longitude, year=2026): #meses de alto risco
    high_risk_months = []
    months = ['Janeiro', 'Fevereiro', 'Março', 'Abril', 'Maio', 'Junho',
              'Julho', 'Agosto', 'Setembro', 'Outubro', 'Novembro', 'Dezembro']
    for month_num in range(1, 13):
        result = predict_fire(latitude, longitude, month_num, 15, year)
        if result['fire_risk'] in ['Alto', 'Muito Alto']:
            high_risk_months.append(months[month_num - 1])
    return high_risk_months

def compare_cities(year=2026, month=1, day=15):
    locations = [
        ("Sydney", -33.87, 151.21),
        ("Melbourne", -37.81, 144.96),
        ("Brisbane", -27.47, 153.03),
        ("Perth", -31.95, 115.86),
        ("Adelaide", -34.93, 138.60),
        ("Canberra", -35.28, 149.13),  
        ("Darwin", -12.46, 130.84)
    ]

    print(f"\nComparando os riscos {day:02d}/{month:02d}/{year}")
    print(f"{'Cidade':<15} {'Localização':<22} {'Risco':<15} {'Intensidade':<12} {'Confiança'}")

    results = []
    for city, lat, lon in locations:
        pred = predict_fire(lat, lon, month, day, year)
        print(f"{city:<15} ({lat:>6.2f}), {lon:>6.2f})  {pred['fire_risk']:<15} {pred['predicted_intensity']:<12} {pred['confidence_score']}")
        results.append({
            'Cidade': city,
            'Latitude': lat,
            'Longitude': lon,
            'Risco': pred['fire_risk'],
            'Intensidade': pred['predicted_intensity'],
            'Confiança': pred['confidence_score']
        })
    
    return pd.DataFrame(results)

#Exemplinhos <3  - Ps: pode apagar depois quando subir

if __name__ == "__main__":
    print("Sistema de predição de queimadas - 2026")
    print("\nPrimeiro: Sydney - Verão")
    print("Localização: Sydney (-33.87, 151.21)")
    print("Data: 15 de Janeiro, 2026")
    
    prediction = predict_fire(
        latitude=-33.87,
        longitude=151.21,
        month=1,
        day=15,
        year=2026
    )
    
    print("\nResultado:")
    for key, value in prediction.items():
        print(f"{key.replace('_', ' ').title()}: {value}")

    print("\nSegundo: Sydney no Inverno")
    print("Localização: Sydney (-33.87, 151.21)")
    print("Data: 15 de Julho, 2026")

    prediction2 = predict_fire(
        latitude=-33.87,
        longitude=151.21,
        month=7,
        day=15,
        year=2026
    )

    print("\nResultado:")
    for key, value in prediction2.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    print("\nOutro tipo (mensal): Análise Mensal para Centro da Austrália")
    monthly_df = predict_monthly_risk(-25.0, 135.0, 2026)
    print("\n" + monthly_df.to_string(index=False))

    print("\nMensal: Meses de Alto Risco em Melbourne")
    print("Localização: Melbourne (-37.81, 144.96)")

    high_risk = find_high_risk_periods(-37.81, 144.96, 2026) #meses de alto risco
    print(f"\nMeses de Alto Risco ({len(high_risk)} meses):")
    for month in high_risk:
        print(f"{month}")


    print("\nComo deve ser na aplicação: Predição Customizada")
    
    latitude = -25.0
    longitude = 135.0
    month = 12
    day = 25
    year = 2026
    
    print(f"Latitude: {latitude}")
    print(f"Longitude: {longitude}")
    print(f"Data: {year}-{month:02d}-{day:02d}")
    
    result = predict_fire(latitude, longitude, month, day, year)
    
    print(f"\nResultado da Predição:")
    print(f"Localização: {result['location']}")
    print(f"Data: {result['date']}")
    print(f"Estação: {result['season']}")
    print(f"Risco de Fogo: {result['fire_risk']}")
    print(f"Intensidade Prevista: {result['predicted_intensity']}")
    print(f"Confiança: {result['confidence_score']}")
    print(f"Temperatura: {result['brightness']}°C")
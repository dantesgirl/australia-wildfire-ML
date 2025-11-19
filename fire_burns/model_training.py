import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import sys
import os
import pickle


def get_training_data():

    try: 
        with open('fire_model.pkl', 'rb') as f:
            model_data = pickle.load(f)

        df_clean = model_data['df_clean']
        feature_importance = model_data['feature_importance']
        accuracy = model_data['accuracy']
        cm = model_data['confusion_matrix']

        print("Dados carregados com sucesso")
        print(f"Dataset: {len(df_clean):,} registros")
        print(f"Acurácia: {accuracy*100:.2f}%")
        
        return df_clean, feature_importance, accuracy, cm
    
    except FileNotFoundError:
        print("Erro: fire_model.pkl não encontrado")
        print("Execute o treinamento primeiro: python fire_burns/model_training.py")
        return None, None, None, None
    except KeyError as e:
        print("Pickle antigo detectado")
        print(f"Variável '{e}' não encontrada")
        print("Treine novamente rodando: python fire_burns/model_training.py")
        return None, None, None, None
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        return None, None, None, None

if __name__ == "__main__":
    df = pd.read_parquet('sorted_fires_australia.parquet')
    print(f"Carregados em {len(df):,} registros")

    if df.empty:
        print("Erro - parquet nao encontrado")
        print(f"Procure em {df}")
        print("Execute o pipeline primeiro")
        exit()
    print(f"Dados carregados do parquet {len(df):,} registros")

    df = df.sample(frac=0.70, random_state=42)
    print(f"Pronto: {len(df):,} dados do fogo")
    print(f"Colunas: {list(df.columns)}")
    print(f"Primeiros 3 dados:")
    print(df.head(3))

    #dados pro fogo
    df['year'] = df['acq_date'].dt.year
    df['month'] = df['acq_date'].dt.month
    df['day'] = df['acq_date'].dt.day
    df['day_of_year'] = df['acq_date'].dt.dayofyear
    df['week_of_year'] = df['acq_date'].dt.isocalendar().week
    df['quarter'] = df['acq_date'].dt.quarter

    def get_season(month):
        if month in [12, 1, 2]:
            return 3
        elif month in [9, 10, 11]:
            return 2 
        elif month in [3, 4, 5]:
            return 1
        else: 
            return 0

    df['season'] = df['month'].apply(get_season)
    le_daynight = LabelEncoder()
    df['daynight_encoded'] = le_daynight.fit_transform(df['daynight'])
    le_satellite = LabelEncoder()
    df['satellite_encoded'] = le_satellite.fit_transform(df['satellite'])

    frp_median = df['frp'].median()
    df['fire_intensity'] = (df['frp'] > frp_median).astype(int)

    print(f"Categoria temporal: year, month, day, day_of_year, week, quarter, season")
    print(f"Cadegoria do codigo: daynight, satellite")
    print(f"Intensidade do fogo: FRP > {frp_median:.2f} MW")
    print(f"Distribuição da intendisade do fogo:")
    print(df['fire_intensity'].value_counts())
    print(f"  0 (Baixo):  {(df['fire_intensity']==0).sum():,} fires")
    print(f"  1 (Alto): {(df['fire_intensity']==1).sum():,} fires")

    #separando pro ml
    df_clean = df.dropna(subset=['frp', 'brightness', 'bright_t31', 'confidence'])
    print(f"\nDataset limpa: {len(df_clean):,} arquivos")

    #selecionando pra predição
    features = [
        'latitude', 'longitude',
        'brightness', 'bright_t31',
        'confidence',
        'scan', 'track',
        'month', 'day_of_year', 'season',
        'quarter', 'week_of_year',
        'daynight_encoded',
        'satellite_encoded'
    ]

    X = df_clean[features]
    y_intensity = df_clean['fire_intensity']
    print(f"Caracterisiticas selecionadas: {len(features)}")
    print(f"Cateristicas: {features}")

    #separando 80% treinando e 20% testando
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_intensity, test_size=0.2, random_state=42, stratify=y_intensity
    )

    print(f"Separando os dados:")
    print(f"Treino: {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Teste:  {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")

    #treinado o ml (100 arvores, 20 maximo de profundidade)
    rf_intensity = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=20,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )

    start_time = datetime.now()
    rf_intensity.fit(X_train, y_train)
    train_time = (datetime.now() - start_time).total_seconds()
    print(f"Modelo treinado em {train_time:.2f} segundos!")

    #predições 
    y_pred = rf_intensity.predict(X_test)
    y_pred_proba = rf_intensity.predict_proba(X_test)

    #calculando a presisão
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Precisão: {accuracy*100:.2f}%")

    #devolvendo
    print(classification_report(y_test, y_pred,
                                target_names=['Low Intensity', 'High Intensity'],
                                digits=3))

    #matriz
    cm = confusion_matrix(y_test, y_pred)
    print("\nMatriz Confunsion:")
    print("                 Baixa Predição  Alta Predição")
    print(f"Baixo atual       {cm[0][0]:>13,}  {cm[0][1]:>14,}")
    print(f"Alto atual      {cm[1][0]:>13,}  {cm[1][1]:>14,}")

    true_negatives = cm[0][0]
    false_positives = cm[0][1]
    false_negatives = cm[1][0]
    true_positives = cm[1][1]

    precision_high = true_positives / (true_positives + false_positives)
    recall_high = true_positives / (true_positives + false_negatives)

    print(f"  Precisão(Alta): {precision_high*100:.2f}%")
    print(f"  Recall (Alta):    {recall_high*100:.2f}%")

    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': rf_intensity.feature_importances_
    }).sort_values('importance', ascending=False)

    # SALVA MODELO + VARIÁVEIS
    model_data = {
        'model': rf_intensity,
        'df_clean': df_clean,
        'feature_importance': feature_importance,
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'features': features
    }

    with open('fire_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)

    print("Modelo e dados salvos em 'fire_model.pkl'")

    print("Top caracteristicas + importantes:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']:.<25} {row['importance']:.4f}")

    #prever quando o fogo occorer
    monthly_fires = df_clean.groupby('month').size()
    print("\nOcorrença por mês:")
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    for month, fires in monthly_fires.items():
        print(f"  {months[month-1]:>3} {fires:>8,} fogo ({fires/len(df_clean)*100:>5.2f}%)")

    #meses de alto risco
    high_risk_months = monthly_fires[monthly_fires > monthly_fires.median()]
    print(f"Meses de risco mais alto (maior que a media):")
    for month in high_risk_months.index:
        print(f"   - {months[month-1]} ({monthly_fires[month]:,} fogo)")

    #por estação
    seasonal_fires = df_clean.groupby('season')['fire_intensity'].agg(['count', 'mean'])
    seasonal_names = {0: 'Winter', 1: 'Fall', 2: 'Spring', 3: 'Summer'}
    print(f"\nPadrões do fogo/queimadas por estação:")
    for season, data in seasonal_fires.iterrows():
        print(f"   {seasonal_names[season]:>6}: {data['count']:>8,} chamas | " + 
              f"Intensidade alta: {data['mean']*100:>5.1f}%")

    #vizualização dos dados
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Ramdom Forest - Predição de queimada', fontsize=18, fontweight='bold')

    ax1 = plt.subplot(2, 3, 1)
    top_features = feature_importance.head(10)
    colors = plt.cm.Reds(np.linspace(0.4, 0.8, len(top_features)))
    ax1.barh(top_features['feature'], top_features['importance'], color=colors)
    ax1.set_xlabel('Pontuação de importancia', fontsize=10)
    ax1.set_title('Top 10', fontsize=12, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)

    #matriz
    ax2 = plt.subplot(2, 3, 2)
    sns.heatmap(cm, annot=True, fmt=',', cmap='YlOrRd', ax=ax2,
                xticklabels=['Low', 'High'],
                yticklabels=['Low', 'High'],
                cbar_kws={'label': 'Count'})
    ax2.set_xlabel('Predição da Intendisdade', fontsize=10)
    ax2.set_ylabel('Intensidade atual', fontsize=10)
    ax2.set_title(f'Confusion Matriz\nPrecisão: {accuracy*100:.2f}%', 
                  fontsize=12, fontweight='bold')

    #por mes
    ax3 = plt.subplot(2, 3, 3)
    colors_month = ['#ff6b6b' if m in high_risk_months.index else '#ffd93d' 
                    for m in range(1, 13)]
    ax3.bar(range(1, 13), monthly_fires, color=colors_month, alpha=0.8, edgecolor='black')
    ax3.set_xlabel('Mês', fontsize=10)
    ax3.set_ylabel('Numero de fogo', fontsize=10)
    ax3.set_title('Distribuição de queimadas por mes\n(Vermelho = Risco alto)', 
                  fontsize=12, fontweight='bold')
    ax3.set_xticks(range(1, 13))
    ax3.set_xticklabels(months, rotation=45)
    ax3.grid(axis='y', alpha=0.3)
    ax3.axhline(y=monthly_fires.median(), color='red', linestyle='--', 
                linewidth=2, label='Media', alpha=0.7)
    ax3.legend()

    ax4 = plt.subplot(2, 3, 4)
    seasons = [seasonal_names[i] for i in sorted(seasonal_fires.index)]
    counts = [seasonal_fires.loc[i, 'count'] for i in sorted(seasonal_fires.index)]
    colors_season = ['#4a90e2', '#f39c12', '#e74c3c', '#c0392b']
    ax4.bar(seasons, counts, color=colors_season, alpha=0.8, edgecolor='black')
    ax4.set_ylabel('Number of Fires', fontsize=10)
    ax4.set_title('Fire Distribution by Season', fontsize=12, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    for i, v in enumerate(counts):
        ax4.text(i, v + max(counts)*0.02, f'{v:,}', ha='center', fontweight='bold')

    #intensidade por estação
    ax5 = plt.subplot(2, 3, 5)
    intensity_pct = [seasonal_fires.loc[i, 'mean']*100 for i in sorted(seasonal_fires.index)]
    ax5.bar(seasons, intensity_pct, color=colors_season, alpha=0.8, edgecolor='black')
    ax5.set_ylabel('Alta intensidade por estação (%)', fontsize=10)
    ax5.set_title('Intensidade do fogo por estação', fontsize=12, fontweight='bold')
    ax5.grid(axis='y', alpha=0.3)
    ax5.set_ylim(0, 100)
    for i, v in enumerate(intensity_pct):
        ax5.text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')

    #anual
    ax6 = plt.subplot(2, 3, 6)
    yearly_data = df_clean.groupby('year').agg({
        'fire_intensity': ['count', 'mean']
    }).reset_index()
    yearly_data.columns = ['year', 'total_fires', 'high_intensity_pct']
    yearly_data['high_intensity_pct'] *= 100

    ax6_twin = ax6.twinx()
    line1 = ax6.plot(yearly_data['year'], yearly_data['total_fires'], 
                     marker='o', color='#e74c3c', linewidth=2, label='Total Fires')
    line2 = ax6_twin.plot(yearly_data['year'], yearly_data['high_intensity_pct'], 
                          marker='s', color='#3498db', linewidth=2, label='High Intensity %')
    ax6.set_xlabel('Year', fontsize=10)
    ax6.set_ylabel('Number of Fires', fontsize=10, color='#e74c3c')
    ax6_twin.set_ylabel('High Intensity (%)', fontsize=10, color='#3498db')
    ax6.set_title('Fire Trends Over Years', fontsize=12, fontweight='bold')
    ax6.tick_params(axis='y', labelcolor='#e74c3c')
    ax6_twin.tick_params(axis='y', labelcolor='#3498db')
    ax6.grid(alpha=0.3)
    plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45)

    lines1, labels1 = ax6.get_legend_handles_labels()
    lines2, labels2 = ax6_twin.get_legend_handles_labels()
    ax6.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), '..', 'fire_prediction_random_forest.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualização salva em: {os.path.abspath(output_path)}")
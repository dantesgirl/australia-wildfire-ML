Projeto acadêmico de Machine Learning para queimadas na Austrália em 2026.

🔥 Predição de Queimadas na Austrália
Projeto acadêmico de Machine Learning para queimadas na Austrália em 2026.
📊 Informações do Dataset

Fonte: NASA MODIS (FIRMS)
Período: 2014-2024
Registros: 2,654,051 detecções de fogo
Região: Austrália
Algoritmo: Random Forest Classifier
Acurácia: 70% ~ 90%
Features: 14 (latitude, longitude, brightness, estação, etc.)

📁 Estrutura do Projeto
├── fire_burns/
│   ├── data_fetch.py        # Coleta de dados (CSV + API NASA)
│   ├── sort_data.py          # Ordenação TimSort + Parquet
│   ├── model_training.py     # Treinamento do modelo
│   ├── predict_model.py      # Sistema de predição
│   └── pipeline.py           # Pipeline completa
├── fire_model.pkl            # Modelo treinado (baixe aqui!)
├── sorted_fires_australia.parquet      # Dados ordenados
└── fire_prediction_random_forest.png    # Visualizações



🚀 Como Usar
1. Baixar o Modelo Treinado
Para usar o modelo sem precisar treinar novamente:
Opção A - Download Direto:

Baixe o arquivo fire_model.pkl do repositório
Coloque na raiz do projeto

Opção B - Treinar localmente (demora ~5 minutos):
bashpython fire_burns/model_training.py

2. Fazer Predições
pythonfrom fire_burns.predict_model import predict_fire

result = predict_fire(
    latitude=-33.87,   # Sydney
    longitude=151.21,
    month=1,           # Janeiro
    day=15,
    year=2026
)

print(f"Risco: {result['fire_risk']}")
print(f"Intensidade: {result['predicted_intensity']}")
print(f"Confiança: {result['confidence_score']}")


🛠️ Instalação

# Instale as dependências
pip install pandas numpy scikit-learn matplotlib seaborn pyarrow na nova versão 

# Rode o Model_training

# Baixe o modelo (ver seção "Como Usar")
📦 Dependências

Python 3.8+
pandas
numpy
scikit-learn
matplotlib
seaborn
pyarrow



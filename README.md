Projeto acadêmico de Machine Learning para queimadas na Austrália em 2026.



\- \*\*Fonte:\*\* NASA MODIS (FIRMS)

\- \*\*Período:\*\* 2014-2024

\- \*\*Registros:\*\* 2,654,051 detecções de fogo

\- \*\*Região:\*\* Austrália



\- \*\*Algoritmo:\*\* Random Forest Classifier

\- \*\*Acurácia:\*\* \[COLOCA A ACURÁCIA DO SEU MODELO]%

\- \*\*Features:\*\* 14 (latitude, longitude, brightness, estação, etc.)



Estrutura

```

├── fire\_burns/

│   ├── data\_fetch.py        Coleta de dados (CSV + API NASA)

│   ├── sort\_data.py          Ordenação TimSort + Parquet

│   ├── model\_training.py     Treinamento do modelo

│   ├── predict\_model.py      Sistema de predição

│   └── pipeline.py           Pipeline completa

├── fire\_model.pkl            Modelo treinado

├── sorted\_fires\_australia.parquet      Dados ordenados

└── fire\_prediction\_random\_forest.png    Visualizações

```





&nbsp;Exemplo de Predição

```python

from fire\_burns.predict\_model import predict\_fire



result = predict\_fire(

&nbsp;   latitude=-33.87,   # Sydney

&nbsp;   longitude=151.21,

&nbsp;   month=1,           # Janeiro

&nbsp;   day=15,

&nbsp;   year=2026

)



print(f"Risco: {result\['fire\_risk']}")

print(f"Intensidade: {result\['predicted\_intensity']}")

print(f"Confiança: {result\['confidence\_score']}")

```






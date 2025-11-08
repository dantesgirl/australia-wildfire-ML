import pandas as pd
from datetime import datetime
import os

def timsort_and_save_data(df, column='acq_date', file_name='sorted_fires_australia.parquet'):
    if df.empty:
        print("Dataframe fazio :(")
        return pd.DataFrame()
    print(f"TimSort - {len(df):,} registros")

    if not pd.api.types.is_datetime64_any_dtype(df[column]):
        print(f"Convertendo '{column}' para datetime...")
        df[column] =  pd.to_datetime(df[column], errors='coerce')

    df_clean = df.dropna(subset=[column])
    if len(df_clean) < len(df):
        print(f"Removidas {len(df) - len(df_clean)} as linhas com informações invalidas")

    #ordenando com o timsort
    print(f"Ordenando por '{column}' usando TimSort pois ordena por data")

    start_time = datetime.now()
    df_sorted = df_clean.sort_values(by=column, ascending=True)
    end_time = datetime.now()

    elapsed = (end_time - start_time).total_seconds()

    print(f"Concluido em {elapsed:.3f} segundos")
    print(f"\nDados ordenados:")
    print(f"Primeira data: {df_sorted[column].iloc[0]}")
    print(f"Última data:   {df_sorted[column].iloc[-1]}")
    print(f"Total de registros: {len(df_sorted):,}")

    #salvando em parquet
    print(f"Arquivo: {file_name}")

    try:
        df_sorted.to_parquet(file_name, index=False, engine='pyarrow')
        if os.path.exists(file_name):
            file_size = os.path.getsize(file_name) / (1024**2) #MB
            print(f"Arquivo salvo.")
            print(f"Tamanho: {file_size:.2f} MB")
            print(f"Localização: {os.path.abspath(file_name)}")
        else:
            print(f"Erro: Arquivo não foi criado")

    except Exception as e:
        print(f"Erro ao salvar: {e}")
        return df_sorted
    
    return df_sorted

def load_from_parquet(file_name='sorted_fires_australia.parquet'):

    if not os.path.exists(file_name):
        print(f"Arquivo '{file_name}' não encontrado")
        return pd.DataFrame()
    
    print(f"Carregando o parquet...")
    try:
        df = pd.read_parquet(file_name)
        print(f"Carregados em {len(df):,} registros")
        print(f"Colunas: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"Erro ao carregar: {e}")
        return pd.DataFrame()
    
#testando
if __name__ == "__main__":
    print("Testando o Timsort + parquet")

# Cria DataFrame de teste - IA ajudando nessa parte de teste (apaga se essa parte for subir
    test_df = pd.DataFrame({
        'acq_date': pd.date_range('2024-01-01', periods=1000),
        'latitude': [-25.0] * 1000,
        'longitude': [135.0] * 1000,
        'brightness': [330.0] * 1000
    })
    
    # Embaralha para testar ordenação
    test_df = test_df.sample(frac=1).reset_index(drop=True)
    
    print(f"\nCriado DataFrame de teste: {len(test_df)} linhas")
    print(f"Datas antes de ordenar: {test_df['acq_date'].iloc[:3].tolist()}")
    
    # Ordena e salva
    df_sorted = timsort_and_save_data(test_df, column='acq_date', file_name='test_output.parquet')
    
    print(f"\nDatas depois de ordenar: {df_sorted['acq_date'].iloc[:3].tolist()}")
    
   
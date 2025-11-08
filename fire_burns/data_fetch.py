import pandas as pd
import requests
import os
import glob
from datetime import datetime, timedelta
import time

MAP_KEY = "29f93e123ee2594a09cfea4e382807d0"
BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/country/csv"
SOURCE = "MODIS_NRT"
COUNTRY = "AUS"
DATA_FOLDER = './fogo'

def ensure_data_folder(): #cria a pasta caso n tenha
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
        print(f"Pasta '{DATA_FOLDER} criada.")
    return DATA_FOLDER

def load_existing_csvs():
    ensure_data_folder()
    all_files = glob.glob(f"{DATA_FOLDER}/*csv")

    if not all_files:
        print(f"Nenhum CSV encontrado no '{DATA_FOLDER}/'")
        return pd.DataFrame()
    
    print(f"Encontrado {len(all_files)} csv:")
    df_list = []

    for file in all_files:
        print(f"Lendo {file}...")
        try:
            df_temp = pd.read_csv(file)
            print(f"{len(df_temp):,} registros")
            df_list.append(df_temp)
        except Exception as e:
            print(f"Erro ao ler {len}: {e}")
    if df_list:
        df_combined = pd.concat(df_list, ignore_index=True)
        print(f"Tudo combinado: {len(df_combined):,} registros")
        return df_combined
    
    return pd.DataFrame()

def download_new_data_from_nasa(start_date, end_date, map_key=MAP_KEY):
    #baixando os dados API 
    url = f"{BASE_URL}/{map_key}/{SOURCE}/{COUNTRY}/1/{start_date}/{end_date}"

    print(f"Baixando os dados...")
    print(f"Periodo: {start_date} até {end_date}")
    print(f"URL: {url[:80]}...")

    try:
        
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
            print(f"Novos registros {len(df):,} baixados")
            return df
        else:
          print(f"Erro http {response.status_code}")
          print(f"Mensagem: {response.text[:200]}")
        return pd.DataFrame
    except Exception as e:
        print(f"Erro na conexão: {e}")
        return pd.DataFrame

def get_complete_dataframe(update_from_nasa=True):
    print("Carregando dados das queimadas na Austrália")

    df_existing = load_existing_csvs()
    if df_existing.empty:
        print("Nenhum dado local encontrado")
        print("Coloque os CSVS na pasta './fogo/' e rode novamente ")
        return pd.DataFrame()
    
    df_existing['acq_date'] = pd.to_datetime(df_existing['acq_date'])
    max_date = df_existing['acq_date'].max()

    print(f"Dados encontrados:")
    print(f"Periodo: {df_existing['acq_date'].min().date()} até {max_date.date()}")
    print(f"Total: {len(df_existing):,} registros.")

    if update_from_nasa: #buscando os dados da nasa
        print("Buscando por novos dados...")

        start_new = (max_date + timedelta(days=1)).strftime('%Y-%m-%d')
        end_new = datetime.now().strftime('%Y-%m-%d')

        if start_new >= end_new:
            print(f"Os dados já estão atualizados até {max_date.date()}")
        else:
            df_new = download_new_data_from_nasa(start_new, end_new)

            if not df_new.empty:
                df_new['acq_date'] = pd.to_datetime(df_new['acq_date'])
            #dados novos + velhos 
            df_combined = pd.concat([df_existing, df_new], ignore_index=True
                                    )
            print(f"Dados atualizados")
            print(f"Novos registros: {len(df_new):,}")
            print(f"Total final: {len(df_combined):,}")
            print(f"Novo período: {df_combined['acq_date'].min().date()} até {df_combined['acq_date'].max().date()}")

            new_file = f"{DATA_FOLDER}/fire_update_{end_new}.csv"
            df_new.to_csv(new_file, index=False)
            print(f"Novos dados salvos em {new_file}")

            return df_combined
    else:
        print(f"Não foi possivel baixar os novos dados (API pode estar com limite)")
    return df_existing

if __name__ == "__main__":
    df =  get_complete_dataframe(update_from_nasa=True)

    if not df.empty:
        print("Resumo final:")
        print(f"Total de registros: {len(df):,}")
        print(f"Colunas: {list(df.columns)}")
        print(f"\nPrimeiras 3 linhas:")
        print(df.head(3))
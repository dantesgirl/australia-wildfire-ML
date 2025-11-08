import pandas as pd
from fire_burns.data_fetch import get_complete_dataframe
from fire_burns.sort_data import timsort_and_save_data, load_from_parquet
import os

def run_full_pipeline(force_update=False):
    #false usa o já existente 
    parquet_file = 'sorted_fires_australia.parquet'
    if os.path.exists(parquet_file) and not force_update: #checando se já foi atualizado
        print("\nArquivo parquet encontrado")
        print(f"Arquivo: {parquet_file}")

        mod_time = os.path.getmtime(parquet_file)
        from datetime import datetime
        mod_date = datetime.fromtimestamp(mod_time)

        print(f"A ultima atualização: {mod_date.strftime('%Y-%m-%d %H:%M:%S')}")
        choice = input("\nUsar dados existentes? (S/N):").strip().lower()

        if choice != 'n':
            print("\nCarregando os dados parquet existentes...")
            df = load_from_parquet(parquet_file)

            if not df.empty:
                print("Pipeline concluida, pronto para o ML")
                print(f"Total dos registros: {len(df):,}")
                print(f"Periodo: {df['acq_date'].min().date()} até {df['acq_date'].max().date()}")
    
    #procurando os dados csvs local + API disponivel
    print("\nCarregando os dados...")
    df = get_complete_dataframe(update_from_nasa=True)
    if df.empty:
        print("\nPipeline falhou - nenhum dado disponivel")
        print("Insira os csvs na pasta './fogo/' e rode de novo")
        return pd.DataFrame()
    
     # Testa carregar
    df_loaded = load_from_parquet('test_output.parquet')
    print("Ordanando e salvando...")
    df_sorted = timsort_and_save_data(df, column='acq_date', file_name=parquet_file )

    if df_sorted.empty:
        print("\nPipeline falhou na ordenação")
        return pd.DataFrame()
    
    print("Pipeline completo")
    print(f"Total: {len(df_sorted):,} registros")
    print(f"Período: {df_sorted['acq_date'].min().date()} até {df_sorted['acq_date'].max().date()}")
    print(f"Arquivo: {parquet_file}")

    return df_sorted
if __name__ == "__main__":
    #como rodar a pipeline
    df_final = run_full_pipeline(force_update=False)

    if not df_final.empty:
        print("Agora:")
        print("Dados carregados e ordenados")
        print("Rode: python model_train.py")
        print("Depois rode: python predict.py")
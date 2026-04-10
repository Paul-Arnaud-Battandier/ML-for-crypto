import pandas as pd
import numpy as np
import os


def load_raw_data():
    return {
        'ohlcv': pd.read_csv("data/raw/ohlcv_raw.csv"),
        'funding': pd.read_csv("data/raw/funding_raw.csv"),
        'fng': pd.read_csv("data/raw/fng_raw.csv"),
        'macro': pd.read_csv("data/raw/macro_raw.csv", header=[0, 1], index_col=0)
    }


def clean_standardize(data_dict):
    # 1. OHLCV (ms)
    data_dict['ohlcv']['timestamp'] = pd.to_datetime(data_dict['ohlcv']['timestamp'], unit='ms', utc=True)
    data_dict['ohlcv'].set_index('timestamp', inplace=True)
    
    # 2. Funding (ms)
    data_dict['funding']['timestamp'] = pd.to_datetime(data_dict['funding']['fundingTime'], unit='ms', utc=True)
    data_dict['funding'].set_index('timestamp', inplace=True)
    data_dict['funding'] = data_dict['funding'][['fundingRate']]

    # 3. Fear & Greed (s)
    data_dict['fng']['timestamp'] = pd.to_datetime(data_dict['fng']['timestamp'], unit='s', utc=True)
    data_dict['fng'].set_index('timestamp', inplace=True)
    data_dict['fng'] = data_dict['fng'][['value']].rename(columns={'value': 'fng_value'}).astype(float)
    
    # 4. Traitement Spécifique Macro (MultiIndex)
    df_m = data_dict['macro'].copy()
    
    # On ne garde que les colonnes 'Close' (ou 'Adj Close')
    # Et on simplifie les noms : ('Close', 'QQQ') devient 'QQQ'
    df_m = df_m['Close'] 
    
    # On s'assure que l'index est en datetime UTC
    df_m.index = pd.to_datetime(df_m.index, utc=True)
    df_m.index.name = 'timestamp'
    
    # On renomme pour avoir des noms de colonnes propres
    df_m = df_m.rename(columns={
        'DX-Y.NYB': 'dxy',
        'QQQ': 'nasdaq'
    })
    
    data_dict['macro'] = df_m
    
    return data_dict


def merge_and_fill(data_dict):
    # On part de l'OHLCV (notre base de temps)
    master_df = data_dict['ohlcv'].copy()

    # Jointure successive
    master_df = master_df.join(data_dict['funding'], how='left')
    master_df = master_df.join(data_dict['fng'], how='left')
    master_df = master_df.join(data_dict['macro'], how='left')

    # LE MOMENT CRUCIAL : Forward Fill
    # On propage la dernière valeur connue (F&G, Macro) sur les lignes vides
    master_df.ffill(inplace=True)

    # On drop les quelques lignes au tout début s'il manque de la macro
    master_df.dropna(inplace=True)

    return master_df


def save_processed_data(data_dict, master_df):
    # Création du dossier si inexistant
    os.makedirs("data/processed", exist_ok=True)
    
    # 1. Sauvegarde des fichiers individuels nettoyés
    for name, df in data_dict.items():
        path = f"data/processed/{name}_cleaned.csv"
        df.to_csv(path)
    
    # 2. Sauvegarde du Master DataFrame (le plus important pour le ML)
    master_path = "data/processed/master_data.csv"
    master_df.to_csv(master_path)


def quality_gate(df):
    # 1. Suppression des doublons d'index (timestamp)
    df = df[~df.index.duplicated(keep='first')]
    
    # 2. Tri chronologique
    df.sort_index(inplace=True)
    
    # 3. Audit des valeurs manquantes
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        print(f"⚠️ Warning: {nan_count} valeurs manquantes détectées. Suppression des lignes...")
        df.dropna(inplace=True)
    
    # 4. Vérification des types
    # On s'assure que tout est numérique pour le XGBoost
    df = df.apply(pd.to_numeric, errors='coerce')
    
    return df


def main():
    # 1. Chargement
    raw_data = load_raw_data()
    
    # 2. Nettoyage & Standardisation
    cleaned_data = clean_standardize(raw_data)
    
    # 3. Fusion & Remplissage (FFill)
    master_df = merge_and_fill(cleaned_data)

    # 4. Qualité des données
    master_df = quality_gate(master_df)

    # 5. Persistence
    save_processed_data(cleaned_data, master_df)

if __name__ == "__main__":
    main()
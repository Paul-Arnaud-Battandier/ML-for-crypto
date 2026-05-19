import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, precision_score
import warnings
warnings.filterwarnings('ignore')

def train_meta_model():
    print("⏳ Chargement du Meta-Dataset...")
    df = pd.read_csv("data/meta_dataset.csv", index_col='timestamp', parse_dates=True)
    
    X = df.drop(columns=['meta_target'])
    y = df['meta_target']
    
    print(f"Taille du dataset : {len(df)} trades à analyser.")
    
    # Validation croisée purgée (on garde les bonnes habitudes)
    tscv = TimeSeriesSplit(n_splits=5, gap=12)
    
    # Le Random Forest : on le veut robuste pour ne pas overfitter le bruit
    meta_model = RandomForestClassifier(
        n_estimators=200,      # On double le nombre d'arbres pour lisser les probabilités
        max_depth=5,           # On lui donne un tout petit peu plus de mémoire
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    precisions = []
        
    # On va stocker les résultats pour différents seuils d'exigence
    thresholds = [0.50, 0.52, 0.54, 0.56, 0.58, 0.60]
    results = {t: {'precisions': [], 'nb_trades': []} for t in thresholds}
    
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        meta_model.fit(X_train, y_train)
        
        # Au lieu de .predict(), on demande les probabilités brutes !
        meta_probas = meta_model.predict_proba(X_test)[:, 1]
        
        # On teste nos différents seuils d'exigence
        for t in thresholds:
            # Si proba > seuil, on trade (1), sinon on annule (0)
            custom_preds = (meta_probas > t).astype(int)
            
            nb_trades_pris = sum(custom_preds)
            results[t]['nb_trades'].append(nb_trades_pris)
            
            if nb_trades_pris > 0:
                prec = precision_score(y_test, custom_preds)
                results[t]['precisions'].append(prec)
            else:
                results[t]['precisions'].append(0.0)

    print("\n📊 --- ANALYSE DES SEUILS DE CONFIANCE ---")
    print("Rappel : Il nous faut > 40.0% de précision pour être rentable (Ratio 1.5:1).\n")
    
    for t in thresholds:
        mean_prec = np.mean(results[t]['precisions']) * 100
        mean_trades = np.mean(results[t]['nb_trades'])
        
        # Un peu de formatage visuel pour voir les gagnants
        if mean_prec > 40.0:
            etat = "✅ RENTABLE"
        else:
            etat = "❌ PERTE"
            
        print(f"Seuil {t*100}% -> Précision: {mean_prec:.2f}% | Trades moyens/fold: {mean_trades:.0f} | {etat}")
        
if __name__ == "__main__":
    train_meta_model()
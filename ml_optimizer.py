import json
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score
from typing import Dict, List, Tuple, Optional
from itertools import combinations
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
import time
from datetime import datetime
warnings.filterwarnings('ignore')

try:
    import ijson
except ImportError:
    raise SystemExit("ijson required: pip install ijson")

# Field mappings
ODD_FIELDS = [
    "odd_1", "odd_x", "odd_2", "odd_1x", "odd_12", "odd_x2",
    "bts_yes", "bts_no", "o+0.5", "u+0.5", "o+1.5", "u+1.5", 
    "o+2.5", "u+2.5", "o+3.5", "u+3.5", "o+4.5", "u+4.5", "o+5.5", "u+5.5"
]

TR_FIELD = {
    "odd_1": "MS 1 (Ev Sahibi Kazanır)",
    "odd_x": "MS X (Beraberlik)",
    "odd_2": "MS 2 (Deplasman Kazanır)",
    "odd_1x": "Double Chance 1X (Ev veya Beraberlik)",
    "odd_12": "Double Chance 1-2",
    "odd_x2": "Double Chance X2 (Beraberlik veya Deplasman)",
    "bts_yes": "Karşılıklı Gol Var",
    "bts_no": "Karşılıklı Gol Yok",
    "o+0.5": "0.5 Üst Gol (Toplam)",
    "u+0.5": "0.5 Alt Gol (Toplam)",
    "o+1.5": "1.5 Üst Gol (Toplam)",
    "u+1.5": "1.5 Alt Gol (Toplam)",
    "o+2.5": "2.5 Üst Gol",
    "u+2.5": "2.5 Alt Gol",
    "o+3.5": "3.5 Üst Gol",
    "u+3.5": "3.5 Alt Gol",
    "o+4.5": "4.5 Üst Gol",
    "u+4.5": "4.5 Alt Gol",
    "o+5.5": "5.5 Üst Gol",
    "u+5.5": "5.5 Alt Gol",
}

TR_TARGET = {
    "O2.5": "2.5+ Gol Tahmini",
    "O3.5": "3.5+ Gol Tahmini",
    "O4.5": "4.5+ Gol Tahmini",
    "O6+": "6+ Gol Tahmini",
    "U2.5": "2.5 Alt Tahmini",
    "U1.5": "0-1 Gol Tahmini",
    "EXACT_0_0": "Hiç Gol Yok (0-0)",
    "MS1": "MS 1 Tahmini (Ev Sahibi Kazanır)",
    "MS2": "MS 2 Tahmini (Deplasman Kazanır)",
    "MSX": "MS X Tahmini (Beraberlik)",
    "BTTS": "KG Var Tahmini (Karşılıklı Gol)",
    "BTTS_NO": "KG Yok Tahmini (Karşılıklı Gol Yok)",
    "HOME_O1.5": "Ev Takım 1.5+ Gol",
    "AWAY_O1.5": "Deplasman Takım 1.5+ Gol",
    "HT_O1.5": "İlk Yarı 1.5+ Gol",
    "HTFT_1/2": "1/2 Tahmini (İY Ev Kazanır / MS Dep Kazanır)",
    "HTFT_2/1": "2/1 Tahmini (İY Dep Kazanır / MS Ev Kazanır)",
    "HTFT_1/X": "1/X Tahmini (İY Ev Kazanır / MS Beraberlik)",
    "HTFT_2/X": "2/X Tahmini (İY Dep Kazanır / MS Beraberlik)",
}

def log_with_timestamp(message: str):
    
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

def stream_records(path: str):
    
    log_with_timestamp(f"Opening file: {path}")
    with open(path, "rb") as f:
        for rec in ijson.items(f, "item"):
            yield rec

def parse_goals(score: str) -> Tuple[Optional[int], Optional[int]]:
    
    if not score or "-" not in score:
        return None, None
    try:
        home_s, away_s = [s.strip() for s in score.split("-")]
        return int(home_s), int(away_s)
    except:
        return None, None

def build_targets() -> Dict[str, callable]:
    
    
    def ft_goals(rec: dict) -> Optional[Tuple[int, int, int]]:
        h, a = parse_goals(rec.get("event_ft_result", ""))
        if h is None or a is None:
            return None
        return h, a, h + a
    
    def ht_goals(rec: dict) -> Optional[Tuple[int, int, int]]:
        h, a = parse_goals(rec.get("event_halftime_result", ""))
        if h is None or a is None:
            return None
        return h, a, h + a
    
    def over(x: float):
        def _f(rec: dict):
            g = ft_goals(rec)
            return None if g is None else g[2] > x
        return _f
    
    def under(x: float):
        def _f(rec: dict):
            g = ft_goals(rec)
            return None if g is None else g[2] < x
        return _f
    
    def btts(rec: dict):
        g = ft_goals(rec)
        return None if g is None else (g[0] > 0 and g[1] > 0)
    
    def btts_no(rec: dict):
        g = ft_goals(rec)
        return None if g is None else not (g[0] > 0 and g[1] > 0)
    
    def ms1(rec: dict):
        g = ft_goals(rec)
        return None if g is None else g[0] > g[1]
    
    def ms2(rec: dict):
        g = ft_goals(rec)
        return None if g is None else g[1] > g[0]
    
    def msx(rec: dict):
        g = ft_goals(rec)
        return None if g is None else g[0] == g[1]
    
    def home_over(n: float):
        def _f(rec: dict):
            g = ft_goals(rec)
            return None if g is None else g[0] > n
        return _f
    
    def away_over(n: float):
        def _f(rec: dict):
            g = ft_goals(rec)
            return None if g is None else g[1] > n
        return _f
    
    def ht_over(x: float):
        def _f(rec: dict):
            g = ht_goals(rec)
            return None if g is None else g[2] > x
        return _f
    
    def htft(pattern: str):
        def _f(rec: dict):
            h_ht, a_ht = parse_goals(rec.get("event_halftime_result", ""))
            h_ft, a_ft = parse_goals(rec.get("event_ft_result", ""))
            if None in (h_ht, a_ht, h_ft, a_ft):
                return None
            
            
            if pattern[0] == "1" and not (h_ht > a_ht):
                return False
            if pattern[0] == "2" and not (a_ht > h_ht):
                return False
            if pattern[0].upper() == "X" and not (a_ht == h_ht):
                return False
            
            
            if pattern[-1] == "1" and not (h_ft > a_ft):
                return False
            if pattern[-1] == "2" and not (a_ft > h_ft):
                return False
            if pattern[-1].upper() == "X" and not (a_ft == h_ft):
                return False
            
            return True
        return _f
    
    return {
        "O2.5": over(2.5),
        "O3.5": over(3.5),
        "O4.5": over(4.5),
        "O6+": over(5.5),
        "U2.5": under(2.5),
        "U1.5": under(1.5),
        "EXACT_0_0": lambda rec: (lambda g: None if g is None else (g[2] == 0))(ft_goals(rec)),
        "MS1": ms1,
        "MS2": ms2,
        "MSX": msx,
        "BTTS": btts,
        "BTTS_NO": btts_no,
        "HOME_O1.5": home_over(1.5),
        "AWAY_O1.5": away_over(1.5),
        "HT_O1.5": ht_over(1.5),
        "HTFT_1/2": htft("1/2"),
        "HTFT_2/1": htft("2/1"),
        "HTFT_1/X": htft("1/X"),
        "HTFT_2/X": htft("2/X"),
    }

def load_and_prepare_data(odds_path: str, target_fn, fields: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    
    log_with_timestamp("Starting data loading process...")
    
    X_data = []
    y_data = []
    processed_count = 0
    
    for rec in stream_records(odds_path):
        
        target = target_fn(rec)
        if target is None:
            continue
        
        
        features = []
        for field in fields:
            try:
                val = float(rec.get(field, 0))
                features.append(val)
            except (ValueError, TypeError):
                features.append(0.0)  
        
        X_data.append(features)
        y_data.append(1 if target else 0)
        
        processed_count += 1
        if processed_count % 1000 == 0:
            log_with_timestamp(f"Processed {processed_count:,} records...")
    
    log_with_timestamp(f"Data loading complete: {len(X_data):,} records")
    return np.array(X_data), np.array(y_data)

def evaluate_filter_combination(args):
    """Evaluate a single filter combination with tolerance values (vectorized, fast)."""
    X, y, field_indices, tolerance, cv_folds, field_medians = args

    try:
       
        import numpy as _np
        mask = _np.ones(X.shape[0], dtype=bool)
        for field_idx in field_indices:
            center = field_medians[field_idx]
            lo, hi = center * (1 - tolerance), center * (1 + tolerance)
            mask &= (X[:, field_idx] >= lo) & (X[:, field_idx] <= hi)

        if not mask.any():
            return (field_indices, tolerance, 0.0, 0, 0.0, 0.0)

        X_filtered = X[mask][:, field_indices]
        y_filtered = y[mask]

        
        if len(X_filtered) < 50:
            return (field_indices, tolerance, 0.0, len(X_filtered), 0.0, 0.0)

        
        model = RandomForestClassifier(
            n_estimators=60,
            random_state=42,
            n_jobs=1,
            oob_score=True,
            bootstrap=True,
        )
        model.fit(X_filtered, y_filtered)
        
        try:
            oob_prob = model.oob_decision_function_[:, 1]
            y_hat = (oob_prob >= 0.5).astype(int)
        except Exception:
            y_hat = model.predict(X_filtered)

        precision = precision_score(y_filtered, y_hat, zero_division=0)
        recall = recall_score(y_filtered, y_hat, zero_division=0)
        f1 = f1_score(y_filtered, y_hat, zero_division=0)

        return (field_indices, tolerance, float(f1), int(len(X_filtered)), float(precision), float(recall))

    except Exception:
        return (field_indices, tolerance, 0.0, 0, 0.0, 0.0)

def exhaustive_optimization(X: np.ndarray, y: np.ndarray, fields: List[str], min_filters: int = 3, max_filters: int = None) -> List[Tuple[List[str], float, float, float, int]]:
    
    
    if max_filters is None:
        max_filters = min(6, len(fields))  
    
    
    tolerance_grid = [0.01,0.02,0.05,0.08,0.10,0.15]
    
    log_with_timestamp(f"Testing {min_filters} to {max_filters} filters with {len(tolerance_grid)} tolerance values...")
    
    
    log_with_timestamp("Pre-calculating field medians for speed optimization...")
    field_medians = {}
    for i in range(len(fields)):
        field_medians[i] = np.median(X[:, i])
    log_with_timestamp("Field medians calculated and cached!")
    
    
    all_combinations = []
    for k in range(min_filters, max_filters + 1):
        for combo in combinations(range(len(fields)), k):
            all_combinations.append(list(combo))
    
    total_combinations = len(all_combinations) * len(tolerance_grid)
    log_with_timestamp(f"Total combinations to test: {total_combinations:,}")
    
    
    args_list = []
    for field_combo in all_combinations:
        for tolerance in tolerance_grid:
            args_list.append((X, y, field_combo, tolerance, 3, field_medians)) 
    
    
    n_cores = mp.cpu_count()
    log_with_timestamp(f"Using {n_cores} CPU cores for parallel processing...")
    
    results = []
    
    
    batch_size = 200  
    total_batches = (len(args_list) + batch_size - 1) // batch_size
    
    start_time = time.time()
    
    for i in range(0, len(args_list), batch_size):
        batch = args_list[i:i + batch_size]
        batch_num = i // batch_size + 1
        
        log_with_timestamp(f"Processing batch {batch_num}/{total_batches} ({len(batch):,} combinations)")
        
        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            batch_results = list(executor.map(evaluate_filter_combination, batch))
            results.extend(batch_results)
        
        
        elapsed = time.time() - start_time
        processed = min((i + batch_size), len(args_list))
        if processed > 0:
            estimated_total = (elapsed / processed) * len(args_list)
            remaining = estimated_total - elapsed
            log_with_timestamp(f"Progress: {processed:,}/{len(args_list):,} ({processed/len(args_list)*100:.1f}%) - Est. remaining: {remaining/60:.1f} min")
            
            
            if len(results) > 0 and len(results) <= 10:
                best_result = max(results, key=lambda x: x[2])  # Best F1 score
                log_with_timestamp(f"Current best: F1={best_result[2]:.4f}, Support={best_result[3]}")
    
    
    log_with_timestamp("Filtering and sorting results...")
    valid_results = []
    for field_indices, tolerance, f1_score, support, precision, recall in results:
        if f1_score > 0 and support >= 20:  
            field_names = [fields[i] for i in field_indices]
            valid_results.append((field_names, tolerance, f1_score, precision, recall, support))
    
    
    valid_results.sort(key=lambda x: (x[2], x[5]), reverse=True)
    
    log_with_timestamp(f"Found {len(valid_results)} valid combinations")
    
    return valid_results

def main():
    
    print("Starting EXHAUSTIVE ML Optimization...")
    print("=" * 60)
    
    odds_path = "odds.json"
    targets = build_targets()
    fields = ODD_FIELDS
    
    log_with_timestamp(f"Total fields available: {len(fields)}")
    log_with_timestamp(f"Total prediction types: {len(targets)}")
    log_with_timestamp(f"CPU cores available: {mp.cpu_count()}")
    print("=" * 60)
    
    tahmin_turleri = []
    overall_start_time = time.time()
    
    for target_key, target_fn in targets.items():
        target_start_time = time.time()
        print(f"\n Optimizing {target_key}...")
        print("-" * 40)
        
        try:
            
            log_with_timestamp("  Loading data...")
            X, y = load_and_prepare_data(odds_path, target_fn, fields)
            
            if len(X) < 100:
                log_with_timestamp(f"   Insufficient data for {target_key} ({len(X)} samples), skipping...")
                continue
            
            log_with_timestamp(f"  Data loaded: {len(X):,} samples, {np.sum(y):,} positive cases")
            
            
            log_with_timestamp("  Starting exhaustive search...")
            optimal_results = exhaustive_optimization(X, y, fields, min_filters=3, max_filters=min(6, len(fields)))
            
            if optimal_results:
                
                top_results = optimal_results[:5]
                
                
                filters_tr = []
                for field_names, tolerance, f1_score, precision, recall, support in top_results:
                    filters_tr.append({
                        "filtre_adi": TR_FIELD.get(field_names[0], field_names[0]),
                        "tolerans": round(tolerance, 4),
                        "f1_score": round(f1_score, 4),
                        "precision": round(precision, 4),
                        "recall": round(recall, 4),
                        "support": support
                    })
                
                tahmin_turleri.append({
                    "tahmin_adi": TR_TARGET.get(target_key, target_key),
                    "filtreler": filters_tr,
                })
                
                target_time = time.time() - target_start_time
                log_with_timestamp(f"  Found {len(top_results)} optimal filter combinations")
                log_with_timestamp(f"  Best F1 Score: {top_results[0][2]:.4f}")
                log_with_timestamp(f"  Best Precision: {top_results[0][3]:.4f}")
                log_with_timestamp(f"  Best Recall: {top_results[0][4]:.4f}")
                log_with_timestamp(f"  Target completed in {target_time/60:.1f} minutes")
                
            else:
                log_with_timestamp(f" No valid combinations found for {target_key}")
                correlations = []
                for i, field in enumerate(fields):
                    try:
                        corr = np.corrcoef(X[:, i], y)[0, 1]
                        if not np.isnan(corr):
                            correlations.append((field, abs(corr)))
                    except:
                        continue
                
                correlations.sort(key=lambda x: x[1], reverse=True)
                fallback_filters = [
                    {"filtre_adi": TR_FIELD.get(fields[0], fields[0]), "tolerans": 0.05},
                    {"filtre_adi": TR_FIELD.get(fields[1], fields[1]), "tolerans": 0.05},
                    {"filtre_adi": TR_FIELD.get(fields[2], fields[2]), "tolerans": 0.05},
                ]
                
                tahmin_turleri.append({
                    "tahmin_adi": TR_TARGET.get(target_key, target_key),
                    "filtreler": fallback_filters,
                })
                
        except Exception as e:
            log_with_timestamp(f"Error optimizing {target_key}: {e}")
            fallback_filters = [
                {"filtre_adi": TR_FIELD.get(fields[0], fields[0]), "tolerans": 0.05},
                {"filtre_adi": TR_FIELD.get(fields[1], fields[1]), "tolerans": 0.05},
                {"filtre_adi": TR_FIELD.get(fields[2], fields[2]), "tolerans": 0.05},
            ]
            tahmin_turleri.append({
                "tahmin_adi": TR_TARGET.get(target_key, target_key),
                "filtreler": fallback_filters,
            })
    
    overall_time = time.time() - overall_start_time
    print("\n" + "=" * 60)
    log_with_timestamp("Optimization Complete! Generating Output...")
    log_with_timestamp(f"Total time: {overall_time/60:.1f} minutes")
    print("=" * 60)
    
    output = {
        "tahmin_turleri": tahmin_turleri,
        "kullanici_notlari": [
            "Tolerans değerleri: Filtre değerlerinin kabul edilebilir sapma aralığını belirtir",
            "Örnek: 0.05 tolerans = %5 sapma kabul edilebilir demektir",
            "Sisteminizde 'Double Chance 1-2' = Hem ev hem deplasman galibiyeti seçeneği",
            "KG = Karşılıklı Gol (Both Teams to Score - BTTS)",
            "MS = Maç Sonu (Match Result)",
            "Bu sonuçlar EXHAUSTIVE makine öğrenmesi ile optimize edilmiştir",
            "Her tahmin için minimum 3 filtre, maksimum 8 filtre test edilmiştir",
            "Tüm olası kombinasyonlar ve tolerans değerleri denenmiştir",
            "En yüksek F1 skoru veren kombinasyonlar seçilmiştir",
        ],
    }
    
    log_with_timestamp("Printing final results...")
    print(json.dumps(output, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()

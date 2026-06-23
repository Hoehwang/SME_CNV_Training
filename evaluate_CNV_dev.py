import os
import sys
import argparse
import random
import subprocess
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import glob

# Ensure robust memory handling for tensorflow if GPU is used
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def count_lines_fast(file_path):
    out = subprocess.check_output(['wc', '-l', file_path])
    return int(out.split()[0])

def load_test_csv(file_path, train_frac, test_frac):
    total_lines = count_lines_fast(file_path)
    data_lines = total_lines - 1
    
    # 1. Reconstruct train_indices exactly as train_CNV.py did
    random.seed(42)
    train_keep_count = int(data_lines * train_frac)
    train_indices = set(random.sample(range(1, total_lines), train_keep_count))
    
    # 2. Find remaining indices
    remaining_indices = sorted(list(set(range(1, total_lines)) - train_indices))
    test_keep_count = int(data_lines * test_frac)
    
    print(f"[{os.path.basename(file_path)}] Total: {total_lines}, Train used: {train_keep_count}, Remaining: {len(remaining_indices)}, Test selecting: {test_keep_count}")
    
    # 3. Sample test_indices from remaining
    random.seed(43)
    test_indices = set(random.sample(remaining_indices, test_keep_count))
    
    # Using a callable for skiprows. We ONLY read if the row is in test_indices.
    # skiprows passes 0-indexed row numbers. Row 0 is header. Data is 1..total_lines-1.
    skip_func = lambda x: x != 0 and x not in test_indices
    return pd.read_csv(file_path, skiprows=skip_func)

def custom_focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true * alpha + (tf.ones_like(y_true) - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (tf.ones_like(y_true) - y_true) * (1 - y_pred)
        focal_loss_val = - alpha_t * tf.math.pow((1 - p_t), gamma) * tf.math.log(p_t)
        return tf.reduce_mean(focal_loss_val)
    return focal_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_frac', type=float, default=0.2, help='Downsample fraction used during training (to exclude it)')
    parser.add_argument('--test_frac', type=float, default=0.1, help='Fraction of total data to use for this independent test set')
    parser.add_argument('--category', type=str, default='dup', choices=['dup', 'del'])
    parser.add_argument('--cache_only', action='store_true', help='Stop execution after generating the test parquet caches')
    parser.add_argument('--model_type', type=str, default='dnn', choices=['dnn', 'lr', 'autokeras', 'mlp', 'linear', 'nb', 'rf'], help='Model type to use for evaluation')
    parser.add_argument('--max_iter', type=int, default=200, help='Max iterations for LR and MLP to find the correct model directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for DNN to find the correct model directory')
    parser.add_argument('--device', type=str, default='GPU', choices=['GPU', 'CPU'], help='Compute device')
    args = parser.parse_args()
    
    if args.device == 'CPU':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        print("Set to use CPU for evaluation.")
    else:
        print("Set to use GPU for evaluation.")
    
    data_dir = 'Synthetic_400K'
    file_prefix = 'duple' if args.category == 'dup' else 'del'
    
    files_to_load = [
        os.path.join(data_dir, f'CNV_{file_prefix}_male_synthetic.csv'),
        os.path.join(data_dir, f'CNV_{file_prefix}_female_synthetic.csv'),
        os.path.join(data_dir, 'CNV_normal_male_synthetic.csv'),
        os.path.join(data_dir, 'CNV_normal_female_synthetic.csv')
    ]
    
    test_cached_paths = []
    for f in files_to_load:
        base_name = os.path.basename(f).replace('.csv', '')
        # Name the cache explicitly so it doesn't collide
        out_path = os.path.join(data_dir, f"{base_name}_EXTERNAL_TEST_tr{args.train_frac}_te{args.test_frac}.parquet")
        test_cached_paths.append(out_path)
        
    def is_cache_valid(path):
        if not os.path.exists(path):
            return False
        try:
            pq.read_metadata(path)
            return True
        except Exception:
            print(f"Cache {path} is incomplete or corrupted. Removing...")
            os.remove(path)
            return False

    valid_caches = all(is_cache_valid(p) for p in test_cached_paths)
    
    if not valid_caches:
        print("Generating independent TEST parquet caches...")
        from multiprocessing import Process
        def worker_task(csv_file, train_frac, test_frac, out_path):
            if not is_cache_valid(out_path):
                print(f"Creating test cache for {os.path.basename(csv_file)}...")
                df = load_test_csv(csv_file, train_frac, test_frac)
                df.to_parquet(out_path, index=False)
                del df
                
        for csv_file, out_path in zip(files_to_load, test_cached_paths):
            p = Process(target=worker_task, args=(csv_file, args.train_frac, args.test_frac, out_path))
            p.start()
            p.join()
            
        print("Test cache generation complete.")
    else:
        print("Found fully intact independent TEST parquet caches.")

    if args.cache_only:
        print("Cache generation complete. Exiting due to --cache_only flag.")
        return

    # Read schemas to know which columns exist
    meta_cnv_m = pq.read_metadata(test_cached_paths[0])
    meta_cnv_f = pq.read_metadata(test_cached_paths[1])
    meta_norm_m = pq.read_metadata(test_cached_paths[2])
    meta_norm_f = pq.read_metadata(test_cached_paths[3])
    
    schema_cnv_m = set(meta_cnv_m.schema.names)
    schema_cnv_f = set(meta_cnv_f.schema.names)
    schema_norm_m = set(meta_norm_m.schema.names)
    schema_norm_f = set(meta_norm_f.schema.names)
    all_cols_set = schema_cnv_m.union(schema_cnv_f).union(schema_norm_m).union(schema_norm_f)

    # Load targets
    d_info = pd.read_excel('Orphanet_Names_filtered_2604.xlsx')
    target_info = d_info[d_info['disease'].str.contains(args.category, na=False, case=False)].reset_index(drop=True)

    total_diseases = len(target_info)
    
    results = []
    if args.model_type in ['lr', 'mlp']:
        model_dir = f'models_{args.category}_{args.model_type}_iter_{args.max_iter}'
    elif args.model_type == 'dnn':
        model_dir = f'models_{args.category}_dnn_epoch_{args.epochs}'
    else:
        model_dir = f'models_{args.category}_{args.model_type}'
    csv_results_path = os.path.join(model_dir, f'{args.category}_external_test_results.csv')
    
    print(f"==================================================")
    print(f"Starting external evaluation for {total_diseases} diseases.")
    print(f"==================================================")
    
    def safe_load(path, schema_set, cols, ur_col):
        read_cols = []
        rename_map = {}
        for c in cols:
            if c in schema_set:
                read_cols.append(c)
            elif c == ur_col and 'UR' in schema_set:
                read_cols.append('UR')
                rename_map['UR'] = c
        
        read_cols = list(set(read_cols))
        df = pd.read_parquet(path, columns=read_cols)
        if rename_map:
            df = df.rename(columns=rename_map)
            
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan
        return df

    for idx, row in target_info.iterrows():
        target_disease = row['disease']
        
        if args.model_type == 'lr':
            model_path = os.path.join(model_dir, f"{target_disease}_lr_model.pkl")
        elif args.model_type == 'mlp':
            model_path = os.path.join(model_dir, f"{target_disease}_mlp_model.pkl")
        elif args.model_type == 'linear':
            model_path = os.path.join(model_dir, f"{target_disease}_linear_model.pkl")
        elif args.model_type == 'nb':
            model_path = os.path.join(model_dir, f"{target_disease}_nb_model.pkl")
        elif args.model_type == 'rf':
            model_path = os.path.join(model_dir, f"{target_disease}_rf_model.pkl")
        else:
            model_path = os.path.join(model_dir, f"{target_disease}_model.h5")
        
        if not os.path.exists(model_path):
            print(f"[{idx+1}/{total_diseases}] Model for {target_disease} not found. Skipping.")
            continue
            
        print(f"\n[{idx+1}/{total_diseases}] Evaluating Model for {target_disease}...")
        
        chr_num = row['chr']
        if str(chr_num) == 'X':
            chr_num = 23
        elif str(chr_num) == 'Y':
            chr_num = 24
            
        prefix = f'{target_disease}_chr{chr_num}_'
        chr_columns = [c for c in all_cols_set if c.startswith(prefix)]
        chr_columns.sort(key=lambda x: int(x.split('_')[-1]))
        
        if not chr_columns:
            print(f"Skipping {target_disease}: No bin columns found starting with {prefix}")
            continue
            
        dep_columns = ['GC', 'snp_FF']
        ur_column = f'{target_disease}_UR'
        read_columns = [ur_column] + chr_columns

        missing_columns = [col for col in dep_columns + read_columns if col not in all_cols_set]
        if missing_columns:
            print(f"Skipping {target_disease}: Missing columns {missing_columns[:5]}...")
            continue

        cols_to_load = dep_columns + read_columns

        cnv_m = safe_load(test_cached_paths[0], schema_cnv_m, cols_to_load, ur_column)
        cnv_f = safe_load(test_cached_paths[1], schema_cnv_f, cols_to_load, ur_column)
        CNV_test = pd.concat([cnv_m, cnv_f], ignore_index=True)
        CNV_test['result'] = 1
        del cnv_m, cnv_f
        
        norm_m = safe_load(test_cached_paths[2], schema_norm_m, cols_to_load, ur_column)
        norm_f = safe_load(test_cached_paths[3], schema_norm_f, cols_to_load, ur_column)
        Normal_test = pd.concat([norm_m, norm_f], ignore_index=True)
        Normal_test['result'] = 0
        del norm_m, norm_f
        
        test_df = pd.concat([Normal_test, CNV_test], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
        del Normal_test, CNV_test

        X_test_dep = test_df[dep_columns].copy()
        X_test_read = test_df[read_columns].copy()
        y_test = test_df['result'].values
        del test_df
        
        from sklearn.preprocessing import StandardScaler
        # Note: train_CNV_dev.py uses StandardScaler. We must match it.
        scaler_gc = StandardScaler()
        scaler_ff = StandardScaler()
        scaler_read = StandardScaler()

        X_test_dep['GC'] = scaler_gc.fit_transform(X_test_dep[['GC']])
        X_test_dep['snp_FF'] = scaler_ff.fit_transform(X_test_dep[['snp_FF']])
        X_test_read_scaled = scaler_read.fit_transform(X_test_read)
        X_test_dep_scaled = X_test_dep.values
        
        # Load Model
        if args.model_type in ['lr', 'mlp', 'linear', 'nb', 'rf']:
            import joblib
            from sklearn.metrics import log_loss, accuracy_score
            model = joblib.load(model_path)
            X_test_combined = np.hstack([X_test_dep_scaled, X_test_read_scaled])
            
            if args.model_type == 'linear':
                y_pred_raw = model.predict(X_test_combined)
                y_pred_prob = np.clip(y_pred_raw, 0, 1)
                y_pred = (y_pred_raw >= 0.5).astype(int)
                loss = log_loss(y_test, y_pred_prob)
                accuracy = accuracy_score(y_test, y_pred)
            else:
                y_pred_prob = model.predict_proba(X_test_combined)[:, 1]
                y_pred = model.predict(X_test_combined)
                loss = log_loss(y_test, y_pred_prob)
                accuracy = model.score(X_test_combined, y_test)
        else:
            custom_objects = {'focal_loss': custom_focal_loss()}
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
            
            loss, accuracy, _ = model.evaluate(
                x=[X_test_dep_scaled, X_test_read_scaled],
                y=y_test,
                verbose=0
            )
            
            y_pred_prob = model.predict([X_test_dep_scaled, X_test_read_scaled], verbose=0)
            y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        
        def format_with_ci(p, n):
            if n == 0:
                return "0.0000"
            margin = 1.96 * np.sqrt((p * (1 - p)) / n)
            lower = max(0.0, p - margin)
            upper = min(1.0, p + margin)
            return f"{p:.4f} ({lower:.4f}-{upper:.4f})"
            
        specificity_val = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sensitivity_val = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        ppv_val = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv_val = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        
        specificity = format_with_ci(specificity_val, tn + fp)
        sensitivity = format_with_ci(sensitivity_val, tp + fn)
        ppv = format_with_ci(ppv_val, tp + fp)
        npv = format_with_ci(npv_val, tn + fn)
        accuracy_str = format_with_ci(accuracy, tn + fp + fn + tp)
        
        print(f"Results for {target_disease} -> Specificity: {specificity} | Sensitivity: {sensitivity} | PPV: {ppv} | NPV: {npv} | Acc: {accuracy_str}")
        
        results.append({
            'Disease': target_disease,
            'Category': args.category.upper(),
            'Specificity': specificity,
            'Sensitivity': sensitivity,
            'PPV': ppv,
            'NPV': npv,
            'Test_Loss': round(loss, 4),
            'Test_Accuracy': accuracy_str
        })
        pd.DataFrame(results).to_csv(csv_results_path, index=False)
        tf.keras.backend.clear_session()

if __name__ == "__main__":
    main()

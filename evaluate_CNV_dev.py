import os
import sys

# --- NVIDIA GPU 라이브러리 자동 환경 변수 추가 시작 ---
if 'LD_LIBRARY_PATH' not in os.environ or 'site-packages/nvidia' not in os.environ['LD_LIBRARY_PATH']:
    try:
        nvidia_dir = os.path.join(sys.prefix, 'lib', f'python{sys.version_info.major}.{sys.version_info.minor}', 'site-packages', 'nvidia')
        if os.path.exists(nvidia_dir):
            libs = ['cublas', 'cudnn', 'cufft', 'curand', 'cusolver', 'cusparse', 'nccl', 'nvjitlink']
            paths = [os.path.join(nvidia_dir, lib, 'lib') for lib in libs]
            paths = [p for p in paths if os.path.exists(p)]
            if paths:
                new_ld_path = ':'.join(paths)
                if 'LD_LIBRARY_PATH' in os.environ:
                    new_ld_path += ':' + os.environ['LD_LIBRARY_PATH']
                os.environ['LD_LIBRARY_PATH'] = new_ld_path
                if sys.argv and os.path.exists(sys.argv[0]):
                    os.execv(sys.executable, [sys.executable] + sys.argv)
    except Exception as e:
        print("Warning: Could not automatically set LD_LIBRARY_PATH for NVIDIA libraries:", e)
# --- NVIDIA GPU 라이브러리 자동 환경 변수 추가 끝 ---

import argparse
import random
import subprocess
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import glob

# Ensure robust memory handling for tensorflow if GPU is used
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def convert_csv_to_parquet(csv_path, parquet_path):
    print(f"Converting CSV {csv_path} -> {parquet_path}...")
    writer = None
    for chunk in pd.read_csv(csv_path, chunksize=10000):
        float_cols = chunk.select_dtypes(include=['float64']).columns
        chunk[float_cols] = chunk[float_cols].astype('float32')
        table = pa.Table.from_pandas(chunk)
        if writer is None:
            writer = pq.ParquetWriter(parquet_path, table.schema)
        writer.write_table(table)
    if writer is not None:
        writer.close()
    print(f"Parquet conversion complete: {parquet_path}")

def custom_focal_loss(alpha=0.25, gamma=3.0):
    def focal_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        modulating_factor = tf.pow(1.0 - p_t, gamma)
        return tf.reduce_mean(alpha_factor * modulating_factor * bce)
    return focal_loss

def main():
    parser = argparse.ArgumentParser(description="Evaluate CNV Models on Test Dataset (Synthetic_Test)")
    parser.add_argument('--data_dir', type=str, default='Synthetic_Test', help='Path to test dataset directory (default: Synthetic_Test)')
    parser.add_argument('--category', type=str, default='del', choices=['dup', 'del'], help='Target category (dup or del)')
    parser.add_argument('--model_type', type=str, default='dnn', choices=['dnn', 'lr', 'autokeras', 'mlp', 'linear', 'nb', 'rf'], help='Model type to use for evaluation')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for DNN to find the correct model directory (default: 10)')
    parser.add_argument('--max_iter', type=int, default=200, help='Max iterations for LR and MLP to find the correct model directory')
    parser.add_argument('--device', type=str, default='GPU', choices=['GPU', 'CPU'], help='Compute device')
    parser.add_argument('--model_dir', type=str, default='', help='Direct path to model directory (e.g., models_del_dnn_epoch_10)')
    parser.add_argument('--test_frac', type=float, default=1.0, help='Fraction of test data to evaluate (default: 1.0 for all)')
    args = parser.parse_args()
    
    if args.device == 'CPU':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        print("Set to use CPU for evaluation.")
    else:
        print("Set to use GPU for evaluation.")
    
    data_dir = args.data_dir
    file_prefix = 'duple' if args.category == 'dup' else 'del'
    
    # 4 dataset targets to resolve
    target_names = [
        f'CNV_{file_prefix}_male_synthetic',
        f'CNV_{file_prefix}_female_synthetic',
        'CNV_normal_male_synthetic',
        'CNV_normal_female_synthetic'
    ]
    
    test_cached_paths = []
    for base in target_names:
        p_parquet = os.path.join(data_dir, f"{base}.parquet")
        p_parquet_up = os.path.join(data_dir, f"{base}_updated.parquet")
        p_csv = os.path.join(data_dir, f"{base}.csv")
        p_csv_up = os.path.join(data_dir, f"{base}_updated.csv")
        
        if os.path.exists(p_parquet):
            test_cached_paths.append(p_parquet)
        elif os.path.exists(p_parquet_up):
            test_cached_paths.append(p_parquet_up)
        elif os.path.exists(p_csv):
            out_p = os.path.join(data_dir, f"{base}.parquet")
            convert_csv_to_parquet(p_csv, out_p)
            test_cached_paths.append(out_p)
        elif os.path.exists(p_csv_up):
            out_p = os.path.join(data_dir, f"{base}_updated.parquet")
            convert_csv_to_parquet(p_csv_up, out_p)
            test_cached_paths.append(out_p)
        else:
            raise FileNotFoundError(f"Could not find {base} (.parquet or .csv) in {data_dir}!")
            
    print(f"Using test datasets from {data_dir}:")
    for p in test_cached_paths:
        print(f"  - {os.path.basename(p)}")

    # Read schemas to know which columns exist
    meta_cnv_m = pq.read_metadata(test_cached_paths[0])
    meta_cnv_f = pq.read_metadata(test_cached_paths[1])
    meta_norm_m = pq.read_metadata(test_cached_paths[2])
    meta_norm_f = pq.read_metadata(test_cached_paths[3])
    
    total_pos = meta_cnv_m.num_rows + meta_cnv_f.num_rows
    total_neg = meta_norm_m.num_rows + meta_norm_f.num_rows
    if args.test_frac < 1.0:
        total_pos = int(total_pos * args.test_frac)
        total_neg = int(total_neg * args.test_frac)
    
    print(f"Test CNV Positive samples: {total_pos:,}")
    print(f"Test Normal Negative samples: {total_neg:,}")
    print(f"Total Evaluation samples: {total_pos + total_neg:,}")
    
    schema_cnv_m = set(meta_cnv_m.schema.names)
    schema_cnv_f = set(meta_cnv_f.schema.names)
    schema_norm_m = set(meta_norm_m.schema.names)
    schema_norm_f = set(meta_norm_f.schema.names)
    all_cols_set = schema_cnv_m.union(schema_cnv_f).union(schema_norm_m).union(schema_norm_f)

    # Load targets
    d_info = pd.read_excel('Orphanet_Names_filtered_2604.xlsx')
    if args.category == 'dup':
        target_info = d_info[d_info['disease'].str.contains('dup|tetra', na=False, case=False, regex=True)].reset_index(drop=True)
    else:
        target_info = d_info[d_info['disease'].str.contains(args.category, na=False, case=False)].reset_index(drop=True)

    total_diseases = len(target_info)
    
    if args.model_dir:
        model_dir = args.model_dir
    elif args.model_type in ['lr', 'mlp']:
        model_dir = f'models_{args.category}_{args.model_type}_iter_{args.max_iter}'
    elif args.model_type == 'dnn':
        model_dir = f'models_{args.category}_dnn_epoch_{args.epochs}'
    else:
        model_dir = f'models_{args.category}_{args.model_type}'
        
    csv_results_path = os.path.join(model_dir, f'{args.category}_synthetic_test_results.csv')
    
    results = []
    evaluated_diseases = set()
    if os.path.exists(csv_results_path):
        try:
            results = pd.read_csv(csv_results_path).to_dict('records')
            evaluated_diseases = {r.get('Disease') for r in results}
            print(f"Loaded {len(results)} existing evaluation results from {csv_results_path}.")
        except Exception:
            pass
            
    print(f"==================================================")
    print(f"Starting evaluation for {total_diseases} diseases in {model_dir}")
    print(f"Results will be saved to: {csv_results_path}")
    print(f"==================================================")
    
    def safe_load(path, schema_set, cols, ur_col, frac=1.0):
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
                
        if frac < 1.0:
            df = df.sample(frac=frac, random_state=42).reset_index(drop=True)
            
        return df

    for idx, row in target_info.iterrows():
        target_disease = row['disease']
        
        if target_disease in evaluated_diseases:
            print(f"\n---> [{idx+1}/{total_diseases}] Skipping {target_disease}: Already evaluated.")
            continue
            
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
            print(f"\n---> [{idx+1}/{total_diseases}] Model for {target_disease} not found at {model_path}. Skipping.")
            continue
            
        print(f"\n---> [{idx+1}/{total_diseases}] Evaluating Model for {target_disease}...")
        
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

        cnv_m = safe_load(test_cached_paths[0], schema_cnv_m, cols_to_load, ur_column, args.test_frac)
        cnv_f = safe_load(test_cached_paths[1], schema_cnv_f, cols_to_load, ur_column, args.test_frac)
        CNV_test = pd.concat([cnv_m, cnv_f], ignore_index=True)
        CNV_test['result'] = 1
        del cnv_m, cnv_f
        
        norm_m = safe_load(test_cached_paths[2], schema_norm_m, cols_to_load, ur_column, args.test_frac)
        norm_f = safe_load(test_cached_paths[3], schema_norm_f, cols_to_load, ur_column, args.test_frac)
        Normal_test = pd.concat([norm_m, norm_f], ignore_index=True)
        Normal_test['result'] = 0
        del norm_m, norm_f
        
        test_df = pd.concat([Normal_test, CNV_test], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
        del Normal_test, CNV_test

        X_test_dep = test_df[dep_columns].copy()
        X_test_read = test_df[read_columns].copy()
        y_test = test_df['result'].values
        del test_df
        
        # Scaling must match train_CNV_dev.py (MinMaxScaler)
        scaler_gc = MinMaxScaler()
        scaler_ff = MinMaxScaler()
        scaler_read = MinMaxScaler()

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
            
            eval_res = model.evaluate(
                x=[X_test_dep_scaled, X_test_read_scaled],
                y=y_test,
                verbose=0
            )
            if isinstance(eval_res, (list, tuple, np.ndarray)):
                loss = float(eval_res[0])
            else:
                loss = float(eval_res)
            
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
        accuracy_val = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        
        specificity = format_with_ci(specificity_val, tn + fp)
        sensitivity = format_with_ci(sensitivity_val, tp + fn)
        ppv = format_with_ci(ppv_val, tp + fp)
        npv = format_with_ci(npv_val, tn + fn)
        accuracy_str = format_with_ci(accuracy_val, tn + fp + fn + tp)
        
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
        del model
        import gc
        gc.collect()

    print("==================================================")
    print(f"Evaluation complete! Results saved to {csv_results_path}")

if __name__ == "__main__":
    main()

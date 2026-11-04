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
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.metrics import confusion_matrix, log_loss, accuracy_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import joblib
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Dropout, Concatenate, Activation
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras import regularizers

def get_args():
    parser = argparse.ArgumentParser(description="Sequential Two-Branch DNN for Duplication CNV")
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size (default: 1024)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--device', type=str, default='GPU', choices=['GPU', 'CPU'], help='Compute device')
    parser.add_argument('--gpu', type=str, default='0', help='Specific GPU index to use (e.g., 0, 1, 2, 3)')
    parser.add_argument('--downsample_frac', type=float, default=1.0, help='Fraction of data to load (e.g., 0.05 for 400K)')
    parser.add_argument('--category', type=str, default='dup', choices=['dup', 'del', 'both'], help='Target category (dup, del, or both)')
    parser.add_argument('--data_dir', type=str, default='Synthetic', help='Dataset directory (default: Synthetic)')
    parser.add_argument('--model_dir', type=str, default='', help='Custom output model directory (e.g. models_del_dnn_400k)')
    parser.add_argument('--model_type', type=str, default='dnn', choices=['dnn', 'lr', 'autokeras', 'mlp', 'linear', 'nb', 'rf'], help='Model type to use for training')
    parser.add_argument('--ak_max_trials', type=int, default=3, help='Max trials for AutoKeras')
    parser.add_argument('--max_iter', type=int, default=200, help='Max iterations for LR and MLP')
    parser.add_argument('--nb_var_smoothing', type=float, default=1e-9, help='var_smoothing parameter for Naive Bayes')
    parser.add_argument('--rf_n_estimators', type=int, default=100, help='n_estimators for Random Forest')
    parser.add_argument('--rf_max_depth', type=int, default=-1, help='max_depth for Random Forest (-1 for None)')
    parser.add_argument('--iqr_filter', type=str, default='none', choices=['none', 'both', 'normal', 'abnormal'], help='Apply IQR outlier removal on UR column')
    parser.add_argument('--patience', type=int, default=5, help='Patience for EarlyStopping (default: 5)')
    parser.add_argument('--min_delta', type=float, default=1e-4, help='Minimum change in val_loss for EarlyStopping (default: 1e-4)')
    parser.add_argument('--tag', type=str, default='', help='Suffix tag for model output directory (e.g. es_400k)')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing trained models instead of skipping')
    parser.add_argument('--memory_growth', action='store_true', default=False, help='Enable GPU memory growth (default: False for maximum speed)')
    return parser.parse_args()

def custom_focal_loss(alpha=0.25, gamma=3.0):
    def focal_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        modulating_factor = tf.pow(1.0 - p_t, gamma)
        return tf.reduce_mean(alpha_factor * modulating_factor * bce)
    return focal_loss

def train_logistic_regression(X_train, y_train, X_test, y_test, target_disease, max_iter):
    print(f"[{target_disease}] Training Logistic Regression...")
    clf = LogisticRegression(max_iter=max_iter, random_state=42, C=1, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    y_pred_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)
    
    loss = log_loss(y_test, y_pred_prob)
    accuracy = accuracy_score(y_test, y_pred)
    
    return clf, loss, accuracy, y_pred_prob, y_pred

def train_linear_regression(X_train, y_train, X_test, y_test, target_disease):
    print(f"[{target_disease}] Training Linear Regression...")
    clf = LinearRegression(n_jobs=-1)
    clf.fit(X_train, y_train)
    
    y_pred_raw = clf.predict(X_test)
    y_pred_prob = np.clip(y_pred_raw, 0, 1)
    y_pred = (y_pred_raw >= 0.5).astype(int)
    
    loss = log_loss(y_test, y_pred_prob)
    accuracy = accuracy_score(y_test, y_pred)
    
    return clf, loss, accuracy, y_pred_prob, y_pred

def train_naive_bayes(X_train, y_train, X_test, y_test, target_disease, var_smoothing):
    print(f"[{target_disease}] Training Naive Bayes...")
    clf = GaussianNB(var_smoothing=var_smoothing)
    clf.fit(X_train, y_train)
    
    y_pred_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)
    
    loss = log_loss(y_test, y_pred_prob)
    accuracy = accuracy_score(y_test, y_pred)
    
    return clf, loss, accuracy, y_pred_prob, y_pred

def train_random_forest(X_train, y_train, X_test, y_test, target_disease, n_estimators, max_depth):
    print(f"[{target_disease}] Training Random Forest...")
    md = None if max_depth == -1 else max_depth
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=md, random_state=1234, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    y_pred_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)
    
    loss = log_loss(y_test, y_pred_prob)
    accuracy = accuracy_score(y_test, y_pred)
    
    return clf, loss, accuracy, y_pred_prob, y_pred

def train_mlp_classifier(X_train, y_train, X_test, y_test, target_disease, max_iter):
    print(f"[{target_disease}] Training MLP Classifier...")
    clf = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=max_iter, random_state=1234, early_stopping=True)
    clf.fit(X_train, y_train)
    
    y_pred_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)
    
    loss = log_loss(y_test, y_pred_prob)
    accuracy = accuracy_score(y_test, y_pred)
    
    return clf, loss, accuracy, y_pred_prob, y_pred

def train_for_category(args, category, d_info):
    if category == 'dup':
        target_info = d_info[d_info['disease'].str.contains('dup|tetra', na=False, case=False, regex=True)].reset_index(drop=True)
    else:
        target_info = d_info[d_info['disease'].str.contains(category, na=False, case=False)].reset_index(drop=True)
        
    if len(target_info) == 0:
        print(f"No {category} targets found in Excel.")
        return

    data_dir = args.data_dir
    file_prefix = 'duple' if category == 'dup' else 'del'

    if file_prefix == 'duple':
        files_to_load = [
            os.path.join(data_dir, f'CNV_{file_prefix}_male_synthetic.csv'),
            os.path.join(data_dir, f'CNV_{file_prefix}_female_synthetic.csv'),
            os.path.join(data_dir, 'CNV_normal_male_synthetic.csv'),
            os.path.join(data_dir, 'CNV_normal_female_synthetic.csv')
        ]

    else:
        files_to_load = [
            os.path.join(data_dir, f'CNV_{file_prefix}_male_synthetic_updated.csv'),
            os.path.join(data_dir, f'CNV_{file_prefix}_female_synthetic_updated.csv'),
            os.path.join(data_dir, 'CNV_normal_male_synthetic.csv'),
            os.path.join(data_dir, 'CNV_normal_female_synthetic.csv')
        ]

    cached_paths = []
    for f in files_to_load:
        base_name = os.path.basename(f).replace('.csv', '')
        out_path = os.path.join(data_dir, f"{base_name}_frac_{args.downsample_frac}.parquet")
        cached_paths.append(out_path)

    if all(os.path.exists(p) for p in cached_paths):
        print(f"Found all 4 cached parquet files for fraction {args.downsample_frac}.")
    else:
        import subprocess
        import random
        
        def count_lines_fast(file_path):
            print(f"Counting exact total lines for {os.path.basename(file_path)} using optimized 'wc' (Disk I/O only)...")
            out = subprocess.check_output(['wc', '-l', file_path])
            return int(out.split()[0])
            
        def load_sampled_csv(file_path, frac):
            if frac >= 1.0:
                print(f"Loading {os.path.basename(file_path)} entirely (no downsampling)...")
                return pd.read_csv(file_path)
            
            total_lines = count_lines_fast(file_path)
            random.seed(42)
            data_lines = total_lines - 1
            keep_count = int(data_lines * frac)
            
            print(f"[{os.path.basename(file_path)}] Total lines: {total_lines}. Sampling exact {keep_count} rows...")
            keep_indices = set(random.sample(range(1, total_lines), keep_count))
            
            # Using a callable for skiprows. Pandas C-engine will skip lines without parsing floats!
            skip_func = lambda x: x != 0 and x not in keep_indices
            return pd.read_csv(file_path, skiprows=skip_func)

        try:
            
            from multiprocessing import Process
            
            def worker_task(file_path, frac, out_path):
                if os.path.exists(out_path):
                    print(f"[{os.path.basename(file_path)}] Cache already exists: {os.path.basename(out_path)}. Skipping CSV parsing!")
                    return

                # If 1.0 parquet already exists, fast downsample directly from it without parsing massive CSVs
                base_name = os.path.basename(file_path).replace('.csv', '')
                frac_1_path = os.path.join(data_dir, f"{base_name}_frac_1.0.parquet")
                if frac < 1.0 and os.path.exists(frac_1_path):
                    print(f"[{os.path.basename(file_path)}] Fast downsampling from existing {os.path.basename(frac_1_path)}...")
                    import pyarrow.parquet as pq
                    from tqdm import tqdm
                    pf = pq.ParquetFile(frac_1_path)
                    total_rg = pf.num_row_groups
                    keep_rg = max(1, int(total_rg * frac))
                    print(f"[{os.path.basename(file_path)}] Streaming {keep_rg}/{total_rg} row groups directly into cache...")
                    writer = None
                    total_rows = 0
                    for rg_idx in tqdm(range(keep_rg), desc=f"Caching {os.path.basename(out_path)}"):
                        rg_table = pf.read_row_group(rg_idx)
                        total_rows += rg_table.num_rows
                        if writer is None:
                            writer = pq.ParquetWriter(out_path, rg_table.schema)
                        writer.write_table(rg_table)
                    if writer is not None:
                        writer.close()
                    print(f"[{os.path.basename(file_path)}] Parquet saved successfully ({total_rows:,} rows).")
                    return
                    
                print(f"[{os.path.basename(file_path)}] Parsing CSV in chunks -> {os.path.basename(out_path)}...")
                
                total_lines = count_lines_fast(file_path)
                data_lines = total_lines - 1
                
                skip_func = None
                if frac < 1.0:
                    import random
                    random.seed(42)
                    keep_count = int(data_lines * frac)
                    print(f"[{os.path.basename(file_path)}] Total lines: {total_lines}. Sampling exact {keep_count} rows...")
                    keep_indices = set(random.sample(range(1, total_lines), keep_count))
                    skip_func = lambda x: x != 0 and x not in keep_indices
                    progress_total = keep_count
                else:
                    print(f"Loading {os.path.basename(file_path)} entirely ({data_lines} rows, no downsampling)...")
                    progress_total = data_lines
                
                import pyarrow as pa
                import pyarrow.parquet as pq
                from tqdm import tqdm
                
                writer = None
                with tqdm(total=progress_total, desc=os.path.basename(file_path)) as pbar:
                    for chunk in pd.read_csv(file_path, skiprows=skip_func, chunksize=5000):
                        float_cols = chunk.select_dtypes(include=['float64']).columns
                        chunk[float_cols] = chunk[float_cols].astype('float32')
                        
                        table = pa.Table.from_pandas(chunk)
                        if writer is None:
                            writer = pq.ParquetWriter(out_path, table.schema)
                        writer.write_table(table)
                        pbar.update(len(chunk))
                
                if writer is not None:
                    writer.close()
                print(f"[{os.path.basename(file_path)}] Parquet saved successfully.")
            
            cached_paths = []
            
            print(f"Avoiding OOM: Processing 4 files sequentially...")
            for i, f in enumerate(files_to_load):
                # Save as individual cache files based on args so they can be reused!
                base_name = os.path.basename(f).replace('.csv', '')
                out_path = os.path.join(data_dir, f"{base_name}_frac_{args.downsample_frac}.parquet")
                cached_paths.append(out_path)
                
                worker_task(f, args.downsample_frac, out_path)
                
            print("Caching complete!")
                    
        except Exception as e:
            print(f"Failed to load synthetic datasets: {e}")
            return

    # Read metadata using PyArrow without loading into memory
    import pyarrow.parquet as pq
    print("Reading metadata from Parquet files...")
    meta_cnv_m = pq.read_metadata(cached_paths[0])
    meta_cnv_f = pq.read_metadata(cached_paths[1])
    meta_norm_m = pq.read_metadata(cached_paths[2])
    meta_norm_f = pq.read_metadata(cached_paths[3])

    print(f"CNV positive samples total: {meta_cnv_m.num_rows + meta_cnv_f.num_rows}")
    print(f"Normal negative samples total: {meta_norm_m.num_rows + meta_norm_f.num_rows}")
    
    # Create models output directory
    if args.model_dir:
        if '{category}' in args.model_dir:
            model_dir = args.model_dir.format(category=category)
        elif args.category == 'both':
            if '_del_' in args.model_dir or '_dup_' in args.model_dir:
                model_dir = args.model_dir.replace('_dup_', f'_{category}_').replace('_del_', f'_{category}_')
            elif args.model_dir.startswith('models_del_') or args.model_dir.startswith('models_dup_'):
                model_dir = args.model_dir.replace('models_del_', f'models_{category}_').replace('models_dup_', f'models_{category}_')
            elif args.model_dir.startswith('models_'):
                model_dir = f"models_{category}_" + args.model_dir[len('models_'):]
            else:
                model_dir = f"{args.model_dir}_{category}"
        else:
            model_dir = args.model_dir
    else:
        suffix = f"_{args.tag}" if args.tag else (f"_{int(args.downsample_frac * 8000)}k" if args.downsample_frac < 1.0 else "")
        if args.model_type in ['lr', 'mlp']:
            model_dir = f'models_{category}_{args.model_type}_iter_{args.max_iter}{suffix}'
        elif args.model_type == 'dnn':
            model_dir = f'models_{category}_dnn_epoch_{args.epochs}{suffix}'
        else:
            model_dir = f'models_{category}_{args.model_type}{suffix}'
    os.makedirs(model_dir, exist_ok=True)
    
    # Generate dataset scale summary markdown file in model directory
    md_summary_path = os.path.join(model_dir, 'dataset_info.md')
    pos_m = meta_cnv_m.num_rows
    pos_f = meta_cnv_f.num_rows
    pos_total = pos_m + pos_f
    neg_m = meta_norm_m.num_rows
    neg_f = meta_norm_f.num_rows
    neg_total = neg_m + neg_f
    grand_total = pos_total + neg_total
    train_samples = int(grand_total * 0.8)
    test_samples = grand_total - train_samples
    
    from datetime import datetime
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    md_content = f"""# Training Dataset Scale & Configuration

- **Created At**: {now_str}
- **Category**: {category.upper()}
- **Model Type**: {args.model_type}
- **Downsample Fraction**: {args.downsample_frac}
- **Total Diseases**: {len(target_info)}

---

## 1. Dataset Scale Summary

| Classification | Subgroup | Number of Samples (Rows) | Proportion |
| :--- | :--- | :--- | :--- |
| **CNV Positive (양성)** | Male (남자) | {pos_m:,} | {pos_m / grand_total * 100:.1f}% |
| | Female (여자) | {pos_f:,} | {pos_f / grand_total * 100:.1f}% |
| | **Positive Subtotal** | **{pos_total:,}** | **{pos_total / grand_total * 100:.1f}%** |
| **Normal Negative (음성)** | Male (남자) | {neg_m:,} | {neg_m / grand_total * 100:.1f}% |
| | Female (여자) | {neg_f:,} | {neg_f / grand_total * 100:.1f}% |
| | **Negative Subtotal** | **{neg_total:,}** | **{neg_total / grand_total * 100:.1f}%** |
| **Grand Total (전체)** | **All 4 Groups Combined** | **{grand_total:,}** | **100.0%** |

---

## 2. Train / Test Split (Per Disease Baseline)

- **Train Split (80%)**: ~{train_samples:,} samples
- **Test Split (20%)**: ~{test_samples:,} samples
*(Note: Per-disease outlier removal such as IQR filter may slightly adjust final sample counts)*

---

## 3. Training Parameters

- **Compute Device**: {args.device}
- **Epochs**: {args.epochs}
- **Batch Size**: {args.batch_size}
- **Learning Rate**: {args.lr}
- **Dropout Rate**: {args.dropout}
- **IQR Filter**: {args.iqr_filter}
"""
    with open(md_summary_path, 'w', encoding='utf-8') as f_md:
        f_md.write(md_content)
    print(f"Saved dataset scale summary to {md_summary_path}")
    
    schema_cnv_m = set(meta_cnv_m.schema.names)
    schema_cnv_f = set(meta_cnv_f.schema.names)
    schema_norm_m = set(meta_norm_m.schema.names)
    schema_norm_f = set(meta_norm_f.schema.names)
    all_cols_set = schema_cnv_m.union(schema_cnv_f).union(schema_norm_m).union(schema_norm_f)

    # Max Index calculation is no longer needed because columns are now disease-specific

    # START SEQUENTIAL LOOP
    total_diseases = len(target_info)
    print(f"==================================================")
    print(f"Starting sequential training for {total_diseases} {category.upper()} diseases in {model_dir}")
    print(f"==================================================")

    validation_results = []
    csv_results_path = os.path.join(model_dir, f'{category}_validation_results.csv')

    if os.path.exists(csv_results_path):
        try:
            existing_df = pd.read_csv(csv_results_path)
            validation_results = existing_df.to_dict('records')
            print(f"Loaded {len(validation_results)} existing validation results.")
        except Exception:
            pass

    for idx, row in target_info.iterrows():
        target_disease = row['disease']
        
        model_save_path = os.path.join(model_dir, f"{target_disease}_model.h5")
        if args.model_type == 'lr':
            model_save_path = os.path.join(model_dir, f"{target_disease}_lr_model.pkl")
        elif args.model_type == 'mlp':
            model_save_path = os.path.join(model_dir, f"{target_disease}_mlp_model.pkl")
        elif args.model_type == 'linear':
            model_save_path = os.path.join(model_dir, f"{target_disease}_linear_model.pkl")
        elif args.model_type == 'nb':
            model_save_path = os.path.join(model_dir, f"{target_disease}_nb_model.pkl")
        elif args.model_type == 'rf':
            model_save_path = os.path.join(model_dir, f"{target_disease}_rf_model.pkl")
        if os.path.exists(model_save_path) and not args.overwrite:
            print(f"\n---> [{idx+1}/{total_diseases}] Skipping {target_disease}: Model already exists.")
            continue

        print(f"\n---> [{idx+1}/{total_diseases}] Training Model for {target_disease}")
        
        # If we train a new model, remove any old result for the same disease before appending at the end
        validation_results = [res for res in validation_results if res.get('Disease') != target_disease]
        
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
        
        # 질병에 맞는 전용 UR 컬럼을 동적으로 선택합니다.
        ur_column = f'{target_disease}_UR'
        read_columns = [ur_column] + chr_columns

        missing_columns = [col for col in dep_columns + read_columns if col not in all_cols_set]
        if missing_columns:
            print(f"Skipping {target_disease}: Missing columns {missing_columns[:5]}...")
            continue

        cols_to_load = dep_columns + read_columns
        
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

        # Load only necessary columns to avoid OOM
        def remove_outliers_iqr(df, column):
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            initial_len = len(df)
            filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
            dropped = initial_len - len(filtered_df)
            return filtered_df, dropped

        cnv_m = safe_load(cached_paths[0], schema_cnv_m, cols_to_load, ur_column)
        cnv_f = safe_load(cached_paths[1], schema_cnv_f, cols_to_load, ur_column)
        CNV_data = pd.concat([cnv_m, cnv_f], ignore_index=True)
        if args.iqr_filter in ['both', 'abnormal']:
            CNV_data, dropped_cnv = remove_outliers_iqr(CNV_data, ur_column)
            print(f"[{target_disease}] IQR Filter: Removed {dropped_cnv} outliers from CNV_data.")
        CNV_data['result'] = 1
        del cnv_m, cnv_f
        
        norm_m = safe_load(cached_paths[2], schema_norm_m, cols_to_load, ur_column)
        norm_f = safe_load(cached_paths[3], schema_norm_f, cols_to_load, ur_column)
        Normal_data = pd.concat([norm_m, norm_f], ignore_index=True)
        if args.iqr_filter in ['both', 'normal']:
            Normal_data, dropped_norm = remove_outliers_iqr(Normal_data, ur_column)
            print(f"[{target_disease}] IQR Filter: Removed {dropped_norm} outliers from Normal_data.")
        Normal_data['result'] = 0
        del norm_m, norm_f
        
        import gc
        gc.collect()

        CNV_train, CNV_test = train_test_split(CNV_data, test_size=0.2, random_state=42)
        del CNV_data
        Normal_train, Normal_test = train_test_split(Normal_data, test_size=0.2, random_state=42)
        del Normal_data
        gc.collect()
        
        train_df = pd.concat([Normal_train, CNV_train], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
        del Normal_train, CNV_train
        test_df = pd.concat([Normal_test, CNV_test], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
        del Normal_test, CNV_test
        gc.collect()

        X_train_dep = train_df[dep_columns].copy()
        X_train_read = train_df[read_columns].copy()
        y_train = train_df['result'].values

        X_test_dep = test_df[dep_columns].copy()
        X_test_read = test_df[read_columns].copy()
        y_test = test_df['result'].values
        
        del train_df, test_df
        gc.collect()

        scaler_gc = MinMaxScaler()
        scaler_ff = MinMaxScaler()
        scaler_read = MinMaxScaler()

        X_train_dep['GC'] = scaler_gc.fit_transform(X_train_dep[['GC']])
        X_train_dep['snp_FF'] = scaler_ff.fit_transform(X_train_dep[['snp_FF']])
        X_train_read_scaled = scaler_read.fit_transform(X_train_read)
        X_train_dep_scaled = X_train_dep.values

        X_test_dep['GC'] = scaler_gc.transform(X_test_dep[['GC']])
        X_test_dep['snp_FF'] = scaler_ff.transform(X_test_dep[['snp_FF']])
        X_test_read_scaled = scaler_read.transform(X_test_read)
        X_test_dep_scaled = X_test_dep.values

        if args.model_type == 'lr':
            X_train_combined = np.hstack([X_train_dep_scaled, X_train_read_scaled])
            X_test_combined = np.hstack([X_test_dep_scaled, X_test_read_scaled])
            
            model, loss, accuracy, y_pred_prob, y_pred = train_logistic_regression(
                X_train_combined, y_train, X_test_combined, y_test, target_disease, args.max_iter
            )
            
        elif args.model_type == 'mlp':
            X_train_combined = np.hstack([X_train_dep_scaled, X_train_read_scaled])
            X_test_combined = np.hstack([X_test_dep_scaled, X_test_read_scaled])
            
            model, loss, accuracy, y_pred_prob, y_pred = train_mlp_classifier(
                X_train_combined, y_train, X_test_combined, y_test, target_disease, args.max_iter
            )
            
        elif args.model_type == 'linear':
            X_train_combined = np.hstack([X_train_dep_scaled, X_train_read_scaled])
            X_test_combined = np.hstack([X_test_dep_scaled, X_test_read_scaled])
            
            model, loss, accuracy, y_pred_prob, y_pred = train_linear_regression(
                X_train_combined, y_train, X_test_combined, y_test, target_disease
            )
            
        elif args.model_type == 'nb':
            X_train_combined = np.hstack([X_train_dep_scaled, X_train_read_scaled])
            X_test_combined = np.hstack([X_test_dep_scaled, X_test_read_scaled])
            
            model, loss, accuracy, y_pred_prob, y_pred = train_naive_bayes(
                X_train_combined, y_train, X_test_combined, y_test, target_disease, args.nb_var_smoothing
            )
            
        elif args.model_type == 'rf':
            X_train_combined = np.hstack([X_train_dep_scaled, X_train_read_scaled])
            X_test_combined = np.hstack([X_test_dep_scaled, X_test_read_scaled])
            
            model, loss, accuracy, y_pred_prob, y_pred = train_random_forest(
                X_train_combined, y_train, X_test_combined, y_test, target_disease, args.rf_n_estimators, args.rf_max_depth
            )
            
        elif args.model_type == 'autokeras':
            import autokeras as ak
            print(f"[{target_disease}] Starting AutoKeras search with {args.ak_max_trials} trials...")
            X_train_combined = np.hstack([X_train_dep_scaled, X_train_read_scaled])
            X_test_combined = np.hstack([X_test_dep_scaled, X_test_read_scaled])
            
            clf = ak.StructuredDataClassifier(
                max_trials=args.ak_max_trials,
                overwrite=True,
                project_name=f"ak_cache_{target_disease}"
            )
            
            clf.fit(X_train_combined, y_train, epochs=args.epochs, validation_split=0.2, verbose=2)
            model = clf.export_model()
            
            eval_res = model.evaluate(X_test_combined, y_test, verbose=0)
            loss, accuracy = eval_res[0], eval_res[1]
            y_pred_prob = model.predict(X_test_combined, verbose=0)
            y_pred = (y_pred_prob > 0.5).astype(int).flatten()
            
        else:
            # Build Manual Model
            dep_input_shape = len(dep_columns)
            read_input_shape = len(read_columns)
            
            dep_initial_nodes = dep_input_shape * 4 
            read_initial_nodes = max(50, int(read_input_shape * 1.5))
            
            dep_input = Input(shape=(dep_input_shape,), name="Dep_Input")
            x_dep = Dense(dep_initial_nodes, kernel_regularizer=regularizers.l2(0.001))(dep_input)
            x_dep = Activation('relu')(x_dep)
            x_dep = Dropout(args.dropout)(x_dep)
            
            read_input = Input(shape=(read_input_shape,), name="Read_Input")
            x_read = Dense(read_initial_nodes, kernel_regularizer=regularizers.l2(0.001))(read_input)
            x_read = Activation('relu')(x_read)
            x_read = Dropout(args.dropout)(x_read)
            
            merged = Concatenate()([x_dep, x_read])
            merged_nodes = dep_initial_nodes + read_initial_nodes
            X_total = Dense(merged_nodes // 2, kernel_regularizer=regularizers.l2(0.001))(merged)
            X_total = Activation('relu')(X_total)
            X_total = Dropout(args.dropout)(X_total)
            
            output = Dense(1, activation='sigmoid')(X_total)
            
            model = Model(inputs=[dep_input, read_input], outputs=output)
            
            decay_step = max(1, (len(y_train) // args.batch_size) * 3)
            lr_schedule = ExponentialDecay(initial_learning_rate=args.lr, decay_steps=decay_step, decay_rate=0.96)
            
            model.compile(optimizer=AdamW(learning_rate=args.lr),
                          loss=custom_focal_loss(),
                          metrics=["accuracy", tf.keras.metrics.Recall(name='sensitivity')])
            
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                min_delta=args.min_delta,
                patience=args.patience,
                restore_best_weights=True,
                verbose=1
            )
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                min_delta=args.min_delta,
                patience=max(1, args.patience - 1),
                min_lr=1e-6,
                verbose=1
            )
    
            history = model.fit(
                x=[X_train_dep_scaled, X_train_read_scaled],
                y=y_train,
                validation_split=0.2, 
                batch_size=args.batch_size,
                epochs=args.epochs,
                callbacks=[early_stopping, reduce_lr],
                verbose=2 # Set verbose to 2 to show training progress (one line per epoch)
            )
    
            eval_res = model.evaluate(
                x=[X_test_dep_scaled, X_test_read_scaled],
                y=y_test,
                verbose=0
            )
            loss, accuracy = eval_res[0], eval_res[1]
            
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
        
        if args.model_type == 'lr':
            model_save_path = os.path.join(model_dir, f"{target_disease}_lr_model.pkl")
            joblib.dump(model, model_save_path)
        elif args.model_type == 'mlp':
            model_save_path = os.path.join(model_dir, f"{target_disease}_mlp_model.pkl")
            joblib.dump(model, model_save_path)
        elif args.model_type == 'linear':
            model_save_path = os.path.join(model_dir, f"{target_disease}_linear_model.pkl")
            joblib.dump(model, model_save_path)
        elif args.model_type == 'nb':
            model_save_path = os.path.join(model_dir, f"{target_disease}_nb_model.pkl")
            joblib.dump(model, model_save_path)
        elif args.model_type == 'rf':
            model_save_path = os.path.join(model_dir, f"{target_disease}_rf_model.pkl")
            joblib.dump(model, model_save_path)
        else:
            model_save_path = os.path.join(model_dir, f"{target_disease}_model.h5")
            model.save(model_save_path)
        print(f"Saved {model_save_path}")
        
        validation_results.append({
            'Disease': target_disease,
            'Category': category.upper(),
            'Specificity': specificity,
            'Sensitivity': sensitivity,
            'PPV': ppv,
            'NPV': npv,
            'Test_Loss': round(loss, 4),
            'Test_Accuracy': accuracy_str
        })
        pd.DataFrame(validation_results).to_csv(csv_results_path, index=False)
        
        if args.model_type == 'dnn':
            tf.keras.backend.clear_session()
        del model
        gc.collect()

    print("==================================================")
    print(f"All sequential training complete for {category.upper()}! Results saved to {model_dir}")

def main():
    args = get_args()
    
    if args.device == 'CPU':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        print("Set to use CPU for training.")
    else:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                gpu_idx = int(args.gpu)
                if 0 <= gpu_idx < len(gpus):
                    tf.config.set_visible_devices(gpus[gpu_idx], 'GPU')
                    if args.memory_growth:
                        tf.config.experimental.set_memory_growth(gpus[gpu_idx], True)
                    print(f"Set to use GPU {args.gpu} for training (memory growth: {args.memory_growth}).")
                else:
                    if args.memory_growth:
                        for gpu in gpus:
                            tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"Warning: GPU {args.gpu} out of range (found {len(gpus)} GPUs). Using default GPU.")
            except Exception as e:
                print(f"Warning configuring GPU {args.gpu}: {e}")
        else:
            print("Set to use GPU for training (no physical GPU detected, fallback to CPU).")

    # Load disease info
    orphanet_excel = 'Orphanet_Names_filtered_2607.xlsx' if os.path.exists('Orphanet_Names_filtered_2607.xlsx') else 'Orphanet_Names_filtered_2604.xlsx'
    print(f"Loading disease info from {orphanet_excel}...")
    try:
        d_info = pd.read_excel(orphanet_excel)
    except Exception as e:
        print(f"Failed to load {orphanet_excel}: {e}")
        return

    categories = ['del', 'dup'] if args.category == 'both' else [args.category]
    for cat in categories:
        train_for_category(args, cat, d_info)

if __name__ == "__main__":
    main()

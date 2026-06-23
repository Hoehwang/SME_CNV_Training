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
                # 변경된 환경 변수를 시스템이 인식하도록 현재 스크립트 재실행
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
from sklearn.linear_model import LogisticRegression
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
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--device', type=str, default='GPU', choices=['GPU', 'CPU'], help='Compute device')
    parser.add_argument('--downsample_frac', type=float, default=1.0, help='Fraction of data to load (e.g., 0.05)')
    parser.add_argument('--category', type=str, default='dup', choices=['dup', 'del'], help='Target category (dup or del)')
    parser.add_argument('--model_type', type=str, default='dnn', choices=['dnn', 'lr', 'autokeras'], help='Model type to use for training')
    parser.add_argument('--ak_max_trials', type=int, default=3, help='Max trials for AutoKeras')
    parser.add_argument('--iqr_filter', type=str, default='none', choices=['none', 'both', 'normal'], help='Apply IQR outlier removal on UR column')
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

def train_logistic_regression(X_train, y_train, X_test, y_test, target_disease):
    print(f"[{target_disease}] Training Logistic Regression...")
    clf = LogisticRegression(max_iter=2, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    y_pred_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)
    
    loss = log_loss(y_test, y_pred_prob)
    accuracy = accuracy_score(y_test, y_pred)
    
    return clf, loss, accuracy, y_pred_prob, y_pred

def main():
    args = get_args()
    
    if args.device == 'CPU':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        print("Set to use CPU for training.")
    else:
        print("Set to use GPU for training.")

    # Load disease info
    print("Loading disease info from Orphanet_Names_filtered_2604.xlsx...")
    try:
        d_info = pd.read_excel('Orphanet_Names_filtered_2604.xlsx')
    except Exception as e:
        print(f"Failed to load Orphanet_Names_filtered_2604.xlsx: {e}")
        return
        
    target_info = d_info[d_info['disease'].str.contains(args.category, na=False, case=False)].reset_index(drop=True)
    if len(target_info) == 0:
        print(f"No {args.category} targets found in Excel.")
        return

    data_dir = 'Synthetic_400K'
    file_prefix = 'duple' if args.category == 'dup' else 'del'
    files_to_load = [
        os.path.join(data_dir, f'CNV_{file_prefix}_male_synthetic.csv'),
        os.path.join(data_dir, f'CNV_{file_prefix}_female_synthetic.csv'),
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
            file_prefix = 'duple' if args.category == 'dup' else 'del'
            files_to_load = [
                os.path.join(data_dir, f'CNV_{file_prefix}_male_synthetic.csv'),
                os.path.join(data_dir, f'CNV_{file_prefix}_female_synthetic.csv'),
                os.path.join(data_dir, 'CNV_normal_male_synthetic.csv'),
                os.path.join(data_dir, 'CNV_normal_female_synthetic.csv')
            ]
            
            from multiprocessing import Process
            
            def worker_task(file_path, frac, out_path):
                if os.path.exists(out_path):
                    print(f"[{os.path.basename(file_path)}] Cache already exists: {os.path.basename(out_path)}. Skipping CSV parsing!")
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
    os.makedirs('models', exist_ok=True)
    
    schema_cnv_m = set(meta_cnv_m.schema.names)
    schema_cnv_f = set(meta_cnv_f.schema.names)
    schema_norm_m = set(meta_norm_m.schema.names)
    schema_norm_f = set(meta_norm_f.schema.names)
    all_cols_set = schema_cnv_m.union(schema_cnv_f).union(schema_norm_m).union(schema_norm_f)

    # Calculate Max Index for each chromosome
    print("Pre-calculating maximum chromosome bins...")
    max_chr_idx = {}
    for c in all_cols_set:
        if c.startswith('chr'):
            parts = c.split('_')
            if len(parts) == 2:
                chr_num_str = parts[0][3:]
                try:
                    idx = int(parts[1])
                    max_chr_idx[chr_num_str] = max(max_chr_idx.get(chr_num_str, 0), idx)
                except ValueError:
                    pass

    # START SEQUENTIAL LOOP
    total_diseases = len(target_info)
    print(f"==================================================")
    print(f"Starting sequential training for {total_diseases} {args.category.upper()} diseases.")
    print(f"==================================================")

    validation_results = []
    csv_results_path = os.path.join('models', f'{args.category}_validation_results.csv')

    if os.path.exists(csv_results_path):
        try:
            existing_df = pd.read_csv(csv_results_path)
            validation_results = existing_df.to_dict('records')
            print(f"Loaded {len(validation_results)} existing validation results.")
        except Exception:
            pass

    for idx, row in target_info.iterrows():
        target_disease = row['disease']
        
        model_save_path = os.path.join('models', f"{target_disease}_model.h5")
        if args.model_type == 'lr':
            model_save_path = os.path.join('models', f"{target_disease}_lr_model.pkl")
        if os.path.exists(model_save_path):
            print(f"\n---> [{idx+1}/{total_diseases}] Skipping {target_disease}: Model already exists.")
            continue

        print(f"\n---> [{idx+1}/{total_diseases}] Training Model for {target_disease}")
        
        # If we train a new model, remove any old result for the same disease before appending at the end
        validation_results = [res for res in validation_results if res.get('Disease') != target_disease]
        
        import math
        chr_num = row['chr']
        start = row['start'] // 300000
        end = int(math.ceil(row['end'] / 300000))
        
        # 19p13.13_del과 같이 크기가 작아 하나의 bin 안에 완전히 포함되는 경우, 최소 1개의 bin을 포함하도록 처리합니다.
        if start == end:
            end = start + 1
        
        if str(chr_num) == 'X':
            chr_num = 23
        elif str(chr_num) == 'Y':
            chr_num = 24
            
        chr_num_str = str(chr_num)
        max_idx = max_chr_idx.get(chr_num_str, start)
        
        margin = 0
        range_start_with_margin = max(0, start - margin)
        range_end_with_margin = min(end + margin, max_idx)

        chr_columns = [f'chr{chr_num}_{i+1}' for i in range(range_start_with_margin, range_end_with_margin)]
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
        if args.iqr_filter in ['both']:
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
                X_train_combined, y_train, X_test_combined, y_test, target_disease
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
            
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    
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
            model_save_path = os.path.join('models', f"{target_disease}_lr_model.pkl")
            joblib.dump(model, model_save_path)
        else:
            model_save_path = os.path.join('models', f"{target_disease}_model.h5")
            model.save(model_save_path)
        print(f"Saved {model_save_path}")
        
        validation_results.append({
            'Disease': target_disease,
            'Category': args.category.upper(),
            'Specificity': specificity,
            'Sensitivity': sensitivity,
            'PPV': ppv,
            'NPV': npv,
            'Test_Loss': round(loss, 4),
            'Test_Accuracy': accuracy_str
        })
        pd.DataFrame(validation_results).to_csv(csv_results_path, index=False)

    print("==================================================")
    print("All sequential training complete!")

if __name__ == "__main__":
    main()

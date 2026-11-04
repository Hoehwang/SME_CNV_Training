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

def get_genomom_plus_mapping(excel_path=None):
    if not excel_path:
        cands = glob.glob('*256*.xlsx')
        if not cands:
            return None, None
        excel_path = cands[0]
        
    if not os.path.exists(excel_path):
        return None, None
        
    try:
        df_geno = pd.read_excel(excel_path, sheet_name='제노맘_플러스', header=1)
        df_geno = df_geno[pd.to_numeric(df_geno['번호'], errors='coerce').notnull()].copy()
        df_geno['번호'] = df_geno['번호'].astype(int)
        
        orphanet_excel = 'Orphanet_Names_filtered_2607.xlsx' if os.path.exists('Orphanet_Names_filtered_2607.xlsx') else 'Orphanet_Names_filtered_2604.xlsx'
        d_info = pd.read_excel(orphanet_excel)
        orpha_to_diseases = {}
        for idx, r in d_info.iterrows():
            code = r.get('orphacode')
            if pd.notnull(code):
                try:
                    c = int(code)
                    if c not in orpha_to_diseases:
                        orpha_to_diseases[c] = []
                    orpha_to_diseases[c].append(r['disease'])
                except Exception:
                    pass

        mapped_models = []
        for idx, r in df_geno.iterrows():
            b_type = r['분류']
            orpha = r['ORPHA No.']
            matched_d = None
            try:
                c = int(str(orpha).strip())
                if c in orpha_to_diseases:
                    cands = orpha_to_diseases[c]
                    if b_type == '결실':
                        type_cands = [x for x in cands if 'del' in x.lower()]
                    else:
                        type_cands = [x for x in cands if 'dup' in x.lower() or 'tetra' in x.lower()]
                    matched_d = type_cands[0] if type_cands else cands[0]
            except Exception:
                pass
            mapped_models.append(matched_d)
            
        df_geno['Model_Disease'] = mapped_models
        return df_geno, excel_path
    except Exception as e:
        print(f"Warning: Failed to load GenoMom Plus mapping: {e}")
        return None, None

def generate_genomom_markdown_report(results, md_path, df_geno, category=None, model_type="dnn", model_dir="", data_dir="Synthetic_Test"):
    from datetime import datetime
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if df_geno is None or df_geno.empty:
        return
        
    res_dict = {r['Disease']: r for r in results if 'Disease' in r}
    
    if category in ['del', 'dup']:
        target_type = '결실' if category == 'del' else '중복'
        target_geno = df_geno[df_geno['분류'] == target_type].copy()
    else:
        target_geno = df_geno.copy()
        
    def parse_stat(val):
        if pd.isnull(val): return 0.0
        if isinstance(val, (int, float)): return float(val)
        try:
            return float(str(val).split()[0])
        except Exception:
            return 0.0

    merged_rows = []
    for idx, r in target_geno.iterrows():
        md_name = r['Model_Disease']
        res = res_dict.get(md_name, {})
        
        sens_str = res.get('Sensitivity', 'N/A')
        spec_str = res.get('Specificity', 'N/A')
        ppv_str = res.get('PPV', 'N/A')
        npv_str = res.get('NPV', 'N/A')
        acc_str = res.get('Test_Accuracy', 'N/A')
        loss_val = res.get('Test_Loss', np.nan)
        
        merged_rows.append({
            '번호': r['번호'],
            '분류': r['분류'],
            '염색체_위치': r['염색체 위치'],
            'ORPHA_No': r['ORPHA No.'],
            '질환명': r['질환명'],
            'Model_Disease': md_name,
            'Sensitivity': sens_str,
            'Specificity': spec_str,
            'PPV': ppv_str,
            'NPV': npv_str,
            'Accuracy': acc_str,
            'Test_Loss': loss_val,
            '_sens_num': parse_stat(sens_str),
            '_spec_num': parse_stat(spec_str),
            '_ppv_num': parse_stat(ppv_str),
            '_npv_num': parse_stat(npv_str),
            '_acc_num': parse_stat(acc_str),
            '_has_result': md_name in res_dict
        })
        
    sub_df = pd.DataFrame(merged_rows)
    eval_df = sub_df[sub_df['_has_result']]
    
    if eval_df.empty:
        return
        
    def calc_stats(d):
        if d.empty:
            return {'mean_acc': 0, 'min_acc': 0, 'max_acc': 0, 'mean_sens': 0, 'min_sens': 0, 'max_sens': 0,
                    'mean_spec': 0, 'min_spec': 0, 'max_spec': 0, 'mean_ppv': 0, 'mean_npv': 0, 'mean_loss': 0,
                    'sens_ge_95': 0, 'spec_ge_99': 0, 'acc_ge_99': 0, 'count': 0}
        accs = d['_acc_num'].values
        sens = d['_sens_num'].values
        specs = d['_spec_num'].values
        ppvs = d['_ppv_num'].values
        npvs = d['_npv_num'].values
        losses = d['Test_Loss'].dropna().values
        return {
            'mean_acc': np.mean(accs) * 100,
            'min_acc': np.min(accs) * 100,
            'max_acc': np.max(accs) * 100,
            'mean_sens': np.mean(sens) * 100,
            'min_sens': np.min(sens) * 100,
            'max_sens': np.max(sens) * 100,
            'mean_spec': np.mean(specs) * 100,
            'min_spec': np.min(specs) * 100,
            'max_spec': np.max(specs) * 100,
            'mean_ppv': np.mean(ppvs) * 100,
            'mean_npv': np.mean(npvs) * 100,
            'mean_loss': np.mean(losses) if len(losses) > 0 else 0.0,
            'sens_ge_95': sum(1 for s in sens if s >= 0.95),
            'spec_ge_99': sum(1 for s in specs if s >= 0.99),
            'acc_ge_99': sum(1 for a in accs if a >= 0.99),
            'count': len(d)
        }

    total_stats = calc_stats(eval_df)
    
    md = []
    title_suffix = f" - {category.upper()}" if category in ['del', 'dup'] else ""
    md.append(f"# 제노맘 플러스 CNV 모델 성능 평가 리포트{title_suffix}\n")
    md.append(f"- **생성 일시**: {now_str}")
    md.append(f"- **대상 시트**: `제노맘_플러스` (평가 완료: {len(eval_df)} / {len(sub_df)}종)")
    if model_dir:
        md.append(f"- **평가 모델 폴더**: `{model_dir}` ({model_type.upper()})")
    if data_dir:
        md.append(f"- **테스트 데이터셋**: `{data_dir}`")
    md.append("\n---\n")
    
    md.append("## 1. 종합 성능 요약 (Macro Average)\n")
    md.append("| 구분 | 평가 질환 수 | 정확도 (Accuracy) | 민감도 (Sensitivity) | 특이도 (Specificity) | 양성예측도 (PPV) | 음성예측도 (NPV) | 평균 Loss |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    md.append(f"| **제노맘 플러스 ({'전체' if category is None else category.upper()})** | **{total_stats['count']}종** | **{total_stats['mean_acc']:.2f}%** | **{total_stats['mean_sens']:.2f}%** | **{total_stats['mean_spec']:.2f}%** | **{total_stats['mean_ppv']:.2f}%** | **{total_stats['mean_npv']:.2f}%** | **{total_stats['mean_loss']:.4f}** |")
    
    if category is None or category == 'all':
        del_eval = eval_df[eval_df['분류'] == '결실']
        dup_eval = eval_df[eval_df['분류'] == '중복']
        if not del_eval.empty:
            ds = calc_stats(del_eval)
            md.append(f"| 결실 (Deletion) | {ds['count']}종 | {ds['mean_acc']:.2f}% | {ds['mean_sens']:.2f}% | {ds['mean_spec']:.2f}% | {ds['mean_ppv']:.2f}% | {ds['mean_npv']:.2f}% | {ds['mean_loss']:.4f} |")
        if not dup_eval.empty:
            ds = calc_stats(dup_eval)
            md.append(f"| 중복 (Duplication) | {ds['count']}종 | {ds['mean_acc']:.2f}% | {ds['mean_sens']:.2f}% | {ds['mean_spec']:.2f}% | {ds['mean_ppv']:.2f}% | {ds['mean_npv']:.2f}% | {ds['mean_loss']:.4f} |")
    md.append("\n### 주요 품질 기준 달성 현황")
    md.append(f"- **정확도 ≥ 99.0% 달성**: **{total_stats['acc_ge_99']} / {total_stats['count']}종** ({total_stats['acc_ge_99']/max(total_stats['count'], 1)*100:.1f}%)")
    md.append(f"- **특이도 ≥ 99.0% 달성**: **{total_stats['spec_ge_99']} / {total_stats['count']}종** ({total_stats['spec_ge_99']/max(total_stats['count'], 1)*100:.1f}%)")
    md.append(f"- **민감도 ≥ 95.0% 달성**: **{total_stats['sens_ge_95']} / {total_stats['count']}종** ({total_stats['sens_ge_95']/max(total_stats['count'], 1)*100:.1f}%)\n")
    md.append("---\n")
    
    md.append("## 2. 제노맘 플러스 질환별 상세 평가 결과\n")
    md.append("| 번호 | 분류 | 염색체 위치 | ORPHA No. | 질환명 | 모델 키 | 민감도 (95% CI) | 특이도 (95% CI) | PPV (95% CI) | NPV (95% CI) | 정확도 (95% CI) | Loss |")
    md.append("| :-: | :-: | :-: | :-: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for idx, r in sub_df.iterrows():
        md.append(f"| {r['번호']} | {r['분류']} | {r['염색체_위치']} | {r['ORPHA_No']} | {r['질환명']} | `{r['Model_Disease']}` | {r['Sensitivity']} | {r['Specificity']} | {r['PPV']} | {r['NPV']} | {r['Accuracy']} | {r['Test_Loss']} |")
        
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md) + '\n')
    print(f"GenoMom Plus markdown report saved to: {md_path}")

def try_generate_combined_genomom_report(df_geno, model_dir="", category=None, model_type="dnn", epochs=50, max_iter=200, data_dir="Synthetic_Test", tag_suffix=""):
    if df_geno is None:
        return None
        
    if model_dir:
        base = os.path.basename(os.path.normpath(model_dir))
        tag = base
        for pfx in ['models_del_', 'models_dup_', 'models_']:
            if tag.startswith(pfx):
                tag = tag[len(pfx):]
                break
        
        parent = os.path.dirname(os.path.normpath(model_dir))
        if base.startswith('models_del_'):
            del_dir = model_dir
            dup_base = base.replace('models_del_', 'models_dup_')
            dup_dir = os.path.join(parent, dup_base) if parent else dup_base
        elif base.startswith('models_dup_'):
            dup_dir = model_dir
            del_base = base.replace('models_dup_', 'models_del_')
            del_dir = os.path.join(parent, del_base) if parent else del_base
        elif base.startswith('models_'):
            del_dir = os.path.join(parent, f"models_del_{tag}") if parent else f"models_del_{tag}"
            dup_dir = os.path.join(parent, f"models_dup_{tag}") if parent else f"models_dup_{tag}"
        else:
            del_dir = model_dir
            dup_dir = model_dir
    else:
        suffix = f"_{tag_suffix}" if tag_suffix else ""
        tag = f"{model_type}_epoch_{epochs}{suffix}" if model_type == 'dnn' else f"{model_type}_iter_{max_iter}{suffix}"
        del_dir = f'models_del_{tag}'
        dup_dir = f'models_dup_{tag}'
        
    del_csv = os.path.join(del_dir, 'del_synthetic_test_results.csv')
    dup_csv = os.path.join(dup_dir, 'dup_synthetic_test_results.csv')
    
    del_exists = os.path.exists(del_csv)
    dup_exists = os.path.exists(dup_csv)
    
    if not del_exists and not dup_exists:
        return None
        
    combined_results = []
    evaluated_models_desc = []
    
    if del_exists:
        try:
            del_records = pd.read_csv(del_csv).to_dict('records')
            combined_results.extend(del_records)
            evaluated_models_desc.append(f"`{del_dir}` (DEL {len(del_records)}종)")
        except Exception as e:
            print(f"Warning reading {del_csv}: {e}")
            
    if dup_exists:
        try:
            dup_records = pd.read_csv(dup_csv).to_dict('records')
            combined_results.extend(dup_records)
            evaluated_models_desc.append(f"`{dup_dir}` (DUP {len(dup_records)}종)")
        except Exception as e:
            print(f"Warning reading {dup_csv}: {e}")
            
    if not combined_results:
        return None

    # Dedicated parameter-combined report path
    out_md_path = f"genomom_plus_{tag}_synthetic_test_results.md"
    models_desc_str = " & ".join(evaluated_models_desc)
    
    generate_genomom_markdown_report(
        combined_results, 
        out_md_path, 
        df_geno, 
        category=None, 
        model_type=model_type, 
        model_dir=models_desc_str, 
        data_dir=data_dir
    )
    print(f"GenoMom Plus ({tag}) Combined Report saved to: {out_md_path}")
    
    # Also save a copy inside each evaluated model folder
    if del_exists and os.path.isdir(del_dir):
        copy_del = os.path.join(del_dir, f"genomom_plus_{tag}_combined_test_results.md")
        generate_genomom_markdown_report(combined_results, copy_del, df_geno, category=None, model_type=model_type, model_dir=models_desc_str, data_dir=data_dir)
        
    if dup_exists and os.path.isdir(dup_dir):
        copy_dup = os.path.join(dup_dir, f"genomom_plus_{tag}_combined_test_results.md")
        generate_genomom_markdown_report(combined_results, copy_dup, df_geno, category=None, model_type=model_type, model_dir=models_desc_str, data_dir=data_dir)

    return out_md_path

def generate_markdown_report(results, md_path, category, model_type, model_dir, data_dir, total_pos, total_neg, total_diseases):
    from datetime import datetime
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if not results:
        return
        
    def parse_stat(val):
        if isinstance(val, (int, float)):
            return float(val)
        try:
            return float(str(val).split()[0])
        except Exception:
            return 0.0

    accs = [parse_stat(r.get('Test_Accuracy', 0.0)) for r in results]
    sens = [parse_stat(r.get('Sensitivity', 0.0)) for r in results]
    specs = [parse_stat(r.get('Specificity', 0.0)) for r in results]
    ppvs = [parse_stat(r.get('PPV', 0.0)) for r in results]
    npvs = [parse_stat(r.get('NPV', 0.0)) for r in results]
    losses = [parse_stat(r.get('Test_Loss', 0.0)) for r in results]
    
    mean_acc = np.mean(accs) * 100
    mean_sens = np.mean(sens) * 100
    mean_spec = np.mean(specs) * 100
    mean_ppv = np.mean(ppvs) * 100
    mean_npv = np.mean(npvs) * 100
    mean_loss = np.mean(losses)
    
    sens_above_95 = sum(1 for s in sens if s >= 0.95)
    spec_above_99 = sum(1 for s in specs if s >= 0.99)
    acc_above_99 = sum(1 for a in accs if a >= 0.99)
    
    md = []
    md.append(f"# CNV {category.upper()} Test Evaluation Report\n")
    md.append(f"- **Evaluated At**: {now_str}")
    md.append(f"- **Model Directory**: `{model_dir}`")
    md.append(f"- **Model Type**: {model_type.upper()}")
    md.append(f"- **Evaluated Diseases**: {len(results):,} / {total_diseases:,}")
    md.append(f"- **Test Dataset**: `{data_dir}` (Positive: {total_pos:,}, Negative: {total_neg:,}, Total: {total_pos + total_neg:,})\n")
    md.append("---\n")
    
    md.append("## 1. Overall Performance Summary (Macro Average)\n")
    md.append("| Metric | Mean | Min | Max | Target Quality Benchmark |")
    md.append("| :--- | :---: | :---: | :---: | :---: |")
    md.append(f"| **Accuracy (정확도)** | **{mean_acc:.2f}%** | {np.min(accs)*100:.2f}% | {np.max(accs)*100:.2f}% | ≥ 99.0% ({acc_above_99}/{len(results)} diseases) |")
    md.append(f"| **Sensitivity (민감도)** | **{mean_sens:.2f}%** | {np.min(sens)*100:.2f}% | {np.max(sens)*100:.2f}% | ≥ 95.0% ({sens_above_95}/{len(results)} diseases) |")
    md.append(f"| **Specificity (특이도)** | **{mean_spec:.2f}%** | {np.min(specs)*100:.2f}% | {np.max(specs)*100:.2f}% | ≥ 99.0% ({spec_above_99}/{len(results)} diseases) |")
    md.append(f"| **PPV (양성 예측도)** | **{mean_ppv:.2f}%** | {np.min(ppvs)*100:.2f}% | {np.max(ppvs)*100:.2f}% | - |")
    md.append(f"| **NPV (음성 예측도)** | **{mean_npv:.2f}%** | {np.min(npvs)*100:.2f}% | {np.max(npvs)*100:.2f}% | - |")
    md.append(f"| **Average Loss** | **{mean_loss:.4f}** | {np.min(losses):.4f} | {np.max(losses):.4f} | - |\n")
    md.append("---\n")
    
    md.append("## 2. Per-Disease Evaluation Results\n")
    md.append("| No | Disease | Sensitivity (95% CI) | Specificity (95% CI) | PPV (95% CI) | NPV (95% CI) | Accuracy (95% CI) | Loss |")
    md.append("| :-: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for i, r in enumerate(results, start=1):
        md.append(f"| {i} | {r['Disease']} | {r['Sensitivity']} | {r['Specificity']} | {r['PPV']} | {r['NPV']} | {r['Test_Accuracy']} | {r['Test_Loss']} |")
        
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md) + '\n')

def main():
    parser = argparse.ArgumentParser(description="Evaluate CNV Models on Test Dataset (Synthetic_Test)")
    parser.add_argument('--data_dir', type=str, default='Synthetic_Test', help='Path to test dataset directory (default: Synthetic_Test)')
    parser.add_argument('--category', type=str, default='del', choices=['dup', 'del', 'both'], help='Target category (dup, del, or both)')
    parser.add_argument('--model_type', type=str, default='dnn', choices=['dnn', 'lr', 'autokeras', 'mlp', 'linear', 'nb', 'rf'], help='Model type to use for evaluation')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for DNN to find the correct model directory (default: 50)')
    parser.add_argument('--max_iter', type=int, default=200, help='Max iterations for LR and MLP to find the correct model directory')
    parser.add_argument('--device', type=str, default='GPU', choices=['GPU', 'CPU'], help='Compute device')
    parser.add_argument('--gpu', type=str, default='0', help='Specific GPU index to use (e.g., 0, 1, 2, 3)')
    parser.add_argument('--batch_size', type=int, default=4096, help='Batch size for evaluation (default: 4096)')
    parser.add_argument('--tag', type=str, default='', help='Model directory suffix tag (e.g. 400k)')
    parser.add_argument('--model_dir', type=str, default='', help='Direct path to model directory (e.g., models_del_dnn_epoch_50)')
    parser.add_argument('--test_frac', type=float, default=1.0, help='Fraction of test data to evaluate (default: 1.0 for all)')
    parser.add_argument('--genomom_only', action='store_true', help='Only evaluate diseases included in "제노맘_플러스" sheet')
    parser.add_argument('--genomom_report', action='store_true', help='Generate GenoMom Plus markdown reports directly from existing results')
    parser.add_argument('--genomom_excel', type=str, default='', help='Path to CNV_256종... excel file')
    parser.add_argument('--overwrite', action='store_true', help='Re-evaluate all diseases even if results CSV already exists')
    args = parser.parse_args()
    
    if args.genomom_report:
        df_geno, excel_path = get_genomom_plus_mapping(args.genomom_excel)
        if df_geno is None:
            print("Error: Could not load GenoMom Plus mapping.")
            return
        print(f"Loaded GenoMom Plus mapping from {excel_path} ({len(df_geno)} diseases).")
        try_generate_combined_genomom_report(df_geno, model_dir=args.model_dir, model_type=args.model_type, epochs=args.epochs, max_iter=args.max_iter, data_dir=args.data_dir, tag_suffix=args.tag)
        return

    if args.device == 'CPU':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        print("Set to use CPU for evaluation.")
    else:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                gpu_idx = int(args.gpu)
                if 0 <= gpu_idx < len(gpus):
                    tf.config.set_visible_devices(gpus[gpu_idx], 'GPU')
                    print(f"Set to use GPU {args.gpu} for evaluation.")
                else:
                    print(f"Warning: GPU {args.gpu} out of range. Using default GPU.")
            except Exception as e:
                print(f"Warning configuring GPU {args.gpu}: {e}")
        else:
            print("Set to use GPU for evaluation (fallback to CPU/default).")

    categories = ['del', 'dup'] if args.category == 'both' else [args.category]
    df_geno, _ = get_genomom_plus_mapping(args.genomom_excel)
    for cat in categories:
        evaluate_for_category(args, cat, df_geno)

def evaluate_for_category(args, category, df_geno):
    data_dir = args.data_dir
    file_prefix = 'duple' if category == 'dup' else 'del'
    
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
    orphanet_excel = 'Orphanet_Names_filtered_2607.xlsx' if os.path.exists('Orphanet_Names_filtered_2607.xlsx') else 'Orphanet_Names_filtered_2604.xlsx'
    d_info = pd.read_excel(orphanet_excel)
    if category == 'dup':
        target_info = d_info[d_info['disease'].str.contains('dup|tetra', na=False, case=False, regex=True)].reset_index(drop=True)
    else:
        target_info = d_info[d_info['disease'].str.contains(category, na=False, case=False)].reset_index(drop=True)

    if args.genomom_only:
        if df_geno is not None:
            target_type = '결실' if category == 'del' else '중복'
            geno_targets = set(df_geno[df_geno['분류'] == target_type]['Model_Disease'].dropna())
            target_info = target_info[target_info['disease'].isin(geno_targets)].reset_index(drop=True)
            print(f"Filtered for GenoMom Plus ({category}): {len(target_info)} diseases to evaluate.")
        else:
            print("Warning: Could not load GenoMom Plus mapping. Evaluating all targets.")

    total_diseases = len(target_info)
    
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
        suffix = f"_{args.tag}" if args.tag else ""
        if args.model_type in ['lr', 'mlp']:
            model_dir = f'models_{category}_{args.model_type}_iter_{args.max_iter}{suffix}'
        elif args.model_type == 'dnn':
            model_dir = f'models_{category}_dnn_epoch_{args.epochs}{suffix}'
        else:
            model_dir = f'models_{category}_{args.model_type}{suffix}'
        
    csv_results_path = os.path.join(model_dir, f'{category}_synthetic_test_results.csv')
    md_results_path = os.path.join(model_dir, f'{category}_synthetic_test_results.md')
    
    results = []
    evaluated_diseases = set()
    combined_md_path = None
    if os.path.exists(csv_results_path) and not args.overwrite:
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
                batch_size=args.batch_size,
                verbose=0
            )
            if isinstance(eval_res, (list, tuple, np.ndarray)):
                loss = float(eval_res[0])
            else:
                loss = float(eval_res)
            
            y_pred_prob = model.predict([X_test_dep_scaled, X_test_read_scaled], batch_size=args.batch_size, verbose=0)
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
            'Category': category.upper(),
            'Specificity': specificity,
            'Sensitivity': sensitivity,
            'PPV': ppv,
            'NPV': npv,
            'Test_Loss': round(loss, 4),
            'Test_Accuracy': accuracy_str
        })
        pd.DataFrame(results).to_csv(csv_results_path, index=False)
        generate_markdown_report(results, md_results_path, category, args.model_type, model_dir, data_dir, total_pos, total_neg, total_diseases)
        combined_md_path = None
        if df_geno is not None:
            geno_category_md = os.path.join(model_dir, f'genomom_plus_{category}_test_results.md')
            generate_genomom_markdown_report(results, geno_category_md, df_geno, category=category, model_type=args.model_type, model_dir=model_dir, data_dir=data_dir)
            combined_md_path = try_generate_combined_genomom_report(df_geno, model_dir=model_dir, category=category, model_type=args.model_type, epochs=args.epochs, max_iter=args.max_iter, data_dir=data_dir, tag_suffix=args.tag)
        tf.keras.backend.clear_session()
        del model
        import gc
        gc.collect()

    if df_geno is not None:
        geno_category_md = os.path.join(model_dir, f'genomom_plus_{category}_test_results.md')
        if not os.path.exists(geno_category_md) and results:
            generate_genomom_markdown_report(results, geno_category_md, df_geno, category=category, model_type=args.model_type, model_dir=model_dir, data_dir=data_dir)
        if not combined_md_path:
            combined_md_path = try_generate_combined_genomom_report(df_geno, model_dir=model_dir, category=category, model_type=args.model_type, epochs=args.epochs, max_iter=args.max_iter, data_dir=data_dir, tag_suffix=args.tag)

    print("==================================================")
    print(f"Evaluation complete for {category.upper()}! Results saved to:")
    print(f"  - CSV: {csv_results_path}")
    print(f"  - Markdown Report: {md_results_path}")
    if df_geno is not None:
        print(f"  - GenoMom Plus Report ({category}): {os.path.join(model_dir, f'genomom_plus_{category}_test_results.md')}")
        if combined_md_path:
            print(f"  - GenoMom Plus Combined Report: {combined_md_path}")

if __name__ == "__main__":
    main()

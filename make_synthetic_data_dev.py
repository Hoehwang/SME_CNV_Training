import os
import itertools
import random
import argparse
import threading
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

DISEASE_INFO_FILE = "Orphanet_Names_filtered_2603.xlsx"
CACHE_DIR = ""
CACHE_ENABLED = False
PRELOAD_MODE = "none"
PRELOADED_CACHE = {}
DISEASE_INFO_CACHE = {}
DISEASE_META_CACHE = {}
CHR_RANGES = []
CHR_COL_NAMES = []
CHR_BINS = []          # build_count_read_pos용 bins 배열 (사전 계산)
CHR_TOTAL_BINS = 0    # 전체 히스토그램 bin 수 (사전할당용)
PRELOADED_ARRAYS = {}  # stem → {chr_int, start, end, gc_count, total, chr_idx}


def enrich_unique_df(df):
    if "Chr" in df.columns:
        df["Chr"] = df["Chr"].replace(["chrX", "chrY"], ["chr23", "chr24"])
    if "Total" not in df.columns:
        df["Total"] = df["Seq"].str.len()
    if "GC_COUNT" not in df.columns:
        df["GC_COUNT"] = df["Seq"].str.count("[GC]")
    if "start" not in df.columns:
        df["start"] = df["Pos"]
    if "end" not in df.columns:
        df["end"] = df["start"] + df["Total"]
    # 원본 Pos는 start로 대체되었으므로 중복 컬럼 제거
    if "Pos" in df.columns and "start" in df.columns:
        df = df.drop(columns=["Pos"])
    # 염색체를 정수로 인코딩 (chr1→1, chr23→23) — 문자열 비교 제거용
    if "Chr_int" not in df.columns:
        df["Chr_int"] = df["Chr"].str[3:].astype(np.uint8)
    return df


def write_cache_df(df, cache_path: Path):
    # Seq는 Total·GC_COUNT 계산 후 불필요 → feather에서 제거해 파일 크기·로딩 시간 절감
    save_df = df.drop(columns=["Seq"], errors="ignore")
    temp_path = cache_path.parent / (
        f"{cache_path.stem}.{os.getpid()}.{threading.get_ident()}.tmp{cache_path.suffix}"
    )
    save_df.to_feather(temp_path)
    os.replace(temp_path, cache_path)



def read_unique_file_with_cache(unique_path: Path, need_copy: bool = False):
    if PRELOAD_MODE == "all":
        preloaded = PRELOADED_CACHE.get(unique_path.stem)
        if preloaded is not None:
            # need_copy=True일 때만 복사 (수정이 필요한 호출자용)
            return preloaded.copy() if need_copy else preloaded

    cache_dir = Path(CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_path = cache_dir / f"{unique_path.stem}.feather"

    # 데이터 원본이 변하지 않는 전제에서, 캐시가 있으면 무조건 캐시를 사용합니다.
    if cache_path.exists():
        df = pd.read_feather(cache_path)
        original_columns = tuple(df.columns)
        df = enrich_unique_df(df)
        if tuple(df.columns) != original_columns:
            write_cache_df(df, cache_path)
        return df

    # 캐시가 없을 때만 원본에서 생성
    df = pd.read_csv(unique_path, sep='\t', header=None, usecols=[2, 3, 9])
    df = df.rename(columns={2: 'Chr', 3: 'Pos', 9: 'Seq'})
    df = enrich_unique_df(df)

    # 병렬 환경에서 충돌을 줄이기 위해 임시 파일 후 원자적 교체
    write_cache_df(df, cache_path)
    return df


def _worker_init(base_dir_val: str, cache_dir_val: str, cache_enabled_val: bool, preload_mode_val: str):
    """각 워커 프로세스가 시작될 때 1회 실행.
    - preload=all  : 메인이 fork 전에 PRELOADED_ARRAYS를 채움 → 워커가 CoW로 상속
                     워커는 preload 재실행 없이 전역 배열을 그대로 사용 (1× 메모리)
    - preload=none : 워커가 feather on-demand 로드 (OS page cache 공유)
    """
    global base_dir, CACHE_DIR, CACHE_ENABLED, PRELOAD_MODE
    base_dir      = base_dir_val
    CACHE_DIR     = cache_dir_val
    CACHE_ENABLED = cache_enabled_val
    PRELOAD_MODE  = preload_mode_val   # 메인과 동일하게 유지 (CoW 상속 활용)
    init_cnv_ranges()
    load_disease_meta('del')
    load_disease_meta('duple')


def preload_all_cache(cache_dir: Path, stems: set | None = None):
    """feather 캐시를 메모리에 로드합니다.
    stems: 로드할 파일 stem 집합. None이면 전체 로드.
    """
    global PRELOADED_CACHE, PRELOADED_ARRAYS
    all_files = sorted(cache_dir.glob("*.feather"))
    feather_files = [fp for fp in all_files if stems is None or fp.stem in stems]
    if not feather_files:
        print(f"[WARN] preload-mode=all 이지만 캐시 파일이 없습니다: {cache_dir}")
        return

    total_str = f"{len(feather_files)}" if stems is None else f"{len(feather_files)}/{len(all_files)}"
    print(f"[INFO] preload all cache start | files={total_str}")
    preloaded = {}
    preloaded_arrays = {}
    progress = tqdm(feather_files, desc="Preload cache")
    for fp in progress:
        df = pd.read_feather(fp)
        original_columns = tuple(df.columns)
        df = enrich_unique_df(df)
        if tuple(df.columns) != original_columns:
            write_cache_df(df, fp)
        preloaded[fp.stem] = df
        # numpy 배열 및 염색체별 인덱스를 미리 추출하여 저장
        ci = df['Chr_int'].to_numpy()
        preloaded_arrays[fp.stem] = {
            'chr_int': ci,
            'start':   df['start'].to_numpy(),
            'end':     df['end'].to_numpy(),
            'gc_count': df['GC_COUNT'].to_numpy(),
            'total':   df['Total'].to_numpy(),
            'chr_idx': {i: np.where(ci == i)[0] for i in range(1, 25)},
        }
    progress.close()
    PRELOADED_CACHE = preloaded
    PRELOADED_ARRAYS = preloaded_arrays
    print(f"[INFO] preload all cache done | loaded={len(PRELOADED_CACHE)}")


def build_mixed_unique_file(working_dir, filenames):
    # 여러 샘플의 배열을 합친 뒤 1/N 비율로 다운샘플합니다. (pd.concat 제거)
    ci_parts, st_parts, en_parts, gc_parts, tot_parts = [], [], [], [], []

    for fname in filenames:
        stem = Path(fname).stem
        arr = PRELOADED_ARRAYS.get(stem)
        if arr is not None:
            # preload된 배열 직접 참조 (복사 없음)
            ci_parts.append(arr['chr_int'])
            st_parts.append(arr['start'])
            en_parts.append(arr['end'])
            gc_parts.append(arr['gc_count'])
            tot_parts.append(arr['total'])
        else:
            # fallback: DataFrame 경유
            df = read_unique_file_with_cache(Path(f'{working_dir}/{fname}'))
            ci_parts.append(df['Chr_int'].to_numpy())
            st_parts.append(df['start'].to_numpy())
            en_parts.append(df['end'].to_numpy())
            gc_parts.append(df['GC_COUNT'].to_numpy())
            tot_parts.append(df['Total'].to_numpy())

    ci_all  = np.concatenate(ci_parts)
    st_all  = np.concatenate(st_parts)
    en_all  = np.concatenate(en_parts)
    gc_all  = np.concatenate(gc_parts)
    tot_all = np.concatenate(tot_parts)

    n_total = len(ci_all)
    sampled_size = max(1, int(n_total / len(filenames)))

    if sampled_size < n_total:
        idx = np.random.choice(n_total, sampled_size, replace=False)
        ci_all  = ci_all[idx]
        st_all  = st_all[idx]
        en_all  = en_all[idx]
        gc_all  = gc_all[idx]
        tot_all = tot_all[idx]

    # chr_idx 사전 계산 — make_CNV_del/duple에서 중복 argsort 제거
    sort_ord   = np.argsort(ci_all, kind='stable')
    sorted_chr = ci_all[sort_ord]
    bnd        = np.searchsorted(sorted_chr, np.arange(1, 26))
    chr_idx    = {i: sort_ord[bnd[i - 1]:bnd[i]] for i in range(1, 25)}

    return {
        'Chr_int': ci_all,
        'start':   st_all,
        'end':     en_all,
        'GC_COUNT': gc_all,
        'Total':   tot_all,
        'chr_idx': chr_idx,
    }


def init_cnv_ranges():
    global CHR_RANGES, CHR_BINS, CHR_TOTAL_BINS
    
    CHR_RANGES = [
        list(np.arange(0, 249300001, 300000)),
        list(np.arange(0, 243300001, 300000)),
        list(np.arange(0, 198300001, 300000)),
        list(np.arange(0, 191400001, 300000)),
        list(np.arange(0, 181200001, 300000)),
        list(np.arange(0, 171300001, 300000)),
        list(np.arange(0, 159300001, 300000)),
        list(np.arange(0, 146400001, 300000)),
        list(np.arange(0, 141300001, 300000)),
        list(np.arange(0, 135600001, 300000)),
        list(np.arange(0, 135300001, 300000)),
        list(np.arange(0, 134100001, 300000)),
        list(np.arange(0, 115200001, 300000)),
        list(np.arange(0, 107400001, 300000)),
        list(np.arange(0, 102600001, 300000)),
        list(np.arange(0, 90600001, 300000)),
        list(np.arange(0, 81300001, 300000)),
        list(np.arange(0, 78300001, 300000)),
        list(np.arange(0, 59400001, 300000)),
        list(np.arange(0, 63300001, 300000)),
        list(np.arange(0, 48300001, 300000)),
        list(np.arange(0, 51600001, 300000)),
        list(np.arange(0, 155400001, 300000)),
        list(np.arange(0, 59400001, 300000)),
    ]

    CHR_BINS = []
    for chr_range in CHR_RANGES:
        bins = np.array(chr_range)
        bins[1:] += 1
        CHR_BINS.append(bins)
    CHR_TOTAL_BINS = sum(len(r) - 1 for r in CHR_RANGES)



def build_hist_dict(chr_int_arr, start_arr):
    hist_dict = {}
    for i, bins in enumerate(CHR_BINS, start=1):
        chr_pos = start_arr[chr_int_arr == i]
        if chr_pos.size > 0:
            hist_dict[i] = np.histogram(chr_pos, bins=bins)[0]
        else:
            hist_dict[i] = np.zeros(len(bins) - 1, dtype=np.int64)
    return hist_dict

def get_disease_bins(hist_dict, disease_name, chr_value, start_value, end_value):
    import math
    start_bin = start_value // 300000
    end_bin = int(math.ceil(end_value / 300000))
    if start_bin == end_bin:
        end_bin += 1
        
    n_bins = len(hist_dict.get(chr_value, []))
    if n_bins == 0:
        n_bins = len(CHR_BINS[chr_value - 1]) - 1
        arr = np.zeros(n_bins, dtype=np.int64)
    else:
        arr = hist_dict[chr_value]
        
    range_start = max(0, start_bin)
    range_end = min(end_bin, n_bins)
    
    col_names = []
    values = []
    for i in range(range_start, range_end):
        col_names.append(f"{disease_name}_chr{chr_value}_{i+1}")
        values.append(arr[i])
    return col_names, values

def extract_all_disease_bins(hist_dict, disease_meta_records):
    col_names = []
    values = []
    for disease_name, chr_value, start_value, end_value, max_range in disease_meta_records:
        c, v = get_disease_bins(hist_dict, disease_name, chr_value, start_value, end_value)
        col_names.extend(c)
        values.extend(v)
    return col_names, values


    for i, (bins, chr_range) in enumerate(zip(CHR_BINS, CHR_RANGES), start=1):
        n_bins = len(chr_range) - 1
        chr_pos = start_arr[chr_int_arr == i]
        if chr_pos.size > 0:
            result[offset:offset + n_bins] = np.histogram(chr_pos, bins=bins)[0]
        offset += n_bins

    return result


def process_one_combination(pair, data_by_id, cnv_typ, working_dir, sex_label='male'):
    component_ids = list(pair)
    entries   = [data_by_id[sid] for sid in component_ids]  # O(1) dict 룩업
    filenames = [e[0] for e in entries]
    urs       = np.array([e[1] for e in entries], dtype=float)
    ffs       = np.array([e[2] for e in entries], dtype=float)
    weighted_ff = np.sum(urs * ffs * 0.01) / np.sum(urs) * 100
    reference_filename = filenames[-1]

    snp_ff = weighted_ff
    ids_prefix = "_".join(component_ids)
    new_filename = f'{ids_prefix}_{round(snp_ff, 2)}_CNV_{cnv_typ}.csv'

    mixed_unique_file = build_mixed_unique_file(
        working_dir=working_dir,
        filenames=filenames,
    )
    ur_base = len(mixed_unique_file['Chr_int'])
    sample_gc = mixed_unique_file['GC_COUNT'].sum() / mixed_unique_file['Total'].sum()

    if cnv_typ == 'del':
        col_names, values = make_CNV_del(
            unique_file=mixed_unique_file,
            new_filename=new_filename,
            sample_gc=sample_gc,
            ur_base=ur_base,
            snp_ff=snp_ff,
            sex_label=sex_label,
        )
    elif cnv_typ == 'duple':
        col_names, values = make_CNV_duple(
            unique_file=mixed_unique_file,
            ref_unique_file=reference_filename,
            working_dir=working_dir,
            new_filename=new_filename,
            sample_gc=sample_gc,
            ur_base=ur_base,
            snp_ff=snp_ff,
            sex_label=sex_label,
        )
    else:  # normal
        col_names, values = make_normal(
            unique_file=mixed_unique_file,
            new_filename=new_filename,
            sample_gc=sample_gc,
            ur_base=ur_base,
            snp_ff=snp_ff,
            sex_label=sex_label,
        )

    return frozenset(component_ids), component_ids, col_names, values


def load_disease_info(cnv_typ):
    type_map = {
        'del': 'Deletion',
        'duple': 'Duplication',
    }
    if cnv_typ not in type_map:
        raise ValueError(f"지원하지 않는 CNV_typ입니다: {cnv_typ}")

    cached = DISEASE_INFO_CACHE.get(cnv_typ)
    if cached is not None:
        return cached

    disease_info = pd.read_excel(f'{base_dir}/{DISEASE_INFO_FILE}')
    disease_info = disease_info[disease_info['Type'] == type_map[cnv_typ]].copy()
    disease_info['chr'] = disease_info['chr'].replace(['X', 'Y'], ['23', '24'])
    DISEASE_INFO_CACHE[cnv_typ] = disease_info
    return disease_info


def load_disease_meta(cnv_typ):
    cached = DISEASE_META_CACHE.get(cnv_typ)
    if cached is not None:
        return cached

    disease_info = load_disease_info(cnv_typ)
    records = [
        (
            row.disease,
            int(row.chr),
            int(row.start),
            int(row.end),
            int(row.end) - int(row.start),  # 엑셀의 range 컬럼의 오기입 방지를 위해 실제 (end - start) 크기로 강제 계산
        )
        for row in disease_info.itertuples(index=False)
    ]
    meta = {
        "diseases": [record[0] for record in records],
        "records": records,
        "range_cols": [
            f"{record[0]}_{'del_range' if cnv_typ == 'del' else 'duple_range'}"
            for record in records
        ],
        "ur_cols": [f"{record[0]}_UR" for record in records],
    }
    DISEASE_META_CACHE[cnv_typ] = meta
    return meta



def load_metadata_from_sample_csv(sample_csv_path, working_dir, sex_symbol):
    sample_df = pd.read_csv(
        sample_csv_path,
        usecols=['tgc_s_id', 'fetus_sex', 'unique_reads_ur', 'genoai', 'gc'],
    )
    sample_df['tgc_s_id'] = sample_df['tgc_s_id'].astype(str).str.strip()
    sample_df['fetus_sex'] = sample_df['fetus_sex'].astype(str).str.strip()
    sample_df = sample_df[sample_df['fetus_sex'] == sex_symbol].copy()

    unique_files = [f for f in os.listdir(working_dir) if f.endswith('.unique')]
    file_by_id = {}
    for sample_id in sample_df['tgc_s_id'].unique():
        matched = [f for f in unique_files if f.startswith(sample_id)]
        if matched:
            file_by_id[sample_id] = matched[0]

    sample_df['filename'] = sample_df['tgc_s_id'].map(file_by_id)
    sample_df = sample_df.dropna(subset=['filename']).copy()

    sample_df = sample_df.rename(columns={'unique_reads_ur': 'UR', 'gc': 'GC'})
    sample_df['ff_genoai'] = pd.to_numeric(sample_df['genoai'], errors='coerce')
    sample_df['GC'] = pd.to_numeric(sample_df['GC'], errors='coerce').fillna(0.0)
    if sample_df['ff_genoai'].isna().any():
        missing_ids = sample_df[sample_df['ff_genoai'].isna()]['tgc_s_id'].tolist()
        preview = ", ".join(missing_ids[:10])
        raise ValueError(
            "genoai(ff) 값이 없는 샘플이 있어 중단합니다. "
            f"(예시: {preview})"
        )
    return sample_df[['tgc_s_id', 'filename', 'UR', 'ff_genoai', 'GC']]


def append_result(col_names, values, fh, header_written, component_ids_str):
    """fh: 결과 CSV 파일 핸들, header_written: set (출력 경로 추적용)"""
    col_names_full = [col_names[0], 'component_ids'] + col_names[1:]
    values_full = [values[0], component_ids_str] + values[1:]
    csv_path = fh.name
    if csv_path not in header_written:
        if fh.tell() == 0:
            fh.write(','.join(str(c) for c in col_names_full) + '\n')
        header_written.add(csv_path)
    fh.write(','.join(str(v) for v in values_full) + '\n')


def _component_index_path(output_csv_path):
    return f"{output_csv_path}.component_ids"


def _component_key_to_index_line(component_key):
    return "|".join(sorted(component_key))


def _write_component_index_snapshot(index_path, processed_pairs):
    temp_path = f"{index_path}.{os.getpid()}.{threading.get_ident()}.tmp"
    with open(temp_path, "w") as idx_fh:
        for key in processed_pairs:
            idx_fh.write(_component_key_to_index_line(key) + "\n")
    os.replace(temp_path, index_path)


def extract_component_key_from_ids(component_ids):
    ids = [i.strip() for i in str(component_ids).split('|') if str(i).strip()]
    if not ids:
        return None
    return frozenset(ids)


def extract_component_key_from_sample_id(sample_id):
    parts = str(sample_id).split('_')
    if len(parts) < 6:
        return None
    id_parts = parts[:-4]
    if len(id_parts) < 6 or len(id_parts) % 3 != 0:
        return None
    ids = ['_'.join(id_parts[i:i + 3]) for i in range(0, len(id_parts), 3)]
    return frozenset(ids)


def load_processed_pairs_from_output(output_csv_path):
    if (not os.path.exists(output_csv_path)) or os.path.getsize(output_csv_path) == 0:
        return set()

    index_path = _component_index_path(output_csv_path)
    if os.path.exists(index_path) and os.path.getsize(index_path) > 0:
        processed_pairs = set()
        with open(index_path, "r") as idx_fh:
            for line in idx_fh:
                pair_key = extract_component_key_from_ids(line.strip())
                if pair_key is not None:
                    processed_pairs.add(pair_key)
        if processed_pairs:
            return processed_pairs

    processed_pairs = set()
    # 필요한 컬럼만 읽어 재시작 시 대용량 CSV 로딩 비용을 줄입니다.
    header_df = pd.read_csv(output_csv_path, nrows=0)
    if 'component_ids' in header_df.columns:
        for chunk in pd.read_csv(output_csv_path, usecols=['component_ids'], chunksize=200000):
            for component_ids in chunk['component_ids'].dropna():
                pair_key = extract_component_key_from_ids(component_ids)
                if pair_key is not None:
                    processed_pairs.add(pair_key)
    elif 'sample_id' in header_df.columns:
        for chunk in pd.read_csv(output_csv_path, usecols=['sample_id'], chunksize=200000):
            for sample_id in chunk['sample_id'].dropna():
                pair_key = extract_component_key_from_sample_id(sample_id)
                if pair_key is not None:
                    processed_pairs.add(pair_key)
    if processed_pairs:
        _write_component_index_snapshot(index_path, processed_pairs)
    return processed_pairs


def make_normal(unique_file, new_filename, sample_gc, ur_base, snp_ff, sex_label='male'):
    disease_meta_del = load_disease_meta('del')
    disease_meta_duple = load_disease_meta('duple')
    
    hist_dict = build_hist_dict(unique_file['Chr_int'], unique_file['start'])
    
    col_names = ['sample_id', 'sex', 'result', 'GC', 'UR', 'snp_FF']
    values = [new_filename, sex_label, 'normal', sample_gc, ur_base, snp_ff]
    
    # We need to output all disease bins for normal sample as well
    # Normal sample will have target_disease_UR missing, which train_CNV_dev.py will fallback to UR
    # First del diseases
    del_records = disease_meta_del['records']
    c, v = extract_all_disease_bins(hist_dict, del_records)
    col_names.extend(c)
    values.extend(v)
    
    # Then duple diseases
    duple_records = disease_meta_duple['records']
    c, v = extract_all_disease_bins(hist_dict, duple_records)
    col_names.extend(c)
    values.extend(v)
    
    return col_names, values


def make_CNV_del(unique_file, new_filename, sample_gc, ur_base, snp_ff, sex_label='male'):
    disease_meta = load_disease_meta('del')

    chr_int = unique_file['Chr_int']
    start_arr = unique_file['start']
    chr_idx = unique_file['chr_idx']

    hist_dict = build_hist_dict(chr_int, start_arr)

    cnv_del_range = []
    cnv_del_UR = []
    
    disease_bin_cols = []
    disease_bin_vals = []

    for disease_name, chr_value, start_value, end_value, max_range in disease_meta["records"]:
        applied_del_range = 0
        updated_ur = ur_base
        
        disease_counts = hist_dict.get(chr_value, np.zeros(len(CHR_BINS[chr_value - 1]) - 1, dtype=np.int64)).copy()
        
        if end_value <= start_value:
            cnv_del_range.append(applied_del_range)
            cnv_del_UR.append(updated_ur)
        else:
            lower_bound = max(1000000, int(max_range * 0.5))
            lower_bound = min(lower_bound, max_range)
            del_range = random.randint(lower_bound, max_range)
            
            if end_value - del_range >= start_value:
                start_mb = random.randint(start_value, end_value - del_range)
                end_mb = start_mb + del_range

                idx = chr_idx.get(chr_value, np.array([], dtype=np.intp))
                if len(idx) > 0:
                    mask = (start_mb <= start_arr[idx]) & (start_arr[idx] <= end_mb)
                    valid_idx = idx[mask]
                    
                    del_cnv = int(len(valid_idx) * snp_ff * 0.005)
                    if del_cnv > 0:
                        to_delete = np.random.choice(valid_idx, size=del_cnv, replace=False)
                        deleted_starts = start_arr[to_delete]
                        
                        bin_edges = CHR_BINS[chr_value - 1]
                        del_counts, _ = np.histogram(deleted_starts, bins=bin_edges)
                        
                        disease_counts -= del_counts
                        applied_del_range = del_range
                        updated_ur = ur_base - del_cnv

            cnv_del_range.append(applied_del_range)
            cnv_del_UR.append(updated_ur)
            
        temp_dict = {chr_value: disease_counts}
        c, v = get_disease_bins(temp_dict, disease_name, chr_value, start_value, end_value)
        disease_bin_cols.extend(c)
        disease_bin_vals.extend(v)

    cnv_del_range_col = disease_meta["range_cols"]
    cnv_del_UR_col = disease_meta["ur_cols"]

    col_names = ['sample_id', 'sex', 'result', 'GC', 'UR', 'snp_FF'] + cnv_del_range_col + cnv_del_UR_col + disease_bin_cols
    values = [new_filename, sex_label, 'CNV_del', sample_gc, ur_base, snp_ff] + cnv_del_range + cnv_del_UR + disease_bin_vals
    return col_names, values

def make_CNV_duple(unique_file, ref_unique_file, working_dir, new_filename, sample_gc, ur_base, snp_ff, sex_label='male'):
    disease_meta = load_disease_meta('duple')

    cur_chr_int = unique_file['Chr_int']
    cur_start   = unique_file['start']

    hist_dict = build_hist_dict(cur_chr_int, cur_start)

    ref_stem = __import__('pathlib').Path(ref_unique_file).stem
    ref_arr = PRELOADED_ARRAYS.get(ref_stem)
    if ref_arr is not None:
        ref_start   = ref_arr['start']
        ref_chr_idx = ref_arr['chr_idx']
    else:
        ref_df = read_unique_file_with_cache(__import__('pathlib').Path(f'{working_dir}/{ref_unique_file}'))
        ref_chr_int = ref_df['Chr_int'].to_numpy()
        ref_start   = ref_df['start'].to_numpy()
        ref_chr_idx = {i: np.where(ref_chr_int == i)[0] for i in range(1, 25)}

    cnv_duple_range = []
    cnv_duple_UR = []
    
    disease_bin_cols = []
    disease_bin_vals = []

    import random
    for disease_name, chr_value, start_value, end_value, max_range in disease_meta["records"]:
        applied_duple_range = 0
        updated_ur = ur_base
        
        disease_counts = hist_dict.get(chr_value, np.zeros(len(CHR_BINS[chr_value - 1]) - 1, dtype=np.int64)).copy()

        if end_value <= start_value:
            cnv_duple_range.append(applied_duple_range)
            cnv_duple_UR.append(updated_ur)
        else:
            lower_bound = max(1000000, int(max_range * 0.5))
            lower_bound = min(lower_bound, max_range)
            duple_range = random.randint(lower_bound, max_range)
            
            if end_value - duple_range >= start_value:
                start_mb = random.randint(start_value, end_value - duple_range)
                end_mb = start_mb + duple_range

                ref_idx = ref_chr_idx.get(chr_value, np.array([], dtype=np.intp))
                if len(ref_idx) > 0:
                    ref_mask = (start_mb <= ref_start[ref_idx]) & (ref_start[ref_idx] <= end_mb)
                    valid_ref = ref_idx[ref_mask]
                    
                    duple_cnv = int(len(valid_ref) * snp_ff * 0.005)
                    if duple_cnv > 0:
                        sampled_ref_idx = np.random.choice(valid_ref, size=duple_cnv, replace=False)
                        added_starts = ref_start[sampled_ref_idx]
                        
                        bin_edges = CHR_BINS[chr_value - 1]
                        add_counts, _ = np.histogram(added_starts, bins=bin_edges)
                        
                        disease_counts += add_counts
                        applied_duple_range = duple_range
                        updated_ur = ur_base + duple_cnv

            cnv_duple_range.append(applied_duple_range)
            cnv_duple_UR.append(updated_ur)
            
        temp_dict = {chr_value: disease_counts}
        c, v = get_disease_bins(temp_dict, disease_name, chr_value, start_value, end_value)
        disease_bin_cols.extend(c)
        disease_bin_vals.extend(v)

    cnv_duple_range_col = disease_meta["range_cols"]
    cnv_duple_UR_col = disease_meta["ur_cols"]

    col_names = ['sample_id', 'sex', 'result', 'GC', 'UR', 'snp_FF'] + cnv_duple_range_col + cnv_duple_UR_col + disease_bin_cols
    values = [new_filename, sex_label, 'CNV_duple', sample_gc, ur_base, snp_ff] + cnv_duple_range + cnv_duple_UR + disease_bin_vals
    return col_names, values

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CNV synthetic data 생성 — del/duple/normal × male/female 6개 태스크 동시 실행"
    )
    parser.add_argument(
        "--sample-csv",
        default="/BiO/SME_Synthetic_CNV/SME_Normal_TF_UR3M_2025_08_to_2026_01_no_outliers_balanced_500.csv",
        help="tgc_s_id,fetus_sex,unique_reads_ur,genoai,gc 컬럼이 있는 샘플 메타 CSV 경로",
    )
    parser.add_argument("--mix-size",     type=int, default=3,        help="조합에 사용할 샘플 수 (2 이상)")
    parser.add_argument("--working-dir",  default="/BiO/SME_Synthetic_CNV/Thermo_Fastq", help=".unique 파일 디렉토리")
    parser.add_argument("--output-dir",   default="/BiO/SME_Synthetic_CNV/Thermo_Synthetic",       help="결과 CSV 저장 루트 디렉토리")
    parser.add_argument("--max-count",    type=int, default=3000000,   help="태스크당 생성할 최대 개수 (기본 3000000)")
    parser.add_argument("--num-workers",  type=int, default=10,        help="전체 worker 수 (기본 10)")
    parser.add_argument("--cache-dir",    default="/BiO/SME_Synthetic_CNV/.unique_cache", help=".feather 캐시 디렉토리")
    parser.add_argument("--no-cache",     action="store_true",         help="캐시 비활성화")
    parser.add_argument("--preload-mode", choices=["none", "all"], default="none", help="캐시 preload 모드")
    parser.add_argument("--sex",          choices=["male", "female", "both"], default="both", help="실행할 성별 (기본: both)")
    args = parser.parse_args()

    if args.mix_size < 2:    raise ValueError("mix-size는 2 이상이어야 합니다.")
    if args.max_count <= 0:  raise ValueError("max-count는 1 이상이어야 합니다.")
    if args.num_workers <= 0: raise ValueError("num-workers는 1 이상이어야 합니다.")

    base_dir    = '/BiO/SME_Synthetic_CNV'
    working_dir = args.working_dir
    CACHE_DIR   = args.cache_dir
    CACHE_ENABLED = not args.no_cache
    PRELOAD_MODE  = args.preload_mode

    if not os.path.isdir(working_dir):
        raise FileNotFoundError(f"working-dir를 찾을 수 없습니다: {working_dir}")

    # fork 전 초기화: CHR_COL_NAMES 등 태스크 구성에 필요
    init_cnv_ranges()

    # 성별별 메타데이터를 한 번씩만 로드 (preload 범위 결정에도 사용)
    ALL_SEX_SPECS = [('male', '+'), ('female', '-')]
    SEX_SPECS = [(l, s) for l, s in ALL_SEX_SPECS if args.sex == 'both' or args.sex == l]
    metadata_by_sex = {}
    for sex_label, sex_symbol in SEX_SPECS:
        try:
            df = load_metadata_from_sample_csv(args.sample_csv, working_dir, sex_symbol)
            if not df.empty and df['tgc_s_id'].nunique() >= args.mix_size:
                metadata_by_sex[sex_label] = df
            else:
                print(f"[WARN] {sex_label}: 샘플 부족, 해당 성별 태스크 건너뜀")
        except Exception as e:
            print(f"[WARN] {sex_label}: 메타 로드 실패 — {e}")

    # preload=all: 선택된 성별 샘플의 feather 파일만 로드 → fork 후 CoW 상속
    if CACHE_ENABLED and PRELOAD_MODE == "all":
        needed_stems = {
            Path(fname).stem
            for df in metadata_by_sex.values()
            for fname in df['filename']
        }
        preload_all_cache(Path(CACHE_DIR), stems=needed_stems)

    # 6개 태스크 컨텍스트 구성 (del/duple/normal × male/female)
    CNV_TYPS = ['del', 'duple', 'normal']
    tasks = []
    for cnv_typ in CNV_TYPS:
        for sex_label in metadata_by_sex:
            dataDf  = metadata_by_sex[sex_label]
            ids     = list(dataDf['tgc_s_id'])
            # dict 룩업으로 변경 — pandas .loc 보다 O(1) 딕셔너리 접근
            data_by_id = {
                sid: (fname, float(ur), float(ff))
                for sid, fname, ur, ff in zip(
                    dataDf['tgc_s_id'], dataDf['filename'],
                    dataDf['UR'], dataDf['ff_genoai']
                )
            }

            out_dir    = f'{args.output_dir}/CNV_Synthetic_{cnv_typ}'
            os.makedirs(out_dir, exist_ok=True)
            output_csv = f'{out_dir}/CNV_{cnv_typ}_{sex_label}_synthetic.csv'

            current_pairs = load_processed_pairs_from_output(output_csv)
            total_working = args.max_count - len(current_pairs)
            if total_working <= 0:
                print(f"[INFO] {cnv_typ}/{sex_label}: 이미 완료 ({len(current_pairs)}/{args.max_count})")
                continue

            filtered_pairs = []
            for combo in itertools.combinations(ids, args.mix_size):
                if frozenset(combo) in current_pairs:
                    continue
                filtered_pairs.append(combo)
                if len(filtered_pairs) >= total_working:
                    break

            if not filtered_pairs:
                print(f"[INFO] {cnv_typ}/{sex_label}: 새 조합 없음")
                continue

            tasks.append({
                'cnv_typ':      cnv_typ,
                'sex_label':    sex_label,
                'output_csv':   output_csv,
                'data_by_id':   data_by_id,
                'current_pairs': current_pairs,
                'filtered_pairs': filtered_pairs,
                'total_planned': len(filtered_pairs),
                'generated': 0,
                'skipped':   0,
            })
            print(
                f"[INFO] {cnv_typ}/{sex_label}: planned={len(filtered_pairs)} "
                f"already_done={len(current_pairs)} target={args.max_count}"
            )

    if not tasks:
        print("[INFO] 모든 태스크가 이미 완료되었습니다.")
        raise SystemExit(0)

    print(f"\n[INFO] 실행 태스크 수={len(tasks)} | workers={args.num_workers} "
          f"cache={'on' if CACHE_ENABLED else 'off'} preload={PRELOAD_MODE}\n")

    # 태스크별 진행률 바 + 출력 파일 핸들 (루프 종료까지 유지)
    output_fhs = {}
    output_index_fhs = {}
    header_written = set()
    for i, task in enumerate(tasks):
        task['pbar'] = tqdm(
            total=task['total_planned'],
            desc=f"{task['cnv_typ']}/{task['sex_label']}",
            position=i,
            leave=True,
        )
        csv_path = task['output_csv']
        if csv_path not in output_fhs:
            output_fhs[csv_path] = open(csv_path, 'a')
            if os.path.getsize(csv_path) > 0:
                header_written.add(csv_path)
        index_path = _component_index_path(csv_path)
        if index_path not in output_index_fhs:
            output_index_fhs[index_path] = open(index_path, 'a')
        task['fh'] = output_fhs[csv_path]
        task['index_fh'] = output_index_fhs[index_path]

    # 전체 작업을 round-robin 인터리빙 → 단일 work_iter
    all_work = []
    tmp_iters = [iter(t['filtered_pairs']) for t in tasks]
    while True:
        added = False
        for tidx, it in enumerate(tmp_iters):
            pair = next(it, None)
            if pair is not None:
                all_work.append((tidx, pair))
                added = True
        if not added:
            break
    work_iter = iter(all_work)

    with ProcessPoolExecutor(
        max_workers=args.num_workers,
        initializer=_worker_init,
        initargs=(base_dir, CACHE_DIR, CACHE_ENABLED, PRELOAD_MODE),
    ) as executor:
        in_flight = {}  # fut → task_idx

        # 초기 투입
        for _ in range(min(args.num_workers, len(all_work))):
            item = next(work_iter, None)
            if item is None:
                break
            tidx, pair = item
            t = tasks[tidx]
            fut = executor.submit(
                process_one_combination,
                pair, t['data_by_id'], t['cnv_typ'],
                working_dir, t['sex_label'],
            )
            in_flight[fut] = tidx

        # 완료 수거 + 다음 투입
        flush_counter = 0
        while in_flight:
            done, _ = wait(in_flight.keys(), return_when=FIRST_COMPLETED)
            for fut in done:
                tidx = in_flight.pop(fut)
                t    = tasks[tidx]
                pair_key, component_ids, col_names, values = fut.result()

                if pair_key not in t['current_pairs']:
                    append_result(col_names, values, t['fh'], header_written, "|".join(component_ids))
                    t['index_fh'].write("|".join(component_ids) + "\n")
                    t['current_pairs'].add(pair_key)
                    t['generated'] += 1
                else:
                    t['skipped'] += 1

                t['pbar'].update(1)
                t['pbar'].set_postfix(
                    generated=t['generated'],
                    skipped=t['skipped'],
                    done=len(t['current_pairs']),
                )
                flush_counter += 1
                if flush_counter % 100 == 0:
                    for fh in output_fhs.values():
                        fh.flush()
                    for idx_fh in output_index_fhs.values():
                        idx_fh.flush()

                item = next(work_iter, None)
                if item is not None:
                    ntidx, npair = item
                    nt = tasks[ntidx]
                    nfut = executor.submit(
                        process_one_combination,
                        npair, nt['data_by_id'], nt['cnv_typ'],
                        working_dir, nt['sex_label'],
                    )
                    in_flight[nfut] = ntidx

    # 파일 핸들 닫기
    for fh in output_fhs.values():
        fh.close()
    for idx_fh in output_index_fhs.values():
        idx_fh.close()

    for task in tasks:
        task['pbar'].close()
        print(
            f"[DONE] {task['cnv_typ']}/{task['sex_label']} | "
            f"generated={task['generated']} skipped={task['skipped']} "
            f"done_total={len(task['current_pairs'])} output={task['output_csv']}"
        )


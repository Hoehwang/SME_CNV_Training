import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pyarrow.parquet as pq
import os
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Interactive CNV Visualization (DUP/DEL vs Normal)")
    parser.add_argument('--disease', type=str, default='16p11.2_dup', help='Target disease to visualize')
    parser.add_argument('--frac', type=float, default=0.2, help='Fraction of parquet file to use')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples to load (for speed)')
    return parser.parse_args()

def main():
    args = get_args()
    
    print(f"Loading disease info for {args.disease}...")
    try:
        d_info = pd.read_excel('Orphanet_Names_filtered_2604.xlsx')
    except Exception as e:
        print(f"Error loading excel file: {e}")
        return
        
    target = d_info[d_info['disease'] == args.disease]
    if target.empty:
        print(f"Disease {args.disease} not found in excel.")
        return
        
    row = target.iloc[0]
    chr_num = row['chr']
    start_bin = row['start'] // 300000
    end_bin = row['end'] // 300000
    
    # 2. Determine DEL/DUP category automatically
    disease_name_lower = args.disease.lower()
    if 'dup' in disease_name_lower or 'tetra' in disease_name_lower:
        category = 'duple'
        disease_label = 'DUP'
    elif 'del' in disease_name_lower or 'mono' in disease_name_lower:
        category = 'del'
        disease_label = 'DEL'
    else:
        # Fallback
        category = 'duple'
        disease_label = 'Disease'
        
    if str(chr_num) == 'X':
        chr_num = 23
    elif str(chr_num) == 'Y':
        chr_num = 24
        
    # Set margin
    margin = 10
    chr_columns = [f'chr{chr_num}_{i+1}' for i in range(max(0, start_bin - margin), end_bin + margin)]
    
    print(f"Category detected: {disease_label}")
    dup_file_m = f'Synthetic_400K/CNV_{category}_male_synthetic_frac_{args.frac}.parquet'
    dup_file_f = f'Synthetic_400K/CNV_{category}_female_synthetic_frac_{args.frac}.parquet'
    norm_file_m = f'Synthetic_400K/CNV_normal_male_synthetic_frac_{args.frac}.parquet'
    norm_file_f = f'Synthetic_400K/CNV_normal_female_synthetic_frac_{args.frac}.parquet'
    
    # Use one file to verify schema
    file_to_check = dup_file_m if os.path.exists(dup_file_m) else dup_file_f
    if not os.path.exists(file_to_check):
        print(f"Parquet files for {category} not found at Synthetic_400K/.")
        return
        
    print("Reading schema...")
    meta = pq.read_metadata(file_to_check)
    schema_names = set(meta.schema.names)
    
    valid_chr_cols = [c for c in chr_columns if c in schema_names]
    cols_to_load = valid_chr_cols + ['GC', 'snp_FF']
    cols_to_load = [c for c in cols_to_load if c in schema_names]
    
    def load_samples(files, cols, num):
        dfs = []
        for f in files:
            if os.path.exists(f):
                try:
                    df = pq.read_table(f, columns=cols).slice(0, num).to_pandas()
                    dfs.append(df)
                except Exception as e:
                    pass
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        return pd.DataFrame()

    print("Loading samples efficiently...")
    disease_df = load_samples([dup_file_m, dup_file_f], cols_to_load, args.num_samples)
    norm_df = load_samples([norm_file_m, norm_file_f], cols_to_load, args.num_samples)
    
    if disease_df.empty or norm_df.empty:
        print("Could not load sufficient data for comparison.")
        return

    # Calculate statistics
    disease_means = disease_df[valid_chr_cols].mean()
    norm_means = norm_df[valid_chr_cols].mean()
    
    disease_std = disease_df[valid_chr_cols].std()
    norm_std = norm_df[valid_chr_cols].std()

    x_labels = valid_chr_cols
    
    print("Generating Interactive Plotly Report...")
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"colspan": 2}, None],
               [{}, {}]],
        subplot_titles=(
            f"Read Depth Comparison: {args.disease} (Chr {row['chr']})",
            "GC Ratio Comparison",
            "snp_FF (Fraction) Comparison"
        )
    )
    
    # ----- Plot 1: Read Depth (Line + Shaded Std) -----
    # Normal Read Depth
    fig.add_trace(go.Scatter(
        x=x_labels, y=norm_means,
        mode='lines+markers', name='Normal (Mean)',
        line=dict(color='blue'),
        hovertemplate="Bin: %{x}<br>Depth: %{y:.3f}<extra></extra>"
    ), row=1, col=1)
    
    # Normal Shade (±1 Std)
    fig.add_trace(go.Scatter(
        x=x_labels + x_labels[::-1],
        y=list(norm_means + norm_std) + list((norm_means - norm_std)[::-1]),
        fill='toself', fillcolor='rgba(0,0,255,0.1)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip", showlegend=False, name='Normal ±1 SD'
    ), row=1, col=1)
    
    # Disease Read Depth
    fig.add_trace(go.Scatter(
        x=x_labels, y=disease_means,
        mode='lines+markers', name=f'{disease_label} (Mean)',
        line=dict(color='red'),
        hovertemplate="Bin: %{x}<br>Depth: %{y:.3f}<extra></extra>"
    ), row=1, col=1)

    # Disease Shade (±1 Std)
    fig.add_trace(go.Scatter(
        x=x_labels + x_labels[::-1],
        y=list(disease_means + disease_std) + list((disease_means - disease_std)[::-1]),
        fill='toself', fillcolor='rgba(255,0,0,0.1)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip", showlegend=False, name=f'{disease_label} ±1 SD'
    ), row=1, col=1)
    
    # Highlight Target Region
    region_labels = [c for c in valid_chr_cols if start_bin <= int(c.split('_')[1])-1 < end_bin]
    if region_labels:
        fig.add_vrect(
            x0=region_labels[0], x1=region_labels[-1],
            fillcolor="orange", opacity=0.15,
            layer="below", line_width=1, line_dash="dash",
            annotation_text="Target Region", annotation_position="top left",
            row=1, col=1
        )
        
    # ----- Plot 2: GC Ratio Boxplot -----
    if 'GC' in cols_to_load:
        fig.add_trace(go.Box(y=norm_df['GC'], name='Normal', marker_color='blue', boxpoints='all', jitter=0.3, pointpos=-1.8), row=2, col=1)
        fig.add_trace(go.Box(y=disease_df['GC'], name=disease_label, marker_color='red', boxpoints='all', jitter=0.3, pointpos=-1.8), row=2, col=1)
        
    # ----- Plot 3: snp_FF Boxplot -----
    if 'snp_FF' in cols_to_load:
        fig.add_trace(go.Box(y=norm_df['snp_FF'], name='Normal', marker_color='blue', boxpoints='all', jitter=0.3, pointpos=-1.8), row=2, col=2)
        fig.add_trace(go.Box(y=disease_df['snp_FF'], name=disease_label, marker_color='red', boxpoints='all', jitter=0.3, pointpos=-1.8), row=2, col=2)
        
    # Report Info Annotation
    target_start_mb = round(row['start'] / 1000000, 2)
    target_end_mb = round(row['end'] / 1000000, 2)
    
    info_text = (
        "<b>CNV Data Info:</b><br>"
        f"• Target Disease: {args.disease} ({disease_label})<br>"
        f"• Target Region: Chr {row['chr']} {target_start_mb}MB ~ {target_end_mb}MB<br>"
        "• Normalization info:<br>"
        "  - The Read Depth is pre-normalized (relative to GC-content & Reference panel).<br>"
        "  - Normal copies ≈ 1.0 <br>"
        "  - DUP copies ≈ 1.5 (Trisomy) or 2.0 (Tetrasomy) <br>"
        "  - DEL copies ≈ 0.5 (Monosomy) or 0.0 <br>"
        "• The GC ratio and snp_FF represent the sequence attributes of the samples."
    )
    
    fig.add_annotation(
        text=info_text,
        xref="paper", yref="paper",
        x=1.0, y=1.08,
        showarrow=False,
        align="left",
        bgcolor="rgba(240, 240, 240, 0.9)",
        bordercolor="black", borderwidth=1,
        font=dict(size=12)
    )
    
    fig.update_layout(
        height=900, width=1400,
        title_text=f"Interactive CNV Analysis Dashboard: {args.disease}",
        showlegend=True,
        template='plotly_white'
    )
    fig.update_xaxes(tickangle=45, row=1, col=1)
    
    out_file = f"visualize_{args.disease}_report.html"
    fig.write_html(out_file)
    print(f"Saved interactive HTML report to: {out_file}")

if __name__ == '__main__':
    main()

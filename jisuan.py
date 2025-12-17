import pandas as pd
import numpy as np
import re
import os

OUTPUT_FILE = "result_summary_output.txt"


def log_message(message, file_path=OUTPUT_FILE):
    """打印并追加写入文件"""
    print(message)
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(message + '\n')


def get_params_from_config(config_str):
    params = {}
    patterns = {
        'bs': r'_bs(\d+)_',
        'nh': r'_nh(\d+)_',
        'ial': r'_ial(\d+)_',
        'pdl': r'_pdl(\d+)_',
        'cal': r'_cal(\d+)_',
        'rec': r'_rec([0-9eE.-]+)',
        'seed': r'_seed(\d+)'
    }
    for key, pat in patterns.items():
        match = re.search(pat, config_str)
        params[key] = match.group(1) if match else 'N/A'
    return params


def parse_file(file_path):
    if not os.path.exists(file_path):
        return pd.DataFrame()

    with open(file_path, 'r') as f:
        content = f.read()

    lines = content.strip().split('\n')
    parsed_data = []
    current_config = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith('exchange_rate') or line.startswith('weather'):
            current_config = line
        elif line.startswith('mse:') or line.startswith('mae:'):
            if current_config:
                pl_match = re.search(r'_pl(\d+)_', current_config)
                pl = int(pl_match.group(1)) if pl_match else None

                parts = line.split(',')
                mse = np.nan
                mae = np.nan

                for p in parts:
                    if ':' in p:
                        k, v = p.split(':')
                        k = k.strip().lower()
                        try:
                            v = float(v.strip())
                            if k == 'mse' and np.isnan(mse):
                                mse = v
                            elif k == 'mae' and np.isnan(mae):
                                mae = v
                            elif k == 'mse' and not np.isnan(mse) and np.isnan(mae):
                                mae = v
                        except:
                            pass

                dataset_name = 'exchange_rate' if 'exchange_rate' in current_config else 'weather'

                if pl is not None and not np.isnan(mse):
                    parsed_data.append({
                        'dataset': dataset_name,
                        'config': current_config,
                        'pl': pl,
                        'mse': mse,
                        'mae': mae
                    })
                current_config = None

    return pd.DataFrame(parsed_data)


def main():
    files = ["result_long_term_forecast_exchange_rate.csv_.txt", "result_long_term_forecast_weather.csv_.txt"]
    df_list = [parse_file(f) for f in files]
    if not df_list: return

    all_data = pd.concat(df_list, ignore_index=True)

    # 优化逻辑：根据配置字符串去重，保留第一次出现的记录
    # 这避免了同一实验运行多次产生的重复记录干扰 Top 3 的选择
    all_data = all_data.drop_duplicates(subset=['config'], keep='first')

    target_pls = [96, 192, 336, 720]
    datasets = ['exchange_rate', 'weather']

    header = f"{'Dataset':<13} | {'PL':<3} | {'Rank':<4} | {'MSE':<8} | {'MAE':<8} | {'Params (bs, nh, ial, pdl, cal, rec, seed)'}"
    log_message("-" * 120)
    log_message(header)
    log_message("-" * 120)

    for ds in datasets:
        for pl in target_pls:
            subset = all_data[(all_data['dataset'] == ds) & (all_data['pl'] == pl)]
            if subset.empty: continue

            # 排序并取前 3
            top3 = subset.sort_values(by='mse', ascending=True).head(3)

            mse_mean = top3['mse'].mean()
            mse_std = top3['mse'].std() if len(top3) > 1 else 0.0
            mae_mean = top3['mae'].mean()
            mae_std = top3['mae'].std() if len(top3) > 1 else 0.0

            log_message(f"{'=' * 120}")
            log_message(f"{ds} PL {pl} Summary: MSE={mse_mean:.3f}±{mse_std:.3f}, MAE={mae_mean:.3f}±{mae_std:.3f}")
            log_message(f"{'-' * 120}")

            for i, (idx, row) in enumerate(top3.iterrows()):
                params = get_params_from_config(row['config'])
                p_str = f"bs={params['bs']}, nh={params['nh']}, ial={params['ial']}, pdl={params['pdl']}, cal={params['cal']}, rec={params['rec']}, seed={params['seed']}"
                log_message(f"{ds:<13} | {pl:<3} | {i + 1:<4} | {row['mse']:.3f}  | {row['mae']:.3f}  | {p_str}")

    print(f"\nOptimization complete. Results have been appended to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
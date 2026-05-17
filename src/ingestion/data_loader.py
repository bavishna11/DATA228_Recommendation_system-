"""Data Ingestion Module.
Converts raw JSONL files into Parquet format.
"""
import os
import glob
import pathlib
import shutil
import pyarrow.json as paj
import pyarrow.parquet as pq

from src.utils.config_loader import load_config

def run_ingestion():
    print("Running Data Ingestion...")
    config, root_dir = load_config()
    
    raw_dir = str(root_dir / config["data"]["raw_dir"])
    parquet_output_path = str(root_dir / config["data"]["raw_path"])
    
    data_files = glob.glob('data/*.jsonl')
    
    if not data_files:
        print("No .jsonl files found in data/ directory. Skipping ingestion.")
        return

    existing_parquets = glob.glob(os.path.join(parquet_output_path, '*.parquet'))
    
    if not existing_parquets:
        if os.path.exists(parquet_output_path):
            shutil.rmtree(parquet_output_path)
            print('Cleaned up incomplete parquet directory.')
        os.makedirs(parquet_output_path, exist_ok=True)
    
        for fpath in data_files:
            category = pathlib.Path(fpath).stem
            out_file = os.path.join(parquet_output_path, f'{category}.parquet')
            size_gb = os.path.getsize(fpath) / 1e9
            print(f'Converting {fpath} ({size_gb:.2f} GB) -> {out_file} ...')
            table = paj.read_json(fpath)
            pq.write_table(table, out_file, compression='snappy')
            written_gb = os.path.getsize(out_file) / 1e9
            print(f'  Done: {written_gb:.2f} GB written, {len(table):,} rows')
    
        print('All files converted to Parquet!')
    else:
        print(f'Found {len(existing_parquets)} existing parquet file(s). Skipping conversion.')

if __name__ == "__main__":
    run_ingestion()

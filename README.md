# Pythia
Structure-based self-supervised learning enables ultrafast prediction of stability changes upon mutations

## Prerequisites

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install -r requirements.txt
```

## Usage

To use the Pythia, you can run it from the command line with the following options:

### Basic Usage

```bash
cd pythia
python masked_ddg_scan.py
```

By default, this will process files in the directory `../s669_AF_PDBs/` using `cuda:0` (GPU 0) if available.

**Command Line Options**  

- `--input_dir`: Specifies the directory path containing the PDB files. Default is `../s669_AF_PDBs/`.
  
  Example: 
  ```bash
  python masked_ddg_scan.py --input_dir "/path/to/directory/"
  ```

- `--pdb_filename`: If you want to process a single PDB file instead of a directory, specify its path with this option.

  Example:
  ```bash
  python masked_ddg_scan.py --pdb_filename "/path/to/file.pdb"
  ```

- `--check_plddt`: Use this flag if you want to filter PDB files based on their pLDDT value. Files with a pLDDT value less than the specified cutoff (see below) will be ignored.

  Example:
  ```bash
  python masked_ddg_scan.py --check_plddt
  ```

- `--plddt_cutoff`: Specifies the pLDDT cutoff value if `--check_plddt` is used. Default is 95.

  Example:
  ```bash
  python masked_ddg_scan.py --check_plddt --plddt_cutoff 90
  ```

- `--n_jobs`: Indicates the number of parallel jobs to run. Default is 2.

  Example:
  ```bash
  python masked_ddg_scan.py --n_jobs 4
  ```

- `--device`: Specifies the device to use for computation. By default, it will use `cuda:0` (GPU 0). If you want to use CPU or another GPU, specify it here. Valid values include `cuda:0`, `cuda:1`, ... for GPUs, or `cpu` for the CPU.

  Example:
  ```bash
  python masked_ddg_scan.py --device cpu
  ```

**Examples**

1. Process all PDB files in the directory `/path/to/directory/`, using the first GPU and checking pLDDT values with a cutoff of 90:
   
   ```bash
   python masked_ddg_scan.py --input_dir "/path/to/directory/" --check_plddt --plddt_cutoff 90 --device cuda:0
   ```

2. Process a single PDB file `/path/to/file.pdb` using the CPU:

   ```bash
   python masked_ddg_scan.py --pdb_filename "/path/to/file.pdb" --device cpu
   ```
[Megascale dataset](./megascale_data.csv), [S2648](./s2648_data.csv), [S669](./s669_data.csv) contains predictions and labels.

### Pocket Prediction
Change the `input_pdb = "aes72_af3.pdb"` in `inference.py` to the desired PDB file and run the following command:
```bash
cd pythia-pocket
python inference.py
```

### Protein Stability Score
Change the `fpath_pdb = "../examples/1pga.pdb"` in `score.py` to the desired PDB file and run the following command:
```
cd pythia
python score.py
```

## Train

1. Download preprocessed files for training at [CATH dataset](https://drive.google.com/file/d/1HlW27bcHX6CB5GpHlSf90pwrX5e21_Ch/view?usp=sharing) or [BioA dataset](https://drive.google.com/file/d/1iGJPXThJj6Vv7noT-RlcXuUPasxhamkg/view?usp=sharing) from the Google Drive:  
    ```bash
    sbatch train_model.sh
    ```

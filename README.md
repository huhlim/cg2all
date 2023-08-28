# cg2all
Convert coarse-grained protein structure to all-atom model

## Web server / Google Colab notebook
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/huhlim/cg2all)</br>
A demo web page is available for conversions of CG model to all-atom structure via Huggingface space.</br>

[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huhlim/cg2all/blob/main/cg2all.ipynb)</br>
A Google Colab notebook is available for tasks:
- Task 1: Conversion of an all-atom structure to a CG model using __convert_all2cg__
- Task 2: Conversion of a CG model to an all-atom structure using __convert_cg2all__
- Task 3: Conversion of a CG simulation trajectory to an atomistic simulation trajectory using __convert_cg2all__

[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huhlim/cg2all/blob/main/cryo_em_minimizer.ipynb)</br>
A Google Colab notebook is available for local optimization of a protein model structure against a cryo-EM density map using __cryo_em_minimizer.py__

## Installation
These steps will install Python libraries including [cg2all (this repository)](https://github.com/huhlim/cg2all), [a modified MDTraj](https://github.com/huhlim/mdtraj), [a modified SE3Transformer](https://github.com/huhlim/SE3Transformer), and other dependent libraries. The installation steps also place executables `convert_cg2all` and `convert_all2cg` in your python binary directory.

This package is tested on Linux (CentOS) and MacOS (Apple Silicon, M1).

#### for CPU only
```bash
pip install git+http://github.com/huhlim/cg2all
```
#### for CUDA (GPU) usage
1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
2. Create an environment with [DGL](https://www.dgl.ai/pages/start.html) library with CUDA support
```bash
# This is an example with cudatoolkit=11.3.
# Set a proper cudatoolkit version that is compatible with your CUDA drivier and DGL library.
conda create --name cg2all pip cudatoolkit=11.3 dgl -c dglteam/label/cu113
```
3. Activate the environment
```bash
conda activate cg2all
```
4. Install this package
```bash
pip install git+http://github.com/huhlim/cg2all
```

#### for cryo_em_minimizer usage
You need additional python package, `mrcfile` to deal with cryo-EM density map.
```bash
pip install mrcfile
```

## Usages
### convert_cg2all
convert a coarse-grained protein structure to all-atom model
```bash
usage: convert_cg2all [-h] -p IN_PDB_FN [-d IN_DCD_FN] -o OUT_FN [-opdb OUTPDB_FN]
                      [--cg {supported_cg_models}] [--chain-break-cutoff CHAIN_BREAK_CUTOFF] [-a]
                      [--fix] [--ckpt CKPT_FN] [--time TIME_JSON] [--device DEVICE] [--batch BATCH_SIZE] [--proc N_PROC]

options:
  -h, --help            show this help message and exit
  -p IN_PDB_FN, --pdb IN_PDB_FN
  -d IN_DCD_FN, --dcd IN_DCD_FN
  -o OUT_FN, --out OUT_FN, --output OUT_FN
  -opdb OUTPDB_FN
  --cg {supported_cg_models}
  --chain-break-cutoff CHAIN_BREAK_CUTOFF
  -a, --all, --is_all
  --fix, --fix_atom
  --ckpt CKPT_FN
  --time TIME_JSON
  --device DEVICE
  --batch BATCH_SIZE
  --proc N_PROC
```
#### arguments
* -p/--pdb: Input PDB file (**mandatory**).
* -d/--dcd: Input DCD file (optional). If a DCD file is given, the input PDB file will be used to define its topology.
* -o/--out/--output: Output PDB or DCD file (**mandatory**). If a DCD file is given, it will be a DCD file. Otherwise, a PDB file will be created.
* -opdb: If a DCD file is given, it will write the last snapshot as a PDB file. (optional)
* --cg: Coarse-grained representation to use (optional, default=CalphaBasedModel).
  - CalphaBasedModel: CA-trace (atom names should be "CA")
  - ResidueBasedModel: Residue center-of-mass (atom names should be "CA")
  - SidechainModel: Sidechain center-of-mass (atom names should be "SC")
  - CalphaCMModel: CA-trace + Residue center-of-mass (atom names should be "CA" and "CM")
  - CalphaSCModel: CA-trace + Sidechain center-of-mass (atom names should be "CA" and "SC")
  - BackboneModel: Model only with backbone atoms (N, CA, C)
  - MainchainModel: Model only with mainchain atoms (N, CA, C, O)
  - Martini: [Martini](http://cgmartini.nl/) model
  - PRIMO: [PRIMO](http://dx.doi.org/10.1002/prot.22645) model
* --chain-break-cutoff: The CA-CA distance cutoff that determines chain breaks. (default=10 Angstroms)
* --fix/--fix_atom: preserve coordinates in the input CG model. For example, CA coordinates in a CA-trace model will be kept in its cg2all output model.
* --ckpt: Input PyTorch ckpt file (optional). If a ckpt file is given, it will override "--cg" option.
* --time: Output JSON file for recording timing. (optional)
* --device: Specify a device to run the model. (optional) You can choose "cpu" or "cuda", or the script will detect one automatically. </br>
  "**cpu**" is usually faster than "cuda" unless the input/output system is really big or you provided a DCD file with many frames because it takes a lot for loading a model ckpt file on a GPU.
* --batch: the number of frames to be dealt at a time. (optional, default=1)
* --proc: Specify the number of threads for loading input data. It is only used for dealing with a DCD file. (optional, default=OMP_NUM_THREADS or 1)

#### examples
Conversion of a PDB file
```bash
convert_cg2all -p tests/1ab1_A.calpha.pdb -o tests/1ab1_A.calpha.all.pdb --cg CalphaBasedModel
```
Conversion of a DCD trajectory file
```bash
convert_cg2all -p tests/1jni.calpha.pdb -d tests/1jni.calpha.dcd -o tests/1jni.calpha.all.dcd --cg CalphaBasedModel
```
Conversion of a PDB file using a ckpt file
```bash
convert_cg2all -p tests/1ab1_A.calpha.pdb -o tests/1ab1_A.calpha.all.pdb --ckpt CalphaBasedModel-104.ckpt
```
<hr/>

### convert_all2cg
convert an all-atom protein structure to coarse-grained model
```bash
usage: convert_all2cg [-h] -p IN_PDB_FN [-d IN_DCD_FN] -o OUT_FN [--cg {supported_cg_models}]

options:
  -h, --help            show this help message and exit
  -p IN_PDB_FN, --pdb IN_PDB_FN
  -d IN_DCD_FN, --dcd IN_DCD_FN
  -o OUT_FN, --out OUT_FN, --output OUT_FN
  --cg
```
#### arguments
* -p/--pdb: Input PDB file (**mandatory**).
* -d/--dcd: Input DCD file (optional). If a DCD file is given, the input PDB file will be used to define its topology.
* -o/--out/--output: Output PDB or DCD file (**mandatory**). If a DCD file is given, it will be a DCD file. Otherwise, a PDB file will be created.
* --cg: Coarse-grained representation to use (optional, default=CalphaBasedModel).
  - CalphaBasedModel: CA-trace (atom names should be "CA")
  - ResidueBasedModel: Residue center-of-mass (atom names should be "CA")
  - SidechainModel: Sidechain center-of-mass (atom names should be "SC")
  - CalphaCMModel: CA-trace + Residue center-of-mass (atom names should be "CA" and "CM")
  - CalphaSCModel: CA-trace + Sidechain center-of-mass (atom names should be "CA" and "SC")
  - BackboneModel: Model only with backbone atoms (N, CA, C)
  - MainchainModel: Model only with mainchain atoms (N, CA, C, O)
  - Martini: [Martini](http://cgmartini.nl/) model
  - PRIMO: [PRIMO](http://dx.doi.org/10.1002/prot.22645) model
  
#### an example
```bash
convert_all2cg -p tests/1ab1_A.pdb -o tests/1ab1_A.calpha.pdb --cg CalphaBasedModel
```

<hr/>

### script/cryo_em_minimizer.py 
Local optimization of protein model structure against given electron density map. This script is a proof-of-concept that utilizes cg2all network to optimize at CA-level resolution with objective functions in both atomistic and CA-level resolutions. It is highly recommended to use **cuda** environment.
```bash
usage: cryo_em_minimizer [-h] -p IN_PDB_FN -m IN_MAP_FN -o OUT_DIR [-a] [-n N_STEP] [--freq OUTPUT_FREQ] [--restraint RESTRAINT]

options:
  -h, --help            show this help message and exit
  -p IN_PDB_FN, --pdb IN_PDB_FN
  -m IN_MAP_FN, --map IN_MAP_FN
  -o OUT_DIR, --out OUT_DIR, --output OUT_DIR
  -a, --all, --is_all
  -n N_STEP, --step N_STEP
  --freq OUTPUT_FREQ, --output_freq OUTPUT_FREQ
  --restraint RESTRAINT
```
#### arguments
* -p/--pdb: Input PDB file (**mandatory**).
* -m/--map: Input electron density map file in the MRC or CCP4 format (**mandatory**).
* -o/--out/--output: Output directory to save optimized structures (**mandatory**).
* -a/--all/--is_all: Whether the input PDB file is atomistic structure or not. (optional, default=False)
* -n/--step: The number of minimization steps. (optional, default=1000)
* --freq/--output_freq: The interval between saving intermediate outputs. (optional, default=100)
* --restraint: The weight of distance restraints. (optional, default=100.0)

#### an example
```bash
./cg2all/script/cryo_em_minimizer.py -p tests/3isr.af2.pdb -m tests/3isr_5.mrc -o 3isr_5+3isr.af2 --all
```

## Datasets
The training/validation/test sets are available at [zenodo](https://zenodo.org/record/8273739).


## Reference
Lim Heo & Michael Feig, "One particle per residue is sufficient to describe all-atom protein structures", _bioRxiv_ (**2023**). [Link](https://www.biorxiv.org/content/10.1101/2023.05.22.541652v1)

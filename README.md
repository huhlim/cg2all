# cg2all
Convert coarse-grained protein structure to all-atom model

## Web server / Google Colab notebook
A demo web page is available via _huggingface spaces_ at [https://huggingface.co/spaces/huhlim/cg2all](https://huggingface.co/spaces/huhlim/cg2all).
A Google Colab notebook is available at [https://colab.research.google.com/github/huhlim/cg2all/blob/main/cg2all.ipynb](https://colab.research.google.com/github/huhlim/cg2all/blob/main/cg2all.ipynb).

## Installation

These steps will install Python libraries including [cg2all (this repository)](https://github.com/huhlim/cg2all), [a modified MDTraj](https://github.com/huhlim/mdtraj), and other dependent libraries. The installation steps also place executables `convert_cg2all` and `convert_all2cg` in your python binary directory.

This package is tested on Linux (CentOS) and MacOS (Apple Silicon, M1).

#### for CPU only
```bash
pip install git+http://github.com/huhlim/cg2all
```
#### for CUDA (GPU) usage
1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
2. Create an environment with CUDA support and activate it
```bash
conda create --name cg2all pip cudatoolkit=11.3 dgl-cuda11.3 -c dglteam
conda activate cg2all
```
3. Install this package
```bash
pip install git+http://github.com/huhlim/cg2all
```

## Usages
### convert_cg2all
convert a coarse-grained protein structure to all-atom model
```bash
usage: convert_cg2all [-h] -p IN_PDB_FN [-d IN_DCD_FN] -o OUT_FN [--cg {CalphaBasedModel,CA,ca,ResidueBasedModel,RES,res,Martini,martini}] [--ckpt CKPT_FN] [--time TIME_JSON] [--device DEVICE]

options:
  -h, --help            show this help message and exit
  -p IN_PDB_FN, --pdb IN_PDB_FN
  -d IN_DCD_FN, --dcd IN_DCD_FN
  -o OUT_FN, --out OUT_FN, --output OUT_FN
  --cg {CalphaBasedModel,CA,ca,ResidueBasedModel,RES,res,Martini,martini}
  --ckpt CKPT_FN
  --time TIME_JSON
  --device DEVICE
```
#### arguments
* -p/--pdb: Input PDB file (**mandatory**).
* -d/--dcd: Input DCD file (optional). If a DCD file is given, the input PDB file will be used to define its topology.
* -o/--out/--output: Output PDB or DCD file (**mandatory**). If a DCD file is given, it will be a DCD file. Otherwise, a PDB file will be created.
* --cg: Coarse-grained representation to use (optional, default=CalphaBasedModel).
  * CalphaBasedModel, CA, ca: the C-alpha atom for a residue
  * ResidueBasedModel, RES, res: the center of mass of a residue for the residue
  * Martini, martini: [MARTINI](http://cgmartini.nl/index.php/martini) model
* --ckpt: Input PyTorch ckpt file (optional). If a ckpt file is given, it will override "--cg" option.
* --time: Output JSON file for recording timing (optional).
* --device: Specify a device to run the model (optional) You can choose "cpu" or "cuda", or the script will detect one automatically. </br>
  "**cpu**" is usually faster than "cuda" unless the input/output system is really big or you provided a DCD file with many frames because it takes a lot for loading a model ckpt file on a GPU.

#### an example
```bash
convert_cg2all -p test/1ab1_A.calpha.pdb -o test/1ab1_A.calpha.all.pdb --cg CalphaBasedModel
```

<hr/>

### convert_all2cg
convert an all-atom protein structure to coarse-grained model
```bash
usage: convert_all2cg [-h] -p IN_PDB_FN [-d IN_DCD_FN] -o OUT_FN [--cg {CalphaBasedModel,CA,ca,ResidueBasedModel,RES,res,Martini,martini}]

optional arguments:
  -h, --help            show this help message and exit
  -p IN_PDB_FN, --pdb IN_PDB_FN
  -d IN_DCD_FN, --dcd IN_DCD_FN
  -o OUT_FN, --out OUT_FN, --output OUT_FN
  --cg {CalphaBasedModel,CA,ca,ResidueBasedModel,RES,res,Martini,martini}
```
#### arguments
* -p/--pdb: Input PDB file (**mandatory**).
* -d/--dcd: Input DCD file (optional). If a DCD file is given, the input PDB file will be used to define its topology.
* -o/--out/--output: Output PDB or DCD file (**mandatory**). If a DCD file is given, it will be a DCD file. Otherwise, a PDB file will be created.
* --cg: Coarse-grained representation to use (optional, default=CalphaBasedModel).
  * CalphaBasedModel, CA, ca: the C-alpha atom for a residue
  * ResidueBasedModel, RES, res: the center of mass of a residue for the residue
  * Martini, martini: [MARTINI](http://cgmartini.nl/index.php/martini) model
  
#### an example
```bash
convert_all2cg -p test/1ab1_A.pdb -o test/1ab1_A.calpha.pdb --cg CalphaBasedModel
```


# cg2all
Convert coarse-grained protein structure to all-atom model

## Installation
```bash
pip install git+http://github.com/huhlim/cg2all
```
This step will install Python libraries including [cg2all (this repository)](https://github.com/huhlim/cg2all), [a modified MDTraj](https://github.com/huhlim/mdtraj), and other dependent libraries. It also places executables `convert_cg2all` and `convert_all2cg` in your python binary directory.

## Usages
### convert_cg2all
convert a coarse-grained protein structure to all-atom model
```bash
usage: convert_cg2all [-h] -p IN_PDB_FN [-d IN_DCD_FN] -o OUT_FN --ckpt CKPT_FN [--time TIME_JSON] [--device DEVICE]

optional arguments:
  -h, --help            show this help message and exit
  -p IN_PDB_FN, --pdb IN_PDB_FN
  -d IN_DCD_FN, --dcd IN_DCD_FN
  -o OUT_FN, --out OUT_FN, --output OUT_FN
  --ckpt CKPT_FN
  --time TIME_JSON
  --device DEVICE
```
#### arguments
* -p/--pdb: Input PDB file (**mandatory**).
* -d/--dcd: Input DCD file (optional). If a DCD file is given, the input PDB file will be used to define its topology.
* -o/--out/--output: Output PDB or DCD file (**mandatory**). If a DCD file is given, it will be a DCD file. Otherwise, a PDB file will be created.
* --ckpt: Input PyTorch ckpt file. (**mandatory**)
* --time: Output JSON file for recording timing (optional).
* --device: Specify a device to run the model (optional) You can choose "cpu", "gpu", or "cuda", or the script will detect one automatically.

<hr/>

### convert_all2cg
convert an all-atom protein structure to coarse-grained model
```bash
usage: all2cg [-h] -p IN_PDB_FN [-d IN_DCD_FN] -o OUT_FN [--cg {CalphaBasedModel,CA,ca,ResidueBasedModel,RES,res,Martini,martini}]

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
* --cg: Coarse-grained representation to use.
  * CalphaBasedModel, CA, ca: the C-alpha atom for a residue
  * ResidueBasedModel, RES, res: the center of mass of a residue for the residue
  * Martini, martini: [MARTINI](http://cgmartini.nl/index.php/martini) model

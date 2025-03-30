# FSP (Feature Space Partition) implementation in Python using GPU

[![DOI](https://zenodo.org/badge/807089780.svg)](https://doi.org/10.5281/zenodo.15048153)

## Licença
THis project is licenced uder Apache 2.0 - details in [LICENSE](LICENSE).

## Abstract 

Implementation of FSP in python running over GPU hardware.

### How Execute

#### Software requirements:

Docker container or Conda/Manba/Miniforge3 environmet!

#### Tested environments:

Env | Result
---------- | ---------
Windows 11 + Docker V19.03.8 | OK
Ubuntu 22.04 + Docker V20.10.14 | OK
Ubuntu 22.04 + Miniforge V20.10.14 | OK


#### Fast Execution (accessing Docker Hub directly)
1 - Run image automatically downloading the last version published in docker hub
```
$ docker run -p 8888:8888 fsp/fsp-python-gpu
```
2 - Open your browser in the same host machine and access jupyter notbook URL

http://127.0.0.1:8888

### Running locally from conda environment
1 - With a Conda/Mamba/Miniforge enviroment installed and initilized, clone the original repository in GitHub
```    
$ git clone https://github.com/sauloaalmeida/fsp-python-gpu --branch=v1.0
```
2 - Go to the cloned directory
```    
$ cd fps-python-gpu
```
3 - Create conda project environment
```    
$ conda env create -f fsp-gpu.yml
```
4 - Activate project environment
```    
$ conda activate fsp-gpu
```

### Running locally unit tests
1 - After activate conda project environment:
```    
$ conda activate fsp-gpu
```

2 - For a detailed report, in root's directory execute:
```    
$ pytest -v
```

### Running on docker locally from scratch
1 - Clone the original repository in github
```    
$ git clone https://github.com/sauloaalmeida/fsp-python-gpu --branch=v1.0
```


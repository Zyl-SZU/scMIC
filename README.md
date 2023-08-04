# scMIC
The source code and input data of scMIC

## Requirement
- Pytorch --- 1.12.1
- Python --- 3.8.16
- Numpy --- 1.24.3
- Scipy --- 1.10.1
- Sklearn --- 1.2.2
- Munkres --- 1.1.4
- tqdm ---4.65.0

## Usage
#### Clone this repo.
```
git clone https://github.com/Zyl-SZU/scMIC.git
```

#### Code structure
- ```data_loader.py```: loads the dataset and construct the cell graph
- ```opt.py```: defines parameters
- ```utils.py```: defines the utility functions
- ```encoder.py```: defines the AE, GAE and q_distribution
- ```scMIC.py```: defines the architecture of the whole network
- ```main.py```: run the model


#### Example command
Take the datasets "PBMC-10k" and "Ma-2020-1" as an example

Using pre-trained model
```
python main.py --name PBMC-10k
```
```
python main.py --name Ma-2020-1
```
Pre-training model command
```
python main.py --name PBMC-10k --pretrain True --dropout 0.4
```
```
python main.py --name Ma-2020-1 --pretrain True --dropout 0.4
```

## Data availability
|  Dataset              | Protocol   | Source |
| --------------------------- | ----------------------- | ----------------------- |
| ***PBMC-10k***             | ***10x Multiome***      | ***https://support.10xgenomics.com/single-cell-multiome-atac-gex/datasets/1.0.0/pbmc_granulocyte_sorted_10k*** |
| ***PBMC-3K***          | ***10x Multiome***      | ***https://www.10xgenomics.com/resources/datasets/pbmc-from-a-healthy-donor-no-cell-sorting-3-k-1-standard-2-0-0***     |
| ***PBMC-CITE***              | ***10x Multiome***           | ***https://support.10xgenomics.com/single-cell-gene-expression/datasets/3.0.0/pbmc_10k_protein_v3*** |
| ***Ma-2020***             | ***SHARE-seq*** | ***https://scglue.readthedocs.io/en/latest/data.html***        |

## Triple-omics integration
#### Quick start
```
python ./src_triple/main.py --name dataset
```


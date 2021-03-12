# Format for samples

## Generate Synthetic Dataset
Run from this folder: `Shi-VAE/src/hmm_dataset`
```
conda activate Shi-VAE
python3 heterHMMdataset.py 
```

## Dataset Structure
A folder is generated with the form:
`hmm_heter_1000_1real_1pos_1bin_1cat_3_100`
with template
```
hmm_heter{N}_{d}{type_d}_..._{K}_{T},
```
where:
- `N`: Number of samples in the dataset.
- `D`: Number of attributes for type `type_d`
- `type_d`: Data type: real, positive, binary, or categorical.
- `K`: Number of hidden states for the Heterogeneous HMM.
- `T`: Time-length for every individual sequence. 

## 

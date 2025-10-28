# cry_analyzer

# Model 
https://drive.google.com/file/d/1szE_3IFeeR5-WK1p4GEbuyg1xlWRdH4O/view?usp=sharing

## Dataset
I borrowed from: 
- [Infant cry audio corpus](https://github.com/gveres/donateacry-corpus)

[image](img/dataset_original.png)

Since the dataset is too imbalanced, I ueed some augmentation techniques to balance the dataset. 
- --target for the number of target data after balancing 

```bash
python balance_data.py cry_data --target 60 --strategy hybrid
```


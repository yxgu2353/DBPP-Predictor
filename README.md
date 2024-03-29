# DBPP-Predictor
The codes and data for the DBPP-Predictor.

## Requipments
* python 3.7
* descriptastorus 2.2.0
* Pytorch 1.5.0+
* dgl-cu101 0.7.0
* dgllife 0.2.8
* rdkit 2021.03.4
* scikit-learn
* numpy
* pandas

## Standalone software available
The DBPP-predictor standalone software is available at https://www.amazon.com/clouddrive/share/f9d5ZQk6UE5ayGSKfnZJu93Cg2SSQE4el9SMM7aZpUK (Amazon Drive)
_or_ 
https://figshare.com/articles/software/DBPP-Predictor_standalone_software/23813805 (Figshare)  
  **Guidance:**  
  1.download: Full version standlone software available at Amazon Drive and Figshare repositories.  
  2.Access: The DBPP-Predictor standalone software does not need to be installed, users can start using it by double clicking DBPP_Software\dist\DBPP_Predictor.exe file.  
  3.Usage: Detailed instructions can be obtained through the "Help" button of the software.  

## Molecular representation
In this study, four representation methods were explored including molecular descriptors, molecular fingerprints, molecular graphs and property profiles. They can be implemented as follows:

### Molecular descriptors
```
python descriptor_calc.py
```
### Molecular fingerprints
```
python FP_calc.py
```
### Property profiles
```
python Property_Profiles_csv.py
```


## Model training
### GNN models
The four GNN models can be trained as follow:

```
python AttentiveFP_classify.py
python GAT_classify.py
python GCN_classify.py
python GraphSAGE_classify.py
```
### Logistic regression model based on QED
```
python LR_QED.py
```
### DBPP-Predictor
```
python DBPP_model.py
```

## Trained model
All the models in this study were available in the ‘models’ folder.

## DBPP score
The DBPP score can be utilized for the assessment of drug-like properties of new compounds. It can be implemented as follows:
```
python Model_Validation.py
```

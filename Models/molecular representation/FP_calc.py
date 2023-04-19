# _*_ coding: utf-8 _*_
# @Time : $[DATE] $[TIME]
# @Author : G

# rdkit Fingerprint Calculate
from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import MACCSkeys
import pandas as pd
def rdkit_fingerprint(input_data, out_name, fp_name):
    with open(out_name+'_'+fp_name+'.csv', 'w') as fp:
        for smi, label in zip(input_data.SMILES, input_data.label):
            mol = Chem.MolFromSmiles(smi)
            # SMILES is error
            if not mol:
                print('Cannot calculate this %s' %smi)
            if fp_name == 'RDKFingerprint':
                fingerprint = rdmolops.RDKFingerprint(mol).ToBitString()
            elif fp_name == 'Morgan':
                fingerprint = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2).ToBitString()
            elif fp_name == 'MACCS':
                fingerprint = MACCSkeys.GenMACCSKeys(mol).ToBitString()
            elif fp_name == 'AtomPairs':
                fingerprint = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol).ToBitString()
            elif fp_name =='TopoTorsion':
                fingerprint = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol).ToBitString()
            else:
                return False
            fp.write(str(label) + ',' + ','.join(fingerprint) + '\n')

if __name__ == '__main__':
    # rdkit FP calc
    target_data = 'Valid_Sample1'  # output file name eg: train or test
    data = pd.read_csv(target_data + '.csv')  # csv data for using
    fp_list = ['RDKFingerprint', 'Morgan', 'MACCS', 'AtomPairs', 'TopoTorsion']
    # fp_list = ['RDKFingerprint', 'AtomPairs', 'TopoTorsion']
    # fp_list = ['MACCS', 'Morgan']
    for fpname in fp_list:
        rdkit_fingerprint(data, target_data, fpname)
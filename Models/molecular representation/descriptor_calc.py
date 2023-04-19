# _*_ coding: utf-8 _*_
# @Time : $[DATE] $[TIME]
# @Author : G

import numpy as np
from descriptastorus.descriptors.DescriptorGenerator import  MakeGenerator
def DB_RdkitEF(df, is_load_path = None, Save_PATH= './cache.npy'):
    print('>>>>>>>>>>--- Generating Features for dictionaries--->>>>>>>>>>')
    if is_load_path != None:
        print('Using cache --->>>>>>>>>>')
        db_ef = np.load(is_load_path, allow_pickle=True).item()
        return db_ef
    dfl = df.values.tolist()
    db_ef = {}
    sum = len(dfl)
    generator = MakeGenerator(('rdkit2d',))
    feature_list = []
    for name in generator.GetColumns():
        name = list(name)
        name_list = name[0]
        feature_list.append(name_list)
    feature_list.remove(feature_list[0])
    print(feature_list)
    with open('Valid_Sample1_Descriptor.csv', 'w') as sf:
        sf.write('label' + ',' +'SMILES'+','+ ','.join('%s'%a for a in feature_list) + '\n')
        for i in range(sum):
            print('Making {}/{} molecule'.format(i, sum))
            Smiles = dfl[i][0]
            label = dfl[i][1]
            feature = generator.process(Smiles)
            features = feature[1:]
            db_ef[Smiles] = features
            sf.write(str(label) + ',' + str(Smiles) + ',' + ','.join('%s'%a for a in features) + '\n')
        np.save(Save_PATH, db_ef)
        return db_ef

import pandas as pd
df = pd.read_csv('Valid_Sample1.csv')
A = DB_RdkitEF(df)

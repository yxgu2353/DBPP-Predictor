# _*_ coding: utf-8 _*_
# @Time : $[DATE] $[TIME]
# @Author : G

import pandas as pd
from descriptastorus.descriptors.DescriptorGenerator import MakeGenerator
import numpy as np
import pickle
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import frozen_dir
import os

frozen = frozen_dir.app_path()

def PC_Profile(smi, label):
    summary = len(smi)
    generator = MakeGenerator(('rdkit2dnormalized',))
    # generator = MakeGenerator(('rdkit2d',))
    feature_list = []
    for name in generator.GetColumns():
        name = list(name)
        name_list = name[0]
        feature_list.append(name_list)
    feature_list.remove(feature_list[0])
    logP = []
    MW1 = []
    HBA1 = []
    HBD1 = []
    NROT1 = []
    TPSA1 = []
    for i in range(summary):
        SMILES = smi[i]
        feature = generator.process(SMILES)
        features = feature[1:]
        LogP = features[46]
        MW = features[48]
        HBA = features[57]
        HBD = features[58]
        nROT = features[61]
        TPSA = features[103]
        logP.append('%.3f' % LogP)
        MW1.append('%.3f' % MW)
        HBA1.append('%.3f' % HBA)
        HBD1.append('%.3f' % HBD)
        NROT1.append('%.3f' % nROT)
        TPSA1.append('%.3f' % TPSA)
    logP2 = pd.DataFrame(logP, columns=['logP'])
    MW2 = pd.DataFrame(MW1, columns=['MW'])
    HBA2 = pd.DataFrame(HBA1, columns=['HBA'])
    HBD2 = pd.DataFrame(HBD1, columns=['HBD'])
    nROT2 = pd.DataFrame(NROT1, columns=['nROT'])
    TPSA2 = pd.DataFrame(TPSA1, columns=['TPSA'])

    PC_property = pd.concat([label, logP2, MW2, HBA2, HBD2, nROT2, TPSA2], axis=1)
    DBPP_Vis_Result = os.path.join(frozen, f'CSV_Feats_DBPP')
    if not os.path.exists(DBPP_Vis_Result):
        os.makedirs(DBPP_Vis_Result)

    PC_property.to_csv(DBPP_Vis_Result + '/' + 'Corr_PC_profile.csv', index=None)
    return PC_property

def ADMET_Profile(input_data, label):
    # ADMET_profile
    fps = []
    for smi in input_data:
        mol = Chem.MolFromSmiles(smi)
        if not mol:
            print('Cannot calculate %s' % smi)
        # Calculate Morgan fingerprint
        fingerprint = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2).ToBitString()
        fps.append(list(fingerprint))
    fps = np.array(fps).astype('int')

    # ADMET Profile
    Model_dir = frozen + '/ADMET_Models/'
    # HIA
    model_load_HIA = open(f'{Model_dir}' + 'HIA_svm_Morgan.model', 'rb')
    model_HIA = pickle.load(model_load_HIA)
    model_load_HIA.close()
    y_pred_HIA = model_HIA.predict(fps)
    y_pred_score_HIA = model_HIA.predict_proba(fps)[:, 1]

    # HOB
    model_load_HOB = open(f'{Model_dir}' + 'HOB_svm_Morgan.model', 'rb')
    model_HOB = pickle.load(model_load_HOB)
    model_load_HOB.close()
    y_pred_HOB = model_HOB.predict(fps)
    y_pred_score_HOB = model_HOB.predict_proba(fps)[:, 1]

    # P-pgi
    model_load_P_pgi = open(f'{Model_dir}' + 'P-pgi_svm_Morgan.model', 'rb')
    model_P_pgi = pickle.load(model_load_P_pgi)
    model_load_P_pgi.close()
    y_pred_P_pgi = model_P_pgi.predict(fps)
    y_pred_score_P_pgi = model_P_pgi.predict_proba(fps)[:, 1]

    # P-pgs
    model_load_P_pgs = open(f'{Model_dir}' + 'P-pgs_svm_Morgan.model', 'rb')
    model_P_pgs = pickle.load(model_load_P_pgs)
    model_load_P_pgs.close()
    y_pred_P_pgs = model_P_pgs.predict(fps)
    y_pred_score_P_pgs = model_P_pgs.predict_proba(fps)[:, 1]

    # Caco-2
    model_load_Caco2 = open(f'{Model_dir}' + 'Caco2_svm_Morgan.model', 'rb')
    model_Caco2 = pickle.load(model_load_Caco2)
    model_load_Caco2.close()
    y_pred_Caco2 = model_Caco2.predict(fps)
    y_pred_score_Caco2 = model_Caco2.predict_proba(fps)[:, 1]

    # BCRPi
    model_load_BCRPi = open(f'{Model_dir}' + 'BCRPi_svm_Morgan.model', 'rb')
    model_BCRPi = pickle.load(model_load_BCRPi)
    model_load_BCRPi.close()
    y_pred_BCRPi = model_BCRPi.predict(fps)
    y_pred_score_BCRPi = model_BCRPi.predict_proba(fps)[:, 1]

    # BSEPi
    model_load_BSEPi = open(f'{Model_dir}' + 'BSEPi_svm_Morgan.model', 'rb')
    model_BSEPi = pickle.load(model_load_BSEPi)
    model_load_BSEPi.close()
    y_pred_BSEPi = model_BSEPi.predict(fps)
    y_pred_score_BSEPi = model_BSEPi.predict_proba(fps)[:, 1]

    # OCT2i
    model_load_OCT2i = open(f'{Model_dir}' + 'OCT2i_svm_Morgan.model', 'rb')
    model_OCT2i = pickle.load(model_load_OCT2i)
    model_load_OCT2i.close()
    y_pred_OCT2i = model_OCT2i.predict(fps)
    y_pred_score_OCT2i = model_OCT2i.predict_proba(fps)[:, 1]

    # OATP1B1i
    model_load_OATP1B1i = open(f'{Model_dir}' + 'OATP1B1i_svm_Morgan.model', 'rb')
    model_OATP1B1i = pickle.load(model_load_OATP1B1i)
    model_load_OATP1B1i.close()
    y_pred_OATP1B1i = model_OATP1B1i.predict(fps)
    y_pred_score_OATP1B1i = model_OATP1B1i.predict_proba(fps)[:, 1]

    # OATP1B3i
    model_load_OATP1B3i = open(f'{Model_dir}' + 'OATP1B3i_svm_Morgan.model', 'rb')
    model_OATP1B3i = pickle.load(model_load_OATP1B3i)
    model_load_OATP1B3i.close()
    y_pred_OATP1B3i = model_OATP1B3i.predict(fps)
    y_pred_score_OATP1B3i = model_OATP1B3i.predict_proba(fps)[:, 1]

    # CL
    model_load_CL = open(f'{Model_dir}' + 'CL_svm_Morgan.model', 'rb')
    model_CL = pickle.load(model_load_CL)
    model_load_CL.close()
    y_pred_CL = model_CL.predict(fps)
    y_pred_score_CL = model_CL.predict_proba(fps)[:, 1]

    # MMP
    model_load_MMP = open(f'{Model_dir}' + 'MMP_svm_Morgan.model', 'rb')
    model_MMP = pickle.load(model_load_MMP)
    model_load_MMP.close()
    y_pred_MMP = model_MMP.predict(fps)
    y_pred_score_MMP = model_MMP.predict_proba(fps)[:, 1]

    # hERG
    model_load_hERG = open(f'{Model_dir}' + 'hERG_svm_Morgan.model', 'rb')
    model_hERG = pickle.load(model_load_hERG)
    model_load_hERG.close()
    y_pred_hERG = model_hERG.predict(fps)
    y_pred_score_hERG = model_hERG.predict_proba(fps)[:, 1]

    # Ames
    model_load_Ames = open(f'{Model_dir}' + 'Ames_svm_Morgan.model', 'rb')
    model_Ames = pickle.load(model_load_Ames)
    model_load_Ames.close()
    y_pred_Ames = model_Ames.predict(fps)
    y_pred_score_Ames = model_Ames.predict_proba(fps)[:, 1]

    # Repro
    model_load_Repro = open(f'{Model_dir}' + 'Repro_svm_Morgan.model', 'rb')
    model_Repro = pickle.load(model_load_Repro)
    model_load_Repro.close()
    y_pred_Repro = model_Repro.predict(fps)
    y_pred_score_Repro = model_Repro.predict_proba(fps)[:, 1]

    # Carc
    model_load_Carc = open(f'{Model_dir}' + 'Carc_svm_Morgan.model', 'rb')
    model_Carc = pickle.load(model_load_Carc)
    model_load_Carc.close()
    y_pred_Carc = model_Carc.predict(fps)
    y_pred_score_Carc = model_Carc.predict_proba(fps)[:, 1]

    # Gene
    model_load_Gene = open(f'{Model_dir}' + 'Gene_svm_Morgan.model', 'rb')
    model_Gene = pickle.load(model_load_Gene)
    model_load_Gene.close()
    y_pred_Gene = model_Gene.predict(fps)
    y_pred_score_Gene = model_Gene.predict_proba(fps)[:, 1]

    # Hepa
    model_load_Hepa = open(f'{Model_dir}' + 'DILI_svm_Morgan.model', 'rb')
    model_Hepa = pickle.load(model_load_Hepa)
    model_load_Hepa.close()
    y_pred_Hepa = model_Hepa.predict(fps)
    y_pred_score_Hepa = model_Hepa.predict_proba(fps)[:, 1]

    # Kidney
    model_load_Kidney = open(f'{Model_dir}' + 'Kidney_svm_Morgan.model', 'rb')
    model_Kidney = pickle.load(model_load_Kidney)
    model_load_Kidney.close()
    y_pred_Kidney = model_Kidney.predict(fps)
    y_pred_score_Kidney = model_Kidney.predict_proba(fps)[:, 1]

    # ROA
    model_load_ROA = open(f'{Model_dir}' + 'ROA_svm_Morgan.model', 'rb')
    model_ROA = pickle.load(model_load_ROA)
    model_load_ROA.close()
    y_pred_ROA = model_ROA.predict(fps)
    y_pred_score_ROA = model_ROA.predict_proba(fps)[:, 1]

    # HIA
    HIA_prob = pd.DataFrame(y_pred_score_HIA, columns=['HIA_Morgan_Prob'])
    # HOB
    HOB_prob = pd.DataFrame(y_pred_score_HOB, columns=['HOB_Morgan_Prob'])
    # P-pgi/s
    P_pgi_prob = pd.DataFrame(y_pred_score_P_pgi, columns=['P_pgi_Morgan_Prob'])
    P_pgs_prob = pd.DataFrame(y_pred_score_P_pgs, columns=['P_pgs_Morgan_Prob'])
    # Caco2
    Caco2_prob = pd.DataFrame(y_pred_score_Caco2, columns=['Caco2_Morgan_Prob'])
    # BCRPi
    BCRPi_prob = pd.DataFrame(y_pred_score_BCRPi, columns=['BCRPi_Morgan_Prob'])
    # BSEPi
    BSEPi_prob = pd.DataFrame(y_pred_score_BSEPi, columns=['BSEPi_Morgan_Prob'])
    # OCT2i
    OCT2i_prob = pd.DataFrame(y_pred_score_OCT2i, columns=['OCT2i_Morgan_Prob'])
    # OATP1B1i
    OATP1B1i_prob = pd.DataFrame(y_pred_score_OATP1B1i, columns=['OATP1B1i_Morgan_Prob'])
    # OATP1B3i
    OATP1B3i_prob = pd.DataFrame(y_pred_score_OATP1B3i, columns=['OATP1B3i_Morgan_Prob'])
    # CL
    CL_prob = pd.DataFrame(y_pred_score_CL, columns=['CL_Morgan_Prob'])
    # MMP
    MMP_prob = pd.DataFrame(y_pred_score_MMP, columns=['MMP_Morgan_Prob'])
    # hERG
    hERG_prob = pd.DataFrame(y_pred_score_hERG, columns=['hERG_Morgan_Prob'])
    # AMES
    Ames_prob = pd.DataFrame(y_pred_score_Ames, columns=['Ames_Morgan_Prob'])
    # Repro
    Repro_prob = pd.DataFrame(y_pred_score_Repro, columns=['Repro_Morgan_Prob'])
    # Carc
    Carc_prob = pd.DataFrame(y_pred_score_Carc, columns=['Carc_Morgan_Prob'])
    # Gene
    Gene_prob = pd.DataFrame(y_pred_score_Gene, columns=['Gene_Morgan_Prob'])
    # Hepa
    Hepa_prob = pd.DataFrame(y_pred_score_Hepa, columns=['Hepa_Morgan_Prob'])
    #  Kidney
    Kidney_prob = pd.DataFrame(y_pred_score_Kidney, columns=['Kidney_Morgan_Prob'])
    # ROA
    ROA_prob = pd.DataFrame(y_pred_score_ROA, columns=['ROA_Morgan_Prob'])
    # label
    Label = pd.DataFrame(label, columns=['label'])

    Prob_results = pd.concat([Label, HIA_prob, HOB_prob, P_pgi_prob, P_pgs_prob, Caco2_prob,
                              BCRPi_prob, BSEPi_prob, OCT2i_prob, OATP1B1i_prob, OATP1B3i_prob,
                              CL_prob, MMP_prob, hERG_prob, Ames_prob, Repro_prob, Carc_prob,
                              Gene_prob, Hepa_prob, Kidney_prob, ROA_prob], axis=1)

    DBPP_Vis_Result = os.path.join(frozen, f'CSV_Feats_DBPP')
    if not os.path.exists(DBPP_Vis_Result):
        os.makedirs(DBPP_Vis_Result)

    Prob_file = Prob_results.to_csv(DBPP_Vis_Result + '/' + 'FDA_ADMET_profile.csv', index=None)
    return Prob_file

if __name__ == "__main__":
    data = pd.read_csv('Con_FDA.csv')
    SMILES = data['SMILES']
    label = data['label']
    # PC_Profile(SMILES, label)
    ADMET_Profile(SMILES, label)


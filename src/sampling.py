from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids, InstanceHardnessThreshold

def resample_smote(feat, tgt):
    sm = SMOTE(random_state = 42)
    return sm.fit_resample(feat, tgt)

def resample_cc(feat, tgt):
    cc = ClusterCentroids(random_state=0)
    return cc.fit_resample(feat, tgt)

def resample_iht(feat, tgt):
    clc = InstanceHardnessThreshold()
    return clc.fit_resample(feat, tgt)
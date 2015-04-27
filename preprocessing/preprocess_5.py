#
# one-hot coding

import scipy.io as sio
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# load mat
matf = '../vessel_data/vessel_delete_update.mat'
vessel_de_up = sio.loadmat(matf)

# non negative
vessel_data = vessel_de_up['vessel_delete_update']
vessel_data = vessel_data+2

# one-hot
enc = OneHotEncoder()
enc.fit(vessel_data)
vessel_onehot = enc.transform(vessel_data).toarray()

sio.savemat('vessel_onehot.mat',{'vessel_onehot':vessel_onehot})

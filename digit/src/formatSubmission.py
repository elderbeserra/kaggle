import pandas as pd
import numpy as np

#orig = pd.read_csv('~/projetos/kaggle/digit/data/rffr_noCV_1.csv')
#
orig = np.genfromtxt(open('/home/elder/projetos/kaggle/digit/data/rffr_noCV_1.csv', 'r'), dtype=np.int32)
ImageId = []
for i in range(1,len(orig)+1):
    ImageId.append(i)
    
#output = np.ndarray(2,28001)

out = pd.DataFrame({"ImageId": ImageId, "Labels": orig})
out.to_csv('~/projetos/kaggle/digit/data/rffr_noCV_1.csv', header=True, sep=',')

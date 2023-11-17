NewModelPath = 'model.pt'
GoodModelPath = 'Goodmodel.pt'

import os
import sys

#delete the old model and rename the new model
os.remove(GoodModelPath)
os.rename(NewModelPath, GoodModelPath)
#remove all checkpoints
for i in range(0, 30,5):
    if os.path.exists(f"{GoodModelPath}_{i}.pt"):
        os.remove(f"{GoodModelPath}_{i}.pt")
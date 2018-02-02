import warnings
warnings.filterwarnings("ignore")
from realdata_experiment.machine_realdata import *


acc_classifer, pred_classifiers, y_test = machineRun(0)
corraltion_range = correlation(pred_classifiers, y_test)
min_corr = round(min([x[2] for x in corraltion_range.values()]), 1)
max_corr = round(max([x[2] for x in corraltion_range.values()]), 1)
corraltion_range = np.arange(min_corr, max_corr + 0.01, 0.1)
min_acc_classifer = min([x[6] for x in acc_classifer])
max_acc_classifer = max([x[6] for x in acc_classifer])

classifer_acc_range = np.arange(min_acc_classifer, max_acc_classifer + 0.01, 0.1)

print (corraltion_range)
print (classifer_acc_range)
from cProfile import label
from tkinter import Label
import matplotlib.pyplot as plt
import numpy as np

AWP = []
SCP = []
AWP_LOSS = []
SCP_LOSS = []
LM = []
for i in range(1024,307200,1024):
    with open("results/eval_results_%s.txt"%i,'r',encoding="utf-8") as f:
        lines = f.readlines()
        AWP.append(float(lines[0].split('=')[1]))
        AWP_LOSS.append(float(lines[1].split('=')[1]))
        SCP.append(float(lines[2].split('=')[1]))
        SCP_LOSS.append(float(lines[3].split('=')[1]))
        LM.append(float(lines[5].split('=')[1]))

X = np.linspace(1024,307200,299)
plt.figure()
plt.plot(X,AWP,label = 'AWP')
plt.plot(X,SCP,Label = 'SCP')
plt.plot(X,LM,Label = 'LM')
# plt.plot(X,AWP_LOSS)
# plt.plot(X,SCP_LOSS)
plt.show()
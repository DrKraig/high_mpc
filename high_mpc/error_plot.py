import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
fig.set_size_inches(7,5)
a = np.load("./dmpc_error.npy")
b = np.load("./hmpc_error.npy")
c = np.load("./mpc_error.npy")
print("dmpc:",np.mean(a[1,:]))
print("hmpc:",np.mean(b[1,:]))
print("mpc:",np.mean(c[1,:]))
data = [[np.mean(a[0,:]),np.mean(a[1,:])],
        [np.mean(b[0,:]),np.mean(b[1,:])],
        [np.mean(c[0,:]),np.mean(c[1,:])]]
data2 = [[np.std(a[0,:]),np.std(a[1,:])],
        [np.std(b[0,:]),np.std(b[1,:])],
        [np.std(c[0,:]),np.std(c[1,:])]]

X = np.arange(2)
ax.bar(X + 0.00, data[0], color = 'b', width = 0.25,label="Deep_high_MPC")
ax.bar(X + 0.25, data[1], color = 'g', width = 0.25,label="High_MPC")
ax.bar(X + 0.50, data[2], color = 'r', width = 0.25,label="MPC")
ax.plot([X[0],X[0]],[0,data[0][0]+data2[0][0]],c='k',marker='_')
ax.plot([X[0]+0.25,X[0]+0.25],[0,data[1][0]+data2[1][0]],c='k',marker='_')
ax.plot([X[0]+0.5,X[0]+0.5],[0,data[2][0]+data2[2][0]],c='k',marker='_')

ax.plot([X[1],X[1]],[0,data[0][1]+data2[0][1]],c='k',marker='_')
ax.plot([X[1]+0.25,X[1]+0.25],[0,data[1][1]+data2[1][1]],c='k',marker='_')
ax.plot([X[1]+0.5,X[1]+0.5],[0,data[2][1]+data2[2][1]],c='k',marker='_')
ax.set_xticks([X[0]+0.25,X[1]+0.25])
ax.set_xticklabels(['Tracking Error','Goal Error'])
ax.set_ylabel("Error")
ax.legend(loc='center left', bbox_to_anchor=(0.0, 0.9))

plt.show()
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
mpl.use('tkagg')


def run(fric, q):
    xys = [[3.32, 1.7], [3.32, 1.7]]
    step = 0.31622776601
    mom = 1.0 - fric * step
    alpha = 2.0 * q * step
    lr = step * step

    init_dist = xys[0][0] ** 2 + xys[0][1]
    for i in range(2000):
        x_prv, y_prv = xys[-2]
        x_cur, y_cur = xys[-1]
        x_new = (x_cur - lr * y_cur +
                 mom * (x_cur - x_prv) -
                 alpha * (y_cur - y_prv))
        x_prv = x_cur + 0.0
        x_cur = x_new
        y_new = (y_cur + lr * x_cur +
                 mom * (y_cur - y_prv) +
                 alpha * (x_cur - x_prv))
        xys += [[x_new, y_new]]

    dist = x_new ** 2 + y_new ** 2
    dist = (dist + 1e-300) / init_dist
    # return min(dist, 1e20)
    return dist

grid_size = 300
fric = np.linspace(0, 2, grid_size)
q = np.linspace(0, 2, grid_size)

frics, qs = np.meshgrid(fric, q)
zs = np.zeros((grid_size, grid_size))

for i in range(grid_size):
    for j in range(grid_size):
        dist = run(frics[i, j], qs[i, j])
        zs[i, j] = -np.log(dist)
        # if dist < 1:
        #     zs[i, j] = -np.log(zs)

plt.figure(figsize=(6, 6))
plt.pcolor(frics, qs, zs, cmap='RdBu', vmin=-zs.max(), vmax=zs.max())
# plt.show()
plt.savefig('out.png')

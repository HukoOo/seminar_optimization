import numpy as np
from scipy.optimize import minimize
from matplotlib import pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d.axes3d import Axes3D

ineq_cons = {'type': 'ineq',
             'fun' : lambda x: np.array([1 - x[0]**2 - x[1]]),
             'jac' : lambda x: np.array([[-2*x[0], -1.0]])}

eq_cons = {'type': 'eq',
           'fun' : lambda x: np.array([2*x[0] + x[1] - 1]),
           'jac' : lambda x: np.array([2.0, 1.0])}

def rosen(x):
    return (1-x[0])**2 + 100*(x[1]-x[0]**2)**2

def rosen_der(x):
    return np.array((-2*(1-x[0])-400*x[0]*(x[1]-x[0]**2),
                     200*(x[1]-x[0]**2)))

x0 = np.array([0.5, 0])
res_cons = minimize(rosen, x0, method='SLSQP', jac=rosen_der,
               constraints=[eq_cons, ineq_cons], options={'ftol': 1e-9, 'disp': False},
               bounds=None)
res_uncons = minimize(rosen, x0, method='SLSQP', jac=rosen_der,
                               options={'ftol': 1e-9, 'disp': False})

print '\nConstrained:'
print res_cons

print '\nUnconstrained:'
print res_uncons

x1, x2 = res_cons['x']
f = res_cons['fun']

x1_unc, x2_unc = res_uncons['x']
f_unc = res_uncons['fun']

# plotting
xgrid = np.mgrid[-2.0:2.0:0.1, -1.0:3.0:0.1]
xvec = xgrid.reshape(2, -1).T
F = np.vstack([rosen(xi) for xi in xvec]).reshape(xgrid.shape[1:])

ax = plt.axes(projection='3d')
ax.plot_surface(xgrid[0], xgrid[1], F, rstride=1, cstride=1,
                cmap=plt.cm.jet, shade=True, alpha=0.9, linewidth=0,
                norm=colors.LogNorm())
ax.plot3D([x1], [x2], [f], 'og', mec='w', label='Constrained minimum')
ax.plot3D([x1_unc], [x2_unc], [f_unc], 'oy', mec='w',
          label='Unconstrained minimum')
ax.legend(fancybox=True, numpoints=1)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('F')
plt.show()
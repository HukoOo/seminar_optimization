import numpy as np
from scipy.optimize import minimize, rosen, rosen_der
from scipy.optimize import BFGS
from matplotlib import pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d.axes3d import Axes3D

def cons_f(x):
    return [1 - x[0]**2 - x[1]]
def cons_J(x):
    return [[-2*x[0], -1.0]]

from scipy.optimize import LinearConstraint
linear_constraint = LinearConstraint([[2, 1]], [1], [1])

from scipy.optimize import NonlinearConstraint
nonlinear_constraint = NonlinearConstraint(cons_f, -np.inf, 1, jac=cons_J, hess=BFGS())

x0 = np.array([0.5, 0])

res_cons = minimize(rosen, x0, method='trust-constr', jac=rosen_der, hess=BFGS(),
               constraints=[linear_constraint,nonlinear_constraint],
               options={'verbose': 1}, bounds=None)

res_uncons = minimize(rosen, x0, method='trust-constr', jac=rosen_der, hess=BFGS(),
                      options={'verbose': 1}, bounds=None)

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
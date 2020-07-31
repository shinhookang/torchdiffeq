import sys, petsc4py
petsc4py.init(sys.argv)
import numpy as np
from petsc4py import PETSc

import torch


class DEMO(object):
    n = 2
    comm = PETSc.COMM_SELF
    def __init__(self, prec_=np.float64,mf_=False):
        self.prec_ = prec_
        self.mf_ = mf_
        if self.mf_:
            self.J_ = PETSc.Mat().createDense([self.n,self.n], comm=self.comm)
            self.J_.setUp()
    def initialCondition(self, u):
        u[0] = 2.0
        u[1] = 0
        u.assemble()
    def evalFunction(self, ts, t, u, f):
        J = np.array([[-0.1, 2.0], [-2.0, -0.1]],dtype=self.prec_)
        ua = u.array
        ua = ua.astype(self.prec_)
        f.array = np.dot(J,ua**3)
    def evalJacobian(self, ts, t, u, A, B):
        J = np.array([[-0.1, 2.0], [-2.0, -0.1]],dtype=self.prec_)
        ua = u.array
        ua = ua.astype(self.prec_)
        Aa = A.getDenseArray()
        Aa = np.matmul(J,np.diag(3.0*ua**2))
        A[:,:] = Aa
        A.assemble()
        return True # same nonzero pattern

    def evalIFunction(self, ts, t, u, udot, f):
        J = np.array([[-0.1, 2.0], [-2.0, -0.1]],dtype=self.prec_)
        ua = u.array
        ua = ua.astype(self.prec_)
        f.array = udot.array - np.dot(J,ua**3)
    def evalIJacobian(self, ts, t, u, udot, shift, A, B):
        J = np.array([[-0.1, 2.0], [-2.0, -0.1]],dtype=self.prec_)
        ua = u.array
        ua = ua.astype(self.prec_)
        Aa = A.getDenseArray()
        Aa = np.matmul(J,np.diag(3.0*ua**2))
        A[:,:] = shift*np.eye(2) - Aa
        A.assemble()
        return True # same nonzero pattern

class JacShell:
    def __init__(self, ode):
        self.ode_ = ode
    def mult(self, A, x, y):
        "y <- A * x"
        self.ode_.J_.mult(x,y)
    def multTranspose(self, A, x, y):
        "y <- A' * x"
        self.ode_.J_.multTranspose(x, y)

OptDB = PETSc.Options()

mf_ = OptDB.getBool('mf', False)

implicitform_ = OptDB.getBool('implicitform', True)
reducedprecision = OptDB.getBool('reducedprecision', False)

if reducedprecision:
  ode = DEMO(np.float32,mf_)
else:
  ode = DEMO(np.float64,mf_)
  #ode = DEMO(torch.float64,mf_)

if not mf_:
    J = PETSc.Mat().createDense([ode.n,ode.n], comm=ode.comm)
    J.setUp()
    Jp = PETSc.Mat().createDense([ode.n,1], comm=ode.comm)
    Jp.setUp()
else:
    J = PETSc.Mat().create()
    J.setSizes([ode.n,ode.n])
    J.setType('python')
    shell = JacShell(ode)
    J.setPythonContext(shell)
    J.setUp()
    J.assemble()
    Jp = PETSc.Mat().create()
    Jp.setSizes([ode.n,1])
    Jp.setType('python')
    shell = JacPShell(ode)
    Jp.setPythonContext(shell)
    Jp.setUp()
    Jp.assemble()

u = PETSc.Vec().createSeq(ode.n, comm=ode.comm)
f = u.duplicate()

ts = PETSc.TS().create(comm=ode.comm)
ts.setEquationType(ts.EquationType.ODE_EXPLICIT)
ts.setProblemType(ts.ProblemType.NONLINEAR)

if implicitform_:
    ts.setType(ts.Type.CN)
    ts.setIFunction(ode.evalIFunction, f)
    ts.setIJacobian(ode.evalIJacobian, J)
else:
    ts.setType(ts.Type.RK)
    ts.setRHSFunction(ode.evalFunction, f)
    ts.setRHSJacobian(ode.evalJacobian, J)

ts.setSaveTrajectory()
ts.setTime(0.0)
ts.setTimeStep(1.0)
ts.setMaxTime(1.0)
ts.setExactFinalTime(PETSc.TS.ExactFinalTime.MATCHSTEP)

ts.setFromOptions()
ode.initialCondition(u)
ts.solve(u)
print(u)
u.view()

del ode, J, u, f, ts
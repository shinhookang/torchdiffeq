import torch
import torch.nn as nn
from .._impl.misc import _flatten, _flatten_convert_none_to_zeros
import petsc4py
from petsc4py import PETSc

class JacShell:
    def __init__(self, ode):
        self.ode_ = ode

    def multTranspose(self, A, X, Y):
        self.x_tensor = torch.from_numpy(X.getArray().reshape(self.ode_.u_tensor.size())).type(torch.FloatTensor)
        y = Y.array
        f_params = tuple(self.ode_.func.parameters())
        with torch.set_grad_enabled(True):
            self.ode_.u_tensor = self.ode_.u_tensor.detach().requires_grad_(True)
            func_eval = self.ode_.func(self.ode_.t, self.ode_.u_tensor)
            vjp_u = torch.autograd.grad(
                func_eval, self.ode_.u_tensor,
                self.x_tensor, allow_unused=True, retain_graph=True
            )
        # autograd.grad returns None if no gradient, set to zero.
        # vjp_u = tuple(torch.zeros_like(y_) if vjp_u_ is None else vjp_u_ for vjp_u_, y_ in zip(vjp_u, y))
        if vjp_u[0] is None: vjp_u[0] = torch.zeros_like(y)
        y[:] = vjp_u[0].numpy().flatten()

class JacPShell:
    def __init__(self, ode):
        self.ode_ = ode

    def multTranspose(self, A, X, Y):
        self.x_tensor = torch.from_numpy(X.getArray().reshape(self.ode_.u_tensor.size())).type(torch.FloatTensor)
        y = Y.array
        f_params = tuple(self.ode_.func.parameters())
        with torch.set_grad_enabled(True):
            # t = t.to(self.u_tensor.device).detach().requires_grad_(False)
            func_eval = self.ode_.func(self.ode_.t, self.ode_.u_tensor)
            vjp_params = torch.autograd.grad(
                func_eval, f_params,
                self.x_tensor, allow_unused=True, retain_graph=True
            )
        # autograd.grad returns None if no gradient, set to zero.
        vjp_params = _flatten_convert_none_to_zeros(vjp_params, f_params)
        y[:] = vjp_params.numpy().flatten()

class ODEPetsc(object):
    comm = PETSc.COMM_SELF

    def __init__(self):
        self.ts = PETSc.TS().create(comm=self.comm)

    def evalFunction(self, ts, t, U, F):
        f = F.array
        self.u_tensor = torch.from_numpy(U.getArray().reshape(self.u_tensor.size())).type(torch.FloatTensor)
        dudt = self.func(t, self.u_tensor).cpu().detach().numpy()
        f[:] = dudt.flatten()

    def evalJacobian(self, ts, t, U, Jac, JacPre):
        """Cache t and U for matrix-free Jacobian """
        self.t = t
        self.u_tensor = torch.from_numpy(U.getArray().reshape(self.u_tensor.size())).type(torch.FloatTensor)

    def evalJacobianP(self, ts, t, U, Jacp):
        """Cache t and U for matrix-free Jacobian """
        self.t = t
        self.u_tensor = torch.from_numpy(U.getArray().reshape(self.u_tensor.size())).type(torch.FloatTensor)

    def setupTS(self, u_tensor, func, step_size=0.01, enable_adjoint=True):
        self.u_tensor = u_tensor
        self.n = u_tensor.numel()
        self.U = PETSc.Vec().createWithArray(self.u_tensor.numpy()) # convert to PETSc vec

        self.func = func
        self.step_size = step_size
        self.flat_params = _flatten(func.parameters())
        self.np = self.flat_params.numel()

        self.ts.reset()
        self.ts.setType(PETSc.TS.Type.RK)
        self.ts.setExactFinalTime(PETSc.TS.ExactFinalTime.MATCHSTEP)

        F = self.U.duplicate()
        self.ts.setRHSFunction(self.evalFunction, F)

        Jac = PETSc.Mat().create()
        Jac.setSizes([self.n, self.n])
        Jac.setType('python')
        shell = JacShell(self)
        Jac.setPythonContext(shell)
        Jac.setUp()
        Jac.assemble()
        self.ts.setRHSJacobian(self.evalJacobian, Jac)

        if enable_adjoint :
            Jacp = PETSc.Mat().create()
            Jacp.setSizes([self.n, self.np])
            Jacp.setType('python')
            shell = JacPShell(self)
            Jacp.setPythonContext(shell)
            Jacp.setUp()
            Jacp.assemble()
            self.ts.setRHSJacobianP(self.evalJacobianP, Jacp)

            self.adj_u = []
            self.adj_u.append(PETSc.Vec().createSeq(self.n, comm=self.comm))
            self.adj_p = []
            self.adj_p.append(PETSc.Vec().createSeq(self.np, comm=self.comm))
            # self.adj_p.append(torch.zeros_like(self.flat_params))
            self.ts.setCostGradients(self.adj_u, self.adj_p)
            self.ts.setSaveTrajectory()
           

        # self.ts.setMaxSteps(1000)
        self.ts.setFromOptions()
        self.ts.setTimeStep(step_size) # overwrite the command-line option

    def odeint(self, u0, t, method='dopri5' ):
        """Return the solutions in tensor"""
        # self.u0 = u0.clone().detach() # clone a new tensor that will be used by PETSc
        U = self.U
        U = PETSc.Vec().createWithArray(u0.numpy()) # convert to PETSc vec
        ts = self.ts
        solution = [u0]
        t = t.to(u0[0].device, torch.float64)
        ts.setTime(t[0])
        ts.setStepNumber(0)
        # choose the solver here. Currently only RK families are included.
        ts.setType('rk')
        if method=='euler':
            ts.setRKType('1fe')
        elif method == 'midpoint':
            ts.setRKType('2a')
        elif method == 'rk4':
            ts.setRKType('4')
        elif method == 'dopri5_fixed':
            ts.setRKType('5dp')
        

        for i in range(1, len(t)):
            ts.setMaxTime(t[i])
            ts.setTimeStep(self.step_size) # reset the step size because the last time step of TSSolve() may be changed even the fixed time step is used.
            ts.solve(U)

            unew = torch.from_numpy(U.getArray().reshape(u0.size())).type(torch.FloatTensor)
            solution.append(unew)
        solution = torch.stack([solution[i] for i in range(len(t))], dim=0)
        return solution

    def petsc_adjointsolve(self, t):
        t = t.to(self.u_tensor.device, torch.float64)
        ts = self.ts
        dt = ts.getTimeStep()
        # print('do {} adjoint steps'.format(round(((t[1]-t[0])/dt).abs().item())))
        ts.adjointSetSteps(round(((t[1]-t[0])/dt).abs().item()))
        ts.adjointSolve()
        adj_u, adj_p = ts.getCostGradients()
        adj_u_tensor = torch.from_numpy(adj_u[0].getArray().reshape(self.u_tensor.size())).type(torch.FloatTensor)
        adj_p_tensor = torch.from_numpy(adj_p[0].getArray().reshape(self.np)).type(torch.FloatTensor)
        return adj_u_tensor, adj_p_tensor

    def odeint_adjoint(self, y0, t, method='dopri5'):
        # We need this in order to access the variables inside this module,
        # since we have no other way of getting variables along the execution path.

        if not isinstance(self.func, nn.Module):
            raise ValueError('func is required to be an instance of nn.Module.')

        ys = OdeintAdjointMethod.apply(y0,t,self.flat_params,self,method)
        return ys

class OdeintAdjointMethod(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args):
        """
        Solve the ODE forward in time
        """
        assert len(args) >= 5, 'Internal error: all arguments required.'
        y0, t, flat_params, ode, method = args[-5], args[-4], args[-3], args[-2], args[-1]

        ctx.ode = ode

        with torch.no_grad():
            ans = ode.odeint(y0, t, method)
        ctx.save_for_backward(t, flat_params, ans)
        return ans

    @staticmethod
    def backward(ctx, *grad_output):
        """
        Compute adjoint using PETSc TSAdjoint
        """
        t, flat_params, ans = ctx.saved_tensors
        T = ans.shape[0]
        with torch.no_grad():
            ctx.ode.adj_u[0].setArray(grad_output[0][-1].numpy())

            for i in range(T-1, 0, -1):
                adj_u_tensor, adj_p_tensor = ctx.ode.petsc_adjointsolve(torch.tensor([t[i], t[i - 1]]))
                adj_u_tensor = adj_u_tensor + grad_output[0][i-1] # add forcing
        return (adj_u_tensor, None, adj_p_tensor, None, None)
import torch
import torch.nn as nn
from .._impl.misc import _flatten, _flatten_convert_none_to_zeros
import petsc4py
from petsc4py import PETSc


class JacShell:
    def __init__(self, ode):
        self.ode_ = ode

    def multTranspose(self, A, X, Y):
        self.x_tensor = torch.from_numpy(X.getArray(readonly=True).reshape(self.ode_.u_tensor.size())).type(torch.FloatTensor)
        y = Y.array
        f_params = tuple(self.ode_.func.parameters())
        with torch.set_grad_enabled(True):
            self.ode_.u_tensor = self.ode_.u_tensor.detach().requires_grad_(True)
            u_tensor = self.ode_.u_tensor.to(self.ode_.device)
            func_eval = self.ode_.func(self.ode_.t, u_tensor)
            vjp_u = torch.autograd.grad(
                func_eval, u_tensor,
                self.x_tensor.to(self.ode_.device), allow_unused=True, retain_graph=True
            )
        # autograd.grad returns None if no gradient, set to zero.
        # vjp_u = tuple(torch.zeros_like(y_) if vjp_u_ is None else vjp_u_ for vjp_u_, y_ in zip(vjp_u, y))
        if vjp_u[0] is None: vjp_u[0] = torch.zeros_like(y)
        y[:] = vjp_u[0].cpu().numpy().flatten()

class JacPShell:
    def __init__(self, ode):
        self.ode_ = ode

    def multTranspose(self, A, X, Y):
        self.x_tensor = torch.from_numpy(X.getArray(readonly=True).reshape(self.ode_.u_tensor.size())).type(torch.FloatTensor)
        y = Y.array
        f_params = tuple(self.ode_.func.parameters())
        with torch.set_grad_enabled(True):
            # t = t.to(self.u_tensor.device).detach().requires_grad_(False)
            u_tensor = self.ode_.u_tensor.to(self.ode_.device)
            func_eval = self.ode_.func(self.ode_.t, u_tensor)
            vjp_params = torch.autograd.grad(
                func_eval, f_params,
                self.x_tensor.to(self.ode_.device), allow_unused=True, retain_graph=True
            )
        # autograd.grad returns None if no gradient, set to zero.
        vjp_params = _flatten_convert_none_to_zeros(vjp_params, f_params)
        y[:] = vjp_params.cpu().numpy().flatten()

class ODEPetsc(object):
    comm = PETSc.COMM_SELF

    def __init__(self):
        self.ts = PETSc.TS().create(comm=self.comm)
        self.has_monitor = False

    def evalFunction(self, ts, t, U, F):
        f = F.array
        self.u_tensor = torch.from_numpy(U.getArray(readonly=True).reshape(self.u_tensor.size())).type(torch.FloatTensor)
        dudt = self.func(t, self.u_tensor.to(self.device)).cpu().detach().numpy()
        f[:] = dudt.flatten()

    def evalJacobian(self, ts, t, U, Jac, JacPre):
        """Cache t and U for matrix-free Jacobian """
        self.t = t
        self.u_tensor = torch.from_numpy(U.getArray(readonly=True).reshape(self.u_tensor.size())).type(torch.FloatTensor)

    def evalJacobianP(self, ts, t, U, Jacp):
        """Cache t and U for matrix-free Jacobian """
        self.t = t
        self.u_tensor = torch.from_numpy(U.getArray(readonly=True).reshape(self.u_tensor.size())).type(torch.FloatTensor)

    def saveSolution(self, ts, stepno, t, U):
        """"Save the solutions at intermediate points"""
        dt = ts.getTimeStep()
        #print(dt)
        if True:#abs(t-self.sol_times[self.cur_index]) < 1E-16:
            unew = torch.from_numpy(U.getArray(readonly=True).reshape(self.u_tensor.size())).type(torch.FloatTensor)
            self.sol_list.append(unew)
            self.cur_index = self.cur_index+1

    def setupTS(self, u_tensor, func, step_size=0.01, method='dopri5_fixed', enable_adjoint=True):
        self.device = u_tensor.device
        self.u_tensor = u_tensor.clone().cpu()
        self.n = u_tensor.numel()
        self.U = PETSc.Vec().createWithArray(self.u_tensor.cpu().numpy()) # convert to PETSc vec
        self.func = func
        self.step_size = step_size
        self.flat_params = _flatten(func.parameters())
        self.np = self.flat_params.numel()

        self.ts.reset()
        self.ts.setType(PETSc.TS.Type.RK)
        # set the solver here. Currently only RK families are included.
        if method=='euler':
            self.ts.setRKType('1fe')
        elif method == 'midpoint':  # 2a is Heun's method, not midpoint.
            self.ts.setRKType('2a')
        elif method == 'rk4':
            self.ts.setRKType('4')
        elif method == 'dopri5_fixed':
            self.ts.setRKType('5dp')
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

        if not self.has_monitor:
          self.ts.setMonitor(self.saveSolution)
          self.has_monitor = True

        # self.ts.setMaxSteps(1000)
        self.ts.setFromOptions()
        self.ts.setTimeStep(step_size) # overwrite the command-line option

    def odeint(self, u0, t):
        """Return the solutions in tensor"""
        self.u0 = u0.clone().detach() # clone a new tensor that will be used by PETSc
        U = self.U
        U = PETSc.Vec().createWithArray(self.u0.cpu().numpy()) # convert to PETSc vec
        ts = self.ts
        
        #self.sol_times = t.to(u0[0].device, torch.float64)
        self.sol_times = self._grid_constructor(t).to(u0[0].device, torch.float64)
        #print(self.sol_times)
        assert self.sol_times[0] == self.sol_times[0] and self.sol_times[-1] == self.sol_times[-1]
        self.sol_times = self.sol_times.to(u0[0])
        self.sol_list = []
        self.cur_index = 0
        ts.setTime(self.sol_times[0])
        ts.setMaxTime(self.sol_times[-1])
        ts.setStepNumber(0)
        ts.setTimeStep(self.step_size) # reset the step size because the last time step of TSSolve() may be changed even the fixed time step is used.
        ts.solve(U)
        solution = torch.stack([torch.reshape(self.sol_list[i],u0.shape) for i in range(len(self.sol_list))], dim=0)

        # j = 1
        # sol_interp = [u0]
        # for j0 in range(len(solution)-1):
        #     t0, t1, u00, u1 = self.sol_times[j0], self.sol_times[j0+1], solution[j0], solution[j0+1]
        #     while j < len(t) and t1 > t[j] - self.step_size/1000:# and t1 > t0:
        #         #print(t1,t[j])
        #         sol_interp.append(self._linear_interp(t0,t1,u00,u1,t[j]))
        #         j += 1
                
        
        # sol_interp = torch.stack([sol_interp[i] for i in range(len(sol_interp))], dim=0)
        #print(sol_interp.shape)
        #print(len(t))
        
        return solution.to(u0.device)

    def _grid_constructor(self, t):
        """Construct uniform time grid with step size self.step_size"""
        start_time = t[0]
        end_time = t[-1]
        niters = torch.ceil((end_time - start_time) / self.step_size + 1).item()
        t_infer = torch.arange(0, niters).to(t) * self.step_size + start_time
        if t_infer[-1] > t[-1]:
            t_infer[-1] = t[-1]
        return t_infer
    
    def _linear_interp(self, t0, t1, u0, u1, tj):
        """ Do linear interpolation if tj falls between t0 and t1 """
        if tj == t0:
            return u0
        if tj == t1:
            return u1
        t0_, t1_, tj_ = t0.to(u0[0]), t1.to(u0[0]), tj.to(u0[0])
        slope = torch.stack([(u1_ - u0_) / (t1_ - t0_) for u0_, u1_, in zip(u0, u1)])
        return torch.stack([u0_ + slope_ * (tj_ - t0_) for u0_, slope_ in zip(u0, slope)  ])


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

    def odeint_adjoint(self, y0, t):
        # We need this in order to access the variables inside this module,
        # since we have no other way of getting variables along the execution path.

        if not isinstance(self.func, nn.Module):
            raise ValueError('func is required to be an instance of nn.Module.')

        ys = OdeintAdjointMethod.apply(y0,t,self.flat_params,self)
        return ys

class OdeintAdjointMethod(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args):
        """
        Solve the ODE forward in time
        """
        assert len(args) >= 4, 'Internal error: all arguments required.'
        y0, t, flat_params, ode = args[-4], args[-3], args[-2], args[-1]

        ctx.ode = ode

        with torch.no_grad():
            ans = ode.odeint(y0, t)
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
            ctx.ode.adj_u[0].setArray(grad_output[0][-1].cpu().numpy())
            ctx.ode.adj_p[0].zeroEntries()

            for i in range(T-1, 0, -1):
                adj_u_tensor, adj_p_tensor = ctx.ode.petsc_adjointsolve(torch.tensor([t[i], t[i - 1]]))
                adj_u_tensor += grad_output[0][i-1].cpu() # add forcing
                ctx.ode.adj_u[0].setArray(adj_u_tensor.cpu().numpy()) # update PETSc work vectors

        return (adj_u_tensor.to(grad_output[0].device), None, adj_p_tensor.to(grad_output[0].device), None)
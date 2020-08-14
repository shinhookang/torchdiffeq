import torch
import torch.nn as nn
import torch.utils.dlpack as dlpack
from .._impl.misc import _flatten, _flatten_convert_none_to_zeros
import petsc4py
from petsc4py import PETSc
import copy


class JacShell:
    def __init__(self, ode):
        self.ode_ = ode

    def multTranspose(self, A, X, Y):
        if self.ode_.use_dlpack:
            self.x_tensor = dlpack.from_dlpack(X.toDlpack()).view(self.ode_.cached_u_tensor.size()).type(self.ode_.tensor_type)
            y = dlpack.from_dlpack(Y.toDlpack()).view(self.ode_.cached_u_tensor.size())
        else:
            self.x_tensor = torch.from_numpy(X.getArray(readonly=True).reshape(self.ode_.cached_u_tensor.size())).type(self.ode_.tensor_type).to(self.ode_.device)
            y = Y.array
        
        with torch.set_grad_enabled(True):
            self.ode_.cached_u_tensor = self.ode_.cached_u_tensor.detach().requires_grad_(True)
            # u_tensor = self.ode_.cached_u_tensor.to(self.ode_.device)
            self.ode_.func_eval = self.ode_.func(self.ode_.t, self.ode_.cached_u_tensor)
            vjp_u = torch.autograd.grad(
                self.ode_.func_eval, self.ode_.cached_u_tensor,
                self.x_tensor, allow_unused=True, retain_graph=True
            )
        # autograd.grad returns None if no gradient, set to zero.
        # vjp_u = tuple(torch.zeros_like(y_) if vjp_u_ is None else vjp_u_ for vjp_u_, y_ in zip(vjp_u, y))
        if vjp_u[0] is None: vjp_u[0] = torch.zeros_like(y)
        if self.ode_.use_dlpack:
            y.copy_(vjp_u[0])
        else:
            y[:] = vjp_u[0].cpu().numpy().flatten()

class JacPShell:
    def __init__(self, ode):
        self.ode_ = ode

    def multTranspose(self, A, X, Y):
        if self.ode_.use_dlpack:
            self.x_tensor = dlpack.from_dlpack(X.toDlpack()).view(self.ode_.cached_u_tensor.size()).type(self.ode_.tensor_type)
            y = dlpack.from_dlpack(Y.toDlpack()).view(self.ode_.np)
        else:
            self.x_tensor = torch.from_numpy(X.getArray(readonly=True).reshape(self.ode_.cached_u_tensor.size())).type(self.ode_.tensor_type).to(self.ode_.device)
            y = Y.array
        f_params = tuple(self.ode_.func.parameters())
        with torch.set_grad_enabled(True):
            # t = t.to(self.cached_u_tensor.device).detach().requires_grad_(False)
            # u_tensor = self.ode_.cached_u_tensor.to(self.ode_.device)
            func_eval = self.ode_.func_eval #self.ode_.func(self.ode_.t, self.ode_.cached_u_tensor)
            vjp_params = torch.autograd.grad(
                func_eval, f_params,
                self.x_tensor, allow_unused=True, retain_graph=True
            )
        # autograd.grad returns None if no gradient, set to zero.
        vjp_params = _flatten_convert_none_to_zeros(vjp_params, f_params)
        if self.ode_.use_dlpack:
            y.copy_(vjp_params)
        else:
            y[:] = vjp_params.cpu().numpy().flatten()

class ODEPetsc(object):
    comm = PETSc.COMM_SELF

    def __init__(self):
        self.ts = PETSc.TS().create(comm=self.comm)
        self.has_monitor = False

    def evalFunction(self, ts, t, U, F):
        if self.use_dlpack:
            # have to call to() or type() to avoid a PETSc seg fault
            u_tensor = dlpack.from_dlpack(U.toDlpack()).view(self.cached_u_tensor.size()).type(self.tensor_type)
            dudt = dlpack.from_dlpack(F.toDlpack()).view(self.cached_u_tensor.size())
            # Restoring the handle set the offloadmask flag to PETSC_OFFLOAD_GPU, but it zeros out the GPU memory accidentally, which is probably a bug
            if torch.cuda.is_initialized():
                hdl = F.getCUDAHandle('w')
                F.restoreCUDAHandle(hdl,'w')
            dudt.copy_(self.func(t, u_tensor))
        else:
            f = F.array
            u_tensor = torch.from_numpy(U.array.reshape(self.cached_u_tensor.size())).type(self.tensor_type).to(self.device)
            dudt = self.func(t, u_tensor).cpu().detach().numpy()
            f[:] = dudt.flatten()

    def evalJacobian(self, ts, t, U, Jac, JacPre):
        """Cache t and U for matrix-free Jacobian """
        self.t = t
        if self.use_dlpack:
            self.cached_u_tensor = dlpack.from_dlpack(U.toDlpack()).view(self.cached_u_tensor.size()).type(self.tensor_type)
        else:
            self.cached_u_tensor = torch.from_numpy(U.array.reshape(self.cached_u_tensor.size())).type(self.tensor_type).to(self.device)

    def evalJacobianP(self, ts, t, U, Jacp):
        """Cache t and U for matrix-free Jacobian """
        self.t = t
        if self.use_dlpack:
            self.cached_u_tensor = dlpack.from_dlpack(U.toDlpack()).view(self.cached_u_tensor.size()).type(self.tensor_type)
        else:
            self.cached_u_tensor = torch.from_numpy(U.array.reshape(self.cached_u_tensor.size())).type(self.tensor_type).to(self.device)

    def saveSolution(self, ts, stepno, t, U):
        """"Save the solutions at intermediate points"""
        dt = ts.getTimeStep()
        #print(t)
        if abs(t-self.sol_times[self.cur_index]) < dt/10:#1E-6:
            if self.use_dlpack:
                unew = dlpack.from_dlpack(U.toDlpack()).view(self.cached_u_tensor.size()).clone()
            else:
                unew = torch.from_numpy(U.array.reshape(self.cached_u_tensor.size())).type(self.tensor_type).to(self.device)
            
            self.sol_list.append(unew)
            self.cur_index = self.cur_index+1

    def setupTS(self, u_tensor, func, step_size=0.01, method='euler', enable_adjoint=True):
        self.device = u_tensor.device
        self.cached_u_tensor = u_tensor
        self.tensor_type = u_tensor.dtype#()
        self.use_dlpack = u_tensor.is_cuda
        if self.use_dlpack:
            self.tensor_type = u_tensor.type()
        self.n = u_tensor.numel()
        if self.use_dlpack:
            self.cached_U = PETSc.Vec().createWithDlpack(dlpack.to_dlpack(u_tensor)) # convert to PETSc vec
        else:
            self.cached_U = PETSc.Vec().createWithArray(u_tensor.cpu().numpy()) # convert to PETSc vec
            

        self.func = func
        self.step_size = step_size
        self.flat_params = _flatten(func.parameters())
        self.np = self.flat_params.numel()

        self.ts.reset()
        self.ts.setType(PETSc.TS.Type.RK)
        self.ts.setEquationType(PETSc.TS.EquationType.ODE_EXPLICIT)
        self.ts.setExactFinalTime(PETSc.TS.ExactFinalTime.MATCHSTEP)

        # set the solver here. Currently only RK families are included.
        if method=='euler':
            self.ts.setRKType('1fe')
        elif method == 'midpoint':  # 2a is Heun's method, not midpoint. 
            self.ts.setRKType('2a')
        elif method == 'rk4':
            self.ts.setRKType('4')
        elif method == 'dopri5_fixed':
            self.ts.setRKType('5dp')
        
        F = self.cached_U.duplicate()
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
            if self.use_dlpack:
                self.adj_u.append(PETSc.Vec().createWithDlpack(dlpack.to_dlpack(torch.zeros_like(u_tensor))))
            else:
                self.adj_u.append(PETSc.Vec().createSeq(self.n, comm=self.comm))
            self.adj_p = []
            if self.use_dlpack:
                self.adj_p.append(PETSc.Vec().createWithDlpack(dlpack.to_dlpack(torch.zeros_like(self.flat_params))))
            else:
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
        # check if time grid is decreasing, as PETSc does not support negative time steps
        # if t[0]>t[1]:
        #     t = -t
        #     _base_reverse_func = self.func
        #     self.func = lambda t, y: torch.tensor( tuple(-f_ for f_ in _base_reverse_func(-t, y)))
        self.u0 = u0.clone().detach() # clone a new tensor that will be used by PETSc
        if self.use_dlpack:
            U = PETSc.Vec().createWithDlpack(dlpack.to_dlpack(self.u0)) # convert to PETSc vec
        else:
            U = PETSc.Vec().createWithArray(u0.cpu().detach().numpy()) # convert to PETSc vec
        ts = self.ts

        self.sol_times = t.to(self.device, torch.float64)
        #self.sol_times = self._grid_constructor(t).to(u0[0].device, torch.float64)
        #assert self.sol_times[0] == self.sol_times[0] and self.sol_times[-1] == self.sol_times[-1]
        #self.sol_times = self.sol_times.to(u0[0])
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
        
        return solution

    # def _grid_constructor(self, t):
    #     """Construct uniform time grid with step size self.step_size"""
    #     start_time = t[0]
    #     end_time = t[-1]
    #     niters = torch.ceil((end_time - start_time) / self.step_size + 1).item()
    #     t_infer = torch.arange(0, niters).to(t) * self.step_size + start_time
    #     if t_infer[-1] > t[-1]:
    #         t_infer[-1] = t[-1]
    #     return t_infer
    
    # def _linear_interp(self, t0, t1, u0, u1, tj):
    #     """ Do linear interpolation if tj falls between t0 and t1 """
    #     if tj == t0:
    #         return u0
    #     if tj == t1:
    #         return u1
    #     t0_, t1_, tj_ = t0.to(u0[0]), t1.to(u0[0]), tj.to(u0[0])
    #     slope = torch.stack([(u1_ - u0_) / (t1_ - t0_) for u0_, u1_, in zip(u0, u1)])
    #     return torch.stack([u0_ + slope_ * (tj_ - t0_) for u0_, slope_ in zip(u0, slope)  ])


    def petsc_adjointsolve(self, t):
        t = t.to(self.device, torch.float64)
        ts = self.ts
        dt = ts.getTimeStep()
        # dt = self.step_size
        # print(dt)
        # print('do {} adjoint steps'.format(round(((t[1]-t[0])/dt).abs().item())))
        ts.adjointSetSteps(round(((t[1]-t[0])/dt).abs().item()))
        ts.adjointSolve()
        adj_u, adj_p = ts.getCostGradients()
        if self.use_dlpack:
            adj_u_tensor = dlpack.from_dlpack(adj_u[0].toDlpack()).view(self.cached_u_tensor.size())
            adj_p_tensor = dlpack.from_dlpack(adj_p[0].toDlpack()).view(self.np)  
        else:
            adj_u_tensor = torch.from_numpy(adj_u[0].getArray().reshape(self.cached_u_tensor.size())).type(self.tensor_type).to(self.device)
            adj_p_tensor = torch.from_numpy(adj_p[0].getArray().reshape(self.np)).type(self.tensor_type).to(self.device)
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
            if ctx.ode.use_dlpack:
                adj_u_tensor = dlpack.from_dlpack(ctx.ode.adj_u[0].toDlpack()).view(ctx.ode.cached_u_tensor.size())
                adj_p_tensor = dlpack.from_dlpack(ctx.ode.adj_p[0].toDlpack()).view(ctx.ode.np)
                adj_u_tensor.copy_(grad_output[0][-1].reshape(adj_u_tensor.shape))
                adj_p_tensor.zero_()
            else:
                ctx.ode.adj_u[0].setArray(grad_output[0][-1].cpu().numpy())
                ctx.ode.adj_p[0].zeroEntries()
            for i in range(T-1, 0, -1):
                adj_u_tensor, adj_p_tensor = ctx.ode.petsc_adjointsolve(torch.tensor([t[i], t[i-1]]))
                adj_u_tensor.add_(grad_output[0][i-1].reshape(adj_u_tensor.shape)) # add forcing
                if not ctx.ode.use_dlpack: # if use_dlpack=True, adj_u_tensor shares memory with adj_u[0], so no need to set the values explicitly
                    ctx.ode.adj_u[0].setArray(adj_u_tensor.cpu().numpy()) # update PETSc work vectors              
        print(torch.norm(adj_u_tensor))
        exit()
        return (adj_u_tensor, None, adj_p_tensor, None)
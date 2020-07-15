import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import sys
import copy

# OptDB = PETSc.Options()
# print("first init: ",OptDB.getAll())


#sys.path.append('/home/zhaow/NODE')
sys.path.append('/home/zhaow/torchdiffeq')
parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5','midpoint','rk4','dopri5_fixed', 'adams','euler'], default='dopri5')
parser.add_argument('--step_size',type=eval, default=1)
parser.add_argument('--data_size', type=int, default=101)
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
args, unknown = parser.parse_known_args()

import petsc4py
sys.argv = [sys.argv[0]] + unknown
petsc4py.init(sys.argv)
from petsc4py import PETSc

# Set these random seeds, so everything can be reproduced. 
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True


import torchdiffeq

from torchdiffeq.petscutil import petsc_adjoint as petsc_adjoint
#from petscutil import petsc_adjoint

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

true_y0 = torch.tensor([[2., 0.]])
t = torch.linspace(0., 1., args.data_size)
true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]])
options = {}
options.update({'step_size':args.step_size})

class Lambda(nn.Module):

    def forward(self, t, y):
        return torch.mm(y**3, true_A)


with torch.no_grad():
    
    options_true = {}
    options_true.update({'step_size':args.step_size})

    true_y2 = odeint(Lambda(), true_y0, t, method=args.method, options=options_true)

    ode0 = petsc_adjoint.ODEPetsc()
    ode0.setupTS(true_y0, Lambda(), step_size=args.step_size, method=args.method, enable_adjoint=False)
    true_y = ode0.odeint_adjoint(true_y0,t)
    print(true_y)
    print(true_y2)
    print('Difference between PETSc and NODE reference solutions: {:.6f}'.format(torch.norm(true_y-true_y2)))
    exit()


def get_batch():
    
    s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=True))
    batch_y0 = true_y[s]  # (M, D)
    batch_t = t[:args.batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
    return batch_y0, batch_t, batch_y


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


if args.viz:
    makedirs('png')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_traj = fig.add_subplot(131, frameon=False)
    ax_phase = fig.add_subplot(132, frameon=False)
    ax_vecfield = fig.add_subplot(133, frameon=False)
    plt.show(block=False)


def visualize(true_y, pred_y, odefunc, itr):

    if args.viz:

        ax_traj.cla()
        ax_traj.set_title('Trajectories')
        ax_traj.set_xlabel('t')
        ax_traj.set_ylabel('x,y')
        ax_traj.plot(t.numpy(), true_y.numpy()[:, 0, 0], t.numpy(), true_y.numpy()[:, 0, 1], 'g-')
        ax_traj.plot(t.numpy(), pred_y.numpy()[:, 0, 0], '--', t.numpy(), pred_y.numpy()[:, 0, 1], 'b--')
        ax_traj.set_xlim(t.min(), t.max())
        ax_traj.set_ylim(-2, 2)
        ax_traj.legend()

        ax_phase.cla()
        ax_phase.set_title('Phase Portrait')
        ax_phase.set_xlabel('x')
        ax_phase.set_ylabel('y')
        ax_phase.plot(true_y.numpy()[:, 0, 0], true_y.numpy()[:, 0, 1], 'g-')
        ax_phase.plot(pred_y.numpy()[:, 0, 0], pred_y.numpy()[:, 0, 1], 'b--')
        ax_phase.set_xlim(-2, 2)
        ax_phase.set_ylim(-2, 2)

        ax_vecfield.cla()
        ax_vecfield.set_title('Learned Vector Field')
        ax_vecfield.set_xlabel('x')
        ax_vecfield.set_ylabel('y')

        y, x = np.mgrid[-2:2:21j, -2:2:21j]
        dydt = odefunc(0, torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 2))).cpu().detach().numpy()
        mag = np.sqrt(dydt[:, 0]**2 + dydt[:, 1]**2).reshape(-1, 1)
        dydt = (dydt / mag)
        dydt = dydt.reshape(21, 21, 2)

        ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
        ax_vecfield.set_xlim(-2, 2)
        ax_vecfield.set_ylim(-2, 2)

        fig.tight_layout()
        plt.savefig('png/{:03d}'.format(itr))
        plt.draw()
        plt.pause(0.001)


class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 2),
        )
        self.nfe = 0
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                #nn.init.constant_(m.weight, val=0.1)
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        self.nfe = self.nfe + 1
        return self.net(y**3)


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


if __name__ == '__main__':

    ii = 0
#    NODE implementation 
    func_NODE = ODEFunc()
    optimizer_NODE = optim.RMSprop(func_NODE.parameters(), lr=1e-3)
#   end of NODE


#    Petsc implementation
    func_PETSC = copy.deepcopy(func_NODE)
    ode = petsc_adjoint.ODEPetsc()
    ode.setupTS(torch.zeros(args.batch_size,1,1,2), func_PETSC, args.step_size, args.method, enable_adjoint=True)
    optimizer_PETSC = optim.RMSprop(func_PETSC.parameters(), lr=1e-3)
#   end of PETSC
    end = time.time()

    time_meter = RunningAverageMeter(0.97)
    loss_meter = RunningAverageMeter(0.97)
        
    for itr in range(1, args.niters + 1):
        
        
        optimizer_NODE.zero_grad()
        optimizer_PETSC.zero_grad()

        batch_y0, batch_t, batch_y = get_batch()
        start_NODE = time.time()    
        pred_y_NODE = odeint(func_NODE, batch_y0, batch_t,method=args.method,options=options)
        loss_NODE = torch.mean(torch.abs(pred_y_NODE - batch_y))
        end_NODE = time.time()
        nfe_f_NODE = func_NODE.nfe
        func_NODE.nfe = 0

        start_PETSC = end_NODE
        
        pred_y_PETSC = ode.odeint_adjoint(batch_y0, batch_t)
        #pred_y_PETSC = torch.reshape(pred_y_PETSC, batch_y.shape)
        nfe_f_PETSC = func_PETSC.nfe
        func_PETSC.nfe = 0


        loss_PETSC = torch.mean(torch.abs(pred_y_PETSC - batch_y))
        end_PETSC = time.time()

        loss_NODE.backward()
        optimizer_NODE.step()
        nfe_b_NODE = func_NODE.nfe
        func_NODE.nfe = 0

        loss_PETSC.backward()
        optimizer_PETSC.step()
        nfe_b_PETSC = func_PETSC.nfe
        func_PETSC.nfe = 0

        #   inner product between the gradients from two implementations
        num_diff = 0
        norm_diff = 0
        total_num = 0
        array = []
        array2 = []
        for p1, p2 in zip(func_NODE.parameters(), func_PETSC.parameters()):
            if np.abs(p1.data.ne(p2.data).sum().cpu()) > 1E-4:
                
                num_diff += 1
                norm_diff += np.abs(p1.data.ne(p2.data).sum().cpu())
                array = array + [p1.grad.min().cpu().detach().numpy().tolist()] 
                array2 = array2 + [p2.grad.min().cpu().detach().numpy().tolist()]
                total_num += 1
        
        
        unit_array = array / (np. linalg. norm(array ) + 1E-16)
        unit_array2 = array2 / (np. linalg. norm(array2 ) + 1E-16)
        dot_product = np.dot(unit_array, unit_array2)
        #   end of comparison

        if itr % args.test_freq == 0:
            with torch.no_grad():
                
                pred_y_NODE = odeint(func_NODE, true_y0, t,method=args.method,options=options)
                loss_NODE = torch.mean(torch.abs(pred_y_NODE - true_y))
                print('NODE : Iter {:04d} | Time {:.6f} | Total Loss {:.6f} | NFE-F {:04d} | NFE-B {:04d}'.format(itr,end_NODE-start_NODE, loss_NODE.item(),nfe_f_NODE, nfe_b_NODE))
                #func_NODE.nfe=0
                
                ode0 = petsc_adjoint.ODEPetsc()
                ode0.setupTS(true_y0, func_PETSC, step_size=args.step_size, method=args.method,enable_adjoint=False)
                pred_y_PETSC = ode0.odeint_adjoint(true_y0, t)
                loss_PETSC = torch.mean(torch.abs(pred_y_PETSC - true_y))
                print('PETSC: Iter {:04d} | Time {:.6f} | Total Loss {:.6f} | NFE-F {:04d} | NFE-B {:04d}'.format(itr,end_PETSC-start_PETSC, loss_PETSC.item(),nfe_f_PETSC, nfe_b_PETSC))
                #func_PETSC.nfe=0
                print('Dot product of normalized gradients: {:.6f} | number of different params: {:04d} / {:04d}\n'.format(dot_product,num_diff,total_num))
                #visualize(true_y, pred_y, func, ii)
                ii += 1
                
        end = time.time()

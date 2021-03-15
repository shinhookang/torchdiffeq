import os
import argparse
import time
import numpy as np
import h5py
import glob

import torch
import torch.nn as nn
import torch.optim as optim

# Specify a path
PATH = "coupler_model.pt"

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

# Prepare data
def ReadCCNS2D_Interface_fromHDF5(ifile):
    with h5py.File(ifile, 'r') as f:
        curTime = float(np.array(f['time']))
        QQ1_interface = np.array(f['QQ1'])
        QQ2_interface = np.array(f['QQ2'])
    return curTime,QQ1_interface[:,7],QQ2_interface[:,7]

def getInterfaceData(path_to_folder):
    t = []
    q = []
    count = 0
    filelist = glob.glob(os.path.join(path_to_folder, '*.h5'))
    for f in sorted(filelist):
        if count <= 400:
            root_ext = os.path.splitext(f)
            qt,q1,q2 = ReadCCNS2D_Interface_fromHDF5(f)
            t.append(qt)
            q.append(torch.from_numpy(q1).float().to(device))
        count = count + 1
    t = torch.tensor(t).to(device)
    true_y = torch.empty(len(t), *q[0].shape, dtype=q[0].dtype, device=device)
    for i in range(0,len(t)):
        true_y[i] = q[i]
    return t,true_y

t,true_y = getInterfaceData('../ml-coupling-datafiles/output-interface')
data_size = len(t)

def get_batch():
    s = torch.from_numpy(np.random.choice(np.arange(data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
    batch_y0 = true_y[s]  # (M, D)
    batch_t = t[:args.batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
    return batch_y0.to(device), batch_t.to(device), batch_y.to(device)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(320, 100),
            nn.Tanh(),
            nn.Linear(100, 320),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y)


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

    func = ODEFunc().to(device)
    
    optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
    end = time.time()

    time_meter = RunningAverageMeter(0.97)
    
    loss_meter = RunningAverageMeter(0.97)

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_batch()
        pred_y = odeint(func, batch_y0, batch_t).to(device)
        loss = torch.mean(torch.abs(pred_y - batch_y))
        loss.backward()
        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())

        if itr % args.test_freq == 0:
            with torch.no_grad():
                pred_y = odeint(func, true_y[0], t)
                loss = torch.mean(torch.abs(pred_y - true_y))
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                ii += 1

        end = time.time()
    # Save
    torch.save(func.net.state_dict(), PATH)
    # Load
    # model = Net()
    # model.load_state_dict(torch.load(PATH))
    # model.eval()

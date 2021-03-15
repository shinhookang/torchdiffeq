import os
import numpy as np
import torch
import sys
import glob

sys.path.append('./')
from ocn_atm_coupler import odeint,device,get_batch,ODEFunc,ReadCCNS2D_Interface_fromHDF5

# Specify a path
PATH = "coupler_model.pt"

#parser = argparse.ArgumentParser('ODE demo')
#parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
#parser.add_argument('--gpu', type=int, default=0)
#parser.add_argument('--adjoint', action='store_true')

#if args.adjoint:
#    from torchdiffeq import odeint_adjoint as odeint
#else:
#    from torchdiffeq import odeint

#device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

def getTestingInterfaceData(path_to_folder):
    t = []
    q = []
    count = 0
    filelist = glob.glob(os.path.join(path_to_folder, '*.h5'))
    for f in sorted(filelist):
        if count > 400:
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

t,true_y = getTestingInterfaceData('../ml-coupling-datafiles/output-interface')
tdata_size = len(t)

if __name__ == '__main__':

    func = ODEFunc().to(device)
    num_samples = 10
    func.net.load_state_dict(torch.load(PATH))
    func.eval()
    with torch.no_grad():
        loss = 0
        for i in np.arange(num_samples):
            batch_y0, batch_t, batch_y = get_batch()  
            pred_y = odeint(func, batch_y0, batch_t).to(device)
            loss += torch.mean(torch.abs(pred_y - batch_y))
        
        print(f'Averaged loss on the test data {float(loss)/float(num_samples):.5f}')

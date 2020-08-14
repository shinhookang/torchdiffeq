#*
# @file ANODE training driver based on arxiv:1902.10298
# This file is part of ANODE library.
#
# ANODE is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ANODE is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ANODE.  If not, see <http://www.gnu.org/licenses/>.
#*
import torch
import time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import argparse
import logging
import numpy as np
from tensorboardX import SummaryWriter
import math
import sys
import os
from pytorch_model_summary import summary

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--network', type = str, choices = ['resnet', 'sqnxt'], default = 'sqnxt')
parser.add_argument('--method', type = str, choices=['Euler', 'RK2', 'RK4','Dopri5','Dopri5_fixed'], default = 'Euler')
#parser.add_argument('--adjoint', type=eval, default=False, choices=[True, False])
parser.add_argument('--num_epochs', type = int, default = 200)
parser.add_argument('--lr', type=float, default = 0.1)
parser.add_argument('--Nt', type=int, default = 1)
parser.add_argument('--batch_size', type = int, default = 256)
parser.add_argument('--test_batch_size', type = int, default = 128)
parser.add_argument('--impl',type=str, choices = ['NODE','ANODE','NODE_adj','PETSc'],default = 'ANODE')
parser.add_argument('--tol', type=float, default=1e-3)
parser.add_argument('--gpu', type=int, default=1)
parser.add_argument('--save',type=str, default=None)
parser.add_argument('--implicit', action='store_true')
parser.add_argument('--double_prec', action='store_true')
args, unknown = parser.parse_known_args()

sys.argv = [sys.argv[0]] + unknown
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
if args.network == 'sqnxt' and not args.impl == 'PETSc':
    from models.sqnxt import SqNxt_23_1x, lr_schedule
elif args.network == 'sqnxt' and args.impl == 'PETSc':
    from models.sqnxt_PETSc import SqNxt_23_1x, lr_schedule
    #writer = SummaryWriter(args.network + '/' + args.method + '_lr_' + str(args.lr) + '_Nt_' + str(args.Nt) + '/')
elif args.network == 'resnet':
    from models.resnet import ResNet18, lr_schedule
    #writer = SummaryWriter('resnet/' + args.method + '_lr_' + str(args.lr) + '_Nt_' + str(args.Nt) + '/')
if args.save == None:
    args.save  = args.network+'/' + args.method + '_impl_' + str(args.impl) + '_Nt_' + str(args.Nt) + '/'
writer = SummaryWriter(args.save)

import sys
sys.path.append("../")
if args.impl == 'ANODE':
    sys.path.append('/home/zhaow/anode')
    from anode import odesolver_adjoint as odesolver


is_use_cuda = torch.cuda.is_available()
torch.cuda.set_device(args.gpu)
device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
if args.double_prec:
    tensor_type = torch.float64
else:
    tensor_type = torch.float32
import torchdiffeq
if args.implicit:
    from torchdiffeq.petscutil import petsc_adjoint_implicit as petsc_adjoint
else:
    from torchdiffeq.petscutil import petsc_adjoint_explicit as petsc_adjoint

if args.impl == 'NODE_adj':
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


num_epochs = int(args.num_epochs)
lr           = float(args.lr)
start_epoch  = 1
batch_size   = int(args.batch_size)
test_batch_size = int(args.test_batch_size)


best_acc    = 0.

class ODEBlock_ANODE(nn.Module):

    def __init__(self, odefunc):
        super(ODEBlock_ANODE, self).__init__()
        if args.double_prec:
            self.odefunc = odefunc.double().to(device)
        else:
            self.odefunc = odefunc.to(device)
        self.options = {}
        self.options.update({'Nt':int(args.Nt)})
        
        if args.method == 'Euler':
            self.options.update({'method':'Euler'})
        elif args.method == 'RK2':
            self.options.update({'method':'RK2'})
        elif args.method == 'RK4':
            #self.options.update({'method':'RK4_alt'})
            self.options.update({'method':'RK4'})
        elif args.method == 'Dopri5' or args.method == 'Dopri5_fixed':
            self.options.update({'method':'Dopri5'})
        print(self.options)

    def forward(self, x):
        out = odesolver(self.odefunc, x.to(tensor_type), self.options)
        return out.to(torch.float32)

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value
        
class ODEBlock_NODE(nn.Module):

    def __init__(self, odefunc):
        super(ODEBlock_NODE, self).__init__()
        if args.double_prec:
            self.odefunc = odefunc.double().to(device)
        else:
            self.odefunc = odefunc.to(device)
        #if args.method == 'Dopri5':
        self.integration_time = torch.tensor(  [0,1] ).float()
        #else:
        #    self.integration_time = torch.tensor(  [1./(float(args.Nt))*i for i in range(args.Nt+1)] ).float()
        self.options = {}
        if not args.method == 'Dopri5':
            self.options.update({'step_size':1./float(args.Nt)})
    

    def forward(self, x):
        
        if args.method == 'Euler':
            Method = 'euler'
            
        elif args.method == 'RK2':
            Method = 'midpoint'
        elif args.method == 'RK4':
            Method = 'rk4'
        elif args.method == 'Dopri5':
            Method = 'dopri5'
        elif args.method == 'Dopri5_fixed':
            Method = 'dopri5_fixed'
            
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x.to(tensor_type), self.integration_time, rtol=args.tol, atol=1E16,method = Method,options=self.options) 
        return out[-1].to(torch.float32)


    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value

class ODEBlock_PETSc(nn.Module):

    def __init__(self, odefunc, input_size, Train):
        super(ODEBlock_PETSc, self).__init__()
        if args.double_prec:
            self.odefunc = odefunc.double().to(device)
        else:
            self.odefunc = odefunc.to(device)
        self.options = {}
        
        self.step_size = 1./float(args.Nt)
        if args.method == 'Euler':
            self.method = 'euler'
            
        elif args.method == 'RK2':
            self.method = 'midpoint'
        elif args.method == 'RK4':
            self.method = 'rk4'
        elif args.method == 'Dopri5':
            self.method = 'dopri5'

        elif args.method == 'Dopri5_fixed':
            self.method = 'dopri5_fixed'
        
        self.ode = petsc_adjoint.ODEPetsc()
        if Train:
            self.ode.setupTS(torch.zeros(args.batch_size,*input_size).to(device,tensor_type), self.odefunc, self.step_size, self.method, enable_adjoint=True)
        else:
            self.ode.setupTS(torch.zeros(args.test_batch_size,*input_size).to(device,tensor_type), self.odefunc, self.step_size, self.method, enable_adjoint=False)
        
        self.integration_time = torch.tensor(  [0,1] ).float()
        
    def forward(self, x):
        
        out = self.ode.odeint_adjoint(x.to(tensor_type), self.integration_time)

        return out[-1].to(torch.float32)

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value 


def conv_init(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1 and m.bias is not None:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif class_name.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)
        

# Data Preprocess
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding = 4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test  = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', transform = transform_train, train = True, download = True)
test_dataset = torchvision.datasets.CIFAR10(root='./data', transform = transform_test, train = False, download = True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, num_workers = 4, shuffle = True, drop_last=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = test_batch_size, num_workers = 4, shuffle = False, drop_last=True)

if args.impl == 'PETSc':
    ODEBlock = ODEBlock_PETSc#(train=True)
    
elif args.impl == 'ANODE':
    ODEBlock = ODEBlock_ANODE
else:
    ODEBlock = ODEBlock_NODE
    

if args.network == 'sqnxt' and not args.impl == 'PETSc':
    net = SqNxt_23_1x(10, ODEBlock)
    net_test = net
elif args.network == 'sqnxt' and args.impl == 'PETSc':
    net = SqNxt_23_1x(10, ODEBlock,Train=True)
    net_test = SqNxt_23_1x(10, ODEBlock,Train=False)
    net_test.load_state_dict(net.state_dict())

    

    #net_test = SqNxt_23_1x(10, ODEBlock_test)
    
    #print(summary(net,torch.zeros((1,3,32,32))))
    #exit()
elif args.network == 'resnet':
    net = ResNet18(ODEBlock)

net.apply(conv_init)
print(net)
if is_use_cuda:
    net.to(device)
    net_test.to(device)
    
criterion = nn.CrossEntropyLoss().to(device)

def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r") as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger

def train(epoch):
    net.train()
    train_loss = 0
    correct    = 0
    total      = 0
    optimizer  = optim.SGD(net.parameters(), lr = lr_schedule(lr, epoch), momentum = 0.9, weight_decay = 5e-4)
    
    print('Training Epoch: #%d, LR: %.4f'%(epoch, lr_schedule(lr, epoch)))
    for idx, (inputs, labels) in enumerate(train_loader):
        if idx == 1:
            exit()
        if is_use_cuda:
            inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        writer.add_scalar('Train/Loss', loss.item(), epoch* 50000 + batch_size * (idx + 1)  )
        train_loss += loss.item()
        _, predict = torch.max(outputs, 1)
        total += labels.size(0)
        correct += predict.eq(labels).cpu().sum().double()
        
        sys.stdout.write('\r')
        sys.stdout.write('[%s] Training Epoch [%d/%d] Iter[%d/%d]\t\tLoss: %.4f Acc@1: %.3f'
                        % (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                           epoch, num_epochs, idx, len(train_dataset) // batch_size, 
                          train_loss / (batch_size * (idx + 1)), correct / total))
        sys.stdout.flush()
    writer.add_scalar('Train/Accuracy', correct / total, epoch )
    logger.info('[%s] Training Epoch [%d/%d] Iter[%d/%d]\t\tLoss: %.4f Acc@1: %.3f'
                        % (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                           epoch, num_epochs, idx, len(train_dataset) // batch_size, 
                          train_loss / (batch_size * (idx + 1)), correct / total) )
        
def test(epoch):
    global best_acc
    net_test.load_state_dict(net.state_dict())
    net_test.eval()
    test_loss = 0
    correct = 0
    total = 0
    for idx, (inputs, labels) in enumerate(test_loader):
        if is_use_cuda:
            inputs, labels = inputs.to(device), labels.to(device)
        outputs = net_test(inputs)
        loss = criterion(outputs, labels)
        
        test_loss  += loss.item()
        _, predict = torch.max(outputs, 1)
        total += labels.size(0)
        correct += predict.eq(labels).cpu().sum().double()
        writer.add_scalar('Test/Loss', loss.item(), epoch* 50000 + test_loader.batch_size * (idx + 1)  )
        
        sys.stdout.write('\r')
        sys.stdout.write('[%s] Testing Epoch [%d/%d] Iter[%d/%d]\t\tLoss: %.4f Acc@1: %.3f'
                        % (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                           epoch, num_epochs, idx, len(test_dataset) // test_loader.batch_size, 
                          test_loss / (100 * (idx + 1)), correct / total))
        sys.stdout.flush()
    writer.add_scalar('Test/Accuracy', correct / total, epoch )
    logger.info('[%s] Testing Epoch [%d/%d] Iter[%d/%d]\t\tLoss: %.4f Acc@1: %.3f'
                        % (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                           epoch, num_epochs, idx, len(test_dataset) // test_loader.batch_size, 
                          test_loss / (100 * (idx + 1)), correct / total) )
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
makedirs(args.save)

logger = get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
logger.info(args)

logger.info(net)
logger.info('NODE: Number of parameters: {}'.format(count_parameters(net)))
    
for _epoch in range(start_epoch, start_epoch + num_epochs):
    
    start_time = time.time()
    train(_epoch)
    print()
    test(_epoch)
    print()
    print()
    end_time   = time.time()
    print('Epoch #%d Cost %ds' % (_epoch, end_time - start_time))
    logger.info('Epoch #%d Cost %ds' % (_epoch, end_time - start_time) )
logger.info( 'Best Acc@1: %.4f' % best_acc * 100)
writer.close()


using PyCall
# using Plots
# https://nextjournal.com/leandromartinez98/tips-to-create-beautiful-publication-quality-plots-in-julia
# https://github.com/boathit/JuliaTorch

torch = pyimport("torch")
optim = pyimport("torch.optim")
nn    = pyimport("torch.nn")

pushfirst!(PyVector(pyimport("sys")."path"), "./")
ocn_atm_coupler = pyimport("ocn_atm_coupler")
 
odeint = ocn_atm_coupler.odeint
device = ocn_atm_coupler.device
get_batch = ocn_atm_coupler.get_batch
ReadCCNS2D_Interface_fromHDF5 = ocn_atm_coupler.ReadCCNS2D_Interface_fromHDF5
ReadCCNS2D_Interface_fromHDF5_ = ocn_atm_coupler.ReadCCNS2D_Interface_fromHDF5_

# 1. Create a model

@pydef mutable struct ODEFunc <: nn.Module
    function __init__(self)
        pybuiltin(:super)(ODEFunc, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(320, 100),
            nn.Tanh(),
            nn.Linear(100, 320),
        )

        for m in self.net
            if pybuiltin(:isinstance)(m, nn.Linear)
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)
            end
        end

    end

    function forward(self, t, y)
      self.net(y)
    end
end
# model = ODEFunc() 
 

# 2. Load state 
ifile = "./coupler_model.pt"
func = ODEFunc() 
func.net.load_state_dict(torch.load(ifile,map_location="cpu"))
print(func.state_dict())
func.eval()

# prediction
batch_y0, batch_t, batch_y = get_batch()  
pred_y = odeint(func, batch_y0, batch_t).to(device)
loss = torch.mean(torch.abs(pred_y - batch_y))

# 3. Test
println("Test...")
func.eval()
num_samples = 10
@pywith torch.no_grad() begin
    loss = 0 
    for i=1:num_samples
        batch_y0, batch_t, batch_y = get_batch()  
        pred_y = odeint(func, batch_y0, batch_t).to(device)
        loss += torch.mean(torch.abs(pred_y - batch_y))
    end
    GC.gc(false)
    println("Averaged loss on the test data: $(loss/num_samples)")
end

# 4. read data from a file
# QQ_Interface has 8 fields
#  0: denstiy
#  1: x-momentum
#  2: y-momentum
#  3: total energy
#  4: x-velocity
#  5: y-velocity
#  6: pressure
#  7: temperature
# ifile="../ml-coupling-datafiles/output-interface/cpde2d-interface-000.h5"
# data = ReadCCNS2D_Interface_fromHDF5_(ifile)
# data2 = ReadCCNS2D_Interface_fromHDF5(ifile) # temperature
print("test")

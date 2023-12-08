#include "../include/mesh.h"
#include "../include/pinn.h"
//- construct computational domain for the PINN instance
mesh2D::mesh2D
(
  Dictionary &meshDict, //mesh parameters
  PinNet &net,
  PinNet &netPrev,
  torch::Device &device, // device info
  thermoPhysical &thermo
):
  net_(net), // pass in current neural net
  netPrev_(netPrev), // pass in other neural net
  dict(meshDict),
  device_(device), // pass in device info
  thermo_(thermo), // pass in thermo class instance
  lbX_(dict.get<float>("lbX")), // read in mesh props from dict
  ubX_(dict.get<float>("ubX")),
  lbY_(dict.get<float>("lbY")),
  ubY_(dict.get<float>("ubY")),
  lbT_(dict.get<float>("lbT")),
  ubT_(dict.get<float>("ubT")),
  deltaX_(dict.get<float>("dx")),
  deltaY_(dict.get<float>("dy")),
  deltaT_(dict.get<float>("dt")),
  xc(dict.get<float>("xc")),
  yc(dict.get<float>("yc"))

{
  TimeStep_ = dict.get<float>("stepSize");
    //- get number of ponits from bounds and step size
  Nx_ = (ubX_ - lbX_)/deltaX_ + 1;
  Ny_ = (ubY_ - lbY_)/deltaY_ + 1;
  Nt_ = (ubT_ - lbT_)/deltaT_ + 1;

  //- total number of points in the entire domain (nDOF)
  Ntotal_ = Nx_*Ny_*Nt_;
  //- populate the individual 1D grids
  xGrid = torch::linspace(lbX_, ubX_, Nx_,device_);
  yGrid = torch::linspace(lbY_, ubY_, Ny_,device_);
  tGrid = torch::linspace(lbT_, ubT_, Nt_,device_);
  //- construct entire mesh domain for transient 2D simulations
  mesh_ = torch::meshgrid({xGrid,yGrid,tGrid});
  //- spatial grid for steady state simulations 
  xyGrid = torch::meshgrid({xGrid,yGrid});
  //- tensor to pass for converged neural net
  xy = torch::stack({xyGrid[0].flatten(),xyGrid[1].flatten()},1);
  xy.set_requires_grad(true);
  //- create boundary grids
  createBC();
}

//- operator overload () to acess main computational domain
torch::Tensor  mesh2D::operator()(int i, int j, int k)  
{
  return torch::stack
  (
    {
      mesh_[0].index({i, j, k}), 
      mesh_[1].index({i, j, k}), 
      mesh_[2].index({i, j, k})
    }
  ); 
}

//- create boundary grids
void mesh2D::createBC()
{
  
  torch::Tensor xLeft = torch::tensor(lbX_,device_);
  torch::Tensor xRight = torch::tensor(ubX_,device_);
  torch::Tensor yBottom = torch::tensor(lbY_, device_);
  torch::Tensor yTop = torch::tensor(ubY_, device_);
  torch::Tensor tInitial = torch::tensor(lbT_,device_);
  if(net_->transient_==1)
  {
    leftWall = torch::meshgrid({xLeft,yGrid,tGrid});
    rightWall = torch::meshgrid({xRight,yGrid,tGrid});
    topWall = torch::meshgrid({xGrid,yTop,tGrid});
    bottomWall = torch::meshgrid({xGrid,yBottom,tGrid});
    initialGrid_ = torch::meshgrid({xGrid,yGrid,tInitial});
  }
  else 
  {
    leftWall = torch::meshgrid({xLeft,yGrid});
    rightWall = torch::meshgrid({xRight,yGrid});
    topWall = torch::meshgrid({xGrid,yTop});
    bottomWall = torch::meshgrid({xGrid,yBottom});
  }
}

void mesh2D::getOutputMesh()
{
	//- update all grids and coarsen by a factor
	//- hard coded for now, will add in dict functionality later
	xGrid = torch::linspace(lbX_, ubX_, Nx_/2,device_);
  yGrid = torch::linspace(lbY_, ubY_, Ny_/2,device_);
  tGrid = torch::linspace(lbT_, ubT_, Nt_/2,device_);
  //- construct entire mesh domain for transient 2D simulations
  mesh_ = torch::meshgrid({xGrid,yGrid,tGrid});
 
}

//- general method to create samples
//- used to create boundary as well as intial state samples
void mesh2D::createSamples
(
  std::vector<torch::Tensor> &grid, 
  torch::Tensor &samples,
  int nSamples
) 
{
  //- vectors to stack
  std::vector<torch::Tensor> vectorStack;
  //- total number of points in the grid
  int ntotal = grid[0].numel();
  //- random indices for PDE loss
  torch::Tensor indices = torch::randperm
  (ntotal,device_).slice(0,0,nSamples);
  
  //- push vectors to vectors stack
  for(int i=0;i<grid.size();i++)
  {
    vectorStack.push_back
    (
      torch::flatten
      (
        grid[i]
      ).index_select(0,indices)
    );
  }
  //- pass stack to get samples
  samples = torch::stack(vectorStack,1);
  //- set gradient =true
  samples.set_requires_grad(true);
}

//- create the total samples required for neural net
//- these samples are the input features to the neural net forward passes
void mesh2D::createTotalSamples
(
  int iter // current iter when looping through the batches
) 
{
  //- generate random indices to generate random samples from grids
  if(iter == 0)
  { 
    //- create Indices in the first iteration itself
    createIndices();
  }
  if(net_->transient_==0)
  {
    //- create samples for intial condition loss only if simulationn is transient
    torch::Tensor batchIndices = torch::slice
    (
      pdeIndices_,
      0,
      iter*net_->BATCHSIZE,
      (iter + 1)*net_->BATCHSIZE,
      1 // step size when slicing
    );
    createSamples(xyGrid,iPDE_,batchIndices);
  }
  else
  {
    torch::Tensor batchIndices = pdeIndices_.slice
    (
      0,
      iter*net_->BATCHSIZE,
      (iter + 1)*net_->BATCHSIZE,
      1 // step size when slicing
    );
    createSamples(mesh_,iPDE_,batchIndices);
  }
  //- create samples only for the first iteration
  if(iter ==0)
  {
    if(net_->transient_ == 1)
    {
      //- update samples for intialGrid
      createSamples(initialGrid_,iIC_,net_->N_IC);
    }
    //- update samples for left wall 
    createSamples(leftWall,iLeftWall_,net_->N_BC);
    //- update samples for right wall 
    createSamples(rightWall, iRightWall_,net_->N_BC);
    //- update samples for top wall 
    createSamples(topWall,iTopWall_,net_->N_BC);
    //- update samples for bottom wall
    createSamples(bottomWall,iBottomWall_,net_->N_BC); 
  }
}

//- forward pass of current batch in batch iteration loop
//- update output features for each batch iteration,
//- pass in the iteration 
// (extremely SHITTY method, but cannot think of a 
//  better one as of now)
void mesh2D::update(int iter)
{ 
  createTotalSamples(iter);
  // std::cout<<"updating solution fields\n";
  //- update all fields
  fieldsPDE_ = net_->forward(iPDE_);
  if(net_->transient_ == 1)
  { 
    fieldsIC_ = net_->forward(iIC_);
  }
  fieldsLeft_ = net_->forward(iLeftWall_);
  fieldsRight_ = net_->forward(iRightWall_);
  fieldsBottom_ = net_->forward(iBottomWall_);
  fieldsTop_ = net_->forward(iTopWall_);
}

//- creates indices tensor for iPDE
void mesh2D::createIndices()
{
  if(net_->transient_==0)
  {
    pdeIndices_ = 
      torch::randperm(xyGrid[0].numel(),device_).slice(0,0,net_->N_EQN,1);
  }
  else
  {
    pdeIndices_ = 
      torch::randperm(mesh_[0].numel(),device_).slice(0,0,net_->N_EQN,1);
    
  }
}

//- createSamples over load to create samples for pde loss as it will buffer
//- passed in batches instead of one go, the other samples being way smaller
//- in size remain unchanged
void mesh2D::createSamples
(
 std::vector<torch::Tensor> &grid,
 torch::Tensor &samples,
 torch::Tensor &indices
)
{
  //- vectors to stack
  std::vector<torch::Tensor> vectorStack;
  //- push vectors to vectors stack
  for(int i=0;i<grid.size();i++)
  {
    vectorStack.push_back
    (
      torch::flatten
      (
        grid[i]
      ).index_select(0,indices)
    );
  }
  //- pass stack to get samples
  samples = torch::stack(vectorStack,1);
  //- set gradient =true
  samples.set_requires_grad(true);
}

void mesh2D::updateMesh()
{
  //- update the lower level of time grid
  lbT_ = lbT_ + TimeStep_;
  ubT_ = ubT_ + TimeStep_;
  //- get new number of time steps in the current time domain
  Nt_ = (ubT_ - lbT_)/deltaT_ + 1;
  //- update tGrid
  tGrid = torch::linspace(lbT_, ubT_, Nt_,device_);
  //- update main mesh
  mesh_ = torch::meshgrid({xGrid,yGrid,tGrid});
  //- update the boundary grids
  createBC();
  //- transfer over parameters of current converged net to 
  //- previous net reference to use as intial condition for 
  //- intial losses
  loadState(net_, netPrev_);
}

//- transfers over learned parameters from one neural net isntance to another,
//- both must have been instantiated and must share the same architecture, they must be clones
//- takes net2 and makes it load the state of net1
//
// Lots of error handling missing, might need to add that, not important right now however
//
void loadState(PinNet& net1, PinNet &net2)
{
  torch::autograd::GradMode::set_enabled(false);
  auto net2_params = net2->named_parameters();
  auto net1_params = net1->named_parameters(true);
  for(auto &param : net1_params)
  {
    auto name = param.key();
    net2_params[name].copy_(param.value());
  }
  torch::autograd::GradMode::set_enabled(true);
} 




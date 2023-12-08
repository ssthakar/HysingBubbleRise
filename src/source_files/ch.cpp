#include "../include/ch.h"
#include "../include/pinn.h"
#include "../include/mesh.h"
#include "../include/derivatives.h"
#include "../include/thermo.h"
//- thermoPhysical properties for mixture
torch::Tensor CahnHillard::thermoProp
(
  float propLiquid, //thermoPhysical prop of liquid  phase 
  float propGas, // thermoPhysical prop of gas phase
  const torch::Tensor &I
)
{
  //- get auxillary phase field var to correct for bounds 
  const torch::Tensor C = CahnHillard::Cbar(I.index({Slice(),3}));
  torch::Tensor mixtureProp = 
    0.5*(1+C)*propLiquid + 0.5*(1-C)*propGas;
  return mixtureProp;
}

//- continuity loss 
torch::Tensor CahnHillard::L_Mass2D
(
  const mesh2D &mesh 
)
{
  const torch::Tensor &u = mesh.fieldsPDE_.index({Slice(),0});
  const torch::Tensor &v = mesh.fieldsPDE_.index({Slice(),1});
  torch::Tensor du_dx = d_d1(u,mesh.iPDE_,0);
  torch::Tensor dv_dy = d_d1(v,mesh.iPDE_,1);
  torch::Tensor loss = du_dx + dv_dy;
  return torch::mse_loss(loss, torch::zeros_like(loss));
}

//- returns the phi term needed 
torch::Tensor CahnHillard::phi
(
  const mesh2D &mesh
)
{
  float &e = mesh.thermo_.epsilon;
  const torch::Tensor &C = mesh.fieldsPDE_.index({Slice(),3});
  torch::Tensor Cxx = d_dn(C,mesh.iPDE_,2,0);
  torch::Tensor Cyy = d_dn(C,mesh.iPDE_,2,1);
  return C*(C*C-1) - e*e*(Cxx + Cyy); 
}

//- returns CahnHillard Loss
torch::Tensor CahnHillard::CahnHillard2D
(
  const mesh2D &mesh
)
{
  const float &e = mesh.thermo_.epsilon;
  const float &Mo = mesh.thermo_.Mo;
  //- u vel
  const torch::Tensor &u = mesh.fieldsPDE_.index({Slice(),0});
  //- v vel
  const torch::Tensor &v = mesh.fieldsPDE_.index({Slice(),1});
  //- phase field var
  const torch::Tensor &C = mesh.fieldsPDE_.index({Slice(),3});
  //- derivatives 
  torch::Tensor dC_dt = d_d1(C,mesh.iPDE_,2);
  torch::Tensor dC_dx = d_d1(C,mesh.iPDE_,0);
  torch::Tensor dC_dy = d_d1(C,mesh.iPDE_,1);
  torch::Tensor phi = CahnHillard::phi(mesh);
  torch::Tensor dphi_dxx = d_dn(phi,mesh.iPDE_,2,0);
  torch::Tensor dphi_dyy = d_dn(phi,mesh.iPDE_,2,1);
  //- loss term
  torch::Tensor loss = dC_dt + u*dC_dx + v*dC_dy - 
    Mo*(dphi_dxx + dphi_dyy);
  return torch::mse_loss(loss,torch::zeros_like(loss));
}

//- returns the surface tension tensor needed in mom equation
torch::Tensor CahnHillard::surfaceTension
(
  const mesh2D &mesh,
  int dim
)
{
  const float &sigma = mesh.thermo_.sigma0;
  const float &e_inv = 1.0/mesh.thermo_.epsilon;
  const torch::Tensor &C = mesh.fieldsPDE_.index({Slice(),3});
  torch::Tensor surf = e_inv*sigma*mesh.thermo_.C*CahnHillard::phi(mesh)
    *d_d1(C,mesh.iPDE_,dim);
  return surf;
} 

//- momentum loss for x direction in 2D 
torch::Tensor CahnHillard::L_MomX2d
(
  const mesh2D &mesh
)
{ 
  float &rhoL = mesh.thermo_.rhoL;
  float &muL = mesh.thermo_.muL;
  float rhoG = mesh.thermo_.rhoG;
  float muG = mesh.thermo_.muG;
  const torch::Tensor &u = mesh.fieldsPDE_.index({Slice(),0});
  const torch::Tensor &v = mesh.fieldsPDE_.index({Slice(),1});
  const torch::Tensor &p = mesh.fieldsPDE_.index({Slice(),2});
  const torch::Tensor &C = mesh.fieldsPDE_.index({Slice(),3});
  //- get density of mixture TODO correct this function to take in just mesh
  torch::Tensor rhoM = CahnHillard::thermoProp(rhoL, rhoG, mesh.fieldsPDE_);
  //- get viscosity of mixture
  torch::Tensor muM = CahnHillard::thermoProp(muL, muG, mesh.fieldsPDE_);
  torch::Tensor du_dt = d_d1(u,mesh.iPDE_,2);
  torch::Tensor du_dx = d_d1(u,mesh.iPDE_,0);
  torch::Tensor du_dy = d_d1(u,mesh.iPDE_,1);
  torch::Tensor dv_dx = d_d1(v,mesh.iPDE_,0);
  torch::Tensor dC_dx = d_d1(C,mesh.iPDE_,0);
  torch::Tensor dC_dy = d_d1(C,mesh.iPDE_,1);
  torch::Tensor dp_dx = d_d1(p,mesh.iPDE_,0);
  //- derivative order first spatial variable later
  torch::Tensor du_dxx = d_dn(u,mesh.iPDE_,2,0);
  torch::Tensor du_dyy = d_dn(u,mesh.iPDE_,2,1);
  //- get x component of the surface tension force
  torch::Tensor fx = CahnHillard::surfaceTension(mesh,0);
  torch::Tensor loss1 = rhoM*(du_dt + u*du_dx + v*du_dy) + dp_dx;
  torch::Tensor loss2 = -0.5*(muL - muG)*dC_dy*(du_dy + dv_dx) - (muL -muG)*dC_dx*du_dx;
  torch::Tensor loss3 = -muM*(du_dxx + du_dyy) - fx;
  //- division by rhoL for normalization, loss starts out very large otherwise
  torch::Tensor loss = (loss1 + loss2 + loss3)/rhoL;
  return torch::mse_loss(loss, torch::zeros_like(loss));
}

//- momentum loss for y direction in 2D
torch::Tensor CahnHillard::L_MomY2d
(
  const mesh2D &mesh
)
{
  float &rhoL = mesh.thermo_.rhoL;
  float &muL = mesh.thermo_.muL;
  float rhoG = mesh.thermo_.rhoG;
  float muG = mesh.thermo_.muG;
  const torch::Tensor &u = mesh.fieldsPDE_.index({Slice(),0});
  const torch::Tensor &v = mesh.fieldsPDE_.index({Slice(),1});
  const torch::Tensor &p = mesh.fieldsPDE_.index({Slice(),2});
  const torch::Tensor &C = mesh.fieldsPDE_.index({Slice(),3});
  //- get density of mixture TODO correct this function to take in just mesh
  torch::Tensor rhoM = CahnHillard::thermoProp(rhoL, rhoG, mesh.fieldsPDE_);
  //- get viscosity of mixture
  torch::Tensor muM = CahnHillard::thermoProp(muL, muG, mesh.fieldsPDE_);
  torch::Tensor dv_dt = d_d1(v,mesh.iPDE_,2);
  torch::Tensor dv_dx = d_d1(v,mesh.iPDE_,0);
  torch::Tensor dv_dy = d_d1(v,mesh.iPDE_,1);
  torch::Tensor du_dx = d_d1(u,mesh.iPDE_,0);
  torch::Tensor dC_dx = d_d1(C,mesh.iPDE_,0);
  torch::Tensor dC_dy = d_d1(C,mesh.iPDE_,1);
  torch::Tensor dp_dy = d_d1(p,mesh.iPDE_,1);
  //- derivative order first spatial variable later
  torch::Tensor dv_dxx = d_dn(v,mesh.iPDE_,2,0);
  torch::Tensor dv_dyy = d_dn(v,mesh.iPDE_,2,1);
  //- get x component of the surface tension force
  torch::Tensor fy = CahnHillard::surfaceTension(mesh,1);
  torch::Tensor gy = torch::full_like(fy,-0.98);
  torch::Tensor loss1 = rhoM*(dv_dt + u*dv_dx + v*dv_dy) + dp_dy;
  torch::Tensor loss2 = -0.5*(muL - muG)*dC_dx*(du_dx + dv_dy) - (muL -muG)*dC_dy*dv_dy;
  torch::Tensor loss3 = -muM*(dv_dxx + dv_dyy) - fy - rhoM*gy;
  torch::Tensor loss = (loss1 + loss2 + loss3)/rhoL;
  return torch::mse_loss(loss, torch::zeros_like(loss));
}

//- get total PDE loss
torch::Tensor CahnHillard::PDEloss(mesh2D &mesh)
{
  //- loss from mass conservation
  torch::Tensor LM = CahnHillard::L_Mass2D(mesh);
  torch::Tensor LMX = CahnHillard::L_MomX2d(mesh);
  torch::Tensor LMY = CahnHillard::L_MomY2d(mesh);
  torch::Tensor LC = CahnHillard::CahnHillard2D(mesh);
  //- return total pde loss
  return LM + LC + LMX + LMY;
}

//- TODO make the function more general by adding in another int for u or v
torch::Tensor CahnHillard::slipWall(torch::Tensor &I, torch::Tensor &X,int dim)
{
  const torch::Tensor &u = I.index({Slice(),0});  
  const torch::Tensor &v = I.index({Slice(),1});
  torch::Tensor dv_dx = d_d1(v,X,dim);
  return torch::mse_loss(dv_dx,torch::zeros_like(dv_dx))
    + torch::mse_loss(u,torch::zeros_like(u));
}

torch::Tensor CahnHillard::noSlipWall(torch::Tensor &I, torch::Tensor &X)
{
  const torch::Tensor &u = I.index({Slice(),0});
  const torch::Tensor &v = I.index({Slice(),1});
  return torch::mse_loss(u,torch::zeros_like(u))  + 
    torch::mse_loss(v,torch::zeros_like(v));
  
}
//- get boundary loss
torch::Tensor CahnHillard::BCloss(mesh2D &mesh)
{
  
  //- get phase field vars at all the boundaries
  torch::Tensor Cleft = mesh.fieldsLeft_.index({Slice(),3});
  torch::Tensor Cright = mesh.fieldsRight_.index({Slice(),3});
  torch::Tensor Ctop = mesh.fieldsTop_.index({Slice(),3});
  torch::Tensor Cbottom = mesh.fieldsBottom_.index({Slice(),3});
  
  //- total boundary loss for u, v and C
  torch::Tensor lossLeft = CahnHillard::slipWall(mesh.fieldsLeft_, mesh.iLeftWall_,0); 
       //+ CahnHillard::zeroGrad(Cleft, mesh.iLeftWall_, 0);
  torch::Tensor lossRight = CahnHillard::slipWall(mesh.fieldsRight_,mesh.iRightWall_, 0);
       //+ CahnHillard::zeroGrad(Cright, mesh.iRightWall_, 0);
  torch::Tensor lossTop = CahnHillard::noSlipWall(mesh.fieldsTop_, mesh.iTopWall_);
       //+ CahnHillard::zeroGrad(Ctop, mesh.iTopWall_, 1);
  torch::Tensor lossBottom = CahnHillard::noSlipWall(mesh.fieldsBottom_, mesh.iBottomWall_);
       //+ CahnHillard::zeroGrad(Cbottom, mesh.iBottomWall_, 1);
  return lossLeft + lossRight + lossTop + lossBottom;
}

//- get the intial loss for the 
torch::Tensor CahnHillard::ICloss(mesh2D &mesh)
{
  //- x vel prediction in current iteration torch::NoGradGuard no_grad;
  const torch::Tensor &u = mesh.fieldsIC_.index({Slice(),0});
  //- y vel prediction in current iteration
  const torch::Tensor &v = mesh.fieldsIC_.index({Slice(),1});
  //- phaseField variable prediction in current iteration
  const torch::Tensor &C = mesh.fieldsIC_.index({Slice(),3});
  //- get all the intial losses
  torch::Tensor uLoss = torch::mse_loss(u,CahnHillard::u_at_InitialTime(mesh));
  torch::Tensor vLoss = torch::mse_loss(v,CahnHillard::v_at_InitialTime(mesh));
  torch::Tensor CLoss = torch::mse_loss(C,CahnHillard::C_at_InitialTime(mesh));
  //- return total loss
  return uLoss +vLoss +CLoss;
}

//- total loss function for the optimizer
torch::Tensor CahnHillard::loss(mesh2D &mesh)
{
  // torch::Tensor pdeloss = CahnHillard::PDEloss(mesh);
  torch::Tensor bcLoss = CahnHillard::BCloss(mesh);
  torch::Tensor pdeLoss = CahnHillard::PDEloss(mesh);
  torch::Tensor icLoss = CahnHillard::ICloss(mesh);
  return bcLoss + pdeLoss + icLoss;
}


//- TODO make radius a variable 
torch::Tensor CahnHillard::C_at_InitialTime(mesh2D &mesh)
{
  if(mesh.lbT_ == 0)
  {
    const float &xc = mesh.xc;
    const float &yc = mesh.yc;
    const float &e = mesh.thermo_.epsilon;
    //- x 
    const torch::Tensor &x = mesh.iIC_.index({Slice(),0});
    //- y
    const torch::Tensor &y = mesh.iIC_.index({Slice(),1});
    //- intial condition
    torch::Tensor Ci =torch::tanh((torch::sqrt(torch::pow(x - xc, 2) + torch::pow(y - yc, 2)) - 0.15)/ (1.41421356237 * e));
    
    return Ci;
  }
  else  
  {
    //- disable gradient tracking for netPrev predictions so optim steps don't update netPrev parameters 
    //  every epoch
    torch::NoGradGuard no_grad;
    //- use previous converged neural net as intial conditions
    torch::Tensor Ci = mesh.netPrev_->forward(mesh.iIC_).index({Slice(),3});
    return Ci;
  }
}
//- intial velocity fields for u and v
torch::Tensor CahnHillard::u_at_InitialTime(mesh2D &mesh)
{
  if(mesh.lbT_ ==0)
  {
    return torch::zeros_like(mesh.iIC_.index({Slice(),0}));
    
  }
  else
  {
    //- disable grad tracking to stop optim.step() from updating netPrev_ parameters
    torch::NoGradGuard no_grad;
    return mesh.netPrev_->forward(mesh.iIC_).index({Slice(),0});
  }
}
//-v at intial time
torch::Tensor CahnHillard::v_at_InitialTime(mesh2D &mesh)
{
  if(mesh.lbT_ ==0)
  {
    return torch::zeros_like(mesh.iIC_.index({Slice(),0}));
  }
  else
  {
    //- disable grad tracking to stop optim from updating netPrev parameters
    torch::NoGradGuard no_grad;
    return mesh.netPrev_->forward(mesh.iIC_).index({Slice(),1});
  }
}


//- auxiliary variable to bound thermophysical properties 
torch::Tensor CahnHillard::Cbar(const torch::Tensor &C)
{
  //- get the absolute value of the phasefield tensor
  torch::Tensor absC = torch::abs(C);
  if(torch::all(absC <=1).item<float>())
  {
    return C;
  }
  else {
    return torch::sign(C);
  }
}

//- zero Grad function for phaseField boundary condtion
torch::Tensor CahnHillard::zeroGrad(torch::Tensor &I, torch::Tensor &X, int dim)
{ 
  torch::Tensor grad = d_d1(I,X,dim);
  return torch::mse_loss(grad, torch::zeros_like(grad));
}



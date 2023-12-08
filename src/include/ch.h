#ifndef ch_h
#define ch_h
#include "pinn.h"
#include "torch/torch.h"
#include "mesh.h"
//- begin CahnHillard namespace
namespace CahnHillard
{
//- returns the surface tension force term for the momentum equation
torch::Tensor surfaceTension
(
  const mesh2D &mesh,
  int dim
);
//- returns phi term used in CH and momentum equation 
torch::Tensor phi
(
  const mesh2D &mesh
);
//- continuity loss in 2D
torch::Tensor L_Mass2D
(
  const mesh2D &mesh // pass reference for gradient input
);
//- momentum loss for x direction in 2D 
torch::Tensor L_MomX2d
(
  const mesh2D &mesh
);
//- momentual loss for y direction in 2D
torch::Tensor L_MomY2d
(
  const mesh2D &mesh
);
//- loss from Cahn-Hillard equation for phase-field transport
torch::Tensor CahnHillard2D
(
  const mesh2D &mesh
);
//- get thermoPhysical properties
torch::Tensor thermoProp
(
  float propLiquid,
  float propGas,
  const torch::Tensor &I
);

torch::Tensor PDEloss(mesh2D &mesh);
torch::Tensor ICloss(mesh2D &mesh);

torch::Tensor slipWall(torch::Tensor &I,torch::Tensor &X, int dim);
torch::Tensor noSlipWall(torch::Tensor &I, torch::Tensor &X);

torch::Tensor BCloss(mesh2D &mesh);
//- total loss function for the net
torch::Tensor loss(mesh2D &mesh);

//- constrains C
torch::Tensor Cbar(const torch::Tensor &C);

//- generates phase field vars at t=0
torch::Tensor C_at_InitialTime(mesh2D &mesh);

torch::Tensor u_at_InitialTime(mesh2D &mesh);

torch::Tensor v_at_InitialTime(mesh2D &mesh);
//- zero gradient at horizontal or vertical wall for now,
//- TOOD make general boundary condition for curved walls based on points
torch::Tensor zeroGrad(torch::Tensor &I,torch::Tensor &X, int dim);
}


#endif // !ch_h

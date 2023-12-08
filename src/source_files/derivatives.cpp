#include "../include/derivatives.h"

using namespace torch::indexing; 

//- first order derivative
torch::Tensor d_d1
(
  const torch::Tensor &I,
  const torch::Tensor &X,
  int spatialIndex
)
{
  torch::Tensor derivative = torch::autograd::grad 
  (
    {I}, // predicted output from net
    {X}, // input features to get the prediction
    {torch::ones_like(I)},
    true, // retain graph, lets us get higher order derivatives
    true, // create graph
    true // allow unused
  )[0].requires_grad_(true);
  return derivative.index({Slice(),spatialIndex});
}

torch::Tensor d_d1
(
  const torch::Tensor &I,
  const torch::Tensor &X
)
{
  torch::Tensor derivative = torch::autograd::grad 
  (
    {I},
    {X},
    {torch::ones_like(I)},
    true,
    true,
    true
  )[0].requires_grad_(true);
  return derivative;
}

//- higher order derivative
torch::Tensor d_dn
(
  const torch::Tensor &I, 
  const torch::Tensor &X, 
  int order, // order of derivative
  int spatialIndex
)
{
  torch::Tensor derivative =  d_d1(I,X,spatialIndex);
  for(int i=0;i<order-1;i++)
  {
    derivative = d_d1(derivative,X,spatialIndex);
  }
  return derivative;
}

//- function overload when X is 1D tensor
torch::Tensor d_dn
(
  const torch::Tensor &I, 
  const torch::Tensor &X, 
  int order // order of derivative
)
{
  torch::Tensor derivative =  d_d1(I,X);
  for(int i=0;i<order-1;i++)
  {
    derivative = d_d1(derivative,X);
  }
  return derivative;
}



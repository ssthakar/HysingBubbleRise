#ifndef derivative_h
#define derivative_h
#include <torch/torch.h>
//- function to calculate higher order derivatives

using namespace torch::indexing; 
torch::Tensor d_dn
(
  const torch::Tensor &I, // dependant scalar var
  const torch::Tensor &X, // independant vars
  int order, // order of derivative
  int spatialIndex // gradient wrt to which independant var? auto generator = torch::cuda::detail::getRandomCUDA()
);

//- function overload when X is a 1D tensor
torch::Tensor d_dn
(
  const torch::Tensor &I, // dependant scalar var
  const torch::Tensor &X, // independant vars
  int order // order of derivative
);

//- first partial derivative 
torch::Tensor d_d1
(
  const torch::Tensor &I, // dependant scalar var
  const torch::Tensor &X, // indpendant vars
  int spatialIndex // gradient wrt to which indpendant var?
);

//- function overload when X is a 1D tensor
torch::Tensor d_d1
(
  const torch::Tensor &I, // dependant scalar var
  const torch::Tensor &X // indpendant vars
);

#endif // !derivative_h

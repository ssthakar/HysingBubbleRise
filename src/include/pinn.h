#ifndef pinn_h
#define pinn_h
#include <torch/torch.h>
#include "utils.h"
using namespace torch::indexing; 
//- Neural network class
class  PinNetImpl:
    public torch::nn::Module
{
    //- private fields
    private:
        //- create and  register sub-modules with the main nn
        void create_layers();
    //- public fields
    public:
        //- parametrized constructor
        PinNetImpl
        (
         const Dictionary &dict
        ); 
    //- public member functions
        // forward propogation with relu activation
        torch::Tensor forward(const torch::Tensor &X);
        //- resets all parameters in the network
        void reset_layers();
    //- public members
        //- Dictionary reference
        const Dictionary& dict;
        // Sequential collection of hidden layers
        torch::nn::Sequential hidden_layers;
        //- input layer
        torch::nn::Linear input = nullptr; 
        //- output layer
        torch::nn::Linear output=nullptr;
        //- number of hidden layers
        //- number of hidden layers
        const int N_HIDDEN_LAYERS;
        //- Dimension of  input layer (no. of input features)
        const int INPUT_DIM;
        //- Dimension of output layer (no. of output features)
        const int OUTPUT_DIM;
        //- Dimension of hidden layers (no. of neurons)
        const int HIDDEN_LAYER_DIM;
        //- grid dimension
        int BATCHSIZE;
        //- maximum number of iterations for optim
        int MAX_STEPS;
        //- tolerance for residual
        float ABS_TOL;
        //- no. of sampling points for PDE loss
        int N_EQN;
        //- no. of sampling points for Boundary condition loss
        int N_BC;
        //- no. of sampling points for intial condition loss
        int N_IC;
        //- max number of epochs during each timestep
        int K_EPOCH;
        //- learning rate
        float L_RATE;
        //- flag for transient
        //- 0 for false else true
        int transient_;
        //- test 
        int test_;
        //- number of iterations in each epoch
        int NITER_;

};
//- create Torch module
TORCH_MODULE(PinNet);

#endif

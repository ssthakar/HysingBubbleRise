#include "../include/pinn.h"
#include "../include/utils.h"
//-------------------PINN definitions----------------------------------------//

//- function to create layers present in the net
void PinNetImpl::create_layers()
{
  //- register input layer 
  input = register_module
  (
    "fc_input",
    torch::nn::Linear(INPUT_DIM,HIDDEN_LAYER_DIM)
  );
  // torch::nn::init::xavier_normal_(input->weight);
  //- register and  hidden layers 
  for(int i=0;i<N_HIDDEN_LAYERS;i++)
  {
    //- hiden layer name
    std::string layer_name = "fc_hidden" + std::to_string(i);
    
    //- create and register each hidden layer
    torch::nn::Linear linear_layer = register_module
    (
      layer_name,
      torch::nn::Linear(HIDDEN_LAYER_DIM,HIDDEN_LAYER_DIM)
    );

    //- intialize network parameters
    torch::nn::init::xavier_normal_(linear_layer->weight);
    
    //- populate sequential with layers
    hidden_layers->push_back(linear_layer);
    
    //- batch normalization layers 
    std::string batchNormName = "fc_batchNorm" + std::to_string(i);
    torch::nn::BatchNorm1d batchNormLayer = register_module
    (
      batchNormName,
      torch::nn::BatchNorm1d(HIDDEN_LAYER_DIM)
    );
    
    //- push back batch-normalization layer
    hidden_layers->push_back(batchNormLayer);

    //- create and register activation functions 
    hidden_layers->push_back
    (
      register_module
      (
        "fc_silu_hidden" + std::to_string(i), 
        torch::nn::SiLU() // swish function X * RELU(X)
      )
    );
  }

  //- register output layer
  output = register_module
  (
    "fc_output",
    torch::nn::Linear(HIDDEN_LAYER_DIM,OUTPUT_DIM)
  );
}

//- resets all the layers in the neural net class instance
//  without needing to reconstruct the net class instance
void PinNetImpl::reset_layers()
{
  //- reset the parameters for the input and output layers
  input->reset_parameters();
  output->reset_parameters();
  //- loop through all the layers in sequential
  for(int i=0;i<hidden_layers->size();i++)
  {
    //- check if the layer being iterated is a linear layer or not
    if (auto linear_layer = dynamic_cast<torch::nn::LinearImpl*>(hidden_layers[i].get()))
    {
      hidden_layers[i]->as<torch::nn::Linear>()->reset_parameters();
    }
  }
}


//- constructor for PinNet module implementation
PinNetImpl::PinNetImpl
(
  const Dictionary &netDict // reference to Dictionary object
)
: 
  dict(netDict), //pass in Dictionary  
  INPUT_DIM(dict.get<int>("inputDim")), // no. of input features  
  HIDDEN_LAYER_DIM(dict.get<int>("hiddenLayerDim")), // no. of neurons in HL
  N_HIDDEN_LAYERS(dict.get<int>("nHiddenLayer")), // no. of hidden layers
  OUTPUT_DIM(dict.get<int>("outputDim")) //- no. of output features
{
  //- set parameters from Dictionary lookup
  N_EQN = dict.get<int>("NEQN");
  N_BC = dict.get<int>("NBC");
  N_IC = dict.get<int>("NIC");
  //- flag for transient or steady state mode
  transient_ = dict.get<int>("transient");
  //- get target loss from dict
  ABS_TOL = dict.get<float>("ABSTOL");
  K_EPOCH = dict.get<int>("KEPOCH");
  //- batch size for pde loss input
  BATCHSIZE=dict.get<int>("BATCHSIZE");
  //- number of iterations in one epoch 
  NITER_ = N_EQN/BATCHSIZE;
  //- create and intialize the layers in the net
  create_layers();
}

//- forward propagation 
torch::Tensor PinNetImpl::forward
(
 const torch::Tensor& X
)
{
  torch::Tensor I = torch::silu(input(X));
  I = hidden_layers->forward(I);
  I = output(I);
  return I;
}

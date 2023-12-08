#ifndef mesh_h
#define mesh_h
#include <torch/torch.h>
#include "pinn.h"
#include "utils.h"
#include "thermo.h"
using namespace torch::indexing; 
//- class to store in computational domain and solution fields
class mesh2D :
  public torch::nn::Module
  {
  //- public fields
  public:
    //- Dictionary reference
    const Dictionary &dict;
    //- current neural net instance
    PinNet &net_;
    //- previous neural net instance
    PinNet &netPrev_;
    //- lower bounds for x 
    const float lbX_;
    //- upper bounds for xGrid
    const float ubX_;
    //- deltaX
    const float deltaX_;
    //- lower bounds for y 
    const float lbY_;
    //- upper bounds for y 
    const float ubY_;
    //- step size for y
    const float deltaY_;
    //- lower bound for time  
    float lbT_;
    //- upper bound for time
    float ubT_;
    //- step size for t
    const float deltaT_;
    //- center of the bubble at time t=0
    const float xc;
    const float yc;
    //- complete computational domain (x X y X t)
    std::vector<torch::Tensor> mesh_;
    //- x grid (1D tensor)
    torch::Tensor xGrid;
    //- y grid
    torch::Tensor yGrid;
    //- t grid
    torch::Tensor tGrid;
    //- spatial grid
    std::vector<torch::Tensor> xyGrid;
    //-xy spatial grid to use for plotting 
    torch::Tensor xy;
    //- left wall grid
    std::vector<torch::Tensor> leftWall;
    torch::Tensor iLeftWall_;
    torch::Tensor il;
    torch::Tensor leftIndices_;
    //- right wall 
    std::vector<torch::Tensor> rightWall;
    torch::Tensor iRightWall_;
    torch::Tensor ir;
    torch::Tensor rightIndices_;
    //- top wall
    std::vector<torch::Tensor> topWall;
    torch::Tensor iTopWall_;
    torch::Tensor it;
    torch::Tensor topIndices_;
    //- bottom walla
    std::vector<torch::Tensor> bottomWall;
    torch::Tensor iBottomWall_;
    torch::Tensor ib;
    torch::Tensor bottomIndices_;
    //- reference to device
    torch::Device &device_;
    //- reference to thermoPhyiscal class
    thermoPhysical &thermo_;
    //- tensor to store in solution fields
    torch::Tensor fieldsPDE_;
    torch::Tensor fieldsIC_;
    torch::Tensor fieldsLeft_;
    torch::Tensor fieldsRight_;
    torch::Tensor fieldsTop_;
    torch::Tensor fieldsBottom_;
    //- sampling points for PDE loss
    torch::Tensor iPDE_;
    torch::Tensor pdeIndices_;
    //-sampling points for IC loss
    std::vector<torch::Tensor> initialGrid_;
    torch::Tensor iIC_;
    torch::Tensor icIndices_;
    //- number of points in x direction 
    int Nx_;
    //- number of points in y direction
    int Ny_;
    //- number of intervals in time 
    int Nt_;
    //- number of points in the spatial domain
    int Nxy_;
    //- total number of points
    int Ntotal_;
    //- time step for adaptive time marching 
    float TimeStep_;
    //- constructor
    mesh2D
    (
      Dictionary &meshDict,
      PinNet &net,
      PinNet &netPrev,
      torch::Device &device,
      thermoPhysical &thermo
    );
    //- operator overload to use index notation for access
    torch::Tensor operator()(int i,int j,int k);
    //- create boundary grids
    void createBC();
    //- after sub-net converges, upadate solution fields
    void update(int iter);
    //- general function to create samples for neural net input
    void createSamples 
    (
      std::vector<torch::Tensor> & grid, // grid to generate samples from
      torch::Tensor &samples, // reference to input feature tensor
      int nSamples // total number of samples to extract from grid
    );
    void createSamples
    (
      std::vector<torch::Tensor> &grid,
      torch::Tensor &samples,
      torch::Tensor &indices
    );
    void createIndices();
    //-  creates total samples 
    void createTotalSamples
    (
      int iter //index for batch for pde
    );
    //- update time parameters for next time interval
    void updateMesh();
    void getOutputMesh();
};

//- function to clone net, transfer learned parameters from 
//  net1 to net2
//  net2 to needs to identical architecture and must be constructed 
//  prior to this function call
void loadState(PinNet& net1, PinNet &net2);


#endif

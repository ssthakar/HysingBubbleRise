#ifndef thermo_h
#define thermo_h
#include "utils.h"
class thermoPhysical
{
  public:
    //- constructor reads in properties from file
    thermoPhysical(Dictionary &dict);
    //- mobility
    float Mo;
    //- interface thickness
    float epsilon;
    //- surface tension coeff
    float sigma0;
    //- viscosity of liquid
    float muL;
    //- viscosity of gas
    float muG;
    //- density of liquid
    float rhoL;
    //- density of gas
    float rhoG;
    //- constant value for 3*sqrt(2)/4
    float C;

};
#endif

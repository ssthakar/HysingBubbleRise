#include "../include/thermo.h"

//---thermophysical class definition
thermoPhysical::thermoPhysical(Dictionary &dict)
{
  Mo = dict.get<float>("Mo");
  epsilon = dict.get<float>("epsilon");
  sigma0 = dict.get<float>("sigma0");
  muL = dict.get<float>("muL");
  muG = dict.get<float>("muG");
  rhoL = dict.get<float>("rhoL");
  rhoG = dict.get<float>("rhoG");
  C = 1.06066017178;
}


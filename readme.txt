The specification is quite simple. This repository only used torch 1.13.1 and numpy 1.24.1, and other versions will work too as long as the packages are compatible with each other. 


Since we randomly initialize the graph neural networks involved in the experiments (to validate our theorems which allow for an arbitrary such network), there is no training/evaluation. Test data consists of the graphs that are fed to these graph neural networks, which are generated according to the Erdos Renyi model in the experiments that validate our proved theorems.

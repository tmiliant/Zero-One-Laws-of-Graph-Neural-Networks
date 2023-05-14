Additional experiments (i.e. not the main ones)

Architecture	   Prefixes of file names meaning
 
(GCN) 	    	   logprob: r decreases as a function of n, namely log(n) / n
(GCN)       	   normal1: initial features generated with normal mean 0.5, standard deviation 1
(GCN)       	   normal5: initial features generated with normal mean 0.5, standard deviation 5
(GCN)       	   pa: graph generated using preferential attachment model 
(GCN)       	   relu: relu activation function for GNN layers
(GCN)              sigmoid: sigmoid activation function for GNN layers
(BaseGNN(SUM, GR)) sum_gr: threshold (-1,1) for activation function rather than (-50,50) 
(BaseGNN(SUM)) 	   sum: threshold (-1,1) for activation function rather than (-50,50)
(GCN)	   	   v: virtual node
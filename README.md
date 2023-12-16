# barrow_cs4775_final

Please see the writeup for details about the implementation of the program.  

The file main.py contains all the original work on this project.  Each method within main.py has an annotated description of its behavior.  To run a simulation, set the three fields in main(): Q_iter, t_depth, num_trees.  These fields tell the simulation how many times to refine the Q matrix, how deep of trees to simulate, and how many trees to simulate, respectively.  If you wish for the program to save .png files of the simulated and reconstructed trees, change the third field in the get_all_stats() call to True.  Note that it will overwrite the .png files for each tree it simulates and reconstructs.  

The output data from the simulation/reconstruction is appended to a text file.  The columns of the text file have the following values, in order:
        depth: depth of tree
        mu: migration rate
        LL: log likelihood of refined tree
        scorr_MLE_Q_knn: spearman correlation of the initial Q
        scorr_MLE_Q_best: spearman correlation of the highest LL Q
        scorr_fitcher: the spearman correlation coefficient for FitchCount  (from Quinn et. al.)
        scorr_naive: scorr for Fitch_naive model  (from Quinn et. al.)
        scorr_mv: scorr for majority vote model  (from Quinn et. al.)

Most other parameters described in the writeup are manipulable, but they will need to be changed within the code.

The folders beyond main.py are packages that have been imported locally.  The folders Analysis, TreeSolver, utilities, and ProcessingPipeline contain an outdated implementation of the Cassiopeia program.  The published code in Quinn et. al. 2021 does not run with the current version of Cassiopeia so these older versions are needed.  Few of the classes and methods are used in this work, mostly it is used for the simulation of single-cell lineage data and the analysis of FitchCount.  

Some of the imported code has been copied into main() for ease of utility.  All of this copied code is contained in the simulate_tree_get_stats() method.  This method is entirely pulled from the old Cassiopeia code aside from the last block of code which copies certain fields of data and returns it.  

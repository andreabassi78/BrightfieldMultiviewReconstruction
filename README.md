Example code for Brightfield Multiview Reconstruction as described in:

Calisesi, G., et al.,
Three-dimensional bright-field microscopy with isotropic resolution based on multi-view acquisition and image fusion reconstruction.
Sci Rep 10, 12771 (2020). https://doi.org/10.1038/s41598-020-69730-4


The code reconstructs a single section (y) of a sample acquired in brightfield mode from multiple views.

The data should be ordered (resliced after multi-view acquisition) in the following order:
                                [angles, z, x]
                                
Please read Calisesi et al. for details on acquisition and on the coordinates definition.  

In order to reconstruct a full 3D sample, the code should be run section by section 
or could be modified to take into account 3D multiview stacks. Please contact
the corresponding author if you need help in performing this step. 

Required dependencies are included in Anaconda, except for tqdm that can be installed with pip.
# State-specific Individualized Network Parcellations



These set of functions are used to generate the anlaysis in Salehi et al. [1]. This paper uses data from 9 functional states (2 rest and 7 task states) in the Human Connectome Project (HCP) S900 release [2]. This repository includes MATLAB functions to generate state-specific individualized functional networks.

The main function in this set is **"parcellation_rest.m"**, which generates the individualzied functional networks from the node-level (i.e., parcellated by grouping voxels into nodes) fMRI data. You can execute this function via the following:

<p align="center">
	<img src ="images/Github_NIMG.png" height="347" />
</p>
\input{README.md}

For further questions please raise an issue [here](https://github.com/YaleMRRC/Network-Parcellation-Rest/issues).


### References

[1] Salehi, M., Karbasi, A., Shen, X., Scheinost, D., & Constable, R. T. (2018). An exemplar-based approach to individualized parcellation reveals the need for sex specific functional networks. NeuroImage, 170, 54-67.

[2] Van Essen, D. C., Smith, S. M., Barch, D. M., Behrens, T. E., Yacoub, E., Ugurbil, K., & Wu-Minn HCP Consortium. (2013). The WU-Minn human connectome project: an overview. Neuroimage, 80, 62-79.

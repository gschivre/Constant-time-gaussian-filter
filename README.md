# Constant time gaussian filter

This is an implementation of the constant time gaussian filter from Sugimoto et al. (https://doi.org/10.3169/mta.3.12 ; https://doi.org/10.1109/ICIP.2013.6738106) in julia.
It can be seen from the benchmark script that this code is actualy much faster than the imfilter from `ImageFiltering.jl` when the filter standard deviation is greater than 10.

In (https://doi.org/10.3169/mta.3.12) Sugimoto et al. also describe fast algorithm for first and second order derivative, I implement them in the function `Fastdgauss` and `Fastd2gauss` respectively. However even if this function works they need futher improvement for performance. Indeed for the moment they only work on matrix of float, so images must be converted first. Check the benchmark file for hint on how use then anyway.

This gaussian filter can be use to derive a constant time bilateral filter (compressive bilateral filter), implemented in the CBLF function as describe in another paper from Sugimoto et al. (https://doi.org/10.1109/TIP.2015.2442916).

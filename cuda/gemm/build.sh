nvcc ${1} -std=c++11 -arch sm_${2} -lineinfo --ptxas-options=-v -lcublas


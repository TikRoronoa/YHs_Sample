nvcc -O3 ${1} -std=c++11 -arch sm_${2} -lineinfo --ptxas-options=-v -lcublas -o ${3:-${1%.cu}}



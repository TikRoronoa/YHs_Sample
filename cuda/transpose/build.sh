nvcc -O3 ${1:-transpose.cu} -std=c++14 -arch sm_${2:-86} -lineinfo --ptxas-options=-v -o ${3:-transpose}

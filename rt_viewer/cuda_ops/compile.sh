/usr/local/cuda/bin/nvcc -std=c++11 cuda_ops.cu -o cuda_ops.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++11 cuda_ops.cc cuda_ops.cu.o -o screenproc.so -shared -fPIC -I/usr/include/python3.5/ -I/usr/local/cuda/include -lboost_python-py35 -lcudart -L /usr/local/cuda/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0

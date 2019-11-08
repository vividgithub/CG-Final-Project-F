#/bin/bash
CUDA_PATH=/usr/local/cuda
${CUDA_PATH}/bin/nvcc tf_sampling_g.cu -o tf_sampling_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')  )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')  )
echo $TF_CFLAGS
echo $TF_LFLAGS
g++ -std=c++11 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC -I $CUDA_PATH/include -lcudart -L $CUDA_PATH/lib64/ ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2

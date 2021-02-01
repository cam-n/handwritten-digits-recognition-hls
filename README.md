# Hardware-for-Handwritten-Digits-Recognition

Model:       Lenet-5.
Ref:         http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

Application: Handwritten Digits Recognition.

Phase:       Train and Inference.

Dataset:     MNIST (http://yann.lecun.com/exdb/mnist/)

This appication is implemented on both of Software and Hardware.

The training phase is implemented in Python to generate parameters.
Then, the parameters are fed to the inference phase on hardware system.

Tools:
1. Anaconda3 (Spyder/ Python3)        : To train + Inference.
2. Visual Studio Code (GCC compiler)  : To Build C/HLS program.
3. Vivado HLS                         : To generate C/HLS program to RTL.
4. Vavado                             : To build hardware for inference phace.

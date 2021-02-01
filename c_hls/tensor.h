/******************************************************************/     
/* Author: V.Cam Nguyen
/* nguyen.van_cam.no6@is.naist.jp
/* Date: Dec 3rd, 2020
/* File: tensor.h  
/* Define model's parameters.
/* 
/* Model:       Lenet-5.
/* Application: Handwritten Digits Recognition.
/* Phase:       Inference.
/* Dataset:     MNIST (http://yann.lecun.com/exdb/mnist/)
/* Ref:         http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
/*
/******************************************************************/


#ifndef TENSOR_H_
#define TENSOR_H_

#define W_CONV1_LEN  6*1*1
#define W_CONV2_LEN  16*6*5*5
#define W_FC1_LEN    400*120
#define W_FC2_LEN    120*84
#define W_FC3_LEN    84*10

#define B_CONV1_LEN  6
#define B_CONV2_LEN  16
#define B_FC1_LEN    120
#define B_FC2_LEN    84
#define B_FC3_LEN    10

#define DATASET_LEN  10000*28*28
#define LABEL_LEN    10000
#define N_CLASS      10

//void Inference_Phase(unsigned char, float, float,float, float, float, float, float, float, float, float, float);

typedef struct _float5D {
  //int    n_conv1, n_conv2, n_fc1, n_fc2, n_fc3;   
  float *conv1, *conv2, *fc1, *fc2, *fc3;
} float5D;


#endif  // TENSOR_H_


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "tensor.h"

void conv1(float in[28][28], float kernel[6][1][1], float bias[6], float out[6][28][28]){
 int channel, row, col; /* OUTPUT  */
 int i,j; /* KERNEL */
 for(channel=0;channel<6;channel++){
   for(row=0;row<28;row++){
     for(col=0;col<28;col++){
       for(i=0;i<1;i++){
         for(j=0;j<1;j++){
           if (i==0 && j==0) out[channel][row][col]  = in[row+i][col+j] * kernel[channel][i][j] + bias[channel];
           else              out[channel][row][col] += in[row+i][col+j] * kernel[channel][i][j];
         }
       }
     }
   }
 }
}

void relu1(float in[6][28][28], float out[6][28][28]){
   int i, j,k;
   for(k=0;k<6;k++){
       for(i=0;i<28;i++){
           for(j=0;j<28;j++){
               out[k][i][j] = (in[k][i][j] < 0.0f)? 0.0f: in[k][i][j];
           }
       }
   }
}

void avgpooling1(float in[6][28][28], float out[6][14][14]){
   int n_channel, i, j;
   for(n_channel=0;n_channel<6;n_channel++){
       for(i=0;i<28;i+=2){
           for(j=0;j<28;j+=2){
               out[n_channel][i/2][j/2] = (in[n_channel][i][j] + in[n_channel][i+1][j] + in[n_channel][i][j+1] + in[n_channel][i+1][j+1])/ (4.0f);
           }
       }
   }
}

void conv2(float in[6][14][14], float kernel[16][6][5][5], float bias[16], float out[16][10][10]){
 int channel, row, col; /* OUTPUT */
 int i,j,k; /* KERNEL */
 for(channel=0;channel<16;channel++){
   for(row=0;row<10;row++){
     for(col=0;col<10;col++){
       for(k=0;k<6;k++){
         for(i=0;i<5;i++){
           for(j=0;j<5;j++){
               if (k==0 && i==0 && j==0) out[channel][row][col]  = in[k][row+i][col+j] * kernel[channel][k][i][j] + bias[channel];
               else                      out[channel][row][col] += in[k][row+i][col+j] * kernel[channel][k][i][j];
           }
         }
       }
     }
   }
 }
}

void relu2(float in[16][10][10], float out[16][10][10]){
   int i, j,k;
   for(k=0;k<16;k++){
       for(i=0;i<10;i++){
           for(j=0;j<10;j++){
               out[k][i][j] = (in[k][i][j] < 0.0f)? 0.0f: in[k][i][j];
           }
       }
   }
}

void avgpooling2(float in[16][10][10], float out[16][5][5]){
   int n_channel, i, j;
   for(n_channel=0;n_channel<16;n_channel++){
       for(i=0;i<10;i+=2){
           for(j=0;j<10;j+=2){
               out[n_channel][i/2][j/2] = (in[n_channel][i][j] + in[n_channel][i+1][j] + in[n_channel][i][j+1] + in[n_channel][i+1][j+1]) / (4.0f);
           }
       }
   }
}

void flatten(float in[16][5][5], float out[16*5*5]){
   int i,j,k;
   int index = 0;
   for(k=0;k<16;k++){
       for(i=0;i<5;i++){
           for(j=0;j<5;j++){
               out[index] = in[k][i][j];
               index++;
           }
       }
   }
}


void fc1(float in[400], float weights[120][400], float bias[120], float out[120]){
   int i,j;
   for(i=0;i<120;i++){
       for(j=0;j<400;j++){
           if(j==0)    out[i]  = (weights[i][j] * in[j])  + bias[i];
           else        out[i] +=  weights[i][j] * in[j];
       }
   }
}

void relu3(float in[120], float out[120]){
   int i;
   for(i=0;i<120;i++){
       out[i] = (in[i] < 0.0f)? 0.0f: in[i];
   }
}

void fc2(float in[120], float weights[84][120], float bias[84], float out[84]){
   int i,j;
   for(i=0;i<84;i++){
       for(j=0;j<120;j++){
           if(j==0)    out[i]  =  (weights[i][j] * in[j]) + bias[i];
           else        out[i] +=   weights[i][j] * in[j];
       }
   }
}

void relu4(float in[84], float out[84]){
   int i;
   for(i=0;i<84;i++){
       out[i] = (in[i] < 0.0f)? 0.0f: in[i];
   }
}

void fc3(float in[84], float weights[10][84], float bias[10], float out[10]){
   int i,j;
   for(i=0;i<10;i++){
       for(j=0;j<84;j++){
           if(j==0)    out[i]  = (weights[i][j] * in[j]) + bias[i];
           else        out[i] +=  weights[i][j] * in[j];
       }
   }
}

void softmax(float in[10], float out[10]){
   int i;
   float sum = 0;
   for(i=0;i<10;i++)
       sum += exp(in[i]);

   for(i=0;i<10;i++){
       out[i] = fabs(exp(in[i]) / (sum * 1.0f));
   }
}


void Prediction(     float image[28][28],
                    float w_conv1[6][1][1],
                    float w_conv2[16][6][5][5],
                    float w_fc1[120][400],
                    float w_fc2[84][120],
                    float w_fc3[10][84],
                    float b_conv1[6],
                    float b_conv2[16],
                    float b_fc1[120],
                    float b_fc2[84],
                    float b_fc3[10],
                    float probs[10]){

   //int i,j;

   /* The input image is re-arranged from 1D to 2D */
   // float image[28][28];
   // for(i=0;i<28;i++)
   //     for(j=0;j<28;j++)
   //         image[i][j] = *(float*)&datain[28*i + j];

   /* Lenet-5 */


   float o_conv1[6][28][28],  o_relu1[6][28][28],  o_avgpooling1[6][14][14];
   float o_conv2[16][10][10], o_relu2[16][10][10], o_avgpooling2[16][5][5];
   float o_flatten[400];
   float o_fc1[120], o_relu3[120];
   float o_fc2[84],  o_relu4[84];
   float o_fc3[10];

   conv1(image, w_conv1, b_conv1, o_conv1);
   relu1(o_conv1, o_relu1);
   avgpooling1(o_relu1, o_avgpooling1);

   conv2(o_avgpooling1, w_conv2, b_conv2, o_conv2);
   relu2(o_conv2, o_relu2);
   avgpooling2(o_relu2, o_avgpooling2);

   flatten(o_avgpooling2, o_flatten);

   fc1(o_flatten, w_fc1, b_fc1, o_fc1);
   relu3(o_fc1, o_relu3);

   fc2(o_relu3, w_fc2, b_fc2, o_fc2);
   relu4(o_fc2, o_relu4);
   fc3(o_relu4, w_fc3, b_fc3, o_fc3);
   softmax(o_fc3, probs);

}



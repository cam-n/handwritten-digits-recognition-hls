
#include <stdio.h>
#include <stdlib.h>
#include "tensor.h"

int main(int argc, char** argv){

   //float image[28][28];
   float w_conv1[6][1][1];
   float w_conv2[16][6][5][5];
   float w_fc1[120][400];
   float w_fc2[84][120];
   float w_fc3[10][84];
   float b_conv1[6];
   float b_conv2[16];
   float b_fc1[120];
   float b_fc2[84];
   float b_fc3[10];
   float probs[10];

   int i,j,m,n,index;
   FILE *fp;

    /* Load Weights from DDR->LMM */
   fp = fopen("../data/weights/w_conv1.txt", "r");
   for(i=0;i<6;i++)
       fscanf(fp, "%f ",  &(w_conv1[i][0][0]));  fclose(fp);

   fp = fopen("../data/weights/w_conv2.txt", "r");
   for(i=0;i<16;i++){
       for(j=0;j<6;j++){
           for(m=0;m<5;m++){
               for(n=0;n<5;n++){
                   index = 16*i + 6*j + 5*m + 5*n;
                   fscanf(fp, "%f ",  &(w_conv2[i][j][m][n]));
               }
           }
       }
   }
   fclose(fp);

   fp = fopen("../data/weights/w_fc1.txt", "r");
   for(i=0;i<120;i++){
       for(j=0;j<400;j++)
           fscanf(fp, "%f ",  &(w_fc1[i][j]));
   }
   fclose(fp);

   fp = fopen("../data/weights/w_fc2.txt", "r");
   for(i=0;i<84;i++){
       for(j=0;j<120;j++)
           fscanf(fp, "%f ",  &(w_fc2[i][j]));
   }
   fclose(fp);

   fp = fopen("../data/weights/w_fc3.txt", "r");
   for(i=0;i<10;i++){
       for(j=0;j<84;j++)
           fscanf(fp, "%f ",  &(w_fc3[i][j]));
   }
   fclose(fp);

   fp = fopen("../data/weights/b_conv1.txt", "r");
   for(i=0;i<6;i++)
       fscanf(fp, "%f ",  &(b_conv1[i]));  fclose(fp);

   fp = fopen("../data/weights/b_conv2.txt", "r");
   for(i=0;i<16;i++)
       fscanf(fp, "%f ",  &(b_conv2[i]));  fclose(fp);

   fp = fopen("../data/weights/b_fc1.txt", "r");
   for(i=0;i<120;i++)
       fscanf(fp, "%f ",  &(b_fc1[i]));  fclose(fp);

   fp = fopen("../data/weights/b_fc2.txt", "r");
   for(i=0;i<84;i++)
       fscanf(fp, "%f ",  &(b_fc2[i]));  fclose(fp);

   fp = fopen("../data/weights/b_fc3.txt", "r");
   for(i=0;i<10;i++)
       fscanf(fp, "%f ",  &(b_fc3[i]));  fclose(fp);

   float *dataset = (float*)malloc(LABEL_LEN*28*28 *sizeof(float));
   int target[LABEL_LEN];

   fp = fopen("../data/MNIST/mnist-test-target.txt", "r");
   for(i=0;i<LABEL_LEN;i++)
       fscanf(fp, "%d ",  &(target[i]));  fclose(fp);

   fp = fopen("../data/MNIST/mnist-test-image.txt", "r");
   for(i=0;i<LABEL_LEN*28*28;i++)
       fscanf(fp, "%f ",  &(dataset[i]));  fclose(fp);

   float image[28][28];
   float *datain;
   int acc = 0;
   int mm, nn;
   for(i=0;i<LABEL_LEN;i++) {

       datain = &dataset[i*28*28];
       for(mm=0;mm<28;mm++)
           for(nn=0;nn<28;nn++)
               image[mm][nn] = *(float*)&datain[28*mm + nn];

       Prediction(   image,
                     w_conv1,
                     w_conv2,
                     w_fc1,
                     w_fc2,
                     w_fc3,
                     b_conv1,
                     b_conv2,
                     b_fc1,
                     b_fc2,
                     b_fc3,
                     probs
                     );

       int index = 0;
       float max = probs[0];
       for (j=1;j<10;j++) {
            if (probs[j] > max) {
                index = j;
                max = probs[j];
            }
       }

       if (index == target[i]) acc++;
       printf("Predicted label: %d\n", index);
       printf("Prediction: %d/%d\n", acc, i+1);
   }
   printf("Accuracy = %f\n", acc*1.0f/LABEL_LEN);

    return 0;
}



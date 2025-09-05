#include <random>
#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstring>
#include <sys/mman.h>
using namespace std;

#define NNpath "../neuralData/NNbinaries"
#define dataPath "../trainingData/dataBinaries"
#define testdPath "../trainingData/TestdataBinaries"

#define ReLUalpha 0.1
#define FLT_MAX (1.0/0.0)
#define learningRate -0.0005
#define epsilon 0.00000001
#define beta1 0.9
#define beta2 0.99
#define maxIter 35
#define L2alpha 0.0005

extern "C" void* readF(const char* filename, int elemSize, long* nElWriteB);
extern "C" void writeF(void* d1, int size, int elSize, const char* filepath);

__global__ void VecCMult(float* A, float* B, float c, int size){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i >= size){
        return;
    }
    B[i] = A[i]*c;
}

__global__ void VecAddMult(float* A, float* B, float t,int size){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i >= size){
        return;
    }
    B[i] = (A[i]*t)+B[i];
}

__global__ void ConvolutionFusedAddAct(float* A, float* B, float* C, int ftMaps, int inputChannels, int wSize, int iSize){
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int x = bx*blockDim.x + tx;
    int ty = threadIdx.y;
    int by = blockIdx.y;
    int y = by*blockDim.y + ty;
    int z = blockIdx.z*blockDim.z + threadIdx.z;

    if(x+(wSize-1) >= iSize || y+(wSize-1) >= iSize || z >= ftMaps){
        return;
    }

    int tileSize = blockDim.x;


    float* localW = B + (1+wSize*wSize*inputChannels)*z;

    //float writeBV = *(localW+wSize*wSize*iSize*iSize);
    float writeBV = *(localW+wSize*wSize*inputChannels);

    //__shared__ float W[1024];

    for(int z0 = 0; z0 < inputChannels; z0++){
        for(int i = 0; i < wSize; i++){
            for(int k = 0; k < wSize; k++){
                writeBV += localW[(k) + (i)*wSize]*A[(y+k)+iSize*(x+i)];
            }
        }
        localW += (wSize*wSize);
        A += iSize*iSize;
    }

    /*
    for(int i = 0; i < wSize; i += tileSize){
        for(int k = 0; k < wSize; k += tileSize){
            if(tx < tileSize && tx+i < wSize && ty < tileSize && ty+k < wSize){
                W[ty + tx*tileSize] = localW[(ty+k) + (tx+i)*wSize];
            }
            __syncthreads();
            for(int i2 = 0; i2 < tileSize && i2+i < wSize; i2++){
                for(int k2 = 0; k2 < tileSize && k2+k < wSize; k2++){
                    //writeBV += W[k2 + i2*tileSize]*A[(y+k2+k)+iSize*((x+i2+i) + iSize*z)];
                    writeBV += localW[(k2+k) + (i2+i)*wSize]*A[(y+k2+k)+iSize*((x+i2+i) + iSize*z)];
                }
            }
            __syncthreads();
        }
    }
    */
    if(writeBV < 0){
        writeBV *= ReLUalpha;
    }

    C[y + ((iSize-wSize) + 1)*(x + ((iSize-wSize) + 1)*z)] = writeBV;
}

__global__ void netPolling(float* A, int* Mmap, float* C, int ftMaps, int pollSize, int iSize){
    int x = threadIdx.x + blockDim.x*blockIdx.x;
    int y = threadIdx.y + blockDim.y*blockIdx.y;
    int z = threadIdx.z + blockDim.z*blockIdx.z;

    int xAcc = x*pollSize;
    int yAcc = y*pollSize;

    if(xAcc >= iSize || yAcc >= iSize || z >= ftMaps){
        return;
    }

    int yS;
    int xS;
    float max = -FLT_MAX;

    for(int i = 0; i < pollSize; i++){
        for(int k = 0; k < pollSize; k++){
            if(A[(yAcc+i) + iSize*((xAcc+k) + iSize*z)] > max){
                max = A[(yAcc+i) + iSize*((xAcc+k) + iSize*z)];
                yS = i;
                xS = k;
            }
        }
    }

    C[y + (iSize/pollSize)*(x + (iSize/pollSize)*z)] = max;
    Mmap[(y + (iSize/pollSize)*(x + (iSize/pollSize)*z))*2] = xS + pollSize*x;
    Mmap[(y + (iSize/pollSize)*(x + (iSize/pollSize)*z))*2 + 1] = yS + pollSize*y;
}

__global__ void MatVecMultFusedAddAct(float* a, float* b, float* c, float* addV, int Arows, int AcBr){
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int tileSize = blockDim.x;
    int ctx = tileSize*bx + tx;

    if((AcBr-(bx*tileSize)) < tileSize){
        tileSize=(AcBr-(bx*tileSize));
    }

    if(ctx >= Arows){
        return;
    }

    if(Arows < tileSize){
        tileSize = Arows;
    }

    //printf("AcBr = %d\nTileSize = %d\n", AcBr, tileSize);

    __shared__ float B[1024];

    float tempVal = 0;
    
    for(int i = 0; i < AcBr; i += tileSize){
        if(tx < tileSize && tx+i < AcBr){
            B[tx] = b[tx+i];
        }
        else{
            B[tx] = 0.0;
        }
        __syncthreads();

        for(int k = 0; k < tileSize && k+i < AcBr; k++){
            tempVal += a[ctx + Arows*(k+i)]*B[k];
            //printf("b[%d]=%f; B[%d]=%f\n", tx+i, b[tx+i], tx, B[tx]);
            //printf("Iter: %d; locIter: %d\n", k+i, k);
        }
        __syncthreads();
    }
    
    tempVal += addV[ctx];
    if(tempVal < 0){
        tempVal *= ReLUalpha;
    }
    //printf("%f\n", tempVal);
    c[ctx] = tempVal;
}

__global__ void sumReduction(float* A, float* writeBack, int eSize, int rSize, int sizeDBBlock){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int k = blockIdx.y*blockDim.y + threadIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockDim.x;
    int bi = blockIdx.x;

    __shared__ float l1A[1024];
    l1A[tx + ty*bx] = A[i + k*eSize];
    __syncthreads();

    int p0 = tx*2;
    int p1 = p0+1;
    float* l1AM = l1A+ty*bx;
    for(int j = 0; j < bx; j++){
        __syncthreads();
        if(p1 >= bx || p1+bx*bi >= eSize){
            if(p0 == 0){
                writeBack[bi + k*sizeDBBlock] = l1AM[0];
            }
            return;
        }
        __syncthreads();
        l1AM[p0] = l1AM[p0]+l1AM[p1];
        p0 *= 2;
        p1 *= 2;
    }
}

__global__ void convLayGradient(float* NN, float* prevOut, float* error, float* errWriteBack, float* gradWriteBack, 
int* offMap, int pollSize, int wSize, int plSize, int flSize, int ftMaps, int inputChannels){
    int xy0 = blockIdx.x*blockDim.x + threadIdx.x;
    int xy = blockIdx.y*blockDim.y + threadIdx.y;
    int z01 = blockIdx.z*blockDim.z + threadIdx.z;

    if(xy0 >= flSize*flSize || xy >= wSize*wSize + plSize*plSize + 1 || z01 >= ftMaps*inputChannels){
        return;
    }

    int z = z01/inputChannels;
    int z0 = z01%inputChannels;

    int xOff = offMap[xy0*2 + flSize*flSize*2*z];
    int yOff = offMap[xy0*2 + 1 + flSize*flSize*2*z];

    float errorP = error[xy0 + flSize*flSize*z];
    if(prevOut[xy0 + plSize*plSize*inputChannels + flSize*flSize*z] < 0){
        errorP *= ReLUalpha;
    }
    int x;
    int y;

    if(xy < plSize*plSize){     
        x = xy/plSize;
        y = xy%plSize;
        if(x < xOff || y < yOff || x-xOff >= wSize || y-yOff >= wSize){
            errWriteBack[xy0 + flSize*flSize*z + ftMaps*flSize*flSize*(xy + (plSize*plSize)*z0)] = 0;
            return;
        }
        errWriteBack[xy0 + flSize*flSize*z + ftMaps*flSize*flSize*(xy + (plSize*plSize)*z0)] = errorP*NN[(y-yOff) + wSize*(x-xOff) + (wSize*wSize*inputChannels+1)*(z) + (wSize*wSize)*z0];
        return;
    }
    else{
        xy -= plSize*plSize;
        if(xy < wSize*wSize + 1){
            if(xy == wSize*wSize){
                if(z0 != 0){
                    return;
                }
                gradWriteBack[xy0 + flSize*flSize*(wSize*wSize*inputChannels + (wSize*wSize*inputChannels + 1)*(z))] = errorP;
                return;
            }
            
            x = xy/wSize;
            y = xy%wSize;

            if(xOff < 0 || yOff < 0 || xOff >= (plSize-wSize)+1 || yOff>=(plSize-wSize)+1){
                gradWriteBack[xy0 + flSize*flSize*(xy + wSize*wSize*z0 + (wSize*wSize*inputChannels + 1)*(z))] = 0;
                return;
            }

            gradWriteBack[xy0 + flSize*flSize*(xy + wSize*wSize*z0 + (wSize*wSize*inputChannels + 1)*(z))] = errorP*prevOut[(y+yOff) + plSize*((x+xOff) + plSize*z0)];
        }
    }
}

//modify this to not require a kernel call in the future
__global__ void cErrorSetup(float* lastOutput, int correctPos, float* writeBack, int llSize){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i >= llSize){
        return;
    }
    writeBack[i] = lastOutput[i];
    if(i == correctPos){
        writeBack[i] -= 1;
    }
    //printf("Desired number=%d; writeBack[%d]=%f\n", correctPos, i, writeBack[i]);
}

__global__ void fflGradientComputation(float* NN, float* prevOutput, float* error, float* errWriteBack, float* gradWriteBack, int plSize, int flSize){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int k = blockIdx.y*blockDim.y + threadIdx.y;

    if(i >= flSize || k >= plSize+1){
        return;
    }

    float errP = error[i];
    if(prevOutput[i+plSize] < 0){
        errP *= ReLUalpha;
    }

    if(k < plSize){
        errWriteBack[i + flSize*k] = errP*NN[i + k*flSize];
        gradWriteBack[i + k*flSize] = errP*prevOutput[k];
        return;
    }
    gradWriteBack[plSize*flSize + i] = errP;
}

__global__ void expSoftMax(float* A, float* B, float maxV, int size){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i >= size){
        return;
    }
    B[i] = exp(A[i]-maxV);
}

__global__ void finalSoftmax(float* A, float* B, float sumV, int size){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i >= size){
        return;
    }
    B[i] = A[i]/(sumV);
}

__global__ void initiateToC(float* A, float b, int size){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i >= size){
        return;
    }
    A[i] = b;
}

__global__ void applyAdam(float* grVector, float* varianceV, float* meanV, int size, int iter){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i >= size){
        return;
    }
    meanV[i] = beta1*meanV[i] + (1.0-beta1)*grVector[i];
    varianceV[i] = beta2*varianceV[i] + (1.0-beta2)*(grVector[i]*grVector[i]);
    float lm = (meanV[i])/(1.0-pow(beta1, iter));
    float lv = (varianceV[i])/(1.0-pow(beta2, iter));
    grVector[i] = (lm)/(sqrt(lv)+epsilon);
}

void sumRedSetup(float* matrix, float* holder, int rows, int cols){
    int nBlocks = rows;
    int pRows = rows;
    
    while(1){
        nBlocks = ((nBlocks-1)/1024)+1;
        sumReduction<<<dim3(nBlocks, cols), dim3(1024, 1)>>>(matrix, holder, pRows, cols, nBlocks);
        pRows = nBlocks;
        cudaDeviceSynchronize();
        cudaMemcpy(matrix, holder, nBlocks*cols*sizeof(float), cudaMemcpyDeviceToDevice);
        if(nBlocks <= 1){
            break;
        }
    }
    
}

void cpuSoftMax(float* vector, int size){
    float* hVector = (float*)malloc(size*sizeof(float));
    cudaDeviceSynchronize();
    cudaMemcpy(hVector, vector, sizeof(float)*size, cudaMemcpyDeviceToHost);
    float m = -FLT_MAX;
    for(int i = 0; i < size; i++){
        if(hVector[i] > m){
            m = hVector[i];
        }
    }

    expSoftMax<<<((size-1)/16)+1, 16>>>(vector, vector, m, size);
    cudaDeviceSynchronize();
    cudaMemcpy(hVector, vector, sizeof(float)*size, cudaMemcpyDeviceToHost);
    float s = 0;
    for(int i = 0; i < size; i++){
        s += hVector[i];
    }
    finalSoftmax<<<((size-1)/16)+1, 16>>>(vector, vector, s, size);
    free(hVector);
    cudaDeviceSynchronize();
}

void computeNetworkOutput(float* NN, float* CMidBuffer, float* outputBuffer, int* mapsBuffer, int* convLayers, int clSize, int* dLayers, int dlSize){
    int midOutputSize;
    int inputSize;
    int outputSize;
    int pollingSize;
    int wSize;
    int inputChannels;
    int ftMaps;
    float* lNN = NN;
    float* lOutput = outputBuffer;
    int* lMaps = mapsBuffer;
    int* lCl = convLayers;
    
    for(int i = 0; i < clSize; i++){
        inputChannels = lCl[0];
        wSize = lCl[1];
        inputSize = lCl[2];
        pollingSize = lCl[3];
        ftMaps = lCl[4];

        midOutputSize = (inputSize-wSize)+1;
        outputSize = (midOutputSize/pollingSize);
        dim3 cGridSize(((midOutputSize-1)/32)+1, ((midOutputSize-1)/32)+1, ftMaps);
        dim3 cBlockSize(32, 32, 1);
        cudaDeviceSynchronize();
        ConvolutionFusedAddAct<<<cGridSize, cBlockSize>>>(lOutput, lNN, CMidBuffer, ftMaps, inputChannels, wSize, inputSize);
        cudaDeviceSynchronize();
        lNN += ftMaps*(1+wSize*wSize*inputChannels);
        lOutput += inputSize*inputSize*inputChannels;
        lCl += 4;
        dim3 cOutGridSize(((outputSize-1)/32)+1, ((outputSize-1)/32)+1, ftMaps);
        dim3 cOutBlockSize(32, 32, 1);
        cudaDeviceSynchronize();
        netPolling<<<cOutGridSize, cOutBlockSize>>>(CMidBuffer, lMaps, lOutput, ftMaps, pollingSize, midOutputSize);
        lMaps += 2*(outputSize*outputSize*ftMaps);
        cudaDeviceSynchronize();
    }

    for(int i = 0; i < dlSize-1; i++){
        inputSize = dLayers[i]; //cols
        outputSize = dLayers[i+1];  //rows
        cudaDeviceSynchronize();
        MatVecMultFusedAddAct<<<(((outputSize-1)/1024)+1), 1024>>>(lNN, lOutput, lOutput+inputSize, lNN+(inputSize*outputSize), outputSize, inputSize);
        cudaDeviceSynchronize();
        lNN += (inputSize+1)*outputSize;
        lOutput += inputSize;
        cudaDeviceSynchronize();
    }
    cpuSoftMax(lOutput, dLayers[dlSize-1]);
}

//{matrix size, input size, pooling size, ftMaps}
void computeNetworkGradient(float* NN, float* netOutput, int* netOutputMaps, float* deltaBuffer, float* grWriteBack, int sPos, int* convL, int cvlSize, 
int* ffLayers, int fflSize, int neuralSize, int sumF, int outMapsOff, float* gradBuffer){
    int relativeNeuralOffset = neuralSize;
    int relativeOutputOffset = sumF - ffLayers[fflSize-1];
    cErrorSetup<<<(((ffLayers[fflSize-1]-1)/16)+1), 16>>>(netOutput+relativeOutputOffset, sPos, deltaBuffer, ffLayers[fflSize-1]);
    cudaDeviceSynchronize();
    for(int i = fflSize-1; i >= 1; i--){
        relativeNeuralOffset -= ffLayers[i]*(ffLayers[i-1]+1);
        relativeOutputOffset -= ffLayers[i-1];
        dim3 locGridSize(((ffLayers[i]-1)/4)+1, ((ffLayers[i-1])/4)+1);
        dim3 locBlockSize(4, 4);
        fflGradientComputation<<<locGridSize, locBlockSize>>>(NN+relativeNeuralOffset, netOutput+relativeOutputOffset, deltaBuffer, deltaBuffer+(ffLayers[i]*ffLayers[i-1]), 
        grWriteBack+relativeNeuralOffset, ffLayers[i-1], ffLayers[i]);
        cudaDeviceSynchronize();
        cudaMemcpy(deltaBuffer, deltaBuffer+(ffLayers[i]*ffLayers[i-1]), ffLayers[i]*ffLayers[i-1]*sizeof(float), cudaMemcpyDeviceToDevice);
        sumRedSetup(deltaBuffer, deltaBuffer+(ffLayers[i]*ffLayers[i-1]), ffLayers[i], ffLayers[i-1]);
        cudaDeviceSynchronize();
    }
    int convOffset = 4*cvlSize + 1;
    int inputSize = ((convL[convOffset-3]-convL[convOffset-4])+1)/(convL[convOffset-2]);
    int outputSize;
    int lMatrixSize;
    int lPollSize;
    int ftMaps;
    int inputChannels;
    
    for(int i = cvlSize-1; i >= 0; i--){
        convOffset -= 4;
        outputSize = inputSize;
        inputSize = convL[convOffset+1];
        lMatrixSize = convL[convOffset];
        lPollSize = convL[convOffset+2];
        ftMaps = convL[convOffset+3];
        inputChannels = convL[convOffset-1];

        relativeNeuralOffset -= (lMatrixSize*lMatrixSize*inputChannels+1)*ftMaps;
        relativeOutputOffset -= (inputSize*inputSize)*inputChannels;

        outMapsOff -= outputSize*outputSize*ftMaps*2;

        dim3 cGridSize((((outputSize*outputSize)-1)/32)+1, ((inputSize*inputSize+lMatrixSize*lMatrixSize)/32)+1, ftMaps*inputChannels);
        dim3 cBlockSize(32, 32, 1);

        convLayGradient<<<cGridSize, cBlockSize>>>(NN+relativeNeuralOffset, netOutput+relativeOutputOffset, deltaBuffer, 
        deltaBuffer+(outputSize*outputSize*inputSize*inputSize*ftMaps*inputChannels), gradBuffer, netOutputMaps+outMapsOff, lPollSize, 
        lMatrixSize, inputSize, outputSize, ftMaps, inputChannels);
        cudaDeviceSynchronize();
 

        cudaMemcpy(deltaBuffer, deltaBuffer+(outputSize*outputSize*inputSize*inputSize*ftMaps*inputChannels), 
        outputSize*outputSize*inputSize*inputSize*ftMaps*inputChannels*sizeof(float), cudaMemcpyDeviceToDevice);

        sumRedSetup(deltaBuffer, deltaBuffer+(outputSize*outputSize*inputSize*inputSize*ftMaps), outputSize*outputSize*ftMaps, inputSize*inputSize*inputChannels);
        sumRedSetup(gradBuffer, gradBuffer+(outputSize*outputSize*(lMatrixSize*lMatrixSize*inputChannels+1)*ftMaps), outputSize*outputSize, 
        (lMatrixSize*lMatrixSize*inputChannels+1)*ftMaps);

        cudaMemcpy(grWriteBack+relativeNeuralOffset, gradBuffer, (lMatrixSize*lMatrixSize*inputChannels+1)*ftMaps*sizeof(float), cudaMemcpyDeviceToDevice);
    }
    
}

void runLearning(float* NN, float* initialInfo, float* secOutBuffer, float* deltaBuffer, float* mainGradBuffer, float* secGradBuffer, float* gradientWriteBack, int batchSize, 
int* convLayers, int clSize, int* ffLayers, int fflSize, int sumF, int neuralSize, int* mapsBuffer, int mapsOffset, int deltaS){
    //-----------------------------------------------------------------------------------------------
    float* mainOutBuffer;

    cudaMalloc(&mainOutBuffer, sumF*sizeof(float));
    cudaMemset(mainOutBuffer, 0, sumF*sizeof(float));
    //-----------------------------------------------------------------------------------------------

    float costAcc = 0;
    float accrc = 0;
    float* cAcc = (float*)malloc(ffLayers[fflSize-1]*sizeof(float));
    int offset0 = sumF-ffLayers[fflSize-1];

    for(int i = 0; i < batchSize; i++){
        cudaDeviceSynchronize();
        int selDigit = (int)initialInfo[0];
        initialInfo += 1;
        cudaMemset(mainOutBuffer, 0, sumF*sizeof(float));
        cudaMemset(secOutBuffer, 0, sumF*sizeof(float));
        cudaDeviceSynchronize();

        for(int k = 0; k < 1; k++){
            cudaMemcpy(mainOutBuffer+(k*convLayers[2]*convLayers[2]*convLayers[0]), initialInfo, convLayers[2]*convLayers[2]*convLayers[0]*sizeof(float), cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();
        }
        initialInfo += convLayers[2]*convLayers[2]*convLayers[0];
        cudaDeviceSynchronize();
        computeNetworkOutput(NN, secOutBuffer, mainOutBuffer, mapsBuffer, convLayers, clSize, ffLayers, fflSize);
        cudaDeviceSynchronize();

        cudaMemcpy(cAcc, mainOutBuffer+offset0, ffLayers[fflSize-1]*sizeof(float), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        float max = 0;
        int mPos;
        costAcc -= log(cAcc[selDigit]);
        for(int t = 0; t < ffLayers[fflSize-1]; t++){
            if(cAcc[t] >= max){
                max = cAcc[t];
                mPos = t;
            }
        }
        if(mPos==selDigit){
            accrc++;
        }

        //-------------------------------------------
        cudaMemset(mainGradBuffer, 0, neuralSize*sizeof(float));
        cudaMemset(deltaBuffer, 0, deltaS*sizeof(float));
        cudaDeviceSynchronize();
        computeNetworkGradient(NN, mainOutBuffer, mapsBuffer, deltaBuffer, mainGradBuffer, selDigit, convLayers, clSize, ffLayers, fflSize, neuralSize, sumF, 
        mapsOffset, secGradBuffer);
        cudaDeviceSynchronize();

        cudaDeviceSynchronize();
        VecAddMult<<<((neuralSize-1)/1024)+1, 1024>>>(mainGradBuffer, gradientWriteBack, 1.0f/((float)batchSize), neuralSize);
        cudaDeviceSynchronize();
        cudaMemset(mainGradBuffer, 0, neuralSize*sizeof(float));
        cudaDeviceSynchronize();
    }

    printf("Cost was %f, Accuracy was %f\n-----------------\n", costAcc/(float)batchSize, accrc/(float)batchSize);

    cudaFree(mainOutBuffer);
}

void shuffleToP(int* list, int limit, int size){
    random_device rnDev;
    mt19937 rng(rnDev());
    for(int i = 0; i < limit; i++){
        uniform_real_distribution<> dist0((float)i, (float)size);
        int p = (int)dist0(rng);
        int temp = list[p];
        list[p] = list[i];
        list[i] = temp;
    }
}

void testNetwork(float* NN, float* testData, int neuralSize, int* clLayers, int clSize, int* mapsBuf, int* dLayers, int dlSize, float* sOutBuf, int sumF, 
int testDataSize){
    //-----------------------------------------------------------------------------------------------
    float* mainOutBuffer;

    cudaMalloc(&mainOutBuffer, sumF*sizeof(float));
    cudaMemset(mainOutBuffer, 0, sumF*sizeof(float));
    //-----------------------------------------------------------------------------------------------

    printf("Beggining of verification\n");

    float costAcc = 0;
    float accrc = 0;
    float* cAcc = (float*)malloc(dLayers[dlSize-1]*sizeof(float));
    int offset0 = sumF-dLayers[dlSize-1];

    for(int i = 0; i < testDataSize; i++){
        int selDigit = (int)testData[0];
        testData += 1;
        cudaMemset(mainOutBuffer, 0, sumF*sizeof(float));
        cudaMemset(sOutBuf, 0, sumF*sizeof(float));
        cudaDeviceSynchronize();

        for(int k = 0; k < 1; k++){
            cudaMemcpy(mainOutBuffer+(k*clLayers[2]*clLayers[2]*clLayers[0]), testData, clLayers[2]*clLayers[2]*clLayers[0]*sizeof(float), cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();
        }
        testData += clLayers[2]*clLayers[2]*clLayers[0];
        computeNetworkOutput(NN, sOutBuf, mainOutBuffer, mapsBuf, clLayers, clSize, dLayers, dlSize);
        cudaDeviceSynchronize();

        cudaMemcpy(cAcc, mainOutBuffer+offset0, dLayers[dlSize-1]*sizeof(float), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        float max = 0;
        int mPos;
        costAcc -= log(cAcc[selDigit]);
        for(int t = 0; t < dLayers[dlSize-1]; t++){
            if(cAcc[t] >= max){
                max = cAcc[t];
                mPos = t;
            }
        }
        if(mPos==selDigit){
            accrc++;
        }
    }

    printf("Cost was %f, Accuracy was %f\n-----------------\n", costAcc/(float)testDataSize, accrc/(float)testDataSize);

    free(cAcc);
    cudaFree(mainOutBuffer);
}

void trainingSetup(float* NN, float* initialInfoF, int* iCL, int iclSize, int ixSize, int* IdenseLayers, int idlSize, int inputChannels, int batchSize, int dataSize,
float* testData, int testDataSize){
    int neuralSize = 0;
    int sumF = 0;
    int mapsOffset = 0;

    int dlSize = idlSize+1;
    int* denseLayers = (int*)malloc(dlSize*sizeof(int));

    int* shufPos = (int*)malloc(dataSize*sizeof(int));
    int* poConvL = (int*)malloc((iclSize*4 + 1)*sizeof(int));
    poConvL[0] = inputChannels;

    int mcInput = 0;
    int mcGrad = 0;

    int ftMaps;
    int inpChn;

    int cInput = ixSize;
    for(int i = 0; i < iclSize; i++){
        poConvL[4*i + 1] = iCL[3*i];
        poConvL[4*i + 2] = cInput;
        poConvL[4*i + 3] = iCL[3*i + 1];
        poConvL[4*i + 4] = iCL[3*i + 2];
        ftMaps = iCL[3*i + 2];
        inpChn = poConvL[4*i];
        

        if(cInput*cInput*(((cInput-poConvL[4*i + 1])+1)/poConvL[4*i + 3])*(((cInput-poConvL[4*i + 1])+1)/poConvL[4*i + 3])*poConvL[4*i]*poConvL[4*i + 4]
             > mcInput){
            mcInput = cInput*cInput*(((cInput-poConvL[4*i + 1])+1)/poConvL[4*i + 3])*(((cInput-poConvL[4*i + 1])+1)/poConvL[4*i + 3])*poConvL[4*i]*poConvL[4*i + 4];
        }

        sumF += cInput*cInput*inpChn;
        neuralSize += (poConvL[4*i + 1]*poConvL[4*i + 1]*inpChn + 1)*ftMaps;

        cInput = (((cInput-poConvL[4*i + 1])+1)/poConvL[4*i + 3]);

        if(cInput*cInput*(iCL[3*i]*iCL[3*i]*inpChn + 1)*ftMaps > mcGrad){
            mcGrad = cInput*cInput*(iCL[3*i]*iCL[3*i]*inpChn + 1)*ftMaps;
        }
        mapsOffset += cInput*cInput*ftMaps*2;
    }
    denseLayers[0] = cInput*cInput*ftMaps;
    sumF += IdenseLayers[idlSize-1];

    for(int i = 0; i < idlSize; i++){
        sumF += denseLayers[i];
        denseLayers[i+1] = IdenseLayers[i];
        neuralSize += (denseLayers[i]+1)*denseLayers[i+1];
    }


    for(int i = 0; i < dataSize; i++){
        shufPos[i] = i;
    }
    shuffleToP(shufPos, batchSize, dataSize);

    int deltaSize = 0;
    if(mcGrad > neuralSize){
        deltaSize = mcGrad;
    }
    else{
        deltaSize = neuralSize;
    }
    if(deltaSize < mcInput){
        deltaSize = mcInput;
    }
    deltaSize *= 2;
    deltaSize += 1;

    float* iSend = (float*)malloc(batchSize*(ixSize*ixSize+1)*inputChannels*sizeof(float));

    //--------------device side mem declarations------------------------
    float* dNN;
    float* sOutBuffer;
    float* deltaBuffer;
    float* mGradBuffer;
    float* secGradBuffer;
    float* gradWB;
    int* mapsBuf;
    float* meanH;
    float* varH;

    cudaMalloc(&dNN, neuralSize*sizeof(float));
    cudaMalloc(&sOutBuffer, sumF*sizeof(float));
    cudaMalloc(&deltaBuffer, deltaSize*sizeof(float));
    cudaMalloc(&mGradBuffer, neuralSize*sizeof(float));
    cudaMalloc(&secGradBuffer, deltaSize*sizeof(float));
    cudaMalloc(&gradWB, neuralSize*sizeof(float));
    cudaMalloc(&mapsBuf, mapsOffset*sizeof(int));
    cudaMalloc(&meanH, neuralSize*sizeof(float));
    cudaMalloc(&varH, neuralSize*sizeof(float));

    cudaMemcpy(dNN, NN, neuralSize*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(gradWB, 0, neuralSize*sizeof(float));
    cudaMemset(meanH, 0, neuralSize*sizeof(float));
    cudaMemset(varH, 0, neuralSize*sizeof(float));
    cudaDeviceSynchronize();
    //------------------------------------------------------------------
    int cIt2 = 0;
    int cIt = 0;
    while(1){
        shuffleToP(shufPos, dataSize, dataSize);
        for(int i = 0; i+batchSize < dataSize; i += batchSize){
            cIt2++;
            cudaMemset(gradWB, 0, sizeof(float)*neuralSize);
            cudaDeviceSynchronize();
            cIt++;
            for(int k = 0; k < batchSize; k++){
                int sMovPos = shufPos[k+i];
                memcpy(iSend+((ixSize*ixSize*inputChannels+1)*k), initialInfoF+((ixSize*ixSize*inputChannels+1)*(sMovPos)), sizeof(float)*(ixSize*ixSize*inputChannels+1));
            }
            runLearning(dNN, iSend, sOutBuffer, deltaBuffer, mGradBuffer, secGradBuffer, gradWB, batchSize, poConvL, iclSize, denseLayers, dlSize, sumF, neuralSize,
            mapsBuf, mapsOffset, deltaSize);

            VecAddMult<<<((neuralSize-1)/32)+1, 32>>>(dNN, gradWB, L2alpha, neuralSize);
            cudaDeviceSynchronize();
            applyAdam<<<((neuralSize-1)/32)+1, 32>>>(gradWB, varH, meanH, neuralSize, cIt);
            cudaDeviceSynchronize();
            VecAddMult<<<((neuralSize-1)/32)+1, 32>>>(gradWB, dNN, learningRate, neuralSize);
            cudaDeviceSynchronize();
            if(cIt >= maxIter){
                //cudaMemcpy(NN, dNN, sizeof(float)*neuralSize, cudaMemcpyDeviceToHost);
                //writeF(NN, neuralSize, sizeof(float), NNpath);
                cudaMemset(meanH, 0, neuralSize*sizeof(float));
                cudaMemset(varH, 0, neuralSize*sizeof(float));
                //initiateToC<<<neuralSize, 1>>>(meanH, 0, neuralSize);
                //initiateToC<<<neuralSize, 1>>>(varH, 0, neuralSize);
                cIt = 0;
            }
        }

        testNetwork(dNN, testData, neuralSize, poConvL, iclSize, mapsBuf, denseLayers, dlSize, sOutBuffer, sumF, testDataSize);

        printf("----------------------------epoch-------------------------\n");
        cudaMemcpy(NN, dNN, sizeof(float)*neuralSize, cudaMemcpyDeviceToHost);
        writeF(NN, neuralSize, sizeof(float), NNpath);
    }

    cudaFree(varH);
    cudaFree(meanH);
    cudaFree(mapsBuf);
    cudaFree(gradWB);
    cudaFree(secGradBuffer);
    cudaFree(mGradBuffer);
    cudaFree(deltaBuffer);
    cudaFree(sOutBuffer);
    cudaFree(dNN);

    free(iSend);
    free(denseLayers);
    free(shufPos);
    free(poConvL);
}

float* randomWiehgtHeGeneration(int* layerVals, int totalLayers, int* iCL, int clSize, int ixSize, int fInputChan){
    int fullsize = 0;

    int ftMaps;
    int inpChan = fInputChan;

    int cInput = ixSize;
    for(int i = 0; i < clSize; i++){
        ftMaps = iCL[3*i + 2];
        fullsize += (iCL[3*i]*iCL[3*i]*inpChan+1)*ftMaps;
        cInput = (((cInput-iCL[3*i])+1)/iCL[3*i + 1]);
        inpChan = ftMaps;
    }
    fullsize += (cInput*cInput*ftMaps+1)*layerVals[0];
    for(int i = 1; i < totalLayers; i++){
        fullsize += (layerVals[i-1] + 1)*layerVals[i];
    }
    float* retVal = (float*)mmap(NULL, fullsize*sizeof(float), 0x3, 0x22, -1, 0);

    float* movR = retVal;
    random_device rd1;
    mt19937 gen1(rd1());
    cInput = ixSize;
    inpChan = fInputChan;
    for(int i = 0; i < clSize; i++){
        ftMaps = iCL[3*i + 2];
        normal_distribution<> dis1(0, sqrt(2.0/(iCL[3*i+2]*(cInput+(cInput-iCL[3*i]+1)))));
        //normal_distribution<> dis1(0, sqrt(2.0/(iCL[2*i])));
        //uniform_real_distribution<> dis1(1, 1);
        cInput = (cInput-iCL[3*i]+1)/(iCL[3*i + 1]);
        for(int k = 0; k < ftMaps; k++){
            for(int t0 = 0; t0 < iCL[3*i]*iCL[3*i]*inpChan; t0++){
                movR[t0] = dis1(gen1);
            }
            movR[iCL[3*i]*iCL[3*i]*inpChan] = 0.0;
            movR += (iCL[3*i]*iCL[3*i]*inpChan)+1;
        }
        inpChan = ftMaps;
    }

    for(int i = 0; i < totalLayers; i++){
        int pll;
        if(i == 0){
            pll = cInput*cInput*ftMaps;
        }
        else{
            pll = layerVals[i-1];
        }
        normal_distribution<> dis1(0, sqrt(2.0/(pll + layerVals[i])));
        for(int k = 0; k < pll*layerVals[i]; k++){
            movR[k] = dis1(gen1);
        }
        for(int k = pll*layerVals[i]; k < (pll+1)*layerVals[i]; k++){
            movR[k] = 0;
        }
        movR += (pll + 1)*layerVals[i];
    }
    return retVal;
}

void TrainDataInitialization(float* TrainData, int size, int tiOffset, int initialOffset, float maxValue){
    for(int i = 0; i < size; i++){
        for(int k = initialOffset; k < tiOffset; k++){
            TrainData[i*tiOffset + k] = TrainData[i*tiOffset + k]/maxValue;
        }
    }
    return;
    /*
    random_device rd1;
    mt19937 gen1(rd1());

    int* positions = (int*)malloc(size*sizeof(int));
    int* finalPositions = positions;
    for(int i = 0; i < size; i++){
        positions[i] = i;
    }
    for(int i = 0; i < size; i++){
        uniform_real_distribution<> dis1(0, ((size-i)-0.00000000001));
        int posPicked = (int)dis1(gen1) + i;
        int tempH = positions[i];
        positions[i] = positions[posPicked];
        positions[posPicked] = tempH;
    }
    float* FinalTrainData = (float*)malloc(tiOffset*size*sizeof(float));
    for(int i = 0; i < size; i++){
        memcpy((FinalTrainData+(tiOffset*i)),(TrainData+(tiOffset*finalPositions[i])), tiOffset*sizeof(float));
    }
    memcpy(TrainData, FinalTrainData, tiOffset*size*sizeof(float));
    free(FinalTrainData);
    free(finalPositions);
    */
}


int main(int argc, char** argv){
    /*
    float weightTest[] = {
        1,
        0,
        0.5, 0.6, 0.7,
        0.8, 0.9, 0.10,
        0.11, 0.12, 0.13,
        0.14, 0.15, 0.16,
        0, 0, 0,
        0.17, 0.18, 0.19,
        0.20, 0.21, 0.22, 
        0.23, 0.24, 0.25,
        0, 0, 0
    };
    float inputTest[] = {
        0, 0.1, 0.2, 0.3, 0.4
    };
    int cL2[] = {1, 2, 1};
    int dL2[] = {4, 3, 3};

    float* dNNTest;
    float* tSecBuf;
    float* tDeltaBuf;
    float* gradWbTest;
    float* tSecGradBuf;
    float* tMainGradBuf;
    int* tMapsBuf;

    cudaMalloc(&dNNTest, 29*sizeof(float));
    cudaMalloc(&tSecBuf, 14*sizeof(float));
    cudaMalloc(&tDeltaBuf, 116*sizeof(float));
    cudaMalloc(&gradWbTest, 29*sizeof(float));
    cudaMalloc(&tSecGradBuf, 116*sizeof(float));
    cudaMalloc(&tMainGradBuf, 29*sizeof(float));
    cudaMalloc(&tMapsBuf, 8*sizeof(int));

    cudaMemcpy(dNNTest, weightTest, 29*sizeof(float), cudaMemcpyHostToDevice);

    runLearning(dNNTest, inputTest, tSecBuf, tDeltaBuf, tMainGradBuf, tSecGradBuf, gradWbTest, 1, cL2, 1, dL2, 3, 1, 14, 29, tMapsBuf, 8, 116);

    return 0;
    */
    
    int initialSize = 28;
    int fInputChannels = 1;
    //int featureMaps = 1;
    //matrixSize, pollingSize
    int convLayers[] = {5, 2, 20, 5, 2, 40};
    //int convLayers[] = {1, 1};
    //should include last layer
    int denseLayers[] = {1024, 1024, 128, 10};
    //int denseLayers[] = {16, 16, 10};
    //-------------------------------------------------------------
    /*
    int ftMaps;
    int inpChan = fInputChannels;
    int fullsize = 0;
    int cInput = initialSize;
    for(int i = 0; i < sizeof(convLayers)/(sizeof(int)*3); i++){
        ftMaps = convLayers[3*i + 2];
        fullsize += (convLayers[3*i]*convLayers[3*i]*inpChan+1)*ftMaps;
        cInput = (((cInput-convLayers[3*i])+1)/convLayers[3*i + 1]);
        inpChan = ftMaps;
    }
    fullsize += (cInput*cInput*ftMaps+1)*denseLayers[0];
    for(int i = 1; i < sizeof(denseLayers)/sizeof(int); i++){
        fullsize += (denseLayers[i-1] + 1)*denseLayers[i];
    }
    //-------------------------------------------------------------
    float* NN = randomWiehgtHeGeneration(denseLayers, sizeof(denseLayers)/sizeof(int), convLayers, sizeof(convLayers)/(sizeof(int)*3), initialSize, fInputChannels);
    writeF(NN, fullsize, sizeof(float), NNpath);
    */
    
    long rndP;
    float* NN = (float*)readF(NNpath, sizeof(float), &rndP);
    float* trainData = (float*)readF(dataPath, sizeof(float), &rndP);
    float* testData = (float*)readF(testdPath, sizeof(float), &rndP);

    TrainDataInitialization(trainData, 60000, 785, 1, 255);
    TrainDataInitialization(testData, 10000, 785, 1, 255);

    trainingSetup(NN, trainData, convLayers, sizeof(convLayers)/(sizeof(int)*3), initialSize, denseLayers, sizeof(denseLayers)/sizeof(int), fInputChannels,
    64, 60000, testData, 10000);
    
    return 0;
}
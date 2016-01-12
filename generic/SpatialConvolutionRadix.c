#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialConvolutionRadix.c"
#else
#include<stdlib.h>
#include<complex.h>

#ifdef TH_REAL_IS_FLOAT
extern void c_pad(const float* input, const long unsigned int * input_dims,float *output,const long unsigned int *output_dims);
extern void c_vector_scale_accum(float* vec1,const float* vec2,float scalar, long unsigned int length);
extern void c_vector_complex_fma(float* accum,const float* a,float* b, const long unsigned int length);
extern void c_extract_valid(const float* input, const long unsigned int* input_dims, float* output,const long unsigned int *output_dims );
extern void c_fft2d_dit2_r2c(float *in_fft, unsigned long ldn, long unsigned int* dims,float _Complex *out_fft);
extern void c_fft2d_dit2_c2r(const float _Complex *in_fft, unsigned long ldn, long unsigned int* dims,float *out_fft);


void frequencyConv(real *r_output, real* t_input, long ir,long ic,real* k_weight,long kr,long kc){
	      	  float normaliser=1/(ir*ic);
			  float* weight_data=(float*)(k_weight);
		      float* input_data=(float*)(t_input);
		      long unsigned int kernelDimensions[]={kr,kc};
		      long unsigned int inputDimensions[]={ir,ic};
		      long or = (ir - kr)  + 1;
		      long oc = (ic - kc)  + 1;
		      long unsigned int outputDimensions[]={or,oc};
		      float* padded=(float*)calloc(ic*ir,sizeof(float)); 
		      float* tmp_output=(float*)calloc(or*oc,sizeof(float));
		      int vol=2*ir*(ic/2+1); 
		      float* fws=(float*)calloc(vol,sizeof(float)); 
		      //for(int i=0;i<nOutputPlane;++i){
		    	  c_pad(weight_data, kernelDimensions, padded, inputDimensions);
		    	  c_fft2d_dit2_r2c(padded,5,inputDimensions,(float _Complex *)fws);	
		    	//  fws+=vol;
		    	 // weight_data+=kH*kW;
		      //}
		     
		      float* frequencyInput=(float*)calloc(vol,sizeof(float)); 
		      
		     // for(int j=0;j<input->size[0];++j){
		    	  c_pad(input_data, inputDimensions, padded, inputDimensions);
		    	  c_fft2d_dit2_r2c(padded,5,inputDimensions,(float _Complex*)frequencyInput);
		    //	  ffs+=vol;
		    //	  input_data+=inputWidth*inputHeight;
		     // }
		      
		      float* frequencyOutput=calloc(vol,sizeof(float));
		      c_vector_complex_fma(frequencyOutput, fws, frequencyInput, vol / 2);
		      c_fft2d_dit2_c2r((float _Complex*)frequencyOutput,5,inputDimensions,padded);
		      c_extract_valid( padded, inputDimensions, (real*)tmp_output, outputDimensions);
		      //vector_scale((real*)tmp_output, or*oc, normaliser);
		      c_vector_scale_accum(r_output,(real*)tmp_output,normaliser,or*oc);

		      free(tmp_output);
		      free(frequencyInput);
		      free(frequencyOutput);
		      free(fws);
		      free(padded);
}

void tensor4dConv(THTensor *r_,real beta, THTensor *t_,THTensor* k_,long srow, long scol){
	  long nInputPlane, nInputRows, nInputCols;
	  long nKernelRows, nKernelCols;
	  long nOutputPlane, nOutputRows, nOutputCols;
	  long kstride0, kstride1;
	  THTensor *input;
	  THTensor* kernel;
	  long nbatch;
	  long nelem;
	  real *input_data;
	  real *weight_data;
	  real *output_data;
	  long p;

	  THArgCheck(t_->nDimension == 4 , 3, "input: 4D Tensor expected");
	  THArgCheck(k_->nDimension == 4 , 4, "kernel: 4D Tensor expected");
	  THArgCheck(srow == 1, 5, "Stride should be a positive integer");// only should be 1
	  THArgCheck(scol == 1, 6, "Stride should be a positive integer");

	  input = THTensor_(newContiguous)(t_);
	  if (!(k_->stride[3] == 1) || !(k_->stride[2] == k_->size[3])) {
	    kernel = THTensor_(newContiguous)(k_);
	  } else {
	    THTensor_(retain)(k_);
	    kernel = k_;
	  }

	  nbatch = input->size[0];
	  nInputPlane = input->size[1];
	  nInputRows  = input->size[2];
	  nInputCols  = input->size[3];

	  kstride0    = kernel->stride[0];
	  kstride1    = kernel->stride[1];
	  nKernelRows = kernel->size[2];
	  nKernelCols = kernel->size[3];
	  nOutputPlane = kernel->size[0];
	  THArgCheck(kernel->size[1] == nInputPlane, 2, "invalid number of input planes");
	  nOutputRows = (nInputRows - nKernelRows) / srow + 1;
	  nOutputCols = (nInputCols - nKernelCols) / scol + 1;
	  nelem = THTensor_(nElement)(r_);
	    THTensor_(resize4d)(r_, nbatch, nOutputPlane, nOutputRows, nOutputCols);

	    input_data = THTensor_(data)(input);
	    weight_data = THTensor_(data)(kernel);
	    output_data = THTensor_(data)(r_);
	    if (nelem == 0 || beta == 0 || nelem != THTensor_(nElement)(r_))
	     {
	       /*THTensor_(zero)(r_);*/
	   #pragma omp parallel for private(p)
	       for (p=0; p < r_->size[0]; p++)//nbatch
	       {
	         long k;
	         for (k = 0; k < r_->size[1]; k++)
	         {
	           real* ptr_output = output_data + p*nOutputPlane*nOutputRows*nOutputCols + k*nOutputCols*nOutputRows;
	           long l;
	           for (l = 0; l < nOutputRows*nOutputCols; l++)
	             ptr_output[l] = 0.0;
	         }
	       }
	     }
	     else if (beta != 1)
	     {
	       /*THTensor_(mul)(r_, beta);*/
	   #pragma omp parallel for private(p)
	       for(p=0; p < r_->size[0]; p++)
	       {
	         long k;
	         for (k = 0; k < r_->size[1]; k++)
	         {
	           real* ptr_output = output_data + p*nOutputPlane*nOutputRows*nOutputCols + k*nOutputCols*nOutputRows;
	           long l;
	           for (l = 0; l < nOutputRows*nOutputCols; l++)
	             ptr_output[l] *= beta;
	         }
	       }
	     }
#pragma omp parallel for private(p)
  for(p=0; p < nbatch; p++)
  {
    long k;
    for(k = 0; k < nOutputPlane; k++)
    {
      long i;
      /* get output */
      real *ptr_output = output_data + p*nOutputPlane*nOutputCols*nOutputRows + k*nOutputCols*nOutputRows;
      for(i = 0; i < nInputPlane; i++)
      {
        /* get kernel */
        real *ptr_weight = weight_data + k*kstride0 + i*kstride1;
        /* get input */
        real *ptr_input = input_data + p*nInputPlane*nInputRows*nInputCols + i*nInputRows*nInputCols;

        /* do image, kernel convolution */
       
         frequencyConv(ptr_output, ptr_input,  nInputRows,  nInputCols,ptr_weight, nKernelRows, nKernelCols);
      }
      /* Next output plane */
      /* output_data += nOutputCols*nOutputRows;*/
    }
  }
  THTensor_(free)(input);
  THTensor_(free)(kernel);
}
#endif //#ifdef TH_REAL_IS_FLOAT


static int nn_(SpatialConvolutionRadix_updateOutput)(lua_State *L){
	THTensor *input = luaT_checkudata(L,2,torch_Tensor);  
	
	int dW=luaT_getfieldcheckint(L,1,"dW");
	int dH=luaT_getfieldcheckint(L,1,"dH");
	THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
	THTensor *bias = luaT_getfieldcheckudata(L, 1, "bias", torch_Tensor);
	THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor); 

	int dimw = 2;
	int dimh = 1;

	  luaL_argcheck(L, input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D(batch mode) tensor expected");
	  
	  if (input->nDimension == 4) {
	    dimw++;
	    dimh++;
	  }


	    long nOutputPlane = weight->size[0];//32
	    long kW           = weight->size[3];//5
	    long kH           = weight->size[2];//5
	    long inputWidth   = input->size[dimw];//32
	    long inputHeight  = input->size[dimh];//32
	    long outputWidth  = (inputWidth - kW) / dW + 1;//28
	    long outputHeight = (inputHeight - kH) / dH + 1;//28


	 

	      real* bias_data;
	      real* output_data;
	      long p;

	      THTensor_(resize4d)(output, input->size[0], nOutputPlane, outputHeight, outputWidth);//size[0]为batchsize

	      bias_data = THTensor_(data)(bias);
	      output_data = THTensor_(data)(output);

	#pragma omp parallel for private(p)
	     
	      for (p=0; p<input->size[0]; p++)//10
	      {
	        /* BIAS */
	        long i;
	        for (i=0; i<bias->size[0]; i++)
	        {//nOutputPlane=32
	          real *ptr_output = output_data + p*nOutputPlane*outputWidth*outputHeight + i*outputWidth*outputHeight;
	          long j;
	          for(j = 0; j < outputWidth*outputHeight; j++)
	            ptr_output[j] = bias_data[i];
	        }
	      }

	      /* do convolutions */
	      /*
	        4D input, 4D kernel, 4D output
	        matrix vector product like
	        y <- Ax + beta*y
	      */
	      //THTensor_(conv2Dmm)(output, 1.0, 1.0, input, weight, dH, dW, "V","X");
	      tensor4dConv(output,1,input,weight,dH, dW);

	  return 1;

}

static int nn_(SpatialConvolutionRadix_updateGradInput)(lua_State *L){
	THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
	  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
	  int dW = luaT_getfieldcheckint(L, 1, "dW");
	  int dH = luaT_getfieldcheckint(L, 1, "dH");
	  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");

	  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
	  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);

	  THTensor *tweight;

	  THArgCheck( nOutputPlane == gradOutput->size[input->nDimension == 4 ? 1 : 0], 1, "Number of output features is not equal to nOutputPlane" );

	  /* gradient to input */
	  tweight = THTensor_(newTranspose)(weight,0,1);
	  THTensor_(conv2Dmm)(gradInput, 0.0, 1.0, gradOutput, tweight, dH, dW, "F","C");

	  THTensor_(free)(tweight);
return 1;
}

static int nn_(SpatialConvolutionRadix_accGradParameters)(lua_State *L){
	THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
	  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
	  real scale = luaL_optnumber(L, 4, 1);
	  int dW = luaT_getfieldcheckint(L, 1, "dW");
	  int dH = luaT_getfieldcheckint(L, 1, "dH");
	  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");

	  THTensor *gradWeight = luaT_getfieldcheckudata(L, 1, "gradWeight", torch_Tensor);
	  THTensor *gradBias = luaT_getfieldcheckudata(L, 1, "gradBias", torch_Tensor);

	  int dimw = 2;
	  int dimh = 1;

	  real *gradBias_data;
	  real *gradOutput_data;
	  long noutSlice;

	  THArgCheck( nOutputPlane == gradOutput->size[input->nDimension == 4 ? 1 : 0], 1, "Number of output features is not equal to nOutputPlane" );

	  if (input->nDimension == 4)
	  {
	    dimw++;
	    dimh++;
	  }

	  /* gradient to bias */
	  gradBias_data = THTensor_(data)(gradBias);
	  gradOutput_data = THTensor_(data)(gradOutput);
	  noutSlice = gradOutput->size[dimh]*gradOutput->size[dimw];
	  /*THTensor* gradOutSlice = THTensor_(new)();*/

	  if (input->nDimension == 3)
	  {
	    long k;
	#pragma omp parallel for private(k)
	    for(k = 0; k < nOutputPlane; k++)
	    {
	      /*THTensor_(select)(gradOutSlice, gradOutput, 0, k);*/
	      real *ptr_gradOutput = gradOutput_data + k*noutSlice;
	      long l;
	      for(l = 0; l < noutSlice; l++)
	        gradBias_data[k] += scale*ptr_gradOutput[l];
	    }

	    /* gradient to kernels */
	    THTensor_(conv2DRevger)(gradWeight, 1.0, scale, input, gradOutput, dH, dW);
	  }
	  else
	  {
	    long k;
	#pragma omp parallel for private(k)
	    for(k = 0; k < nOutputPlane; k++)
	    {
	      long p;
	      for(p = 0; p < input->size[0]; p++)
	      {
	        /* BIAS */
	        real *ptr_gradOutput = gradOutput_data + p*nOutputPlane*noutSlice + k*noutSlice;
	        long l;
	        for(l = 0; l < noutSlice; l++)
	          gradBias_data[k] += scale*ptr_gradOutput[l];
	      }
	    }
	    /* gradient to kernels */
	    THTensor_(conv2DRevgerm)(gradWeight, 1.0, scale, input, gradOutput, dH, dW);
	  }
	  return 0;
}





static const struct luaL_Reg nn_(SpatialConvolutionRadix__) [] = {
  {"SpatialConvolutionRadix_updateOutput", nn_(SpatialConvolutionRadix_updateOutput)},
  {"SpatialConvolutionRadix_updateGradInput", nn_(SpatialConvolutionRadix_updateGradInput)},
  {"SpatialConvolutionRadix_accGradParameters", nn_(SpatialConvolutionRadix_accGradParameters)},
  {NULL, NULL}
};

static void nn_(SpatialConvolutionRadix_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(SpatialConvolutionRadix__), "nn");
  lua_pop(L,1);
}


#endif

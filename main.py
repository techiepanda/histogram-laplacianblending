# Instructions:
# For question 1, only modify function: histogram_equalization
# For question 2, only modify functions: low_pass_filter, high_pass_filter, deconvolution
# For question 3, only modify function: laplacian_pyramid_blending

import os
import sys
import cv2
import numpy

def help_message():
   print("Usage: [Question_Number] [Input_Options] [Output_Options]")
   print("[Question Number]")
   print("1 Histogram equalization")
   print("2 Frequency domain filtering")
   print("3 Laplacian pyramid blending")
   print("[Input_Options]")
   print("Path to the input images")
   print("[Output_Options]")
   print("Output directory")
   print("Example usages:")
   print(sys.argv[0] + " 1 " + "[path to input image] " + "[output directory]")                                # Single input, single output
   print(sys.argv[0] + " 2 " + "[path to input image1] " + "[path to input image2] " + "[output directory]")   # Two inputs, three outputs
   print(sys.argv[0] + " 3 " + "[path to input image1] " + "[path to input image2] " + "[output directory]")   # Two inputs, single output
   
# ===================================================
# ======== Question 1: Histogram equalization =======
# ===================================================
def calHist(img):
    hist,bins = numpy.histogram(img.flatten(),256,[0,256])
    # cumsum calculate the cumulative sum
    cdf = hist.cumsum()
    # Masking the cdf array
    cdf_m = numpy.ma.masked_equal(cdf,0)
    # Normalize cdf
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    # Replacing masked data in cdf
    cdf = numpy.ma.filled(cdf_m,0).astype('uint8')
    return cdf[img]   # Returning image with updated pixel values based on cdf

def histogram_equalization(img_in):

   # Write histogram equalization here
   # Split three channel to apply histogram to each channel separately and then merge the result
   b, g, r = cv2.split(img_in)
   b = calHist(b);
   g = calHist(g);
   r = calHist(r);
   img_out = cv2.merge((b,g,r))   # Merging back all threee channels to create the RGB Image
   
   return True, img_out
   
def Question1():

   # Read in input images
   input_image = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR);
   
   # Histogram equalization
   succeed, output_image = histogram_equalization(input_image)
   
   # Write out the result
   output_name = sys.argv[3] + "1.jpg"
   cv2.imwrite(output_name, output_image)

   return True
   
# ===================================================
# ===== Question 2: Frequency domain filtering ======
# ===================================================
# This function calculates the discrete fourier transform of the image passed and shifts the image.
# Shifting is required so that low frequencies are at the centre of the image and it will be easier to then mask those frequencies.
def ft(im):
    dft = cv2.dft(numpy.float32(im),flags = cv2.DFT_COMPLEX_OUTPUT)
    return numpy.fft.fftshift(dft)

# This function first inverse the shift done in the ft function and then perform inverse discrete fourier transform.
# As, inverse FFT, expects the same output(or after frequency filtering, LPF, HPF, etc.) as given by applying forward transform(cv2.dft).
def ift(shift):
    f_ishift = numpy.fft.ifftshift(shift)
    img_back = cv2.idft(f_ishift, flags=cv2.DFT_SCALE)
    # magnitude combines real and imaginary part and return the result.
    return cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

def low_pass_filter(img_in):
	
   # Write low pass filter here
   # Convert input image to grayscale as input image is loaded in BGR mode.
   grayScale = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
   dft_shift =ft(grayScale)
   
   # creating a low pass filter by creating a 20*20 submatrix with values 1 inside a matrix of same size
   # as the input image. All the remaining values are 0 so that we can remove high frequencies.
   rows, cols = grayScale.shape
   centreRow,centreCol = rows/2 , cols/2
   # create a mask first, center square is 1, remaining all zeros
   mask = numpy.zeros((rows,cols,2),numpy.uint8)
   mask[centreRow-10:centreRow+10, centreCol-10:centreCol+10] = 1
   # apply the mask by multiplying the image in magnitude spectrum with the mask.
   dft_shift = dft_shift*mask
   # apply inverse dft to generate output image.
   img_out = ift(dft_shift) # Low pass filter result

   return True, img_out

def high_pass_filter(img_in):

   # Write high pass filter here
   # Convert input image to grayscale as input image is loaded in BGR mode.
   grayScale = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
   
   dft_shift =ft(grayScale)
   
   # creating a high pass filter by creating a 20*20 submatrix with values 0 inside a matrix of same size
   # as the input image. All the remaining values are 1 so that we can remove low frequencies which are at the
   # centre of the magnitude spectrum.
   rows, cols = grayScale.shape
   centreRow,centreCol = rows/2 , cols/2
   # create a mask first, center square is 0, remaining all ones
   mask = numpy.ones((rows,cols,2),numpy.uint8)
   mask[centreRow-10:centreRow+10, centreCol-10:centreCol+10] = 0
   # apply the mask by multiplying the image in magnitude spectrum with the mask.
   dft_shift = dft_shift*mask
   # apply inverse dft to generate output image.
   img_out = ift(dft_shift) # High pass filter result
   
   return True, img_out
   
def deconvolution(img_in):
   
   # Write deconvolution codes here
   # Creating Gaussian kernel
   gk = cv2.getGaussianKernel(21,5)
   gk = gk * gk.T
   
   # Create magnitude spectrum of the input image and apply the shift to get low frequencies at the centre.
   imageFourier = numpy.fft.fft2(numpy.float32(img_in),(img_in.shape[0],img_in.shape[1]))
   imageFourierShifted = numpy.fft.fftshift(imageFourier)
   
   # Create magnitude spectrum of the gaussian kernel and apply the shift to get low frequencies at the centre.
   # Passing the size of the image so that magnitude spectrum for kernel and image can be of same size.
   gaussianFourier = numpy.fft.fft2(numpy.float32(gk),(img_in.shape[0],img_in.shape[1]))
   gaussianFourierShifted = numpy.fft.fftshift(gaussianFourier)
   
   imconvf = imageFourierShifted / gaussianFourierShifted
   
   # inverse the shift as ifft expects same output(or after applying some mask) as given by dft.
   f_ishift =numpy.fft.ifftshift(imconvf)
   img_back = numpy.fft.ifft2(f_ishift)
   
   img_out = numpy.abs(img_back) # Deconvolution result
   img_out = img_out * 255

   return True, img_out

def Question2():

   # Read in input images
   input_image1 = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR);
   input_image2 = cv2.imread(sys.argv[3], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH);

   # Low and high pass filter
   succeed1, output_image1 = low_pass_filter(input_image1)
   succeed2, output_image2 = high_pass_filter(input_image1)
   
   # Deconvolution
   succeed3, output_image3 = deconvolution(input_image2)
   
   # Write out the result
   output_name1 = sys.argv[4] + "2.jpg"
   output_name2 = sys.argv[4] + "3.jpg"
   output_name3 = sys.argv[4] + "4.jpg"
   cv2.imwrite(output_name1, output_image1)
   cv2.imwrite(output_name2, output_image2)
   cv2.imwrite(output_name3, output_image3)
   
   return True
   
# ===================================================
# ===== Question 3: Laplacian pyramid blending ======
# ===================================================

def laplacian_pyramid_blending(img_in1, img_in2):

   # Write laplacian pyramid blending codes here
   img1 = img_in1[:,:img_in1.shape[0]]
   img2 = img_in2[:img1.shape[0],:img1.shape[0]]
   
   levels = 5
   # generating Gaussian pyramids for both images
   gpImg1 = [img1.astype('float32')]
   gpImg2 = [img2.astype('float32')]
   for i in xrange(6):
      img1 = cv2.pyrDown(img1)   # Downsampling using Gaussian filter
      gpImg1.append(img1.astype('float32'))
      img2 = cv2.pyrDown(img2)
      gpImg2.append(img2.astype('float32'))
   
   # Generating Laplacin pyramids for both images
   lpImg1 = [gpImg1[5]]
   lpImg2 = [gpImg2[5]]

   for i in xrange(5,0,-1):
   	  # Upsampling and subtracting from upper level Gaussian pyramid image to get Laplacin pyramid image
      tmp = cv2.pyrUp(gpImg1[i]).astype('float32')
      lpImg1.append(np.subtract(gpImg1[i-1],tmp))
            
      tmp = cv2.pyrUp(gpImg2[i]).astype('float32')
      lpImg2.append(np.subtract(gpImg2[i-1],tmp))

   laplacianList = []
   for lImg1,lImg2 in zip(lpImg1,lpImg2):
      rows,cols,dpt = lImg1.shape
      # Merging first and second half of first and second images respectively at each level in pyramid
#      tmp = numpy.hstack((lImg1[:,0:cols/2], lImg2[:,cols/2:]))
        mask1 = np.zeros(lImg1.shape)
        mask2 = np.zeros(lImg2.shape)
        mask1[:, 0:cols/ 2,:] = 1
        mask2[:, cols / 2:,:] = 1
    
        tmp1 = np.multiply(lImg1, mask1.astype('float32'))
        tmp2 = np.multiply(lImg2, mask2.astype('float32'))
        tmp = np.add(tmp1, tmp2)
        laplacianList.append(tmp)
                    
   img_out = laplacianList[0]
   for i in xrange(1,6):
      img_out = cv2.pyrUp(img_out)   # Upsampling the image and merging with higher resolution level image
      img_out = np.add(img_out, laplacianList[i])
   
   np.clip(img_out, 0, 255, out=img_out)
   return True, img_out.astype('uint8')

def Question3():

   # Read in input images
   input_image1 = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR);
   input_image2 = cv2.imread(sys.argv[3], cv2.IMREAD_COLOR);
   
   # Laplacian pyramid blending
   succeed, output_image = laplacian_pyramid_blending(input_image1, input_image2)
   
   # Write out the result
   output_name = sys.argv[4] + "5.jpg"
   cv2.imwrite(output_name, output_image)
   
   return True

if __name__ == '__main__':
   question_number = -1
   
   # Validate the input arguments
   if (len(sys.argv) < 4):
      help_message()
      sys.exit()
   else:
      question_number = int(sys.argv[1])
	  
      if (question_number == 1 and not(len(sys.argv) == 4)):
         help_message()
	 sys.exit()
      if (question_number == 2 and not(len(sys.argv) == 5)):
	 help_message()
         sys.exit()
      if (question_number == 3 and not(len(sys.argv) == 5)):
	 help_message()
         sys.exit()
      if (question_number > 3 or question_number < 1 or len(sys.argv) > 5):
	 print("Input parameters out of bound ...")
         sys.exit()

   function_launch = {
   1 : Question1,
   2 : Question2,
   3 : Question3,
   }

   # Call the function
   function_launch[question_number]()

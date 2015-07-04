## kompresni.py
## This is an example of using Wavelets to decompose an image
## using Python libraries numpy, scipy, PyWavelets
## 
## 
## main reference that inspired me was this webpage:
## http://kastnerkyle.github.io/blog/2014/04/17/wavelets/
##
##  
#####################################################################################
## Copyleft 2015, Ernest Yeung <ernestyalumni@gmail.com>                 
##                                                                                 
## 20150704
##                                                                          
## This program, along with all its code, is free software; 
## you can redistribute it and/or modify  
## it under the terms of the GNU General Public License as published by                
## the Free Software Foundation; either version 2 of the License, or        
## (at your option) any later version.                               
##                                                                
## This program is distributed in the hope that it will be useful,             
## but WITHOUT ANY WARRANTY; without even the implied warranty of                      
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the    
## GNU General Public License for more details.                      
##                                                                       
## You can have received a copy of the GNU General Public License              
## along with this program; if not, write to the Free Software Foundation, Inc.,  
## S1 Franklin Street, Fifth Floor, Boston, MA                      
## 02110-1301, USA                                              
##                                                
## Governing the ethics of using this program, I default to the Caltech Honor Code:  
## ``No member of the Caltech community shall take unfair advantage of               
## any other member of the Caltech community.''                       
##                                                                                  
## If you like what I'm doing and would like to help and contribute support,  
## please take a look at my crowdfunding campaign at ernestyalumni.tilt.com 
## read my mission statement and give your financial support, 
## no matter how small or large, 
## if you can        
## and to keep checking my ernestyalumni.wordpress.com blog and 
## various social media channels    
## for updates as I try to keep putting out great stuff.                          
##                                                                              
## Fund Science! Help my physics education outreach and research efforts at 
## Open/Tilt ernestyalumni.tilt.com  - Ernest Yeung
##                                                                            
## ernestyalumni.tilt.com                                                         
##                                                                                    
## Facebook     : ernestyalumni                                                       
## gmail        : ernestyalumni                                               
## google       : ernestyalumni                                                    
## linkedin     : ernestyalumni                                                  
## Tilt/Open    : ernestyalumni                                                   
## tumblr       : ernestyalumni                                                       
## twitter      : ernestyalumni                                               
## youtube      : ernestyalumni                                                 
## wordpress    : ernestyalumni                                      
##  
##                                                                           
################################################################################
## 

# EY : 20150704 Install a PIL (Python Image Library).  I installed Pillow with pip install pillow 

import numpy as np
import scipy
from scipy import misc, ndimage
import pywt

import matplotlib.pyplot as plt

filename = "3-IS61836062.jpg"  # EY : 20150704 obviously, you can use your own image or the built-in lena
simona   = ndimage.imread(filename)

# let's get only the R,G,B values
simonaRGB = [simona[:,:,k] for k in range(3)] # cf. http://stackoverflow.com/questions/2725750/slicing-arrays-in-numpy-scipy

simona_dwt2_db4 = [pywt.dwt2(color, 'db4') for color in simonaRGB]

simona_dwt2_db4_cA = np.array([simona_dwt2_db4[0][0],simona_dwt2_db4[1][0],simona_dwt2_db4[2][0]])
simona_dwt2_db4_cA = simona_dwt2_db4_cA.reshape(903,602,3)

simona_dwt2_db4_cH = np.array([simona_dwt2_db4[0][1][0],simona_dwt2_db4[1][1][0],simona_dwt2_db4[2][1][0]])
simona_dwt2_db4_cH = simona_dwt2_db4_cA.reshape(903,602,3)

simona_dwt2_db4_cV = np.array([simona_dwt2_db4[0][1][1],simona_dwt2_db4[1][1][1],simona_dwt2_db4[2][1][1]])
simona_dwt2_db4_cV = simona_dwt2_db4_cA.reshape(903,602,3)

simona_dwt2_db4_cD = np.array([simona_dwt2_db4[0][1][2],simona_dwt2_db4[1][1][2],simona_dwt2_db4[2][1][2]])
simona_dwt2_db4_cD = simona_dwt2_db4_cD.reshape(903,602,3)

simona_idwt2_db4 = [pywt.idwt2(simona_dwt2_db4[0],'db4'),pywt.idwt2(simona_dwt2_db4[1],'db4'),pywt.idwt2(simona_dwt2_db4[2],'db4')]
simona_idwt2_db4 = np.array(simona_idwt2_db4)
simona_idwt2_db4 = simona_idwt2_db4.reshape(1800,1198,3)

def colorwavedec2(data,res=5,wavelet='db4'):
    return [pywt.wavedec2(color,wavelet,level=res) for color in data]

def colorwaverec2(colorcoeffs, wavelet='db4'):
    N = len(colorcoeffs)
    img = np.array([pywt.waverec2(colorcoeff, wavelet) for colorcoeff in colorcoeffs])
    m,n = img[0].shape
    img = img.reshape(m,n,N)
    return img

simona_wavedec2_db4 = colorwavedec2( simonaRGB )
simona_waverec2_db4 = colorwaverec2( simona_wavedec2_db4)


plt.figure(1)
plt.subplot(4,1,1)
plt.imshow( simona_dwt2_db4[0][0] )
#plt.title("cA approximation detail coefficients of Daubechies 4 single level discrete wavelet transform of Red (cerveny)")
plt.subplot(4,1,2)
plt.imshow( simona_dwt2_db4[0][1][0] )
#plt.title("cH horizontal detail coefficients of Daubechies 4 single level discrete wavelet transformor Red (cerveny)")
plt.subplot(4,2,1)
plt.imshow( simona_dwt2_db4[0][1][1] )
#plt.title("cV vertical detail coefficients of Daubechies 4 single level discrete wavelet transform of Red (cerveny)")
plt.subplot(4,2,2)
plt.imshow( simona_dwt2_db4[0][1][2] )
#plt.title("cD diagonal detail coefficients of Daubechies 4 single level discrete wavelet transform of Red (cerveny)")

plt.figure(2)
plt.subplot(4,1,1)
plt.imshow( simona_dwt2_db4[1][0] )
plt.title("cA approximation detail coefficients of Daubechies 4 single level discrete wavelet transform of Green (zeleny)")
plt.subplot(4,1,2)
plt.imshow( simona_dwt2_db4[1][1][0] )
plt.title("cH horizontal detail coefficients of Daubechies 4 single level discrete wavelet transformor Green (zeleny)")
plt.subplot(4,2,1)
plt.imshow( simona_dwt2_db4[1][1][1] )
plt.title("cV vertical detail coefficients of Daubechies 4 single level discrete wavelet transform of Green (zeleny)")
plt.subplot(4,2,2)
plt.imshow( simona_dwt2_db4[1][1][2] )
plt.title("cD diagonal detail coefficients of Daubechies 4 single level discrete wavelet transform of Green (zeleny)")

plt.figure(3)
plt.subplot(4,1,1)
plt.imshow( simona_dwt2_db4[2][0] )
plt.title("cA approximation detail coefficients of Daubechies 4 single level discrete wavelet transform of Blue (modry)")
plt.subplot(4,1,2)
plt.imshow( simona_dwt2_db4[2][1][0] )
plt.title("cH horizontal detail coefficients of Daubechies 4 single level discrete wavelet transformor Blue (modry)")
plt.subplot(4,2,1)
plt.imshow( simona_dwt2_db4[2][1][1] )
plt.title("cV vertical detail coefficients of Daubechies 4 single level discrete wavelet transform of Blue (modry)")
plt.subplot(4,2,2)
plt.imshow( simona_dwt2_db4[2][1][2] )
plt.title("cD diagonal detail coefficients of Daubechies 4 single level discrete wavelet transform of Red (cerveny)")

plt.figure(4)
plt.imshow( simona_dwt2_db4_cA)
plt.title("cA approx coeff of Daub 4 single level DWT")

plt.figure(5)
plt.imshow( simona_dwt2_db4_cH)
plt.title("cH hori coeff of Daub 4 single level DWT")

plt.figure(6)
plt.imshow( simona_dwt2_db4_cV)
plt.title("cV vert coeff of Daub 4 single level DWT")

plt.figure(7)
plt.imshow( simona_dwt2_db4_cD)
plt.title("cD diag coeff of Daub 4 single level DWT")

plt.figure(8)
plt.imshow(simona_idwt2_db4)
plt.title("IDWT using Daub 4 on single level DWT using Daub 4")

plt.figure(9)
plt.imshow(simona_waverec2_db4)
plt.title("5 level reconstruction using Daub 4")

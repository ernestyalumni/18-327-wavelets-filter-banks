## Handout16_examples.py
## This is my implementation of the Handout and Slide examples for the Lecture Notes of 
## using Python libraries numpy, scipy, sympy for 
## Handout 16
##
## The main reference that I'll use is
## Gilbert Strang, and Kevin Amaratunga. 18.327 Wavelets, Filter Banks and Applications, Spring 2003. (Massachusetts Institute of Technology: MIT OpenCourseWare), http://ocw.mit.edu (Accessed 19 Jun, 2015). License: Creative Commons BY-NC-SA
## 
## 
## 
#####################################################################################
## Copyleft 2015, Ernest Yeung <ernestyalumni@gmail.com>                 
##                                                                                 
## 20150707
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
## and subscription-based Patreon   
## read my mission statement and give your financial support, 
## no matter how small or large, 
## if you can        
## and to keep checking my ernestyalumni.wordpress.com blog and 
## various social media channels    
## for updates as I try to keep putting out great stuff.                          
##                                                                              
## Fund Science! Help my physics education outreach and research efforts at 
## Open/Tilt or subscription Patreon - Ernest Yeung
##                                                                            
## ernestyalumni.tilt.com                                                         
##                                                                                    
## Facebook     : ernestyalumni                                                       
## gmail        : ernestyalumni                                               
## google       : ernestyalumni                                                    
## linkedin     : ernestyalumni                                                  
## Patreon      : ernestyalumni
## Tilt/Open    : ernestyalumni                                                   
## tumblr       : ernestyalumni                                                       
## twitter      : ernestyalumni                                               
## youtube      : ernestyalumni                                                 
## wordpress    : ernestyalumni                                      
##  
##                                                                           
################################################################################
## 
## 
## 
##

####################
## MIT OCW 18.327 
####################

###############
# Handout 16
###############

#import sympy

#from sympy import fourier_transform, inverse_fourier_transform, sqrt, exp, I, pi
#from sympy.abc import t, f

import numpy as np 
import matplotlib.pyplot as plt

import pywt

print pywt.wavelist()
choice = 'bior2.2' # choice of wavelet
Wchoice = pywt.Wavelet(choice)
#Wsym5 = pywt.Wavelet('sym5')

# EY : 20150708 note Why are 5/3 wavelets called bior2.2, in Matlab as well? 
# cf. http://www.mathworks.com/matlabcentral/newsreader/view_thread/307357

plt.figure(1)
plt.subplot(4,1,1)
plt.plot(Wchoice.filter_bank[0],)
plt.title("lowpass decomposition filter coefficients for "+choice)
plt.subplot(4,1,2)
plt.plot(Wchoice.filter_bank[1])
plt.title("highpass decomposition filter coefficients for "+choice)
plt.subplot(4,1,3)
plt.plot(Wchoice.filter_bank[2])
plt.title("lowpass reconstruction filter coefficients for "+choice)
plt.subplot(4,1,4)
plt.plot(Wchoice.filter_bank[3])
plt.title("highpass reconstruction filter coefficients for "+choice)

print "The number of vanishing moments for the scaling function phi is ", Wchoice.vanishing_moments_phi
print "The number of vanishing moments for the wavelet psi is ", Wchoice.vanishing_moments_psi

assert Wchoice.biorthogonal # the wavelet to choose for this example should be biorthogonal

[phi_d, psi_d, phi_r, psi_r, x] = Wchoice.wavefun()

print "The default level of resolution for pywt, PyWavelets, is 8"

plt.figure(2)
plt.subplot(4,1,1)
plt.plot(x,phi_d)
plt.title("scaling function phi decomposed for "+choice)
plt.subplot(4,1,2)
plt.plot(x,psi_d)
plt.title("wavelet psi decomposed for "+choice)
plt.subplot(4,1,3)
plt.plot(x,phi_r)
plt.title("scaling function phi reconstructed for "+choice)
plt.subplot(4,1,4)
plt.plot(x,psi_r)
plt.title("wavelet psi reconstructed for "+choice)

# plt.show()

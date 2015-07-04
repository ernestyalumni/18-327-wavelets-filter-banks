## This is my implementation of example6.py
## Example 6: Generation of orthogonal scaling functions and wavelets.  
## using Python libraries numpy, scipy, matlibplot, PyWavelets
## this needs phivals.py (just import it in from the same directory!)  and daub.py
##
## The main reference that I'll use is
## Gilbert Strang, and Kevin Amaratunga. 18.327 Wavelets, Filter Banks and Applications, Spring 2003. (Massachusetts Institute of Technology: MIT OpenCourseWare), http://ocw.mit.edu (Accessed 19 Jun, 2015). License: Creative Commons BY-NC-SA
## 
## 
## 
#####################################################################################
## Copyleft 2015, Ernest Yeung <ernestyalumni@gmail.com>                 
##                                                                                 
## 20150702
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

import numpy as np
import matplotlib.pyplot as plt

from phivals import phivals
from daub import daub

import pywt

# Example 6: Compute the samples of Daubechies scaling function and
# wavelet using the inverse DWT. (Discrete Wavelet Transform)

p = 2                                                                      # number of zeros at pi 
N = 2*p-1                                                                  # Support of the scaling function
numlevels = 5                                                              # Number of iterations/levels
M = 2**numlevels            
L = M*N
f0 = daub(N+1)/2.                                                          # Synthesis lowpass filter.
f1 = np.multiply(np.vstack(np.power((-1),np.array(range(N+1)))),f0[::-1])  # Synthesis highpass filter.

# For the scaling function, we need to compute the inverse DWT with a delta
# for the approximation coefficients. (All detail coefficients are set 
# to zero.)

# EY : 20150703 note on using PyWavelets
# to make a custom Wavelet, you need to specify lowpass and highpass decomposition filters and lowpass and highpass reconstruction filters for a Filter Bank
# how do we get the lowpass and highpass decomposition filters from the highpass reconstruction filters?
# see my note in my pdf/LaTeX writeup, note on Sec. 6.2. of Strang and Nguyen
# the answer is to just reverse the order cf. http://www.pybytes.com/pywavelets/regression/wavelet.html
c = f0[::-1] 
d = f1[::-1]
dec_lo, dec_hi, rec_lo, rec_hi = np.hstack(c), np.hstack(d), np.hstack(f0), np.hstack(f1)
filter_bank = [dec_lo, dec_hi, rec_lo, rec_hi ]
DaubechiesWavelet = pywt.Wavelet(name="DaubechiesWavelet", filter_bank=filter_bank)

y =  pywt.upcoef('a', [1.,0.], DaubechiesWavelet, numlevels)
phi = M*np.append(0, y[0:L])

# For the wavelet, we need to compute the inverse DWT with a delta for the 
# detail coefficients. (All approximation coefficients and all detail
# coefficients at finer scales are set to zero.)
y = pywt.upcoef('d', [1.,0.], DaubechiesWavelet, numlevels)                       # Inverse DWT
w = M*np.append(0,y[0:L])

# Determine the time vector.
t = np.vstack(np.array(range(L+1)))/M

# Plot the results.
plt.figure(1)
plt.plot(t,phi,'-',label="Scaling function")
plt.plot(t,w,'--',label="Wavelet")
plt.legend()
plt.title("Scaling function and wavelet by iteration of synthesis filter bank.")
plt.xlabel('t')

#
# EY : 20150703 Instead of upcoef, use the wavefun function for the Wavelet Object
#
[phi_d,psi_d,phi_r,psi_r,x] = DaubechiesWavelet.wavefun(numlevels)
phi_r = np.sqrt(M) * phi_r
psi_r = np.sqrt(M) * psi_r
plt.figure(2)
plt.plot(x,phi_r,'-',label="Scaling function")
plt.plot(x,psi_r,'--',label="Wavelet")
plt.legend()
plt.title("Scaling function and wavelet by iteration of synthesis filter bank.")
plt.xlabel('t')


# Now compute the scaling function and wavelet by recursion.
# phivals (not part of the Matlab toolbox; EY : 20150703 maybe it's in GNU Octave or PyWavelets?) does this.

[t1, phi1, w1] = phivals(daub(2*p), numlevels)

# Plot the results
plt.figure(3)
plt.plot(t1,phi1,'-',label="Scaling function")
plt.plot(t1,w1,'--',label="Wavelet")
plt.legend()
plt.title("Scaling function and wavelet by recursion.")
plt.xlabel('t')

# View the scaling functions side by side.
plt.figure(4)
plt.plot(x,phi_r,'-',label="Scaling function using iteration")
plt.plot(t1,phi1,'--',label="Scaling function using recursion")
plt.legend()
plt.title('Comparison of the two methods (recursion is exact.)')
plt.xlabel('t')




# View the scaling functions side by side

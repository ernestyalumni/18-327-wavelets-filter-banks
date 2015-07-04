# daub.py
## This is my implementation of daub.m
## Computation of Daubechies' filter coefficients (cepstrum method)
## using Python libraries numpy, scipy
## 
## The main reference that I'll use is
## Gilbert Strang, and Kevin Amaratunga. 18.327 Wavelets, Filter Banks and Applications, Spring 2003. (Massachusetts Institute of Technology: MIT OpenCourseWare), http://ocw.mit.edu (Accessed 19 Jun, 2015). License: Creative Commons BY-NC-SA
## daub.py dovetails with Handout 8 and I explain it in my notes On 18.327 Wavelets pdf.   
## 
## 
#####################################################################################
## Copyleft 2015, Ernest Yeung <ernestyalumni@gmail.com>                 
##                                                                                 
## 20150630
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
## 

import numpy as np
import scipy

# much credit should go to Kevin Amaratunga for writing the daub.m Matlab file implementing 
# Daubeschies orthogonal wavelets first, of which this Python file is based upon.  
# I hope that translating daub.m to Python, and the documentation (see my wordpress.com blog and look for the Wavelets pdf write up on 18.327) I'm continually doing will help
# extend the its use

def daub(Nh):

    """
    function h = daub(Nh)

    Generate filter coefficients for the Daubechies orthogonal wavelets.

    Kevin Amaratunga
    9 December, 1994.

    h = filter coefficients of Daubechies orthonormal compactly supported
        wavelets
    Nh = length of filter.

    EY:20150630 I've implemented daub.m in Python with numpy, scipy ernestyalumni.wordpress.com
    """

    K = Nh/2
    L = Nh/2.  # EY : 20150630 you need L to be a float to match the results of daub.m
    N = 512 # Use a 512 point FFT by default
    k = np.array(range(N))
    
    # Determine samples of the z transform of Mz (= Mz1 Mz2) on the unit circle.
    # Mz2 = z.^L .* ((1 + z.^(-1)) / 2).^(2*L); # or in Python numpy
    # Mz2 = np.multiply( np.power(z, L) , np.power( (( 1+ np.power(z,-1))/2.) , (2*L) ) )

    z = np.exp( 2j*np.pi*k/N)
    tmp1 = (1 + np.power( z,-1 ) )/2
    tmp2 = (-z + 2 - np.power(z,-1) )/4 # sin^2(w/2)

    Mz1 = np.zeros(N)
    vec = np.ones(N)
    for l in range(K):
        # Mz1 = Mz1 + binomial(L+l-1,l) * tmp2.^l;
        Mz1 = Mz1 + vec
        vec = np.multiply( vec, tmp2*(L+l)/(l+1) )  # you need L to be a float to match the results of daub.m
    
    Mz1 = 4 * Mz1

    # Mz1 has no zeros on the unit circle, so use the complex cepstrum to find 
    # its minimum phase spectral factor

    Mz1hat = np.log(Mz1)
    m1hat = np.fft.ifft( Mz1hat )     # Real cepstrum of np.fft.ifft(Mz1). (= cmplx
                                      # cepstrum since Mz1 real, +ve.)
    m1hat[N/2:N] = np.zeros(N/2)  # Retain just the causal part.
    m1hat[0] = m1hat[0]/2             # Value at zero is shared between
                                      # the causal and anticausal part.
    G = np.exp(np.fft.fft(m1hat,N))   # Min phase spectral factor of Mz1
    
    # Mz2 has zeros on the unit circle, but its minimum phase spectral factor
    # is just tmp1.^L i.e. np.power( tmp1, L )
    Hz = np.multiply( G, np.power( tmp1, L ) )  # you need L to be a float to match the results of daub.m
    h = np.real( np.fft.ifft(Hz) ) # there are problems here with precision

# EY : 20150630 there is a difference of precision between GNU Octave and Python numpy
# cf. http://stackoverflow.com/questions/8717099/ifft-in-matlab-and-numpy-give-different-results
# This might be promising for increasing speed and making grids, including numba
# cf. https://jakevdp.github.io/blog/2015/02/24/optimizing-python-with-numpy-and-numba/

    h = np.vstack( h[0:Nh] )
    return h


    
    
                   
        

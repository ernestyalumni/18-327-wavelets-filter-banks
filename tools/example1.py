## example1.py
## This is my implementation of Example 1 - Basic filters, upsampling and downsampling
## using Python libraries numpy, scipy
## 
## The main reference that I'll use is
## Gilbert Strang, and Kevin Amaratunga. 18.327 Wavelets, Filter Banks and Applications, Spring 2003. (Massachusetts Institute of Technology: MIT OpenCourseWare), http://ocw.mit.edu (Accessed 19 Jun, 2015). License: Creative Commons BY-NC-SA
## 
## 
## 
#####################################################################################
## Copyleft 2015, Ernest Yeung <ernestyalumni@gmail.com>                 
##                                                                                 
## 20150619
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

# Example 1 - Basic filters, upsampling and downsampling

import numpy as np
import matplotlib.pyplot as plt

#import scipy

N = 1024
# W = np.arange(-N/2, N/2-1)/(N/2.)

# This is a better way to get the \omega frequency axis, because it's built into numpy's fft
T = 2. # 2.0*np.pi # Sampling period (in seconds)
omegas = np.fft.fftfreq(N,1./T) # rad./sec
omegas = np.fft.fftshift(omegas) # Shift zero frequency to center

# Low pass filter
h0 = np.array([0.5,0.5])
H0 = np.fft.fft(h0,N) # pad the input with zeros
H0 = np.fft.fftshift(H0) # Shift zero frequency to center

Fig00, ax00 = plt.subplots(1,1,sharex=True)
ax00.plot( omegas, np.absolute(H0) ) # Plot spectral power
ax00.set_ylabel(r'Fourier transform magnitude', size='x-large')
ax00.set_xlabel(r'Angular frequency (normalized by $\pi$)')
ax00.set_title(r'Frequency response of Haar lowpass filter: [1/2 1/2]')

# High pass filter
h1 = np.array([0.5,-0.5])
H1 = np.fft.fft(h1,N) # pad the input with zeros
H1 = np.fft.fftshift(H1) # Shift zero frequency to center

Fig01, ax01 = plt.subplots(1,1,sharex=True)
ax01.plot( omegas, np.absolute(H1) ) # Plot spectral power
ax01.set_ylabel(r'Fourier transform magnitude', size='x-large')
ax01.set_xlabel(r'Angular frequency (normalized by $\pi$)')
ax01.set_title(r'Frequency response of Haar highpass filter: [1/2 -1/2]')

# Linear interpolating lowpass filter.
hlin = np.array( [0.5,1.,0.5])
H = np.fft.fft(hlin,N)
H = np.fft.fftshift(H)
Fig02, ax02 = plt.subplots(1,1,sharex=True)
ax02.plot( omegas, np.absolute(H))
ax02.set_ylabel(r'Fourier transform magnitude', size='x-large')
ax02.set_xlabel(r'Angular frequency (normalized by $\pi$)')
ax02.set_title(r'Frequency response of lowpass filter: [1/2 1 1/2]')

# Upsampling
u0 = np.array([0.5,0.,0.5,0.])
U0 = np.fft.fft(u0,N)
U0 = np.fft.fftshift(U0)
Fig03, ax03 = plt.subplots(1,1,sharex=True)
ax03.plot( omegas, np.absolute(U0))
ax03.set_ylabel(r'Fourier transform magnitude', size='x-large')
ax03.set_xlabel(r'Angular frequency (normalized by $\pi$)')
ax03.set_title(r'Fourier transform of [1/2 0 1/2 0]')

# Downsampling
x = np.array([-1,0,9,16,9,0,-1])/16.
X = np.fft.fft(x,N)
X = np.fft.fftshift(X)
Fig04, ax04 = plt.subplots(1,1,sharex=True)
ax04.plot( omegas, np.absolute(X))
ax04.set_ylabel(r'Fourier transform magnitude', size='x-large')
ax04.set_xlabel(r'Angular frequency (normalized by $\pi$)')
ax04.set_title(r'Fourier transform of x=[-1 0 9 16 9 0 -1]/16')

x2 = np.array([-1,9,9,-1])/16.
X2 = np.fft.fftshift( np.fft.fft(x2,N) )
Fig05, ax05 = plt.subplots(1,1,sharex=True)
ax05.plot( omegas, np.absolute(X2))
ax05.set_ylabel(r'Fourier transform magnitude', size='x-large')
ax05.set_xlabel(r'Angular frequency (normalized by $\pi$)')
ax05.set_title(r'Fourier transform of x=[-1 9 9 -1]/16')

# This is downsampling
XX = np.fft.fftshift( np.fft.fft( x,2*N))  # X(\omega)
XX2 = XX[N/2+1:3*N/2+1]
XXPi = np.fft.fftshift( XX)
XX2Pi = XXPi[N/2+1:3*N/2+1]
Y = (XX2 + XX2Pi)/2.
Fig06, ax06 = plt.subplots(1,1,sharex=True)
ax06.plot( omegas, np.absolute(Y))
ax06.set_ylabel(r'Fourier transform magnitude', size='x-large')
ax06.set_xlabel(r'Angular frequency (normalized by $\pi$)')
ax06.set_title(r'[X($\omega/2$) + X($\omega/2+pi$)]/2')




## Handout_examples.py
## This is my implementation of the Handout and Slide examples for the Lecture Notes of 
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

###############
# Handout 1
###############

import numpy as np 
import matplotlib.pyplot as plt

# Shout outs to ESCI 386 - Scientific Programming, Analysis and Visualization with Python 
# LEsson 17 - Fourier Transforms, the lecture slides are good for espousing on the examples with Python
# http://snowball.millersville.edu/~adecaria/ESCI386P/esci386-lesson17-Fourier-Transforms.pdf

N = 100 # Number of data points
dt = 1.0 # Sampling period (in seconds)

time  = dt*np.arange(0,N) # time coordinates

ht = np.zeros(N)
ht[0] = 0.5
ht[1] = 0.5

hhatf = np.fft.fft(ht)
freqs = np.fft.fftfreq(N,dt)
hhatf = np.fft.fftshift(hhatf) # Shift zero frequency to center
freqs = np.fft.fftshift(freqs) # Shift zero frequence to center

Fig0101, ax0101 = plt.subplots(3,1,sharex=True)
ax0101[0].plot( freqs, np.real( hhatf) ) # Plot Cosine terms
ax0101[0].set_ylabel(r'$Re[\widehat{h}(2\pi f)]$', size='x-large')
ax0101[1].plot( freqs, np.imag( hhatf) ) # Plot Sine terms
ax0101[1].set_ylabel(r'$Im[\widehat{h}(2\pi f)]$', size='x-large')
ax0101[2].plot( freqs, np.absolute( hhatf)**2 ) # Plot spectral power
ax0101[2].set_ylabel(r'$|\widehat{h}(2\pi f)|^2$', size='x-large')
ax0101[2].set_xlabel(r'$f$', size='x-large')

# plt.show()

# if you want this in radians (as I do)

T = 2.0*np.pi # Sampling period (in seconds)

omegas = np.fft.fftfreq(N,1./T) # rad./sec
omegas = np.fft.fftshift(omegas) # Shift zero frequence to center

Fig0101b, ax0101b = plt.subplots(3,1,sharex=True)
ax0101b[0].plot( omegas, np.real( hhatf) ) # Plot Cosine terms
ax0101b[0].set_ylabel(r'$Re[\widehat{h}(\omega)]$', size='x-large')
ax0101b[1].plot( omegas, np.imag( hhatf) ) # Plot Sine terms
ax0101b[1].set_ylabel(r'$Im[\widehat{h}(\omega)]$', size='x-large')
ax0101b[2].plot( omegas, np.absolute( hhatf)**2 ) # Plot spectral power
ax0101b[2].set_ylabel(r'$|\widehat{h}(\omega)|^2$', size='x-large')
ax0101b[2].set_xlabel(r'$\omega \, (rad/sec)$', size='x-large')
#Fig0101b.suptitle("Low pass Filter example", fontsize=10)  # add a centered title to the figure
# plt.show()

Fig0101ba, axba = plt.subplots(1,1)
# axba.plot( omegas, np.arccos( np.real( hhatf)/np.absolute(hhatf) ) )
axba.plot( omegas, np.arctan2( np.imag( hhatf), np.real(hhatf)) )
axba.set_ylabel(r'$\phi(\omega)$',size='x-large')
axba.set_xlabel(r'$\omega \, (rad/sec)$', size='x-large')
# axba.title(0,0,"Low-pass filter phase")

#####
## High-pass filter example
#####
ht[1]=-0.5
hhatf = np.fft.fft(ht)
hhatf = np.fft.fftshift(hhatf) # Shift zero frequency to center

Fig0101c, ax0101c = plt.subplots(3,1,sharex=True)
ax0101c[0].plot( omegas, np.real( hhatf) ) # Plot Cosine terms
ax0101c[0].set_ylabel(r'$Re[\widehat{h}(\omega)]$', size='x-large')
ax0101c[1].plot( omegas, np.imag( hhatf) ) # Plot Sine terms
ax0101c[1].set_ylabel(r'$Im[\widehat{h}(\omega)]$', size='x-large')
ax0101c[2].plot( omegas, np.absolute( hhatf)**2 ) # Plot spectral power
ax0101c[2].set_ylabel(r'$|\widehat{h}(\omega)|^2$', size='x-large')
ax0101c[2].set_xlabel(r'$\omega \, (rad/sec)$', size='x-large')
#Fig0101c.suptitle("Low pass Filter example", fontsize=10)  # add a centered title to the figure
# plt.show()

Fig0101ca, axca = plt.subplots(1,1)
axca.plot( omegas, np.arctan2( np.imag( hhatf), np.real( hhatf)))
axca.set_ylabel(r'$\phi(\omega)$',size='x-large')
axca.set_xlabel(r'$\omega \, (rad/sec)$', size='x-large')

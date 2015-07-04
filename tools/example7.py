## This is my implementation of example6.py
## Example 7: Generation of biorthogonal scaling functions and wavelets. 
## using Python libraries numpy, scipy, matlibplot, PyWavelets
## this needs biphivals.py (just import it in from the same directory!)  
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
import pywt
from biphivals import biphivals

# Example 3a: Compute the samples of the biorthogonal scaling functions 
# and wavelets

# 9/7 filters
# create the biorthogonal spline wavelet with 4 vanishing moments object
w_bior = pywt.Wavelet('bior4.4')
[h0,h1,f0,f1] =  w_bior.dec_lo, w_bior.dec_hi, w_bior.rec_lo, w_bior.rec_hi
[x,phi,phitilde,psi,psitilde] = biphivals(h0,h1,f0,f1,5)

plt.figure(1)
plt.plot(x,phi,'-',label="Primary scaling function")
plt.plot(x,psi,'-.',label="Primary wavelet")
plt.legend()
plt.title("Primary Daubachies 9/7 Pair")

plt.figure(2)
plt.plot(x,phitilde,'--',label="Dual scaling function")
plt.plot(x,psitilde,':',label="Dual wavelet")
plt.legend()
plt.title('Dual Daubachies 9/7 pair')




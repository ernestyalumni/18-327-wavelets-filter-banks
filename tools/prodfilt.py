## prodfilt.py
## This is my implementation of prodfilt.m
## Function to generate half-band product filters
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
## 20150628
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


def prodfilt(p):
    """
    prodfilt = prodfilt(p)
    prodfilt(p) = [p0,b,q]

    Generate the halfband product filter of degree 4p-2.

    OUTPUTS
    p0 = coefficients of product filter of degree 4p-2.
    b = coefficients of binomial (spline) filter of degree 2p
    q = coefficients of filter of degree 2p-2 that produces the halfband
        filter p0 when convolved with b.

    """

    # Binomial filter (1 + z^-1)^2p
    tmp1 = np.array([1,1])
    b=1
    for k in range(2*p):
        b = np.convolve(b,tmp1)


    # Q(z)
    tmp2 = np.array( [-1,2,-1])/4.
    q = np.zeros(2*p-1)
    vec = np.zeros(2*p-1)
    vec[p-1] = 1
    for k in range(p):
        q = q + vec
        vec = np.convolve(vec,tmp2)*(p+k)/(k+1.)
        vec = wkeep(vec,2*p-1)
    q = q/2.**(2*p-1)

    # Halfband filter, P0(z)
    p0 = np.convolve(b,q)

    return p0, b, q


##########
## wkeep
##########

def wkeep(x,l):
    assert type(l) == type(1)
    N_0 = len(x)
    if N_0%2 != 0:
        K_0 = N_0/2+1
        if l%2 != 0:
            p = l/2+1
            return x[K_0-p: K_0+p-1]
        else:
            p = l/2
            return x[K_0-p-1:K_0+p-1]
    else:
        K_0 = N_0/2
        if l%2 == 0:
            p=l/2
            return x[K_0-p:K_0+p]
        else:
            p = l/2+1
            return x[K_0 -p:K_0+p-1]

#################
## test values
#################



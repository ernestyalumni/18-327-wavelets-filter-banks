## polyfilt.py
## This is my implementation of polyfilt.m
## Polyphase implementation of a filter
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

def polyfilt(H,X):
    """
    polyfilt = polyfilt(H,X)
    Y = polyfilt(H,X)

    Polyphase filter implementation (2 channels)
    
    X = input signal, separated into even and odd phases.
        first row = even phase
        second row = odd phase
    Y = output signal, separated into even and odd phases.
    H = 2x2 polyphase matrix
        H(1,1,:) = h0,even[n]
        H(1,2,:) = h0,odd[n]
        H(2,1,:) = h1,even[n]
        H(2,2,:) = h1,odd[n]

    that was Kevin Amaratunga's originally explanation.  Here's mine (EY: 20150628)
    H \in \mathbb{R}^{2^2} 
    H[0,0] = h_0, even(n)
    H[0,1] = h_0, odd(n)
    H[1,0] = h_1, even(n)
    H[1,1] = h_1, odd(n)

    If X is a 2xj matrix, then output Y is a 2xj matrix 
        
    """

    y0 = np.convolve(H[0,0],X[0]) + np.convolve(H[0,1],X[1])
    y1 = np.convolve(H[1,0],X[0]) + np.convolve(H[1,1],X[1])
    Y = np.vstack( (y0,y1))
    return Y


###############
## TEST VALUES
###############

H = np.random.rand(2,2)
X = np.random.rand(2,3)

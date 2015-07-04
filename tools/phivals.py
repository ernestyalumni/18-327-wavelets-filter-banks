## philvals.py
## This is my implementation of phivals.m
## Computation of scaling function and wavelet by recursion
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
## 20150621
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
## EY : 20150621 on the MIT OCW website for 18.327, and in the download, phivals.m isn't even a text file; it's in html for some reason.  However, on the web.mit.edu website, the formating is correct, although it's still a html file.

import numpy as np
import scipy
from scipy.linalg import toeplitz

def phivals(h,i):
    """
    phivals = phivals(h,i)
    Generate a scaling function and its associated wavelet 
    using the given filter coefficients

    Matlab original version (but it was a html file!) by Kevin Amaratunga 5 March 1993
    INPUTS: 
    h = filter coefficients (sum(h)=2)
    i = discretization parameter. The number of points per integer
    step is 2^i.  Thus, setting i = 0 gives the scaling function
    and wavelet values at integer points

    OUTPUTS: x,phi,psi
    """

    assert i>=0, "phivals: i must be non-negative"
    
    m,n = h.shape
    assert n==1, "input h is not a column vector"
    
    g = np.multiply( np.array( [(-1)**i for i in range(0,m)]), h[::-1,0] )

    # The Haar filter produces a singular matrix, but since we know the solution
    # already we treat this as a special case.

    if m == 2 and h == np.vstack( np.array([1,1])):
        phi = np.vstack( np.append(np.ones(2**i), np.zeros(1) ) )

        if i > 0:
            psi = np.vstack( np.append( np.append(np.ones(2**(i-1)),-np.ones(2**(i-1))), np.zeros(1) ))

        elif i==0:
            psi = np.vstack( np.array([1,0]) )
    else:
        ch = np.vstack( (h,np.zeros((m,1)) ))
        rh = np.append( np.array([h[0,0]]), np.zeros(m-1))
        tmp = toeplitz(ch,rh)

        M = tmp.flatten('F')[0:-1:2].reshape((m,m)).T - np.identity(m)

        M[-1,:] = np.ones(m)
        tmp = np.vstack( np.append(np.zeros(m-1),np.identity(1)) )
        phi = np.linalg.solve( M,tmp)
            
        # Integer values of phi
        if i > 0:
            for k in range(0,i):
                p = 2**(k+1)*(m-1)+1   # No of rows in toeplitz matrix 
                q = 2**k *(m-1)+1      # No of columns toeplitz matrix
                if k==0:
                    ch0 = np.vstack( np.append( h, np.zeros(p-1-m)) )
                    ch  = np.vstack( ( ch0, np.zeros(1) ))
                    cg0 = np.vstack( np.append(g, np.zeros(p-1-m)))
                else:
                    ch =  np.vstack( np.append( np.identity(1), np.zeros(2**k-1))).dot(ch0.T)
                    ch = np.vstack( np.append( ch.flatten('F'), np.zeros(1) ) )
                rh = np.append( ch[0], np.zeros(q-1) )
                Th = toeplitz(ch,rh)
                if k == i-1:
                    cg = (np.vstack(np.append(np.identity(1),np.zeros(2**k-1)))).dot( cg0.T)
                    cg = cg.flatten('F') # flatten
                    cg = np.vstack( np.append(cg,np.zeros(1)))
                    rg = np.append( cg[0], np.zeros(q-1) )
                    Tg = toeplitz(cg,rg)
                    psi = Tg.dot(phi)
                phi = Th.dot(phi)
        elif i==0:
            cg0 = np.vstack( np.append( g, np.zeros(m-1) ) )
            cg = np.vstack((cg0, np.zeros(1) ) )
            rg = np.append(cg[0],np.zeros(m-1))
            Tg = toeplitz(cg,rg)
            psi = Tg.dot(phi)
            psi=psi[::2]
    
    a,b = phi.shape
    x = np.vstack( np.arange(0,a)/2.**i )

    return x,phi,psi

            

#################
## test values
#################
h_test = np.random.rand(3,1)


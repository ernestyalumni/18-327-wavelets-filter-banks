## This is my implementation of biphivals.py which
## Generate biorthogonal scaling functions and their associated
## using Python libraries numpy, scipy, matlibplot, PyWavelets
## the given filter coefficients. 
##
## The main reference that I'll use is
## Gilbert Strang, and Kevin Amaratunga. 18.327 Wavelets, Filter Banks and Applications, Spring 2003. (Massachusetts Institute of Technology: MIT OpenCourseWare), http://ocw.mit.edu (Accessed 19 Jun, 2015). License: Creative Commons BY-NC-SA
## Note that even though biphivals.m was needed in the MIT OCW 18.327, 
## it was NOT included in the MIT OCW; I found it here:
## http://web.mit.edu/1.130/WebDocs/1.130/Software/Examples/biphivals.m
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
import scipy
from scipy.linalg import toeplitz

def biphivals(h0,h1,f0,f1,i):
    """
    Kevin Amaratunga (4 August, 1993) wrote the Matlab code but I 
    (Ernest Yeung ernestyalumni) implemented biphivals 
    using Python with numpy (20150702) 
    
    Here's Dr. Amaratunga's original comments:

    function [x,phi,phitilde,psi,psitilde] = biphivals(h0,h1,f0,f1,i)
    Generate biorthogonal scaling functions and their associated
    wavelets using the given filter coefficients
    Kevin Amaratunga
    4 August, 1993

    h0, h1, f0, f1 = wavelet filters (from BIORFILT).
    i = discretization parameter.  The number of points per integer
    step is 2^i.  Thus, setting i = 0 gives the scaling function
    and wavelet values at integer points.
    """
    dum = len(f0)
    f0 = np.vstack(np.array(f0))
    f1 = np.vstack(np.array(f1))
    h0 = np.vstack(np.array(h0))
    h1 = np.vstack(np.array(h1))
    N = dum

    tmp,dum = h0.shape
    
    assert i>=0, "biphivals: i must be non-negative"
    
    m,n = f0.shape
    tmp,dum = h0.shape
    
    assert m == tmp, "biphivals: filters f0 and h0 must be the same length"

    #
    # Make sure the lowpass filters sum up to 2
    #
    fac = 2./np.sum( h0 )
    h0 = np.multiply( h0[::-1],fac)
    h1 = np.multiply( h1[::-1],fac)
    f0 = np.multiply( f0, fac)
    f1 = np.multiply( f1, fac)
     
    cf0 = np.vstack( (f0 , np.vstack( np.zeros(m)  ) ) )
    rf0 = np.hstack( ( f0[0], np.zeros(m-1) ) )
    tmp = toeplitz(cf0,rf0)
    M = np.zeros((m,m))
    
    M = tmp.flatten('F')[0:-1:2].reshape((m,m)).T - np.identity(m)

    M[-1,:] = np.ones(m)
    tmp = np.vstack( np.append(np.zeros(m-1),np.identity(1)) )
    phi = np.linalg.solve( M,tmp)  # Integer values of phi 

    ch0 = np.vstack( (h0 , np.vstack( np.zeros(m)  ) ) )
    rh0 = np.hstack( ( h0[0], np.zeros(m-1) ) )
    tmp = toeplitz(ch0,rh0)
    M = np.zeros((m,m))    
    M = tmp.flatten('F')[0:-1:2].reshape((m,m)).T - np.identity(m)
    M[-1,:] = np.ones(m)
    tmp = np.vstack( np.append(np.zeros(m-1),np.identity(1)) )
    phitilde = np.linalg.solve( M,tmp)  # Integer values of phi 

    if i > 0:
        for k in range(0,i):
            p = 2**(k+1)*(m-1)+1   # No of rows in toeplitz matrix 
            q = 2**k *(m-1)+1      # No of columns toeplitz matrix
            if k==0:
                cf00 = np.vstack( np.append(f0, np.zeros(p-1-m)) )
                cf0  = np.vstack(( cf00, np.zeros(1) ))
                ch10 = np.vstack(np.append(h1, np.zeros(p-1-m)))
                ch00 = np.vstack(np.append(h0, np.zeros(p-1-m)))
                ch0  = np.vstack(( cf00, np.zeros(1) ))
                cf10 = np.vstack(np.append(f1, np.zeros(p-1-m)))
            else:
                cf0 = np.vstack( np.append( np.identity(1), np.zeros(2**k-1))).dot(cf00.T)
                cf0 = np.vstack( np.append( cf0.flatten('F'), np.zeros(1) ) )
                ch0 = np.vstack( np.append( np.identity(1), np.zeros(2**k-1))).dot(ch00.T)
                ch0 = np.vstack( np.append( ch0.flatten('F'), np.zeros(1) ) )
            rf0 = np.append( cf0[0], np.zeros(q-1) )
            Tf0 = toeplitz(cf0,rf0)                
            rh0 = np.append( ch0[0], np.zeros(q-1) )
            Th0 = toeplitz(ch0,rh0)
            if k == i-1:
                ch1 = (np.vstack(np.append(np.identity(1),np.zeros(2**k-1)))).dot( ch10.T)
                ch1 = ch1.flatten('F') # flatten
                ch1 = np.vstack(np.append(ch1,np.zeros(1)))
                rh1 = np.append( ch1[0], np.zeros(q-1) )
                Th1 = toeplitz(ch1,rh1)
                cf1 = (np.vstack(np.append(np.identity(1),np.zeros(2**k-1)))).dot( cf10.T)
                cf1 = cf1.flatten('F') # flatten
                cf1 = np.vstack(np.append(cf1,np.zeros(1)))
                rf1 = np.append( cf1[0], np.zeros(q-1) )
                Tf1 = toeplitz(cf1,rf1)
                psi = Tf1.dot(phi)
                psitilde = Th1.dot(phitilde)
            phi = Tf0.dot(phi)
            phitilde = Th0.dot(phitilde)

    elif i==0:
        ch10 = np.vstack( np.append( h1, np.zeros(m-1) ) )
        ch1 = np.vstack((ch10, np.zeros(1) ) )
        rh1 = np.append(ch1[0],np.zeros(m-1))
        Th1 = toeplitz(ch1,rh1)
        cf10 = np.vstack( np.append( f1, np.zeros(m-1) ) )
        cf1 = np.vstack((cf10, np.zeros(1) ) )
        rf1 = np.append(cf1[0],np.zeros(m-1))
        Tf1 = toeplitz(cf1,rf1)
        psi = Tf1.dot(phi)
        psi = psi[::2]
        psitilde=Th1.dot(phitilde)
        psitilde = psitilde[::2]

    a,b = phi.shape
    x = np.vstack( np.arange(0,a)/2.**i )

    return x, phi, phitilde, psi, psitilde




## This is my implementation of example2.m
## Example 2: Product filter examples  
## using Python libraries numpy, scipy
## this needs prodfilt.py (just import it in from the same directory!) 
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
import prodfilt
from prodfilt import prodfilt

from collections import namedtuple

Prodfiltresults = namedtuple("Prodfiltresults", ["p0","b","q"])

# Product filter examples

def cases_eg(caseno,p=2):
    
    cases_available = {1: 
                       # Degree 2
                       Prodfiltresults(
            b = np.array([1,2,1]), # (1 + z^{-1})^2
            q = 1/2.,
            p0 = np.array([1,2,1])/2. # convolve(b,q)
            ), 

                       2:
                           # Degree 6
                       Prodfiltresults(
            b = np.array([1,4,6,4,1]), # (1+z^{-1})^4
            q = np.array([-1,4,-1])/16.,
            p0 = np.array([-1,0,9,16,9,0,-1])/16. # convolve(b,q)
            ),
         
                       3: 
                       # Degree 10
                       Prodfiltresults(
            b = np.array([1,6,15,20,15,6,1]), # (1+z^{-1})^6
            q = np.array([3,-18,38,-18,3])/256.,
            p0 = np.array([3,0,-25,0,150,256,150,0,-25,0,3])/256. # convolve(b,q)
            ),
         
                       4:
                           # Degree 14
                       Prodfiltresults(
            b = np.array([1,8,28,56,70,56,28,8,1]), # (1+z^{-1})^8
            q = np.array([-5,40,-131,208,-131,40,-5])/2048.,
            p0 = np.array([-5,0,49,0,-245,0,1225,2048,1225,0,-245,0,49,0,-5])/2048. # convolve(b,q)
            ),

                       "otherwise":
                           Prodfiltresults( *prodfilt(p) )
                       }
    return cases_available[caseno]

# Let
p0,b,q = cases_eg("otherwise", 2)

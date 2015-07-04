## sunspots.py
## This is an example of using Wavelets to analyze Sun spots
## using Python libraries numpy, scipy, PyWavelets
##
##
## main reference that inspired me was this webpage:
## http://kastnerkyle.github.io/blog/2014/04/17/wavelets/
##
## source of data is from the Royal Observatory of Belgium, Brussels
## and much credits goes to them for making their data freely available:
## Source: WDC-SILSO, Royal Observatory of Belgium, Brussels
##  
#####################################################################################
## Copyleft 2015, Ernest Yeung <ernestyalumni@gmail.com>                 
##                                                                                 
## 20150703
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
## read my mission statement and give your financial support, 
## no matter how small or large, 
## if you can        
## and to keep checking my ernestyalumni.wordpress.com blog and 
## various social media channels    
## for updates as I try to keep putting out great stuff.                          
##                                                                              
## Fund Science! Help my physics education outreach and research efforts at 
## Open/Tilt ernestyalumni.tilt.com  - Ernest Yeung
##                                                                            
## ernestyalumni.tilt.com                                                         
##                                                                                    
## Facebook     : ernestyalumni                                                       
## gmail        : ernestyalumni                                               
## google       : ernestyalumni                                                    
## linkedin     : ernestyalumni                                                  
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

import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

import pywt

try:
    pkl_daily_tot = open('SN_daily_tot.pkl','rb')
    pkl_mm_tot    = open('SN_mm_tot.pkl','rb')

    daily_tot = pickle.load(pkl_daily_tot)
    mm_tot    = pickle.load(pkl_mm_tot)
    
    pkl_daily_tot.close()
    pkl_mm_tot.close()
except IOError:
    print "It's not there!"

d_tot_t   = np.array([float(row[3]) for row in daily_tot])
d_tot_SN  = np.array([int(row[4]) for row in daily_tot])

mm_tot_t  = np.array([float(row[2]) for row in mm_tot])
mm_tot_SN = np.array([float(row[3]) for row in mm_tot])

plt.figure(1)
plt.plot(d_tot_t,d_tot_SN,'-',label="Daily total sunspot number vs. time (years) Source:WDC-SILSO")
plt.plot(mm_tot_t,mm_tot_SN,'--',label="Monthly mean total sunspot no. vs. time (years) Source:WDC-SILSO")
plt.legend()
plt.title("Daily total sunspot number and Monthly mean total sunspot number vs. time (years)")
plt.xlabel('t (years)')

(cA_d,cD_d) = pywt.dwt(d_tot_SN,'db2') 
(cA_mm,cD_mm) = pywt.dwt(mm_tot_SN,'db2') 

plt.figure(2)
plt.subplot(2,1,1)
plt.plot(d_tot_t[::2], cA_d[:-1],'-',label="approximation coefficients of Daily total sunspot number")
plt.plot(d_tot_t[::2], cD_d[:-1],'--',label="detail coefficients of Daily total sunspot number")
plt.legend()
plt.xlabel('t (years)')
plt.title("approximation and detail coefficients using Daubechies 2 for Daily total sunspot number")

plt.subplot(2,1,2)
plt.plot(mm_tot_t[::2],cA_mm[:-1],'-',label="approximation coefficients of Monthly mean total sunspot number")
plt.plot(mm_tot_t[::2],cD_mm[:-1],'--',label="detail coefficients of Monthly mean total sunspot number")
plt.legend()
plt.xlabel('t (years)')
plt.title("approximation and detail coefficients using Daubechies 2 for Monthly mean total sunspot number")

(cA_d_db16,cD_d_db16) = pywt.dwt(d_tot_SN,'db16') 
(cA_mm_db16,cD_mm_db16) = pywt.dwt(mm_tot_SN,'db16') 
#peakind_cD_d_db16 = signal.find_peaks_cwt(cD_d_db16, np.arange(60,120))
#peakind_cD_mm_db16 = signal.find_peaks_cwt(cD_mm_db16, np.arange(60,120))

plt.figure(3)
plt.subplot(2,1,1)
plt.plot(d_tot_t[::2], cA_d_db16[:len(d_tot_t[::2])],'-',label="approximation coefficients of Daily total sunspot number")
plt.plot(d_tot_t[::2], cD_d_db16[:len(d_tot_t[::2])],'--',label="detail coefficients of Daily total sunspot number")
# plt.scatter( d_tot_t[::2][peakind_cD_d_db16] , cD_d_db16[:len(d_tot_t[::2])][peakind_cD_d_db16] , 'x')
plt.legend()
plt.xlabel('t (years)')
plt.title("approximation and detail coefficients using Daubechies 16 for Daily total sunspot number")

plt.subplot(2,1,2)
plt.plot(mm_tot_t[::2],cA_mm_db16[:len(mm_tot_t[::2])],'-',label="approximation coefficients of Monthly mean total sunspot number")
plt.plot(mm_tot_t[::2],cD_mm_db16[:len(mm_tot_t[::2])],'--',label="detail coefficients of Monthly mean total sunspot number")
plt.legend()
plt.xlabel('t (years)')
plt.title("approximation and detail coefficients using Daubechies 16 for Monthly mean total sunspot number")

#plt.figure(4)
#plt.scatter( (d_tot_t[::2])[peakind_cD_mm_db16] , (cD_mm_db16[:len(d_tot_t[::2])])[peakind_cD_mm_db16])

# EY : 201507014 I'm not sure how to get the peaks with scipy.signal.find_peaks_cwt



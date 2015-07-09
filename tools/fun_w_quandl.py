## fun_w_quandl.py
## Fun with Quandl; Fun Wavelets with Quandl
## These are (fun) examples of using Wavelets to analyze (open, public, and open sourced) data sets
## from Quandl
## using Python libraries numpy, scipy, PyWavelets
##
## PS I'm going to use arrow for dealing with time
## arrow is a good replacement for Python's datatime
##  
#####################################################################################
## Copyleft 2015, Ernest Yeung <ernestyalumni@gmail.com>                 
##                                                                                 
## 20150708
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

import Quandl 
import numpy as np
import scipy
from scipy import signal
import matplotlib.pyplot as plt

import pywt
import arrow

from collections import namedtuple

quandl_data_loc = namedtuple('Quandl_Data_loc', ['desc', 'code', 'permalink'])
data_locs_dic = {
    "Shanghai Composite Index (China)":
        quandl_data_loc("Shanghai Composite Index (China)","YAHOO/INDEX_SSEC","https://www.quandl.com/data/YAHOO/INDEX_SSEC"), 
    "Interest Rate Spread, 2 yrs vs 10 yrs, LCY Bond - China":
        quandl_data_loc("Interest Rate Spread, 2 yrs vs 10 yrs, LCY Bond - China","ABMI/INT_RATE_SPREAD_2YRVS10YR_CHN","https://www.quandl.com/data/ABMI/INT_RATE_SPREAD_2YRVS10YR_CHN")
    }
# Shanghai Composite Index (China) 
# https://www.quandl.com/data/YAHOO/INDEX_SSEC

# Interest Rate Spread, 2 yrs vs 10 yrs, LCY Bond - China
# https://www.quandl.com/data/ABMI/INT_RATE_SPREAD_2YRVS10YR_CHN
    
def obtenir(data_locs_dic):
    """
    obtenir (Fr. get)

    obtenir = obtenir(data_locs_dic) 
    where 
    INPUTS
    data_locs_dic is a dictionary of quandl_data_loc, Quandl data locations

    obtenir gets the data, given the Quandl code
    """
    data = {}
    for ele in data_locs_dic:
        data[data_locs_dic[ele].desc] = Quandl.get(data_locs_dic[ele].code, returns="numpy")
    return data

import pickle

def pkl_data(stuff,pkl_file_name = "Quandl_data.pkl"):
    """
    pkl_data = pkl_data(stuff,pkl_file_name)
    
    where
    INPUTS:
    stuff is a Python object to pickle
    pkl_file_name is a string that is the file name of the file to pickle

    pkl_data pickles data and results in a binary file created
    """
    pkl_file = open(pkl_file_name,'wb')
    pickle.dump(stuff,pkl_file,-1)
    pkl_file.close()
    return stuff

def lire(pkl_file_name="Quandl_data.pkl"):
    """
    lire = lire(pkl_file_name="Quandl_data.pkl")
    lire (Fr.read) reads in the pkl file and does the processing
    
    where 
    INPUTS 
    pkl_file_name    is a string
    
    which 
    OUTPUTS
    processed_data             Python object that had been processed
    """

    data_file = open(pkl_file_name,'rb')
    data = pickle.load(data_file)
    data_file.close()

    # Processing part
    processed_data = {}
    for desc in data.keys():
        x = [ pt[0] for pt in data[desc] ]
        y = [ pt[1] for pt in data[desc] ]
        processed_data[desc] = (x,y)
    return processed_data

def make_dwt(processed_data,waveletchoice='haar'):
    """
    make_dwt = make_dwt(data)
    where
    INPUTS:
    data is a dictionary of numpy arrays
    
    OUTPUTS:
    dwt_dic is a dictionary of dwt tuple (discrete wavelet transform)
    """
    dwted_data = {}
    for desc in processed_data.keys():
        dwted_data[desc] = pywt.dwt(processed_data[desc][1],waveletchoice,mode="cpd") # cpd - constant padding - border values are replicated cf. http://www.pybytes.com/pywavelets/ref/signal-extension-modes.html#ref-modes
    return dwted_data

def make_arrow_x(processed_data):
    arrow_x_dict = {}
    for desc in processed_data.keys():
        arrowed = [arrow.get(date) for date in processed_data[desc][0] ]
    # From this point on, this can change, to satisfy different formats to convert into with dates
        arrowed = [date.float_timestamp for date in arrowed ]
        arrow_x_dict[desc] = arrowed
    return arrow_x_dict

def make_frac_year_x(processed_data):
    frac_year_x_dict = {}
    for desc in processed_data.keys():
        yearFractions = [toYearFraction(date) for date in processed_data[desc][0] ]
        frac_year_x_dict[desc] = yearFractions
    return frac_year_x_dict

# commented out because I need to understand the x value for the dwt
"""
def plot_stuff(frac_year_x_dict, dwted_data):
    fig_no = len( dwted_data.keys() )
    for i in range(1,fig_no+1):
        desc = dwted_data.keys()[(i-1)]
        plt.figure(i)
        plt.subplot(2,1,1)
        plt.plot( frac_year_x_dict[desc], dwted_data[desc][0] ,'ko-')
        plt.title('cA approx coeffs for single-level dwt of '+desc)
        plt.subplot(2,1,2)
        plt.plot( frac_year_x_dict[desc], dwted_data[desc][1],'ko-')
        plt.title('cD detail coeffs for single-level dwt of '+desc)
    plt.show()
    return 1
"""

def plot_dwt(dwted_data):
    fig_no = len( dwted_data.keys() )
    for i in range(1,fig_no+1):
        desc = dwted_data.keys()[(i-1)]
        plt.figure(i)
        plt.subplot(2,1,1)
        plt.plot( dwted_data[desc][0] ,'-')
        plt.title('cA approx coeffs for single-level dwt of '+desc)
        plt.subplot(2,1,2)
        plt.plot( dwted_data[desc][1],'ko-')
        plt.title('cD detail coeffs for single-level dwt of '+desc)
    plt.show()
    return 1

        
def main():
    print "Do the command \n data=obtenir(data_locs_dic) \n to get the data from Quandl and assign it to data \n pkl_data(data) \n will get the data into a pickle file \n"

    print "ONCE YOU HAVE A PICKLED FILE: \n processed_data = lire() \n reads in the data from the default file name into Python. \n Then \n dwted_data = make_dwt(processed_data) yields the data after dwt, discrete wavelet transform \n"

    print "WITH processed_data, do make_arrow_x(processed_data) to get the 'x' values (time) in arrow floats or make_frac_year_x(processed_data) for fractional year values \n Also, PLOT your DWT'ed data with plot_dwt(dwted_data)"
    try:
        processed_data = lire()
    except:
        print "File is not there!"

    print "Remember to use the dictionary module (function) of .keys to get the data keys."
        
    return 0

# cf. http://stackoverflow.com/questions/6451655/python-how-to-convert-datetime-dates-to-decimal-years

from datetime import datetime as dt
import time

def toYearFraction(date):
    """
    cf. http://stackoverflow.com/questions/6451655/python-how-to-convert-datetime-dates-to-decimal-years
    """
    def sinceEpoch(date): # returns seconds since epoch
        return time.mktime(date.timetuple())
    s = sinceEpoch

    year = date.year
    startOfThisYear = dt(year=year, month=1, day=1)
    startOfNextYear = dt(year=year+1, month=1, day=1)

    yearElapsed = s(date) - s(startOfThisYear)
    yearDuration = s(startOfNextYear) - s(startOfThisYear)
    fraction = yearElapsed/yearDuration

    return date.year + fraction

if __name__ == "__main__":
    main()



"""
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

# EOAY : 201507014 I'm not sure how to get the peaks with scipy.signal.find_peaks_cwt


"""

## sunspots_get.py
## This is an example of using Wavelets to analyze Sun spots
## using Python libraries numpy, scipy, PyWavelets
## and requests for I/O
## NOTE : this file only gets, web scrapes directly from the main website, the 
## sunspot number data.  See sunspots.py for actually analyzing the data after
## running this file once.   
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

import requests
from urlparse import urlparse, urlunparse

from bs4 import BeautifulSoup as BS4

sidc_main_url = "http://sidc.be/silso/home"
sidc_main_o   = urlparse(sidc_main_url)

sidc_main_r = requests.get(sidc_main_url)
sidc_main_soup = BS4(sidc_main_r.text)

sidc_data_links = [tag for tag in sidc_main_soup.find_all('a') if tag.text == u'Data']
sidc_data_link  = sidc_data_links[0]['href']

sidc_data_url = urlunparse((sidc_main_o[0],sidc_main_o[1],sidc_data_link,'','',''))
sidc_data_r = requests.get(sidc_data_url)
sidc_data_soup = BS4(sidc_data_r.text)

#sidc_subsec_total = [tag for tag in sidc_data_soup.find_all('h4') if tag.text == u'Total sunspot number']
#sidc_subsec_total = sidc_subsec_total[0]

#sidc_tds = [next for next in sidc_subsec_total.nextGenerator() if type(next)==type(sidc_subsec_total) and next.

dwnloads = [lnk for lnk in sidc_data_soup.find_all("a") if 'download' in lnk.attrs.keys() ]
daily_tot_url        = sidc_main_url[: sidc_main_url.rfind('/')+1 ] + dwnloads[0]['href']
monthly_mean_tot_url = sidc_main_url[: sidc_main_url.rfind('/')+1 ] + dwnloads[1]['href']

daily_tot_txt = requests.get(daily_tot_url)
mm_tot_txt    = requests.get(monthly_mean_tot_url)

daily_tot_txt = daily_tot_txt.text.split('\n')
daily_tot_txt = [row.split() for row in daily_tot_txt]
daily_tot_txt = daily_tot_txt[:-1]

mm_tot_txt    = mm_tot_txt.text.split('\n')
mm_tot_txt    = [row.split() for row in mm_tot_txt]
mm_tot_txt    = mm_tot_txt[:-1]

for row in daily_tot_txt:
    try:
        row[3] = float(row[3])
        row[4] = int(row[4])
        row[5] = float(row[5])
        row[6] = int(row[6])
    except IndexError:
        print "Problem:", row

for row in mm_tot_txt:
    try:
        row[2] = float(row[2])
        row[3] = float(row[3])
        row[4] = float(row[4])
        row[5] = int(row[5])
    except IndexError:
        print "Problem:", row

# We shouldn't be accessing the website over and over so let's pickle the data we've obtained
# once and work with it
import pickle

pkl_daily_tot = open('SN_daily_tot.pkl','wb')
pkl_mm_tot    = open('SN_mm_tot.pkl','wb')
pickle.dump(daily_tot_txt,pkl_daily_tot,-1)
pickle.dump(mm_tot_txt,pkl_mm_tot,-1)
pkl_daily_tot.close()
pkl_mm_tot.close()

# Always remember to close your connections!
sidc_main_r.close()
sidc_data_r.close()




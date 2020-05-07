# This code retrieves TESS light curves for an object given its TIC number.
# It will combine data for all existing sectors and perform a period search
# using a Lomb-Scargle periodogram.
# It all searches for the object in Gaia DR2.
# The output is a plot showing the periodogram, the Gaia CMD, and the phase-
# folded light to both the period and twice the period (useful for binary
# systems where the dominant peak is often an alias).

__version__ = '1.0'
__author__ = 'Ingrid Pelisoli'

# Importing relevant packages:

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.mast import Observations
from astroquery.gaia import Gaia
from astropy.io import fits
from astropy.io import ascii
from astropy.timeseries import LombScargle
from astropy.stats import sigma_clip
from astropy.io.votable import parse_single_table
from math import pi
import sys

# First we define the object name using the TIC

TIC = np.int(sys.argv[1])
obj_name = "TIC " + str(TIC)

# Output ascii light curve?
flag_lc = int(input("Would you like an ascii file of the processed light curve?\n0 = no, 1 = yes: "))
# Output ascii periodogram?
flag_ls = int(input("Would you like an ascii file of the Lomb-Scargle periodogram?\n0 = no, 1 = yes: "))
# Output ascii phase?
flag_ph = int(input("Would you like an ascii file of the phased data?\n0 = no, 1 = yes: "))
# Is the period actually 2*P?
flag_p2 = int(input("Would you like to multiply the period by two?\n"
                    "(useful for ellipsoidal variables and some eclipsing systems)\n"
                    "0 = no, 1 = yes: "))

# Then we check data at MAST
# WARNING! Even with this tiny search radius, sometimes there is still more
# than one object selected due to the huge TESS pixel size.
# I will implement something to take care of this in the future.

obsTable = Observations.query_criteria(dataproduct_type="timeseries",
                                       project="TESS",
                                       objectname=obj_name,
                                       proposal_pi = "Ricker, George",
                                       radius="0.00000001 deg")

# Download the light curves

data = Observations.get_product_list(obsTable)
download = Observations.download_products(data,productSubGroupDescription="LC")

infile = download[0][:]

print("I have found a total of " + str(len(infile)) + " light curve(s).")

# Download data for the first sector

crowdsap = []

with fits.open(infile[0]) as TESSdata:
    data=TESSdata[1].data
    BJD = np.array(data['TIME'])
    flux = np.array(data['PDCSAP_FLUX'])
    err_flux = np.array(data['PDCSAP_FLUX_ERR'])
    err_flux = err_flux / np.nanmean(flux)
    flux = flux / np.nanmean(flux)
    header=TESSdata[1].header
    crowdsap.append(header['CROWDSAP'])

# If there are more sectors, download data for the remaning sectors

if (len(infile) > 1):
    for i in range(1,len(infile)):
        with fits.open(infile[i]) as TESSdata:
            data=TESSdata[1].data
            BJD = np.append(BJD, np.array(data['TIME']))
            f = np.array(data['PDCSAP_FLUX'])
            ef = np.array(data['PDCSAP_FLUX_ERR'])
            flux = np.append(flux, f / np.nanmean(f))
            err_flux = np.append(err_flux, ef / np.nanmean(f))
            header=TESSdata[1].header
            crowdsap.append(header['CROWDSAP'])

# Data pre-processing: removing nan values

index = ~(np.isnan(BJD) | np.isnan(flux))

flux = flux[index]
err_flux = err_flux[index]
BJD = BJD[index]

# Data pre-processsing: sigma-clipping

filtered_data = sigma_clip(flux, sigma=5, maxiters=None)
index = ~(filtered_data.mask)

flux = flux[index]
err_flux = err_flux[index]
BJD = BJD[index]


t = (BJD - BJD[0])*24.0 #time in hours

if (flag_lc == 1):
    ascii.write([BJD, flux, err_flux], 'TIC%09d_lc.dat'%(TIC),
                names=['BJD','RelativeFlux','Error'], overwrite=True)

# Calculate input parameters for the periodogram

#calculates the Nyquist frequency that determines the upper limit in frequency
dt = [ t[i+1] - t[i-1] for i in range(1,len(t)-1)]
fmax = 1.0/np.median(dt)

#the lower limit is set by the duration of the light curve
fmin = 2.0/(max(t))

# Calculates the periodogram

ls = LombScargle(t, flux, err_flux)
# Oversampling by a factor of 10 to achieve frequency resolution
freq, power = ls.autopower(minimum_frequency=fmin, maximum_frequency=fmax,
                           samples_per_peak=10)

if (flag_ls == 1):
    ascii.write([1/freq, power], 'TIC%09d_ls.dat'%(TIC),
                names=['Period[h]','Power'], overwrite=True)


# Folds the data to the dominant peak

best_f = freq[np.argmax(power)]
period = 1.0/best_f #period from the LS periodogram

if (flag_p2 == 1):
    period = 2.0*period

phase = ((t - t[0]) / period) % 1.0 #phases the data

flux_phased = [flux for phase, flux in sorted(zip(phase, flux))]
err_flux_phased = [err_flux for phase, err_flux in sorted(zip(phase, err_flux))]
phase = np.sort(phase)

if (flag_ph == 1):
    ascii.write([phase, flux_phased, err_flux_phased], 'TIC%09d_phase.dat'%(TIC),
                names=['Phase','RelativeFlux','Error'], overwrite=True)


# Calculates a running average every 100 points for better visualisation

phase_avg = np.convolve(phase, np.ones((100,))/100, mode='valid')
flux_phased_avg = np.convolve(flux_phased, np.ones((100,))/100, mode='valid')
err_flux_phased_avg = np.convolve(err_flux_phased, np.ones((100,))/100, mode='valid')

# Fits the averaged phase, only for the original period.
# It won't look good for eclipsing binaries anyway

def sine_func(x, a, b):
    if (flag_p2 == 1):
        return 1.0+a*np.sin(4.*pi*x + b)
    else:
        return 1.0+a*np.sin(2.*pi*x + b)
params, params_covariance = optimize.curve_fit(sine_func, phase_avg,
                                               flux_phased_avg,
                                               p0=[np.mean(flux), 0.0])
if (flag_p2 == 1):
    y_fit = 1.0 + params[0] * np.sin(4.*pi*phase_avg + params[1])
else:
    y_fit = 1.0 + params[0] * np.sin(2.*pi*phase_avg + params[1])

# Can we find this thing in Gaia?

coord = SkyCoord(ra=obsTable[0][5], dec=obsTable[0][6],
                 unit=(u.degree, u.degree), frame='icrs')
radius = u.Quantity(5.0, u.arcsec)
q = Gaia.cone_search_async(coord, radius)
gaia = q.get_results()

# High proper motion objects are sometimes not found, so we might need a
# larger search radius
warning = (len(gaia) == 0)
if warning:
    radius = u.Quantity(10.0, u.arcsec)
    q = Gaia.cone_search_async(coord, radius)
    gaia = q.get_results()

# Check if more than one object was found
duplicated = (len(gaia) > 1)
if duplicated:
    id_list = np.array(gaia['source_id'])
    G_list = np.array(gaia['phot_g_mean_mag'])
    plx_list = np.array(gaia['parallax'])
    bprp_list = np.array(gaia['bp_rp'])
    brightest = np.argmin(G_list)
    gaia_id = id_list[brightest]
    MG = 5 + 5*np.log10(plx_list[brightest]/1000) + G_list[brightest]
    bprp = bprp_list[brightest]
else:
    gaia_id = gaia['source_id']
    MG = 5 + 5*np.log10(gaia['parallax']/1000) + gaia['phot_g_mean_mag']
    bprp = gaia['bp_rp']

    gaia_id = np.int(gaia_id)
    MG = np.float(MG)
    bprp = np.float(bprp)

# Writing log file

log = open('TIC%09d.log'%(TIC), "w")
log.write("TIC %09d\n\n"%(TIC))
if warning:
    log.write("Warning! No Gaia match within 5 arcsec. Increased search radius to 10 arcsec.\n")
if duplicated:
    log.write("Warning! More than one match in Gaia. Brightest match was chosen.\n")
log.write("Gaia DR2 source_id = %20d\n"%(gaia_id))
log.write("MG = %5.3f, bp_rp = %5.3f\n\n"%(MG, bprp))
log.write("Number of sectors: %2d\n"%(len(infile)))
log.write("CROWDSAP: %5.3f\n"%(np.mean(crowdsap)))
log.write("Best period = %9.5f hours, Amplitude =  %7.5f per cent"%(period, 100*abs(params[0])))
log.close()

table = parse_single_table("SampleC.vot")
data = table.array

s_MG = 5 + 5*np.log10(table.array['parallax']/1000) + table.array['phot_g_mean_mag']
s_bprp = table.array['bp_rp']


# Let's plot all this

fig = plt.figure(figsize=(18,21))
#fig.title('TIC %d'%(TIC))

plt.rcParams.update({'font.size': 22})

gridspec.GridSpec(3,2)

plt.subplot2grid((3,2), (0,0), colspan=1, rowspan=1)
plt.plot(1.0/freq, power, color ='k')
plt.xlim(min(1.0/freq),max(1.0/freq))
plt.axhline(ls.false_alarm_level(0.01),color='b')
#plt.axvspan(100., max(1.0/freq), alpha=0.5, color='red')
plt.title('TIC %d'%(TIC))
plt.xscale('log')
plt.xlabel('P [h]')
plt.ylabel('Power')

plt.subplot2grid((3,2), (0,1), colspan=1, rowspan=1)
plt.plot(s_bprp,s_MG,'.k', markersize=0.5, zorder=0)
plt.gca().invert_yaxis()
plt.title('$Gaia$ HR-diagram')
plt.plot(bprp,MG,'or',markersize=5.0,zorder=1)
plt.ylabel('$M_G$')
plt.xlabel('$G_{BP}-G_{RP}$')

plt.subplot2grid((3,2), (1,0), colspan=2, rowspan=1)
plt.ylabel('Relative flux')
plt.title('%s sector/s, P = %5.2f h'%(len(infile),period))
plt.xlim(0,2)
plt.errorbar(phase,flux_phased, fmt='.', color='0.5', markersize=0.75, elinewidth=0.5, zorder=0)
plt.plot(phase_avg,flux_phased_avg,'.k', zorder=1)
plt.plot(phase_avg,y_fit,'-r',zorder=2)
plt.errorbar(phase+1.0,flux_phased, fmt='.', color='0.5', markersize=0.75, elinewidth=0.5, zorder=0)
plt.plot(phase_avg+1.0,flux_phased_avg,'.k', zorder=1)
plt.plot(phase_avg+1.0,y_fit,'-r',zorder=2)

plt.subplot2grid((3,2), (2,0), colspan=2, rowspan=1)
plt.xlabel('Phase')
plt.ylabel('Relative flux')
plt.xlim(0,2)
#plt.errorbar(phase,flux_phased, fmt='.', color='0.5', markersize=0.75, elinewidth=0.5, zorder=0)
plt.plot(phase_avg,flux_phased_avg,'.k', zorder=1)
plt.plot(phase_avg,y_fit,'-r',zorder=2)
#plt.errorbar(phase+1.0,flux_phased, fmt='.', color='0.5', markersize=0.75, elinewidth=0.5, zorder=0)
plt.plot(phase_avg+1.0,flux_phased_avg,'.k', zorder=1)
plt.plot(phase_avg+1.0,y_fit,'-r',zorder=2)


fig.savefig('TIC%09d.png'%(TIC))

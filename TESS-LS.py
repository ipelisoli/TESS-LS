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
from astropy import wcs
from astropy.coordinates import SkyCoord, Distance, Angle
from astropy.time import Time
from astroquery.mast import Observations
from astroquery.gaia import Gaia
from astropy.io import fits
from astropy.io import ascii
from astropy.timeseries import LombScargle
from astropy.stats import sigma_clip
from astropy.io.votable import parse_single_table
from math import pi
import sys

# Defining useful functions

def periodogram(t, flux, flux_err):
    dt = [ t[i+1] - t[i-1] for i in range(1,len(t)-1)]
    fmax = 1.0/np.median(dt)
    fmin = 2.0/(max(t))
    ls = LombScargle(t, flux, flux_err)
    #Oversampling a factor of 10 to achieve frequency resolution
    freq, power = ls.autopower(minimum_frequency=fmin,
                               maximum_frequency=fmax,
                               samples_per_peak=10)
    best_f = freq[np.argmax(power)]
    period = 1.0/best_f #period from the LS periodogram
    fap_p = ls.false_alarm_probability(power.max())
    fap_001 = ls.false_alarm_level(0.01)
    return freq, power, period, fap_p, fap_001

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)
    #return np.convolve(x, np.ones((N,))/N, mode='valid')

def chi_sq(guess, x, y, err, factor):
    a, b = guess
    model = 1.0+a*np.sin(factor*2.0*pi*x + b)
    var = np.array(err)*np.array(err)
    chisq = np.sum((model - y) * (model - y)/var)
    return chisq

def phase_data(t, flux, flux_err, period, factor):
    period = factor*period
    phase = ((t - t[0]) / period) % 1.0
    flux_phased = [flux for phase, flux in sorted(zip(phase, flux))]
    flux_err_phased = [flux_err for phase, flux_err in sorted(zip(phase, flux_err))]
    phase = np.sort(phase)
    # Fit the data
    initial = np.array([np.mean(flux), 0.0])
    solution = optimize.minimize(chi_sq, initial,args=(phase,
                                                        flux_phased,
                                                        flux_err_phased,
                                                        factor))
    flux_fit = 1.0 + solution.x[0] * np.sin(factor*2.*np.pi*phase + solution.x[1])
    return phase, flux_phased, flux_err_phased, flux_fit, solution.x[0]

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
download_lc = Observations.download_products(data,productSubGroupDescription="LC")
infile = download_lc[0][:]

print("I have found a total of " + str(len(infile)) + " light curve(s).")

# Dowload target pixel file for plotting

tp_data = Observations.filter_products(data,productSubGroupDescription="TP")
tp_id = str(tp_data['obsID'][0]) # We only need one TP
download_tp = Observations.download_products(tp_id,productSubGroupDescription="TP")
tp = download_tp[0][0]
with fits.open(tp) as TPdata:
    # Create WCS object
    tp_wcs = wcs.WCS(naxis=2)
    tp_wcs.wcs.crpix = [TPdata[1].header['1CRPX4'], TPdata[1].header['2CRPX4']]
    tp_wcs.wcs.cdelt = [TPdata[1].header['1CDLT4'], TPdata[1].header['2CDLT4']]
    tp_wcs.wcs.crval = [TPdata[1].header['1CRVL4'], TPdata[1].header['2CRVL4']]
    tp_wcs.wcs.ctype = [TPdata[1].header['1CTYP4'], TPdata[1].header['2CTYP4']]

    data=TPdata[1].data
    flux_map = data['FLUX']
    flux_map = flux_map[0]

# Open data for the first sector

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

# If there are more sectors, open data for the remaning sectors

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
BJD_or = BJD
flux_or = flux

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

# Calculates the periodogram

freq, power, period, fap_p, fap_001 = periodogram(t, flux, err_flux)

if (flag_ls == 1):
    ascii.write([1/freq, power], 'TIC%09d_ls.dat'%(TIC),
                names=['Period[h]','Power'], overwrite=True)

# Folds the data to the dominant peak
phase, flux_phased, flux_err_phased, flux_fit, amp = phase_data(t, flux, err_flux, period, 1.0)

# Folds the data to twice the dominant peak
phase2, flux_phased2, flux_err_phased2, flux_fit2, amp2 = phase_data(t, flux, err_flux, period, 2.0)

if (flag_ph == 1):
    if (flag_p2 == 1):
        ascii.write([phase2, flux_phased2, flux_err_phased2], 'TIC%09d_phase.dat'%(TIC),
                     names=['Phase','RelativeFlux','Error'], overwrite=True)
    else:
        ascii.write([phase, flux_phased, flux_err_phased], 'TIC%09d_phase.dat'%(TIC),
                     names=['Phase','RelativeFlux','Error'], overwrite=True)

# Can we find this thing in Gaia?

# First do a large search using 30 arcsec

coord = SkyCoord(ra=obsTable[0][5], dec=obsTable[0][6],
                 unit=(u.degree, u.degree), frame='icrs')
radius = u.Quantity(30.0, u.arcsec)
q = Gaia.cone_search_async(coord, radius)
gaia = q.get_results()
gaia = gaia[ np.nan_to_num(gaia['parallax']) > 0 ]
warning = (len(gaia) == 0)

# Then propagate the Gaia coordinates to 2000, and find the best match to the
# input coordinates
if not warning:
    ra2015 = np.array(gaia['ra']) * u.deg
    dec2015 = np.array(gaia['dec']) * u.deg
    parallax = np.array(gaia['parallax']) * u.mas
    pmra = np.array(gaia['pmra']) * u.mas/u.yr
    pmdec = np.array(gaia['pmdec']) * u.mas/u.yr
    c2015 = SkyCoord(ra=ra2015, dec=dec2015,
                     distance=Distance(parallax=parallax, allow_negative=True),
                     pm_ra_cosdec=pmra, pm_dec=pmdec,
                     obstime=Time(2015.5, format='decimalyear'))
    c2000 = c2015.apply_space_motion(dt=-15.5 * u.year)

    idx, sep, _ = coord.match_to_catalog_sky(c2000)

    # All objects
    id_all = gaia['source_id']
    plx_all = np.array(gaia['parallax'])
    g_all = np.array(gaia['phot_g_mean_mag'])
    MG_all = 5 + 5*np.log10(plx_all/1000) + g_all
    bprp_all = np.array(gaia['bp_rp'])

    id_all = np.array(id_all)
    MG_all = np.array(MG_all)
    bprp_all = np.array(bprp_all)

    # The best match object
    best = gaia[idx]
    gaia_id = best['source_id']

    MG = 5 + 5*np.log10(best['parallax']/1000) + best['phot_g_mean_mag']
    bprp = best['bp_rp']

    gaia_id = np.int(gaia_id)
    MG = np.float(MG)
    bprp = np.float(bprp)

    # Coordinates for plotting
    radecs = np.vstack([c2000.ra, c2000.dec]).T
    coords = tp_wcs.all_world2pix(radecs, 0)
    sizes = 10000.0 / 2**(g_all/2)

# Writing log file

log = open('TIC%09d.log'%(TIC), "w")
log.write("TIC %09d\n\n"%(TIC))
if warning:
    log.write("Warning! No object with measured parallax within 30 arcsec.\n")
log.write("Gaia DR2 source_id = %20d\n"%(gaia_id))
log.write("MG = %5.3f, bp_rp = %5.3f\n\n"%(MG, bprp))
log.write("Number of sectors: %2d\n"%(len(infile)))
log.write("CROWDSAP: %5.3f\n"%(np.mean(crowdsap)))
if (flag_p2 == 1):
    log.write("Period = %9.5f hours, Amplitude =  %7.5f per cent\n\n"%(2.0*period, 100*abs(amp2)))
else:
    log.write("Best period = %9.5f hours, Amplitude =  %7.5f per cent\n\n"%(period, 100*abs(amp2)))
if (len(gaia)>0):
    log.write("Other matches within 30 arcsec:\n")
    log.write("source_id            MG    bp_rp\n")
    for i in range(0, len(gaia)):
        if i != idx:
            log.write("%20d %5.3f %5.3f\n"%(id_all[i], MG_all[i], bprp_all[i]))

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
plt.title('TIC %d'%(TIC))
plt.imshow(flux_map, interpolation='nearest')
plt.scatter(coords[:, 0], coords[:, 1], c='firebrick', alpha=0.5, edgecolors='r', s=sizes)
plt.scatter(coords[:, 0], coords[:, 1], c='None', edgecolors='r', s=sizes)
plt.ylabel('Pixel count')
plt.xlabel('Pixel count')
plt.xlim(0,10)
plt.ylim(10,0)

plt.subplot2grid((3,2), (0,1), colspan=1, rowspan=1)
plt.scatter(s_bprp,s_MG,c='0.75', s=0.5, zorder=0)
plt.scatter(bprp_all,MG_all,marker='s',c='b', s=10, zorder=1)
plt.gca().invert_yaxis()
plt.title('$Gaia$ HR-diagram')
plt.plot(bprp,MG,'or',markersize=10,zorder=2)
plt.ylabel('$M_G$')
plt.xlabel('$G_{BP}-G_{RP}$')

plt.subplot2grid((3,2), (1,0), colspan=2, rowspan=1)
plt.title('%s sector/s'%len(infile))
plt.xlabel("BJD - 2457000")
plt.ylabel('Relative flux')
plt.xlim(np.min(BJD), np.max(BJD))
plt.scatter(BJD_or, flux_or, c='0.25', zorder=1, s = 0.5)
plt.scatter(BJD, flux, c='k', zorder=1, s = 0.5)

plt.subplot2grid((3,2), (2,0), colspan=1, rowspan=1)
plt.title("Period = %5.2f h"%period)
plt.plot(1.0/freq, power, color ='k')
plt.xlim(min(1.0/freq),max(1.0/freq))
plt.axhline(fap_001, color='b')
#plt.axvspan(100., max(1.0/freq), alpha=0.5, color='red')
plt.xscale('log')
plt.xlabel('P [h]')
plt.ylabel('Power')

plt.subplot2grid((3,2), (2,1), colspan=2, rowspan=1)
plt.xlabel('Phase')
plt.ylabel('Relative flux')
plt.xlim(0,2)
plt.errorbar(phase,flux_phased, fmt='.', color='0.5', markersize=0.75, elinewidth=0.5, zorder=0)
plt.plot(running_mean(phase,100), running_mean(flux_phased,100),'.k', zorder=1)
plt.plot(phase, flux_fit, '-r', zorder=2)
plt.errorbar(phase+1.0,flux_phased, fmt='.', color='0.5', markersize=0.75, elinewidth=0.5, zorder=0)
plt.plot(running_mean(phase,100)+1.0, running_mean(flux_phased,100),'.k', zorder=1)
plt.plot(phase + 1.0, flux_fit,'-r',zorder=2)

plt.tight_layout()

fig.savefig('TIC%09d.png'%(TIC))

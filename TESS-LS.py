# This code retrieves TESS light curves for an object given its TIC number.
# It will combine data for all existing sectors and perform a period search
# using a Lomb-Scargle periodogram.
# It all searches for the object in Gaia DR2.
# The output is a plot showing the periodogram, the Gaia CMD, and the phase-
# folded light to both the period and twice the period (useful for binary
# systems where the dominant peak is often an alias).

__version__ = '2.0'
__author__ = 'Ingrid Pelisoli'

#####  IMPORTING PACKAGES  ######

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy.io import ascii
from astropy.io import fits
from astropy.io.votable import parse_single_table
from astroquery.mast import Observations
from astroquery.gaia import Gaia
from astropy import wcs
from astropy.coordinates import SkyCoord, Distance, Angle
from astropy.time import Time
import astropy.units as u
import sys
import TESSutils as tul

################################

###  FUNCTION FOR PLOTTING  ####

def make_plot(f, pow, fap, bjd0, flux0, bjd, flux, phi,
              flux_phi, fit, phi2, flux_phi2, fit2, period, crowd, ns):

    fig = plt.figure(figsize=(24,15))

    plt.rcParams.update({'font.size': 22})

    gridspec.GridSpec(6,10)

    plt.subplot2grid((6,10), (0,0), colspan=2, rowspan=2)
    plt.title('TIC %d'%(TIC))
    plt.xlim(0,10)
    plt.ylim(10,0)
    plt.imshow(flux_map, interpolation='nearest')
    if not warning:
        plt.scatter(coords[:, 0], coords[:, 1], c='firebrick', alpha=0.5, edgecolors='r', s=sizes)
        plt.scatter(coords[:, 0], coords[:, 1], c='None', edgecolors='r', s=sizes)
    plt.text(0.1, 9.9, 'crowdsap = %4.2f' % np.mean(crowd), color='w')
    plt.ylabel('Pixel count')
    plt.xlabel('Pixel count')

    plt.subplot2grid((6,10), (0,2), colspan=2, rowspan=2)
    plt.scatter(s_bprp, s_MG, c='0.75', s=0.5, zorder=0)
    if (len(gaia)>1):
        plt.scatter(bprp_all, MG_all, marker='s',c='b', s=10, zorder=1)
    plt.gca().invert_yaxis()
    plt.title('$Gaia$ HR-diagram')
    if not warning:
        plt.plot(bprp, MG, 'or',markersize=10,zorder=2)
    plt.ylabel('$M_G$')
    plt.xlabel('$G_{BP}-G_{RP}$')

    plt.subplot2grid((6,10), (2,0), colspan=4, rowspan=2)
    plt.title("Period = %5.2f h"%period)
    plt.plot(1.0/f, pow, color ='k')
    plt.xlim(min(1.0/f), max(1.0/f))
    plt.axhline(fap, color='b')
    plt.axvline(period, color='r', ls='--', zorder=0)
    #plt.axvspan(100., max(1.0/freq), alpha=0.5, color='red')
    plt.xscale('log')
    plt.xlabel('P [h]')
    plt.ylabel('Power')

    plt.subplot2grid((6,10), (4,0), colspan=4, rowspan=2)
    plt.title('%s sector/s'% ns)
    plt.xlabel("BJD - 2457000")
    plt.ylabel('Relative flux')
    plt.xlim(np.min(bjd), np.max(bjd))
    plt.scatter(bjd0, flux0, c='0.25', zorder=1, s = 0.5)
    plt.scatter(bjd, flux, c='k', zorder=1, s = 0.5)

    phi_avg = tul.avg_array(phi,50)
    fphi_avg = tul.avg_array(flux_phi,50)

    plt.subplot2grid((6,10), (0,4), colspan=6, rowspan=3)
    plt.title('Phased to dominant peak')
    plt.xlabel('Phase')
    plt.ylabel('Relative flux')
    plt.xlim(0,2)
    #plt.errorbar(phi, flux_phi, fmt='.', color='0.5', markersize=0.75, elinewidth=0.5, zorder=0)
    plt.scatter(phi_avg, fphi_avg, marker='.', color='0.5', zorder=0)
    plt.plot(tul.running_mean(phi_avg,15), tul.running_mean(fphi_avg,15),'.k', zorder=1)
    plt.plot(phi, fit, 'r--', lw = 3, zorder=2)
    #plt.errorbar(phi+1.0, flux_phi, fmt='.', color='0.5', markersize=0.75, elinewidth=0.5, zorder=0)
    plt.scatter(phi_avg+1.0, fphi_avg, marker='.', color='0.5', zorder=0)
    plt.plot(tul.running_mean(phi_avg,15)+1.0, tul.running_mean(fphi_avg,15),'.k', zorder=1)
    plt.plot(phi + 1.0, fit,'r--', lw = 3, zorder=2)

    phi_avg = tul.avg_array(phi2,50)
    fphi_avg = tul.avg_array(flux_phi2,50)

    plt.subplot2grid((6,10), (3,4), colspan=6, rowspan=3)
    plt.title('Phased to twice the peak')
    plt.xlabel('Phase')
    plt.ylabel('Relative flux')
    plt.xlim(0,2)
    #plt.errorbar(phi2, flux_phi2, fmt='.', color='0.5', markersize=0.75, elinewidth=0.5, zorder=0)
    plt.scatter(phi_avg, fphi_avg, marker='.', color='0.5', zorder=0)
    plt.plot(tul.running_mean(phi_avg,15), tul.running_mean(fphi_avg,15),'.k', zorder=1)
    plt.plot(phi2, fit2, 'r--', lw = 3, zorder=2)
    #plt.errorbar(phi2+1.0, flux_phi2, fmt='.', color='0.5', markersize=0.75, elinewidth=0.5, zorder=0)
    plt.scatter(phi_avg+1.0, fphi_avg, marker='.', color='0.5', zorder=0)
    plt.plot(tul.running_mean(phi_avg,15)+1.0, tul.running_mean(fphi_avg,15),'.k', zorder=1)
    plt.plot(phi2 + 1.0, fit2, 'r--', lw = 3, zorder=2)

    plt.tight_layout()

    return fig

################################

#########  USER INPUT  #########

# Define the object name using the TIC

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

################################

#######  DOWNLOAD DATA  ########

# Searching for data at MAST

obsTable = Observations.query_criteria(dataproduct_type="timeseries",
                                       project="TESS",
                                       target_name=TIC)

# Download the 2-minute cadence light curves

data = Observations.get_product_list(obsTable)
download_lc = Observations.download_products(data, productSubGroupDescription="LC")
infile = download_lc[0][:]
n_slow = len(infile)

print("I have found a total of " + str(len(infile)) + " 2-min light curve(s).")

# Download the 20-second cadence light curves

download_fast_lc = Observations.download_products(data,
                                                  productSubGroupDescription="FAST-LC")

if download_fast_lc is None:
    print("I have found no 20-sec light curves.")
    fast = False
else:
    infile_fast = download_fast_lc[0][:]
    n_fast = len(infile_fast)
    print("I have found a total of " + str(len(download_fast_lc[0][:])) + " 20-sec light curve(s).")
    fast = True

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
    # This does't seem to always work:
    flux_map = np.flip(flux_map, axis=0)
    flux_map = np.flip(flux_map, axis=1)
    # Figuring this out is on top of the to-do list.

################################

#########  GAIA MATCH  #########

# First do a large search using 11 pixels

coord = SkyCoord(ra=obsTable[0][5], dec=obsTable[0][6],
                 unit=(u.degree, u.degree), frame='icrs')
radius = u.Quantity(105.0, u.arcsec)
q = Gaia.cone_search_async(coord, radius)
gaia = q.get_results()
# Select only those brighter than 18.
gaia = gaia[gaia['phot_g_mean_mag'] < 18.]
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
    g_all = np.array(gaia['phot_g_mean_mag'])
    MG_all = np.array(MG_all)
    bprp_all = np.array(bprp_all)

    # The best match object
    best = gaia[idx]
    gaia_id = best['source_id']

    MG = 5 + 5*np.log10(best['parallax']/1000) + best['phot_g_mean_mag']
    bprp = best['bp_rp']

    gaia_id = np.int(gaia_id)
    G = np.float(best['phot_g_mean_mag'])
    MG = np.float(MG)
    bprp = np.float(bprp)

    # Coordinates for plotting
    radecs = np.vstack([c2000[idx].ra, c2000[idx].dec]).T
    coords = tp_wcs.all_world2pix(radecs, 0)
    sizes = 8000.0 / 2**(g_all[idx]/2)

# Reference sample

table = parse_single_table("SampleC.vot")
data = table.array

s_MG = 5 + 5*np.log10(table.array['parallax']/1000) + table.array['phot_g_mean_mag']
s_bprp = table.array['bp_rp']

################################

#######  2-MINUTE DATA  ########

slow_lc = tul.LCdata(TIC)

slow_lc.read_data(infile)
BJD_or = slow_lc.bjd
flux_or = slow_lc.flux

slow_lc.clean_data()
if (flag_lc == 1):
    ascii.write([slow_lc.bjd, slow_lc.flux, slow_lc.flux_err],
                'TIC%09d_lc.dat'%(TIC), names=['BJD','RelativeFlux','Error'],
                overwrite=True)

# Calculates the periodogram
slow_lc.periodogram()
if (flag_ls == 1):
    ascii.write([1/slow_lc.freq, slow_lc.power], 'TIC%09d_ls.dat'%(TIC),
                names=['Period[h]','Power'], overwrite=True)

# Folds the data to the dominant peak
slow_lc.phase_data(1.0)
phase = slow_lc.phase
flux_phased = slow_lc.flux_phased
flux_err_phased = slow_lc.flux_err_phased
flux_fit = slow_lc.flux_fit
amp = slow_lc.amp
# Folds the data to twice the dominant peak
slow_lc.phase_data(2.0)
phase2 = slow_lc.phase
flux_phased2 = slow_lc.flux_phased
flux_err_phased2 = slow_lc.flux_err_phased
flux_fit2 = slow_lc.flux_fit
amp2 = slow_lc.amp

if (flag_ph == 1):
    if (flag_p2 == 1):
        ascii.write([phase2, flux_phased2, flux_err_phased2], 'TIC%09d_phase.dat'%(TIC),
                     names=['Phase','RelativeFlux','Error'], overwrite=True)
    else:
        ascii.write([phase, flux_phased, flux_err_phased], 'TIC%09d_phase.dat'%(TIC),
                     names=['Phase','RelativeFlux','Error'], overwrite=True)

plot = make_plot(slow_lc.freq, slow_lc.power, slow_lc.fap_001, BJD_or, flux_or,
                 slow_lc.bjd, slow_lc.flux, phase, flux_phased, flux_fit,
                 phase2, flux_phased2, flux_fit2, slow_lc.period, slow_lc.crowdsap,
                 n_slow)

plot.savefig('TIC%09d.png'%(TIC))

################################

######  20-SECOND DATA  ########

if fast:

    fast_lc = tul.LCdata(TIC)

    fast_lc.read_data(infile_fast)
    BJD_or = fast_lc.bjd
    flux_or = fast_lc.flux

    fast_lc.clean_data()
    if (flag_lc == 1):
        ascii.write([fast_lc.bjd, fast_lc.flux, fast_lc.flux_err],
                    'TIC%09d_lc_fast.dat'%(TIC), names=['BJD','RelativeFlux','Error'],
                    overwrite=True)

    # Calculates the periodogram
    fast_lc.periodogram()
    if (flag_ls == 1):
        ascii.write([1/fast_lc.freq, fast_lc.power], 'TIC%09d_ls_fast.dat'%(TIC),
                    names=['Period[h]','Power'], overwrite=True)

    # Folds the data to the dominant peak
    fast_lc.phase_data(1.0)
    phase = fast_lc.phase
    flux_phased = fast_lc.flux_phased
    flux_err_phased = fast_lc.flux_err_phased
    flux_fit = fast_lc.flux_fit
    fast_amp = fast_lc.amp
    # Folds the data to twice the dominant peak
    fast_lc.phase_data(2.0)
    phase2 = fast_lc.phase
    flux_phased2 = fast_lc.flux_phased
    flux_err_phased2 = fast_lc.flux_err_phased
    flux_fit2 = fast_lc.flux_fit
    fast_amp2 = fast_lc.amp

    if (flag_ph == 1):
        if (flag_p2 == 1):
            ascii.write([phase2, flux_phased2, flux_err_phased2], 'TIC%09d_phase_fast.dat'%(TIC),
                        names=['Phase','RelativeFlux','Error'], overwrite=True)
        else:
            ascii.write([phase, flux_phased, flux_err_phased], 'TIC%09d_phase_fast.dat'%(TIC),
                        names=['Phase','RelativeFlux','Error'], overwrite=True)

    plot_fast = make_plot(fast_lc.freq, fast_lc.power, fast_lc.fap_001, BJD_or, flux_or,
                          fast_lc.bjd, fast_lc.flux, phase, flux_phased, flux_fit,
                          phase2, flux_phased2, flux_fit2, fast_lc.period, fast_lc.crowdsap,
                          n_fast)

    plot_fast.savefig('TIC%09d_fast.png'%(TIC))

################################


#########  WRITE LOG  ##########

log = open('TIC%09d.log'%(TIC), "w")
log.write("TIC %09d\n\n"%(TIC))
if warning:
    log.write("Warning! No object with measured parallax within 30 arcsec.\n")
else:
    log.write("Gaia DR2 source_id = %20d\n"%(gaia_id))
    log.write("G = %6.3f, MG = %6.3f, bp_rp = %6.3f\n\n"%(G, MG, bprp))

log.write("2-minute cadence data\n")
log.write("Number of sectors: %2d\n"%(len(infile)))
log.write("CROWDSAP: %5.3f\n"%(np.mean(slow_lc.crowdsap)))

if (flag_p2 == 1):
    log.write("Period = %9.5f hours, Amplitude =  %7.5f per cent\n"%(2.0*slow_lc.period, 100*abs(amp2)))
else:
    log.write("Best period = %9.5f hours, Amplitude =  %7.5f per cent\n"%(slow_lc.period, 100*abs(amp)))
log.write("FAP = %7.5e\n\n"%(slow_lc.fap_p))

if fast:
    log.write("20-second cadence data\n")
    log.write("Number of sectors: %2d\n"%(len(infile_fast)))
    log.write("CROWDSAP: %5.3f\n"%(np.mean(fast_lc.crowdsap)))
    if (flag_p2 == 1):
        log.write("Period = %9.5f hours, Amplitude =  %7.5f per cent\n"%(2.0*fast_lc.period, 100*abs(fast_amp2)))
    else:
        log.write("Best period = %9.5f hours, Amplitude =  %7.5f per cent\n"%(fast_lc.period, 100*abs(fast_amp)))
    log.write("FAP = %7.5e\n\n"%(fast_lc.fap_p))
else:
    log.write("No fast-candence data available.\n\n")

if (len(gaia)>0):
    log.write("Other G < 18. matches within 5 pixels:\n")
    log.write("source_id            G       MG    bp_rp\n")
    for i in range(0, len(gaia)):
        if i != idx:
            log.write("%20d %6.3f %6.3f %6.3f\n"%(id_all[i], g_all[i], MG_all[i], bprp_all[i]))

log.close()


################################

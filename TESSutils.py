import numpy as np
from scipy import optimize
from astropy.io import fits
from astropy.timeseries import LombScargle
from astropy.stats import sigma_clip

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
    model = 1.0+a*np.sin(factor*2.0*np.pi*x + b)
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
    solution = optimize.minimize(chi_sq, initial,args=(running_mean(phase,100),
                                                       running_mean(flux_phased,100),
                                                       running_mean(flux_err_phased,100),
                                                        factor))
    flux_fit = 1.0 + solution.x[0] * np.sin(factor*2.*np.pi*phase + solution.x[1])
    return phase, flux_phased, flux_err_phased, flux_fit, solution.x[0]

def read_data(list):
    # Open data for the first sector

    crowdsap = []

    with fits.open(list[0]) as TESSdata:
        data=TESSdata[1].data
        BJD = np.array(data['TIME'])
        flux = np.array(data['PDCSAP_FLUX'])
        err_flux = np.array(data['PDCSAP_FLUX_ERR'])
        err_flux = err_flux / np.nanmean(flux)
        flux = flux / np.nanmean(flux)
        header=TESSdata[1].header
        crowdsap.append(header['CROWDSAP'])

        # If there are more sectors, open data for the remaning sectors

        if (len(list) > 1):
            for i in range(1,len(list)):
                with fits.open(list[i]) as TESSdata:
                    data=TESSdata[1].data
                    BJD = np.append(BJD, np.array(data['TIME']))
                    f = np.array(data['PDCSAP_FLUX'])
                    ef = np.array(data['PDCSAP_FLUX_ERR'])
                    flux = np.append(flux, f / np.nanmean(f))
                    err_flux = np.append(err_flux, ef / np.nanmean(f))
                    header=TESSdata[1].header
                    crowdsap.append(header['CROWDSAP'])

    err_flux = err_flux / np.nanmean(flux)
    flux = flux / np.nanmean(flux)

    return BJD, flux, err_flux, crowdsap

def clean_data(t, f, err_f):
    # removing nan values
    index = ~(np.isnan(t) | np.isnan(f))

    f = f[index]
    err_f = err_f[index]
    t = t[index]

    # sigma-clipping
    filtered_data = sigma_clip(f, sigma=5, maxiters=None)
    index = ~(filtered_data.mask)

    f = f[index]
    err_f = err_f[index]
    t = t[index]

    return t, f, err_f

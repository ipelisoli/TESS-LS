import numpy as np
from scipy import optimize
from astropy.io import fits
from astropy.timeseries import LombScargle
from astropy.stats import sigma_clip

class LCdata:

    def __init__(self, tic):
        self.tic = tic

        self.bjd = []
        self.t = []
        self.flux = []
        self.flux_err = []
        self.crowdsap = []

        self.freq = []
        self.power = []
        self.period = []
        self.fap_p = []
        self.fap_001 = []

        self.phase = []
        self.flux_phased = []
        self.flux_err_phased = []
        self.flux_fit = []
        self.amp = []

    def read_data(self, list):
        crowdsap = []

        # Open data for the first sector
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

        self.bjd = np.array(BJD)
        self.flux = np.array(flux)
        self.flux_err = np.array(err_flux)
        self.crowdsap = np.array(crowdsap)

    def clean_data(self):

        # removing nan values
        index = ~(np.isnan(self.bjd) | np.isnan(self.flux))

        self.flux = self.flux[index]
        self.flux_err = self.flux_err[index]
        self.bjd = self.bjd[index]

        # sigma-clipping
        filtered_data = sigma_clip(self.flux, sigma=5, maxiters=None)
        index = ~(filtered_data.mask)

        self.flux = self.flux[index]
        self.flux_err = self.flux_err[index]
        self.bjd = self.bjd[index]
        self.t = (self.bjd - self.bjd[0])*24.0

        # renormalising
        self.flux_err = self.flux_err/np.mean(self.flux)
        self.flux = self.flux/np.mean(self.flux)
        
    def periodogram(self):
        dt = [ self.t[i+1] - self.t[i-1] for i in range(1,len(self.t)-1)]
        fmax = 1.0/np.median(dt)
        fmin = 2.0/(max(self.t))
        ls = LombScargle(self.t, self.flux, self.flux_err)
        #Oversampling a factor of 10 to achieve frequency resolution
        freq, power = ls.autopower(minimum_frequency=fmin,
                                   maximum_frequency=fmax,
                                   samples_per_peak=10)
        # Find the dominant peak
        best_f = freq[np.argmax(power)]
        period = 1.0/best_f #period from the LS periodogram
        fap_p = ls.false_alarm_probability(power.max())
        fap_001 = ls.false_alarm_level(0.01)
        if (period > 30.0):
            # Long periods are often spurious, search for a shorter minimum one

            # Calculates treshold using a running median every 2000 points
            mean_freq = avg_array(freq,2000)
            mean_power = avg_array(power,2000)
            treshold = np.interp(freq, mean_freq, mean_power)
            # Finds the period looking for the local maximum
            max_loc = np.argmax(power/treshold)
            best_f = freq[max_loc]
            period = 1.0/best_f
            fap_p = ls.false_alarm_probability(power[max_loc])
            fap_001 = ls.false_alarm_level(0.01)

        self.freq = np.array(freq)
        self.power = np.array(power)
        self.period = period
        self.fap_p = fap_p
        self.fap_001 = fap_001

    def phase_data(self, factor):
        period = factor*self.period
        phase = ((self.t - self.t[0]) / period) % 1.0
        f = self.flux
        ef = self.flux_err
        flux_phased = [f for phase, f in sorted(zip(phase, f))]
        flux_err_phased = [ef for phase, ef in sorted(zip(phase, ef))]
        phase = np.sort(phase)
        # Fit the data
        initial = np.array([np.mean(self.flux), 0.0])
        solution = optimize.minimize(chi_sq, initial, args=(running_mean(phase,100),
                                                            running_mean(flux_phased,100),
                                                            running_mean(flux_err_phased,100),
                                                            factor))
        flux_fit = 1.0 + solution.x[0] * np.sin(factor*2.*np.pi*phase + solution.x[1])

        self.phase = np.array(phase)
        self.flux_phased = np.array(flux_phased)
        self.flux_err_phased = np.array(flux_err_phased)
        self.flux_fit = np.array(flux_fit)
        self.amp = solution.x[0]

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)
    #return np.convolve(x, np.ones((N,))/N, mode='valid')

def avg_array(ar, n):
    ar2 = np.nanmedian(np.pad(ar, (0, n - ar.size%n), mode='constant', constant_values=np.NaN).reshape(-1, n), axis=1)
    return ar2

def chi_sq(guess, x, y, err, factor):
    a, b = guess
    model = 1.0+a*np.sin(factor*2.0*np.pi*x + b)
    var = np.array(err)*np.array(err)
    chisq = np.sum((model - y) * (model - y)/var)
    return chisq

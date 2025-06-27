import glob
import numpy as np
import warnings
import sys
import os
import argparse
import oneftools
from os.path import exists as file_exists
from astropy.io import fits
from jwst.clean_flicker_noise import CleanFlickerNoiseStep
from jwst.ramp_fitting import RampFitStep

parser = argparse.ArgumentParser(description = 'Optimal 1/f correction parameter value selection ' \
                                 'algorithm. \d Input file is an _uncal.fits file', 
                                 add_help = True)
parser.add_argument('fitsfile', type=str)
args = parser.parse_args()

infits = args.fitsfile
ind = infits.find('_uncal.fits')
if ind < 0:
    print('Input file must be an uncalibrated _uncal.fits file')
    sys.exit()
fileroot = infits[:ind]
# First run calwebb_detector1 with default parameters
# for clean_flicker_noise
hduin = fits.open(infits)
if len(hduin) < 7:
    # Set up logfile for calwebb_detector1
    # and run with initial clean_flicker_noise parameters
    cfgfile = open('defpipe-log.cfg','w')
    lines = ['[*]','handler = file:default_one_f.log', 'level = INFO']
    cfgfile.writelines('%s\n' % l for l in lines)
    cfgfile.close()
    result = oneftools.rundet1_default_1f(infits, configfile='defpipe-log.cfg')

hduin.close()
ratefile = fileroot + '_rate.fits'
if not file_exists(ratefile):
    print('{} not found. Need to exit.'.format(ratefile))
    sys.exit()

maskfile = ratefile.replace('_rate.fits', '_mask.fits')
if not file_exists(maskfile):
    print('{} not found. Need to exit.'.format(maskfile))
    sys.exit()

hdr0 = fits.getheader(ratefile, ext=0)
xsize, ysize = (hdr0['SUBSIZE1'], hdr0['SUBSIZE2'])
# Note: This script is only relevant for imaging (and WFSS), which are full-frame.
if np.min([xsize, ysize]) < 2048:
    print('{} is not a full-frame exposure. Need to exit.'.format(ratefile))
    sys.exit()

flatted_bkg, maskfrac = oneftools.make_bkg_from_rate(ratefile, maskfile)
# average row and column from middle quarter of columns/rows
averx = oneftools.myround(flatted_bkg.shape[1]/4).astype(int)
avery = oneftools.myround(flatted_bkg.shape[0]/4).astype(int)
xmid, ymid = (flatted_bkg.shape[1]/2, flatted_bkg.shape[0]/2)
aver_row = oneftools.averrow(flatted_bkg, ymid, avery)
aver_col = oneftools.avercol(flatted_bkg, xmid, averx)
# Get stats on the average rows and columns
nx, ny = (flatted_bkg.shape[1], flatted_bkg.shape[0])
x = np.arange(nx)
y = np.arange(ny)
colstatsarr = oneftools.one_f_colstats(y, aver_col)
rowstatsarr = oneftools.one_f_rowstats(x, aver_row)
slopestats = oneftools.slopestats(colstatsarr[0], rowstatsarr[0])
# Also create an average row for the upper channel (assuming full-frame)
toprow = oneftools.myround(flatted_bkg.shape[0]*7/8).astype(int)
topaver = oneftools.myround(flatted_bkg.shape[0]/12).astype(int)
aver_row_channel1 = oneftools.averrow(flatted_bkg, toprow, topaver)
# Stats to get stddev around fit to average row in upper channel
ch1rowstatsarr = oneftools.one_f_rowstats(x, aver_row_channel1)
# Get max. channel-to-channel offset and overall mean sky level
max_offset, meansky = oneftools.max_channeloffset(aver_col)
by_channel_choice, bkgmethod = oneftools.one_f_parvalues(max_offset, maskfrac, slopestats[0],
                                                         slopestats[1], ch1rowstatsarr[2])
# If necessary,
# re-run clean_flicker_noise and ramp_fitting with the optimal parameters
#
if (bkgmethod == 'model') or by_channel_choice:
    jumpfile = ratefile.replace('_rate.fits', '_jump.fits')
    if not file_exists(jumpfile):
        print('Jumpfile {} not found. Need to exit.'.format(jumpfile))
        print('For the record:')
        print('background_method = {} and fit_by_channel = {}'.format(bkgmethod,
                                                                      by_channel_choice))
        sys.exit()
    cfgfile = open('finaldet1-log.cfg','w')
    lines = ['[*]','handler = file:final_detector1_one_f.log', 'level = INFO']
    cfgfile.writelines('%s\n' % l for l in lines)
    cfgfile.close()
    onefresult = CleanFlickerNoiseStep.call(jumpfile, skip=False,
                                            apply_flat_field=True,
                                            background_method=bkgmethod,
                                            fit_by_channel=by_channel_choice,
                                            logcfg='finaldet1-log.cfg')
    result = RampFitStep.call(onefresult, save_results=True,
                              logcfg='finaldet1-log.cfg')
    os.rename(ratefile.replace('_rate.fits','_0_rampfitstep.fits'), ratefile)
    os.rename(ratefile.replace('_rate.fits','_1_rampfitstep.fits'),
              ratefile.replace('_rate.fits','rateints.fits'))
#
# Now one can continue with stage 2 pipeline processing of resulting _rate and/or _rateints file
#

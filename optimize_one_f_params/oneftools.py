import glob
import numpy as np
import warnings
import sys
from astropy.io import ascii,fits
from astropy.table import Table
from astropy.stats import sigma_clipped_stats, SigmaClip
from photutils.background import Background2D, MedianBackground, MeanBackground
from jwst.pipeline import Detector1Pipeline
import crds.rmap as rmap

def get_flatfile(infits):
    CRDS_PATH = '/grp/crds/jwst/references/jwst/'
    hdulist = fits.open(infits)
    im = hdulist[1].data
    hdr0 = hdulist[0].header
    pmap = hdr0['CRDS_CTX']
    pmap_items = rmap.get_cached_mapping(pmap)
    for item in pmap_items.mapping_names():
        if ('niriss' in item) and ('imap' in item):
            imap = item
    nis = rmap.get_cached_mapping(imap)
    hdr = nis.get_minimum_header(infits)
    flatfile = nis.get_best_references(hdr)['flat']
    flatfilepath = CRDS_PATH + flatfile
    #flatimg = fits.getdata(CRDS_PATH + flatfile, ext=1)
    return flatfilepath


def bkg2d(fitsfile, boxsize=32, n_sigma=3, makemask=False, bkgmeth='median'):
    # This function performs Background2D on the input image after masking out bad pixels
    # and pixels with value < and > 3 sigma from the clipped median or mode.
    # Outputs the files as <rootname>_bkg.fits.
    findex = fitsfile.find('.fits')
    froot = fitsfile[:findex]
    bkgfile = froot + '_bkg.fits'
    image = (fits.open(fitsfile))[1].data
    hdr = (fits.open(fitsfile))[1].header
    dq = (fits.open(fitsfile))[3].data
    # mask of bad pixels from the DQ array
    mask = np.zeros(image.shape).astype(int)
    bady,badx = np.where((np.bitwise_and(dq, dqflags.group['DO_NOT_USE']) == 1))
    mask[bady,badx] = 1
    # Add pixels > n_sigma * sigma away from mode of background to mask
    gooddata = image[np.where(mask == 0)]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Input data contains invalid values")
        mn, median, std = stats.sigma_clipped_stats(gooddata, sigma=n_sigma, maxiters=10)
    thres = median
    if bkgmeth.lower() == 'mode': 
        thres = 3*median-2*mn
    lowpixels = np.where(gooddata < thres)
    bgpixels = np.concatenate(((gooddata[lowpixels]-thres),(thres-gooddata[lowpixels])))
    mask[np.where(image > (thres+3.0*np.std(bgpixels)))] = 1
    mask[np.where(image < (thres-3.0*np.std(bgpixels)))] = 1
    if makemask:
        # Create single-extension FITS file with mask image
        maskfile = froot + '_mask.fits'
        fits.writeto(maskfile, mask, overwrite=True)
    # Derive background 
    sigma_clip_forbkg = stats.SigmaClip(sigma=3., maxiters=5)
    bkg_estimator = MedianBackground()
    # the mask applied to Background2D needs to be Boolean where 1 is masked.
    boolmask = mask.astype(bool)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Input data contains invalid values")
        bkg = Background2D(image, (boxsize, boxsize),filter_size=(5, 5), mask=boolmask,
                           sigma_clip=sigma_clip_forbkg, bkg_estimator=bkg_estimator)
    bkgimage = bkg.background
    hdu = fits.PrimaryHDU(data=bkgimage)
    hdu.writeto(bkgfile, overwrite=True)
    return bkgimage


def bkg2d_mask(fitsfile, maskfile, writefits=False, boxsize=32, n_sigma=3):
    # This function performs Background2D on the input image after masking out pixels
    # flagged in an input mask fits file.
    # Outputs the files as <rootname>_bkg.fits.
    findex = fitsfile.find('.fits')
    froot = fitsfile[:findex]
    bkgfile = froot + '_bkg.fits'
    f = fits.open(fitsfile)
    multiext = False
    if len(f) > 1:
        image = f[1].data
        hdr = f[1].header
        dq = f[3].data
        multiext = True
    else:
        image = f[0].data
    # NOTE: The JWST pipeline-created mask files are multi-extension and treats
    # good pixels as mask = 1, while this function treads good pixels as mask = 0.
    fmask = fits.open(maskfile)
    if len(fmask) > 1:
        mask = fmask[1].data
        if 'JWST' in fmask[0].header['TELESCOP']:
            mask = 1 - mask
    else:
        mask = fits.getdata(maskfile)
    fmask.close()
    # Derive background
    sigma_clip_forbkg = SigmaClip(sigma=n_sigma, maxiters=5)
    bkg_estimator = MedianBackground()
    # the mask applied to Background2D needs to be Boolean where 1 is masked.
    boolmask = mask.astype(bool)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Input data contains invalid values")
        bkg = Background2D(image, (boxsize, boxsize),filter_size=(5, 5), mask=boolmask,
                           sigma_clip=sigma_clip_forbkg, bkg_estimator=bkg_estimator)
    bkgimage = bkg.background
    if writefits:
        hdu = fits.PrimaryHDU(data=bkgimage)
        hdu.writeto(bkgfile, overwrite=True)
    return bkgimage


def make_bkg_from_rate(infits, inmask):
    hdulist = fits.open(infits)
    maskhdu = fits.open(inmask)
    if len(maskhdu) > 1:
        maskim = maskhdu[1].data
        if 'JWST' in maskhdu[0].header['TELESCOP']:
            maskim = 1 - maskim
    else:
        maskim = fits.getdata(inmask)
    inimg = hdulist[1].data
    flatfile = get_flatfile(infits)
    flatimg = fits.getdata(flatfile, ext=1)
    newhdul = fits.HDUList(hdulist)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        flatted = inimg/flatimg
    newhdul[1].data = flatted
    newhdul.writeto('tmp1.fits', overwrite=True)
    bkg2 = bkg2d_mask('tmp1.fits', inmask)
    masked = np.where(maskim == 1)
    flatted[masked] = bkg2[masked]
    maskfrac = len(masked[0]) / (inimg.shape[0] * inimg.shape[1])
    return flatted, maskfrac


def myround(a):
    # Round X.5 up to X+1 no matter whether X is odd or even
    b = np.copy(a)
    partype = type(a).__name__
    if partype == 'ndarray' or partype == 'list':
        for i in range(len(a)):
            b[i] = np.floor(a[i]+0.5)
    else:
        b = np.floor(a+0.5)
    return b


def blkavg2d(in_arr, blockshape, doprint=False):
    """
    Simple block average on 2-D array.
    .. notes::
        #. This is the slow version.
        #. Only works when array is even divided by block size.
    Parameters
    ----------
    in_arr: array_like
        2-D input array.
    blockshape: tuple of int
        Blocking factors for Y and X.
    Returns
    -------
    out_arr: array_like
       Block averaged array with smaller size.
    """
    yblock, xblock = blockshape

    # Calculate new dimensions
    x_bin = int(in_arr.shape[1] / xblock)
    y_bin = int(in_arr.shape[0] / yblock)
    if doprint: 
        print(x_bin,y_bin)
    out_arr = np.zeros((y_bin, x_bin))

    # Average each block
    for j, y1 in enumerate(range(0, in_arr.shape[0], yblock)):
        y2 = y1 + yblock
        for i, x1 in enumerate(range(0, in_arr.shape[1], xblock)):
            x2 = x1 + xblock
            out_arr[j, i] = in_arr[y1:y2, x1:x2].mean()

    return out_arr


def averrow(in_arr, cenrow, nrows):
    ysub = in_arr[myround(cenrow-nrows/2).astype(int):myround(cenrow+nrows/2).astype(int),:]
    rowav = blkavg2d(ysub,(nrows,1))
    return rowav[0,:]


def avercol(in_arr, cencol, ncols):
    xsub = in_arr[:,myround(cencol-ncols/2).astype(int):myround(cencol+ncols/2).astype(int)]
    colav = blkavg2d(xsub,(1,ncols))
    return colav[:,0]


def one_f_rowstats(x, averrow):
    # Get a, b in f(x) = a * x + b and stddev around the linear fit for row case
    good = np.where(~np.isnan(averrow) & (averrow != np.inf) & (averrow != -1*np.inf))
    tmpx, tmpy = (x[good], averrow[good])
    rowfitcoef = np.polyfit(tmpx, tmpy, 1)
    yfit = rowfitcoef[0] * tmpx + rowfitcoef[1]
    yres = tmpy-yfit
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        _, _, rowsig = sigma_clipped_stats(yres, sigma=10.0)
    outarr = [rowfitcoef[0], rowfitcoef[1], rowsig]
    return outarr


def one_f_colstats(y, avercol):
    # Get a, b in f(x) = a * x + b and stddev around the linear fit for row case
    good = np.where(~np.isnan(avercol) & (avercol != np.inf) & (avercol != -1*np.inf))
    tmpx, tmpy = (y[good], avercol[good])
    colfitcoef = np.polyfit(tmpx, tmpy, 1)
    yfit = colfitcoef[0] * tmpx + colfitcoef[1]
    yres = tmpy-yfit
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        _, _, colsig = sigma_clipped_stats(yres, sigma=10.0)
    outarr = [colfitcoef[0], colfitcoef[1], colsig]
    return outarr


def slopestats(colslope, rowslope):
    maxslope = np.max([np.abs(rowslope), np.abs(colslope)])
    minslope = np.min([np.abs(rowslope), np.abs(colslope)])
    return maxslope, maxslope/minslope


def max_channeloffset(avercol):
    # Get channel-to-channel offsets. Only to be run on data with 4 512-row sections.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        ch1av, ch1med, ch1sig = sigma_clipped_stats(avercol[0:511], sigma=3.0) 
        ch2av, ch2med, ch2sig = sigma_clipped_stats(avercol[512:1023], sigma=3.0) 
        ch3av, ch3med, ch3sig = sigma_clipped_stats(avercol[1024:1535], sigma=3.0) 
        ch4av, ch4med, ch4sig = sigma_clipped_stats(avercol[1536:], sigma=3.0)
    meanlevel = np.mean([ch1av, ch2av, ch3av, ch4av])
    diff_1_2 = np.abs(ch1av-ch2av)
    diff_2_3 = np.abs(ch2av-ch3av)
    diff_3_4 = np.abs(ch3av-ch4av)
    maxdiff = np.max([diff_1_2, diff_2_3, diff_3_4])
    return maxdiff, meanlevel


def one_f_parvalues(maxdiff, maskfrac, maxslope, sloperatio, upper_row_sigma):
    if ((maxdiff > 0.02) & (maskfrac < 0.07)):
        bkgmethod = 'median'
        fit_by_channel = True
    elif (maskfrac > 0.07):
        fit_by_channel = False
        if maxdiff < 0.07:
            if maxslope > 1.e-5:
           bkgmethod = 'model'
            else:
                bkgmethod = 'median'
        else:
            if maxslope > 1.e-4:
                bkgmethod = 'model'
            else:
                bkgmethod = 'median'
    else:
        if maxslope > 1.e-4:
            bkgmethod = 'model'
        elif maxslope > 1.e-5:
            if sloperatio < 5.:
                bkgmethod = 'model'
            else:
                bkgmethod = 'median'
        else:
            bkgmethod = 'median'
        if upper_row_sigma > 0.02:
            fit_by_channel = True
        else:
            fit_by_channel = False
    return fit_by_channel, bkgmethod


def rundet1_default_1f(infits, configfile='stpipe-log.cfg'):
    res1 = Detector1Pipeline()
    result = res1.call(infits, save_results=True, 
                       steps={'jump': {'save_results': True},
                              'clean_flicker_noise':{'skip': False,
                                                     'background_method': 'median',
                                                     'fit_by_channel': False,
                                                     'save_mask': True, 
                                                     'apply_flat_field': True}},
                       logcfg=configfile)
    return result


def onefcorr(fitsfile, backsub=False, fit_by_channel=True, boxsize=32, makemask=False,
             user_mask=None, makebkg=False, bkgmethod='median', doprint=False):
   # This function corrects for 1/f noise in JWST H2RG images due to the ASIC.
   # For full-frame images, do this pre-amp by pre-amp (512 rows at a time).
   # Otherwise all rows in one step.
   # Outputs the file as <rootname>_corr.fits 
   findex = fitsfile.find('.fits')
   froot = fitsfile[:findex]
   corrfile = os.getcwd() + '/' + froot + '_corr.fits'
   shutil.copyfile(fitsfile, corrfile)
   if doprint:
       print('  Doing 1/f correction for {}. Output file is {}'.format(fitsfile,corrfile))
   hdulist = fits.open(fitsfile)
   image = hdulist[1].data
   hdr = hdulist[1].header
   dq = hdulist[3].data
   slowaxis = abs(hdulist[0].header['SLOWAXIS'])
   # Only perform automatic mask creation if no user mask provided
   if user_mask is None:
      # mask of bad pixels from the DQ array
      mask = np.zeros(image.shape).astype(int)
      bady,badx = np.where((np.bitwise_and(dq, dqflags.group['DO_NOT_USE']) == 1))
      mask[bady,badx] = 1
      if makemask:
         # Create single-extension FITS file with DQ mask image
         maskfile = froot + '_dqmask.fits'
         fits.writeto(maskfile, mask, overwrite=True)
      # Add pixels > 3 sigma away from "real" background to mask.
      # 11/14/24: Add option to use estimated mode, since the distribution can be skewed
      gooddata = image[np.where(mask == 0)]
      with warnings.catch_warnings():
         warnings.filterwarnings("ignore", message="Input data contains invalid values")
         mn, median, std = stats.sigma_clipped_stats(gooddata, sigma=3, maxiters=10)
      thres = median
      if bkgmethod.lower() == 'mode':
         thres = 3*median-2*mn
      lowpixels = np.where(gooddata < thres)
      bgpixels = np.concatenate(((gooddata[lowpixels]-thres),(thres-gooddata[lowpixels])))
      mask[np.where(image > (thres+3.0*np.std(bgpixels)))] = 1
      mask[np.where(image < (thres-3.0*np.std(bgpixels)))] = 1
      if makemask:
         # Create single-extension FITS file with mask image
         maskfile = froot + '_mask.fits'
         fits.writeto(maskfile, mask, overwrite=True)
   else:
      if not os.path.exists(user_mask):
         print('user_mask {} not found. Have to exit.'.format(user_mask))
         sys.exit()
      else:
         mask = fits.getdata(user_mask, ext=0)
   theimage = np.copy(image)
   if backsub:
      # If selected, do background subtraction to separate background variations
      # from 1/f stripe determination
      sigma_clip_forbkg = stats.SigmaClip(sigma=3., maxiters=10)
      bkg_estimator = MedianBackground()
      # the mask applied to Background2D needs to be Boolean where 1 is masked.
      boolmask = mask.astype(bool)
      with warnings.catch_warnings():
         warnings.filterwarnings("ignore", message="Input data contains invalid values")
         bkg = Background2D(image, (boxsize, boxsize),filter_size=(5, 5), mask=boolmask,
                            sigma_clip=sigma_clip_forbkg, bkg_estimator=bkg_estimator)
      bkgimage = bkg.background
      bksubimage = image - bkgimage
      if makebkg:
         bkgfits = froot + '_bkg.fits'
         hdu = fits.PrimaryHDU(data=bkgimage)
         hdu.writeto(bkgfits, overwrite=True)
      # Remake mask on background-subtracted data, using same method as before
      # but now clipping at < or > 2 sigma of the flux distribution
      if user_mask is None:
         mask = np.zeros(image.shape)
         mask[bady,badx] = 1
         gooddata = bksubimage[np.where(mask==0)]
         mn, median, std = stats.sigma_clipped_stats(gooddata, sigma=3, maxiters=10)
         negpixels = np.where(gooddata < median)
         bgpixels = np.concatenate(((gooddata[negpixels]-median),(median-gooddata[negpixels])))
         mask[np.where(bksubimage > (median+2.0*np.std(bgpixels)))] = 1
         mask[np.where(bksubimage < (median-2.0*np.std(bgpixels)))] = 1
      theimage = np.copy(bksubimage)
   if ((len(image) == 2048) & (len(image[0]) == 2048) & fit_by_channel):
      # Extract subimages, one for each pre-amp. Don't include non-illuminated pixels.
      if slowaxis == 1:
         medians = np.zeros((len(image[0]),4))
         sub1 = theimage[4:512,4:2044]
         sub2 = theimage[512:1024,4:2044]
         sub3 = theimage[1024:1536,4:2044]
         sub4 = theimage[1536:2044,4:2044]
         mask1 = mask[4:512,4:2044]
         mask2 = mask[512:1024,4:2044]
         mask3 = mask[1024:1536,4:2044]
         mask4 = mask[1536:2044,4:2044]
         # For each subimage, make masked array and then take median along slow axis.
         # Subtract overall mean level off of the stripes before subtracting from image.
         # (for backsub = True, that overall mean level should be zero if there are
         #  no changes across the different sub-amp regions.)
         maskedsub1 = np.copy(sub1)
         maskedsub1[np.where(mask1 == 1)] = median
         maskedsub2 = np.copy(sub2)
         maskedsub2[np.where(mask2 == 1)] = median
         maskedsub3 = np.copy(sub3)
         maskedsub3[np.where(mask3 == 1)] = median
         maskedsub4 = np.copy(sub4)
         maskedsub4[np.where(mask4 == 1)] = median
         for i in range(len(sub1[0])):
            mean, medians[i,0], std = stats.sigma_clipped_stats(maskedsub1[:,i], sigma=3, maxiters=10)
            mean, medians[i,1], std = stats.sigma_clipped_stats(maskedsub2[:,i], sigma=3, maxiters=10)
            mean, medians[i,2], std = stats.sigma_clipped_stats(maskedsub3[:,i], sigma=3, maxiters=10)
            mean, medians[i,3], std = stats.sigma_clipped_stats(maskedsub4[:,i], sigma=3, maxiters=10)
         mean1, med, sig = stats.sigma_clipped_stats(medians[:,0], sigma=3, maxiters=10)
         mean2, med, sig = stats.sigma_clipped_stats(medians[:,1], sigma=3, maxiters=10)
         mean3, med, sig = stats.sigma_clipped_stats(medians[:,2], sigma=3, maxiters=10)
         mean4, med, sig = stats.sigma_clipped_stats(medians[:,3], sigma=3, maxiters=10)
         themean = (mean1+mean2+mean3+mean4)/4.
         for i in range(len(sub1[0])):
            sub1[:,i] = sub1[:,i] - medians[i,0] + themean
            sub2[:,i] = sub2[:,i] - medians[i,1] + themean
            sub3[:,i] = sub3[:,i] - medians[i,2] + themean
            sub4[:,i] = sub4[:,i] - medians[i,3] + themean
         # if backsub = True, add the 2-D background back in.
         if backsub:
            sub1 = sub1 + bkg.background[4:512,4:2044]
            sub2 = sub2 + bkg.background[512:1024,4:2044]
            sub3 = sub3 + bkg.background[1024:1536,4:2044]
            sub4 = sub4 + bkg.background[1536:2044,4:2044]
         # insert subimages back into the original image
         image[4:512,4:2044] = sub1
         image[512:1024,4:2044] = sub2
         image[1024:1536,4:2044] = sub3
         image[1536:2044,4:2044] = sub4
      else:
         medians = np.zeros((len(image),4))
         sub1 = theimage[4:2044,4:512]
         sub2 = theimage[4:2044,512:1024]
         sub3 = theimage[4:2044,1024:1536]
         sub4 = theimage[4:2044,1536:2044]
         mask1 = mask[4:2044,4:512]
         mask2 = mask[4:2044,512:1024]
         mask3 = mask[4:2044,1024:1536]
         mask4 = mask[4:2044,1536:2044]
         # For each subimage, make masked array and then take median along slow axis.
         # Subtract overall mean level off of the stripes before subtracting from image.
         # (for backsub = True, that overall mean level should be zero if there are
         #  no changes across the different sub-amp regions.)
         maskedsub1 = np.copy(sub1)
         maskedsub1[np.where(mask1 == 1)] = median
         maskedsub2 = np.copy(sub2)
         maskedsub2[np.where(mask2 == 1)] = median
         maskedsub3 = np.copy(sub3)
         maskedsub3[np.where(mask3 == 1)] = median
         maskedsub4 = np.copy(sub4)
         maskedsub4[np.where(mask4 == 1)] = median
         for i in range(len(sub1)):
            mean, medians[i,0], std = stats.sigma_clipped_stats(maskedsub1[i,:], sigma=3, maxiters=10)
            mean, medians[i,1], std = stats.sigma_clipped_stats(maskedsub2[i,:], sigma=3, maxiters=10)
            mean, medians[i,2], std = stats.sigma_clipped_stats(maskedsub3[i,:], sigma=3, maxiters=10)
            mean, medians[i,3], std = stats.sigma_clipped_stats(maskedsub4[i,:], sigma=3, maxiters=10)
         mean1, med, sig = stats.sigma_clipped_stats(medians[:,0], sigma=3, maxiters=10)
         mean2, med, sig = stats.sigma_clipped_stats(medians[:,1], sigma=3, maxiters=10)
         mean3, med, sig = stats.sigma_clipped_stats(medians[:,2], sigma=3, maxiters=10)
         mean4, med, sig = stats.sigma_clipped_stats(medians[:,3], sigma=3, maxiters=10)
         themean = (mean1+mean2+mean3+mean4)/4.
         for i in range(len(sub1)):
            sub1[i,:] = sub1[i,:] - medians[i,0] + themean
            sub2[i,:] = sub2[i,:] - medians[i,1] + themean
            sub3[i,:] = sub3[i,:] - medians[i,2] + themean
            sub4[i,:] = sub4[i,:] - medians[i,3] + themean
         # if backsub = True, add the 2-D background back in.
         if backsub:
            sub1 = sub1 + bkg.background[4:2044,4:512]
            sub2 = sub2 + bkg.background[4:2044,512:1024]
            sub3 = sub3 + bkg.background[4:2044,1024:1536]
            sub4 = sub4 + bkg.background[4:2044,1536:2044]
         # insert subimages back into the original image
         image[4:2044,4:512] = sub1
         image[4:2044,512:1024] = sub2
         image[4:2044,1024:1536] = sub3
         image[4:2044,1536:2044] = sub4
   else:
      # This is for NOT fit_by_channel
      # First get all medians of rows, then get mean of all of those, then
      # do "row = row-median(row)+mean(median(rows))"
      maskedimg = theimage
      maskedimg[np.where(mask == 1)] = median
      if slowaxis == 1:
         medians = np.zeros(len(image[0]))
         for i in range(len(image[0])):
            mean, medians[i], std = stats.sigma_clipped_stats(maskedimg[:,i], sigma=3, maxiters=10)
         themean, med, sig = stats.sigma_clipped_stats(medians[:], sigma=3, maxiters=10)
         for i in range(len(image[0])):
            image[:,i] = image[:,i] - medians[i] + themean
      else:
         medians = np.zeros(len(image))
         for i in range(len(image)):
            mean, medians[i], std = stats.sigma_clipped_stats(maskedimg[i,:], sigma=3, maxiters=10)
         themean, med, sig = stats.sigma_clipped_stats(medians[:], sigma=3, maxiters=10)
         for i in range(len(image)):
            image[i,:] = image[i,:] - medians[i] + themean

   fits.update(corrfile, image, header=hdr, ext=1)
   newhdul = fits.HDUList(hdulist)
   newhdul[1].data = image
   #newhdul.writeto(corrfile)
   return newhdul

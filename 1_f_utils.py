#!/usr/bin/env python

import sys
import os
import numpy as np
import warnings
from astropy.io import fits
from astropy import stats
from photutils.background import Background2D, MedianBackground, MeanBackground
from photutils.background import ModeEstimatorBackground
from jwst.datamodels import dqflags

def onefcorr(fitsfile, fit_by_channel=True, backsub=False, boxsize=32, 
             bkgmethod='median', makebkg=False, makemask=False, user_mask=None, 
             sigma_bgmask=3.0, sigma_1fmask=2.0):
   """
   Perform 1/f correction for JWST H2RG imaging data.
   Input is a level 2 calibrated NIRISS or NIRCam image (i.e., a _cal.fits file).
   Subtracts a background and masks sources to determine 1/f stripes.
   Uses 'SLOWAXIS' header keyword to determine detector orientation.

   Output:
   Returns FITS HDUList of corrected image

   Parameters:
   fitsfile - FITS file with input level 2 calibrated image (_cal.fits file)
   fit_by_channel - fit each of the 4 amps separately when full-frame (default = True, applicable when scene is sparse)
   backsub - Optionally subtract a smooth 2-D background prior to fitting 1/f pattern (default = False)
   boxsize - Size of square box used during fit of 2-D background (default = 32)
   bkgmethod - Statistical method to use to determine background level ('median' [the default] or 'mode')
   makebkg - Optionally create FITS file containing the fitted 2-D background (default = False)
   makemask - Optionally create FITS file containing the mask to flag sources (default = False)
   user_mask - Optional parameter to input user-supplied mask FITS file (string, default = None).
               If provided, this will skip the automatic mask creation.
   sigma_bgmask - sigma level of outliers for masking when creating background (default = 3.0)
   sigma_1fmask - sigma level of outliers for masking when performing 1/f correction (default = 2.0)

   P. Goudfrooij, STScI, Jan 2022
   Based on image1overf by Chris Willott at chriswillott/jwst
   """
   findex = fitsfile.find('.fits')
   froot = fitsfile[:findex]
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
      # 11/14/24: Add option to use estimated mode, since the background 
      #           distribution can be asymmetric
      gooddata = image[np.where(mask == 0)]
      with warnings.catch_warnings():
         warnings.filterwarnings("ignore", message="Input data contains invalid values")
         mn, median, std = stats.sigma_clipped_stats(gooddata, sigma=3, maxiters=10)
      thres = median
      if bkgmethod.lower() == 'mode':
         thres = 3*median-2*mn
      lowpixels = np.where(gooddata < thres)
      bgpixels = np.concatenate(((gooddata[lowpixels]-thres),(thres-gooddata[lowpixels])))
      mask[np.where(image > (thres+sigma_bgmask*np.std(bgpixels)))] = 1
      mask[np.where(image < (thres-sigma_bgmask*np.std(bgpixels)))] = 1
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
      # from 1/f pattern determination
      sigma_clip_forbkg = stats.SigmaClip(sigma=3., maxiters=10)
      bkg_estimator = MedianBackground()
      # the mask applied to Background2D needs to be Boolean where 1 is masked (True).
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
         mask[np.where(bksubimage > (median+sigma_1fmask*np.std(bgpixels)))] = 1
         mask[np.where(bksubimage < (median-sigma_1fmask*np.std(bgpixels)))] = 1
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
   #
   # Output copy of input HDUList with 1/f-corrected image in 1st ['SCI'] extension
   # 
   newhdul = fits.HDUList(hdulist)
   newhdul[1].data = image
   return newhdul

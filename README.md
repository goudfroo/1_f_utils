# 1_f_utils
1/f correction tools for JWST data

## one_f_utils.py
one_f_utils.py performs a correction for 1/f-pattern noise on JWST NIRISS or NIRCam data in imaging or wide field slitless spectroscopy (WFSS) modes. The appropriate function for imaging data is `onefcorr`, for WFSS (grism) data the function is `onefcorr_wfss`.

For imaging data, the correction using `onefcorr` is best performed on the level 2 calibrated data (the files ending with 'cal.fits'). Pixels containing sources are being masked in the process. For scenes with significant structure in the background, there is an option to remove a smooth 2-D fit to the background prior to 1/f pattern removal, after which the 2-d background is added back into the image.

For WFSS grism data, the correction using `onefcorr_wfss` needs to be performed on level 1 calibrated data (the files ending with 'rate.fits'). Due to the structure of WFSS grism data, a smooth 2-D fit to the background is subtracted <i>by default</i> prior to 1/f pattern removal for this function (and then added back in after 1/f pattern removal). **Note:** For the `onefcorr_wfss` function to work correctly, users not at STScI **must** have environment variables CRDS_PATH and CRDS_SERVER_URL defined (see https://jwst-crds.stsci.edu/docs/cmdline_bestrefs for details). 

The script should be used with care, and results should be inspected for any unintended consequences.

### Usage

A default calling sequence on a level 2 calibrated direct image called `myimgfile_cal.fits` is as follows:
```
from one_f_utils import onefcorr
from astropy.io import fits
result = onefcorr('myimgfile_cal.fits')
result.writeto('myimgfile_cal_1fcorr.fits')
```

A default calling sequence on a level 1 calibrated WFSS grism image called `mywfssfile_rate.fits` is as follows:
```
from one_f_utils import onefcorr_wfss
from astropy.io import fits
result = onefcorr_wfss('mywfssfile_rate.fits')
result.writeto('mywfssfile_1fcorr_rate.fits')
```

Optional parameters to `onefcorr` and `onefcorr_wfss` are listed and explained within the one_f_utils script.

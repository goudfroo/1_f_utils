# 1_f_utils
1/f correction tools for JWST data

## 1_f_utils.py
1_f_utils.py performs a correction for 1/f-pattern noise on JWST NIRISS or NIRCam data in imaging or wide field slitless spectroscopy (WFSS) modes. The correction is best performed on level 2 calibrated data (the files ending with 'cal.fits'). Pixels containing sources are being masked in the process. For scenes with significant structure in the background, there is an option to remove a smooth 2-D fit to the background prior to 1/f pattern removal, after which the 2-d background is added back into the image.

The script should be used with care, and results should be inspected for any unintended consequences.

### Usage

A default calling sequence on a level 2 calibrated image called `myfile_cal.fits` is as follows:
```
from 1_f_utils import onefcorr
from astropy.io import fits
result = onefcorr('myfile_cal.fits')
result.writeto('myfile_cal_1fcorr.fits')
```
Optional parameters are listed and explained within the 1_f_utils script.

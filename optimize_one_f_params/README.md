# optimize_one_f_params
tools for automatic optimal parameter value selection for JWST/NIRISS imaging mode data

## optimize_one_f_params.py
optimize_one_f_params.py runs `jwst.pipeline.Detector1Pipeline()` including the `clean_flicker_noise` step with a certain choice of parameter values. It then performs a series of statistical algorithms on the science extension of the resulting `rate.fits` file and performs a selection algorithm in order to determine the optimal set of parameter values to be applied for the `clean_flicker_noise` step. If those parameter values are the same as those applied in the previous run of `Detector1Pipeline()`, then the script stops and keeps the `rate.fits` and `rateints.fits` file as is. If on the other hand those parameter values are different from the first run, then the script runs the `clean_flicker_noise` step on the `jump.fits` file which was saved during the first run of `Detector1Pipeline()`, using the previously determined optimal parameter values, followed by the `ramp_fit` step to produce updated versions of `rate.fits` and `rateints.fits`. 

Even though the algorithm is very likely to select the optimal parameter values for `clean_flicker_noise`, results should be inspected to confirm a proper correction for 1/f noise.

### Usage

A default calling sequence on an uncalibrated direct image called `myimgfile_uncal.fits` is as follows:
```
python optimize_one_f_params.py myimgfile_uncal.fits
```

There are no optional parameters to `optimize_one_f_params.py`.


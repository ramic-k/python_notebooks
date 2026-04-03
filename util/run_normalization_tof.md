# `run_normalization_tof.py`

Non-interactive driver for the `normalization_tof` workflow. Use this when you want to run the notebook logic without clicking through widgets.

## Recommended environment

On the analysis machine, run from the repo checkout itself:

```bash
cd /SNS/users/ykr/Desktop/VENUS_python_notebooks_normalization
```

Use the repo-local interpreter:

```bash
./.pixi/envs/default/bin/python util/run_normalization_tof.py --help
```

This is the recommended path for actual runs.

Why:
- it uses the exact environment locked for this checkout
- it avoids some of the extra overhead/noise from `pixi run`
- it makes debugging easier

You can still use Pixi directly if you want:

```bash
/SNS/users/ykr/.pixi/bin/pixi run python util/run_normalization_tof.py --help
```

but the repo-local interpreter is preferred.

## Minimal structure

Every run needs:
- sample input: `--sample-run` or `--sample-path`
- OB input: `--ob-run` or `--ob-path`
- output root: `--output-folder`

Common extras:
- ROI profile: `--roi left,top,width,height`
- rebinning: `--rebin-mode ...`
- TPX3 manual spectra fallback: `--tof-bin-size-ns ...`

## Input selection

You can provide inputs in either of these ways:

1. By run number:

```bash
--sample-run 17109 --ob-run 17110
```

2. By explicit folder path:

```bash
--sample-path /SNS/.../sample_run_folder
--ob-path /SNS/.../ob_run_folder
```

Dark current is optional:

```bash
--dc-run 17100
```

or

```bash
--dc-path /SNS/.../dc_run_folder
```

## Detector choices

Valid detector values:

```text
tpx1
tpx1-new
tpx1_legacy
tpx1-old
tpx3
```

Use:
- `tpx1` for current TPX1 new naming
- `tpx1_legacy` for old `Run_<number>` style layouts
- `tpx3` for TPX3

## ROI options

Enable ROI spectrum normalization explicitly or rely on the default:

```bash
--full-spectrum-roi
```

Disable it:

```bash
--no-full-spectrum-roi
```

Set the ROI directly:

```bash
--roi 50,50,406,406
```

Format is:

```text
left,top,width,height
```

If `--roi` is omitted and ROI mode is enabled, the driver creates a centered square ROI using:

```bash
--default-roi-size 200
```

## Proton charge and uncertainty

Proton charge normalization:

```bash
--proton-charge
--no-proton-charge
```

Experimental uncertainty export:

```bash
--experimental-uncertainties
--no-experimental-uncertainties
```

Current defaults:
- TPX1: experimental uncertainties on by default
- TPX3: experimental uncertainties off by default

## Rebin modes

Valid modes:

```text
none
linear-tof
linear-lambda
log-tof
log-lambda
inverse-log-lambda
custom-schedule
```

### 1. No rebin

```bash
--rebin-mode none
```

### 2. Linear TOF

```bash
--rebin-mode linear-tof --delta-tof-us 30
```

### 3. Linear lambda

```bash
--rebin-mode linear-lambda --delta-lambda-a 0.01
```

### 4. Log TOF

```bash
--rebin-mode log-tof --delta-tof-over-tof 0.01
```

### 5. Log lambda

```bash
--rebin-mode log-lambda --delta-lambda-over-lambda 0.005
```

### 6. Inverse log lambda

```bash
--rebin-mode inverse-log-lambda --delta-lambda-squared-a2 0.01
```

### 7. Custom schedule

Use this for segmented bin widths.

Required:
- `--rebin-mode custom-schedule`
- `--custom-basis`
- `--custom-scale`
- one or more `--segment`

Example:

```bash
--rebin-mode custom-schedule \
--custom-basis tof \
--custom-scale linear \
--segment 560,70 \
--segment 2700,60 \
--segment 5500,50 \
--segment ,40
```

This means:
- `TOF_min -> 560 us` with `70 us` bins
- `560 -> 2700 us` with `60 us` bins
- `2700 -> 5500 us` with `50 us` bins
- `5500 us -> TOF_max` with `40 us` bins

Custom basis choices:

```text
tof
lambda
lambda2
lambda^2
```

Custom scale choices:

```text
linear
log
reverse-log
```

`--segment` format is always:

```text
end_value,step
```

The final open-ended segment is:

```text
,40
```

Use full bins only:

```bash
--full-bins-only
```

Disable it:

```bash
--no-full-bins-only
```

Current default is `--full-bins-only`.

## TPX3 manual spectra fallback

If no `*_Spectra.txt` exists, provide a bin size in nanoseconds:

```bash
--tof-bin-size-ns 700
```

This is typically needed for TPX3 workflows that use a manually constructed uniform TOF axis.

## Output selection

The driver always writes into the root given by `--output-folder`.

Default exports:
- normalized stack: on
- normalized integrated image: off
- x-axis file: on
- ROI profile: on if ROI mode is enabled

Optional toggles:

```bash
--export-normalized-stack
--no-export-normalized-stack
--export-normalized-integrated
--no-export-normalized-integrated
--export-sample-stack
--export-sample-integrated
--export-ob-stack
--export-ob-integrated
```

## Common examples

### TPX1 by run number

```bash
cd /SNS/users/ykr/Desktop/VENUS_python_notebooks_normalization
./.pixi/envs/default/bin/python util/run_normalization_tof.py \
  --ipts 36914 \
  --detector tpx1 \
  --sample-run 17174 \
  --ob-run 17175 \
  --output-folder /SNS/VENUS/IPTS-36914/shared/norm_tof_upd_notebook_analysis \
  --roi 124,151,257,218 \
  --rebin-mode linear-tof \
  --delta-tof-us 30 \
  --experimental-uncertainties \
  --export-normalized-integrated
```

### TPX1 custom segmented TOF schedule

```bash
cd /SNS/users/ykr/Desktop/VENUS_python_notebooks_normalization
./.pixi/envs/default/bin/python util/run_normalization_tof.py \
  --ipts 36914 \
  --detector tpx1 \
  --sample-run 17109 \
  --ob-run 17110 \
  --output-folder /SNS/VENUS/IPTS-36914/shared/norm_tof_upd_notebook_analysis/driver_cli_20260403 \
  --roi 50,50,406,406 \
  --rebin-mode custom-schedule \
  --custom-basis tof \
  --custom-scale linear \
  --segment 560,70 \
  --segment 2700,60 \
  --segment 5500,50 \
  --segment ,40 \
  --full-bins-only \
  --experimental-uncertainties \
  --export-normalized-integrated
```

### TPX3 with manual TOF bin size

```bash
cd /SNS/users/ykr/Desktop/VENUS_python_notebooks_normalization
./.pixi/envs/default/bin/python util/run_normalization_tof.py \
  --working-dir /SNS/VENUS/IPTS-35167 \
  --detector tpx3 \
  --sample-path /SNS/users/ykr/data/SNS/VENUS/IPTS-35167/images/tpx3/raw/radiography/... \
  --ob-path /SNS/users/ykr/data/SNS/VENUS/IPTS-35167/images/tpx3/raw/... \
  --tof-bin-size-ns 700 \
  --rebin-mode log-lambda \
  --delta-lambda-over-lambda 0.05 \
  --output-folder /SNS/users/ykr/data/SNS/VENUS/IPTS-35167/shared/out \
  --no-experimental-uncertainties
```

## Notes

- The driver uses the same underlying engine as the notebook.
- For TPX1 new naming, `--sample-run` and `--ob-run` resolve the real data folders through the NeXus files.
- For TPX3, if no spectra file exists, `--tof-bin-size-ns` is required.
- The output folder you pass is the root; the engine creates the detailed normalized output folder underneath it.

## Quick self-check

Before a real run:

```bash
cd /SNS/users/ykr/Desktop/VENUS_python_notebooks_normalization
./.pixi/envs/default/bin/python util/run_normalization_tof.py --help
```

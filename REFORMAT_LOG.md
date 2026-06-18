# Reformat Log

This GitHub-ready version was normalized from the uploaded graduation-design package.

## Main changes

- Removed IDE metadata, Python bytecode caches, and unused AdminLTE development files.
- Renamed runnable experiment scripts to PEP 8 style English filenames.
- Rewrote Python comments and docstrings in English.
- Added `README.md`, `.gitignore`, `.env.example`, and `requirements.txt`.
- Replaced the hard-coded Flask secret key with an environment-variable fallback.
- Added an `admin_required` decorator to reduce repeated permission checks.
- Kept `Flask_app.py` as a compatibility entry point and added `app.py` as the preferred entry point.
- Renamed garbled Unicode static filenames back to readable Chinese filenames.
- Excluded the original `CF1.mat` raw data file by default for public-portfolio safety.

## Important note

The data-cleaning page requires `CF1.mat`. If the data is authorized for public release, place it in the project root. Otherwise, provide a desensitized sample dataset and update `DATA_FILE` in `data_cleaning.py`.

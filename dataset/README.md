# English Premier League dataset (Kaggle original data)

Source: Kaggle "English Premier League" (saife245), data originally from football-data.co.uk.
Every file here is the original Kaggle content, byte-for-byte. Files were only renamed and sorted into folders.
No cleaning, processing or feature engineering has been applied.

## Layout
- `seasons/`   one CSV per season, 2000-01 to 2021-22 (22 files)
- `reference/` final league positions and column documentation
- `MANIFEST.csv`  every file with row/column counts and a description

## Things to know about the original data
- 2018-19 (160 matches) and 2019-20 (260 matches) are incomplete in the source. They are marked `_PARTIAL_` in the filename.
- Columns differ between seasons: early seasons have ~23-28 columns, 2009-13 have ~71, 2020-22 have 106 (more betting markets).
- Files are Latin-1 encoded and dates mix `dd/mm/yy` and `dd/mm/yyyy`.
- The Kaggle author's preprocessed files (`final_dataset.csv`, `test_set.csv`) are not included because they are processed/feature data.
- Column meanings: see `reference/column_dictionary.txt`.

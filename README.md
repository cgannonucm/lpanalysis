# Lensing Perspective Analysis
Analysis for Gannon et al. 2025 (https://arxiv.org/abs/2501.17362)

## Instalation / Running
1. Clone this repository
2. ```
git clone https://github.com/cgannonucm/lpanalysis
```

3. Download galacticus outputs 
Download files from https://ucmerced.box.com/s/gl3lg60e80783m0hq3po4zrzgi9t63bt and place in "data/galacticus" directory
Note that the "scaling" directory is large ~177gb. If this is too large for you, you can download all directories except the scaling directory and can instead download the summarized version here: https://ucmerced.box.com/s/ugtvs2rto2x5opbv2mlesbdnze2un87m . Place the contents of this dodownloaded file in the "out/" directory.

4. Install python dependencies
Install dependencies from requirments.txt file
```
pip install -r requirements.txt
```

4. Download symphony halo data 
Download the symphony halo data on both the group and Wilky Way scales. You will need symphony credentials to do this. A script is included to do this. Simply place your symphony credentials in a file named symphony_credentials.txt with your username as the first line and password in thesecond line. For example,

username
password

then run the symphony_download.py script.
```
chmod u+x symphony_analysis.py
./symphony_analysis.py
```

5. (Optional) Install Fonts
Create fonts directory
```
mkdir fonts
```
download and extract Source Sans 3 into the fonts directory from https://fonts.google.com/specimen/Source+Sans+3

6. Run analysis
Run the "run_analysis.py" script
```
chmod u+x run_analysis.py
./run_analysis.py
```

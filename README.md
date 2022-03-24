# ULISSE
Code for "ULISSE: A Tool for One-shot Sky Exploration and its Application to Active Galactic Nuclei Detection". 


## Loading data

Change line 15	to read in the desired csv file with a column ra and dec for each object, for example from a query to the SDSS.

## Running ULISSE

Using the .npz file obtained from get_data.py (loaded in the get_data() function), the notebook can be used to find nearest neighbours for any given query, by giving their ra and dec coordinates.

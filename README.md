# NASA ExoArchive Aggregator

NASAExoArchive_download_data.py: Downloads all transiting, confirmed planets from the Exoplanet achive and saves them into two csv files

- default_params0.csv: All confirmed planets from not default publications 
- default_params1.csv: All confirmed planets from default publication  

NASAExoArchive_aggregator.py: aggregates the data on the NASA Exoplanet Archive. 

NASAExoArchive_2021-03-12_aggregate.csv: creates a csv file for all planets with the most recently published value for each parameter, as well as the TSM and ESM (Kempton et al. 2018).


# Notes:

## Temperature calculation
To calculate planet temperature, which is often missing, the code 
 
- first populate with insolation                                                 
- if insolation is unavailable, try equilibrium temperature                      
- if equilibrium temperature is unavailable, calculate from a/Rs and Teff        
- if a/Rs is unavailable, calculate it from a [AU] and Rs [Rsun]                 

## Planet masses
The code currently takes the planet masses as published on the NASA Exoplanet Archive. It does not fill in missing masses from a M/R relation, and it does not filter out planets with larger error bars on the published mass. 


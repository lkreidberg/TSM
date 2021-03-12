# NASA ExoArchive Aggregator

NASAExoArchive_download_data.py: Downloads all transiting, confirmed planets from the Exoplanet achive and saves them into two csv files

- default_params0.csv: All confirmed planets from not default publications 
- default_params1.csv: All confirmed planets from default publication  

NASAExoArchive_aggregator.py: It aggregates the data on the NASA Exoplanet Archive. It excludes planets with high uncertainties in mass or radius. Can be removed if not wanted in the code. 

NASAExoArchive_2021-03-12_aggregate.csv: Final csv file for all planets including the calculation for TSM and ESM 
 

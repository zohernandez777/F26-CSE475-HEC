import nflreadpy as nfl 
import polars as pl


#I will be pulling my data from seasons 2010 to 2025
seasons = range(2010, 2026)
#Load the seasons
for season in seasons:
    pbp = nfl.load_pbp(season)
    pbp.write_parquet(f"data/pbp_{season}.parquet") #save that seasonsn data

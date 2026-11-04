import nflreadpy as nfl 
import polars as pl




seasons = range(2010, 2026)
cleaned = []
features = ["score_differential", "game_seconds_remaining", "down", "ydstogo",
            "yardline_100", "posteam_timeouts_remaining", "defteam_timeouts_remaining"]

def cleaned_season(pbp):
    pbp = posteam_won(pbp)
    #drop null features 
    pbp = pbp.drop_nulls(subset=features)
    pbp = pbp.select(["season","game_id"] + features + ["posteam_won"])
    return pbp

def posteam_won(pbp):
    plays = pbp.filter(pl.col("posteam").is_not_null()) 
    plays = plays.filter(pl.col("result") != 0) #drops result = 0 rows
    
    #see if the team with postion won
    home_won = (pl.col("posteam") == pl.col("home_team")) & (pl.col("result") > 0)
    away_won = (pl.col("posteam") == pl.col("away_team")) & (pl.col("result") < 0) #result is calc home score - away score so pos means home won, neg means away team won
    plays = plays.with_columns(
        pl.when(home_won | away_won).then(1).otherwise(0).alias("posteam_won")
    )

    return plays
    





for season in seasons:
    pbp =  pl.read_parquet(f"data/pbp_{season}.parquet")
    pbp = cleaned_season(pbp)
    cleaned.append(pbp) 

all_plays = pl.concat(cleaned)
all_plays.write_parquet("data/plays_clean.parquet")
print(all_plays.height)



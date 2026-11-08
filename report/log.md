## NOV 4TH: 10am

**Summary:**
- I have expplored polar featuires and the dataset. What I have done is then prepared a fiel which cleans the season data for me. I am testing on 16 seasons where I calen the data by showing which team with possesion won and dropping any of lseected features which are null in the dataset. Then I sleected the cleaned data with my features and oushed that output through my "cleaned_season" function". I then looped through each season, called my clean function on them, add those new seasons to a list, concate them, and then save the data to a parquet file

**Next Steps:**
- create train/test splits. I plan to use seasosn 2010-2022 and then test on seasons 2023-2025. A lot of my features are on different scales so I plan on standarizing the features to amke the graident descent implmentation easier


## NOV 7th: 8pm

**Summary**
- I have loaded the data into pandas where I split the data into a training and testing set. I then calculated the mean and standard devlativion to apply standarizing features so that they have a mean of 0 and eveiation of 1. I applied my standarizding features to X only, created my Y(labels) dataset, and then converted everything to numpy arrays for future work

**Next Steps:**
- My next steps are to start workign on the model itself, mroe specfically the sigmoid function.
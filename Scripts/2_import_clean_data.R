## billboard prediction analysis
## db fowler
## 10/13/2016

# import packages
library(dplyr); library(readr); library(lubridate);
library(ggplot2); library(ggthemes)

# set wd
setwd('/Users/bradyfowler/Documents/Fall Semester/Modeling_6021/predicting_billboard_success')

####################################################################################
#
# load charts and song attributes data
#
####################################################################################
charts <- read_csv('Raw Data/all_charts_final.csv')
colnames(charts) <- tolower(colnames(charts))
charts <- charts[!is.na(names(charts))]

# add track ID:
charts <- transform(charts,song_id=as.numeric(factor(
  ifelse(is.na(spotifyid)==FALSE, spotifyid, 
         paste(tolower(title), tolower(artist), sep="-")))))

# add artist id
charts <- transform(charts,artist_id=as.numeric(factor(tolower(art_0))))

traits <- read_csv('Raw Data/audio_traits.csv')
colnames(traits) <- tolower(colnames(traits))
traits <- traits[!is.na(names(traits))]

####################################################################################
#
# create distinct list of songs to be our base analysis table
#
####################################################################################
# check to make sure we have a parse artist object
charts %>% filter(is.na(art_0)==TRUE) %>% select(artist) %>% unique()

weeks.till.top.ten <- charts %>% 
  group_by(song_id) %>%
  filter(peakpos < 11) %>% 
  mutate(reenter = max(ifelse(change=="Re-Entry", 1, 0))) %>% 
  summarise(birthday = min(chartdate),
            top.ten.day = as.Date(min(ifelse(rank < 11, chartdate, 9999999999)), origin="1970-01-01"), 
            reenter = max(reenter)) %>% 
  mutate(weeks.till.top.ten = as.numeric(top.ten.day - birthday)/7.0, 
         yr = year(birthday)) %>% 
  filter(weeks.till.top.ten < 60) ## remove the massive outliers that were reentries

ggplot(weeks.till.top.ten, aes(x=factor(yr), y=weeks.till.top.ten, fill=factor(yr))) + 
  geom_boxplot(notch = TRUE, varwidth = TRUE) + theme_fivethirtyeight() +
  guides(fill=FALSE) + ggtitle("# Weeks to Hit Top Ten by Year :: Distribution")

# look for re-entry problems
left_join(charts, 
          charts %>% group_by(song_id) %>% summarise(birthday = min(chartdate)), 
          by="song_id") %>% 
  filter(chartdate == birthday & year(birthday) > 2013) %>% 
  group_by(song_id) %>%
  group_by(change) %>% 
  summarise(cnt = n()) %>% 
  mutate(pct = cnt*100.0/sum(cnt)) %>% arrange(desc(pct))

# identify the tracks that were reentries in our test-data
left_join(charts, 
          charts %>% group_by(song_id) %>% summarise(birthday = min(chartdate)), 
          by="song_id") %>% 
  filter(chartdate == birthday & year(birthday) > 2013 & change=="Re-Entry") %>% 
  select(song_id) %>% unique()

# ggplot(charts %>% filter(song_id == 5999), aes(x=chartdate, y=rank)) + geom_line() + scale_y_continuous(trans = "reverse")
# look at song that had "reenter" to make sure its okay

analysis <- charts %>%
  group_by(title, art_0, artist, artist_id, song_id, spotifyid) %>% 
  summarise(birthday = min(chartdate), 
            top.rank = min(rank), 
            total.weeks = max(weeks),
            weeks.at.top.ten = sum(ifelse(rank<11,1,0))) %>% ungroup()

# missing by date:
ggplot(charts %>% 
  filter(is.na(spotifyid)==TRUE) %>% group_by(yr=year(chartdate)) %>%
  summarise(count=n_distinct(song_id)), 
  aes(x=yr, y=count, group=1)) + geom_smooth()
# bit less frequent as time goes on

# determine how many songs we are missing audio details for
charts %>% 
  filter(year(chartdate) > 1999) %>% 
  group_by(has_details = -is.na(spotifyid)) %>% 
  summarise(count=n_distinct(song_id)) %>% 
  mutate(pct_with_spotify_details = count*100.0/sum(count))

####################################################################################
#
# determine song and chart details derived from chart table
#
#################################################################################### 
##################################################################
#
## PRIMARY ARTIST :: Number of Previous Top 10 Hits
## PRIMARY ARTIST :: Number of Previous Non Top 10 Hits
## PRIMARY ARTIST :: Number of Weeks on Charts before
## PRIMARY ARTIST :: Average Weekly Lifetime of Previous Songs
#
##################################################################
# how many hits did this artist have prior to a new song's release?
  prior.hits <- left_join(analysis, analysis, by=c("art_0"), suffix=c(".current", ".prior")) %>% 
    filter(birthday.prior < birthday.current) %>% 
    group_by(art_0, song_id.current, birthday.current) %>% 
    summarise(prior.top.ten.hits     = sum(ifelse(top.rank.prior<11,1,0)),
              prior.not.top.ten.hits = sum(ifelse(top.rank.prior<11,0,1)),
              prior.weeks.at.top.ten = sum(weeks.at.top.ten.prior)) 
  
  ## note that some spotify IDs are duplicated. this is an error of billboard data
      charts %>% filter(spotifyid=="0MsrWnxQZxPAcov7c74sSo") %>% distinct(title)

# add in the prior hits
analysis <- left_join(analysis, prior.hits, by=c("song_id"="song_id.current", "birthday"="birthday.current", "art_0")) %>% 
  mutate(prior.top.ten.hits     = ifelse(is.na(prior.top.ten.hits    )==1,0,prior.top.ten.hits    ), 
         prior.not.top.ten.hits = ifelse(is.na(prior.not.top.ten.hits)==1,0,prior.not.top.ten.hits), 
         prior.weeks.at.top.ten = ifelse(is.na(prior.weeks.at.top.ten)==1,0,prior.weeks.at.top.ten))

##################################################################
#
## AVERAGE AGE OF TOP TEN AT SONG X RELEASE
#
##################################################################
# average age of all songs in top ten at time of release excluding that song?
  top.tens <- charts %>% 
    filter(peakpos < 11) %>% 
    distinct(title, artist, artist_id, song_id, chartdate, weeks) 
  
  top.tens <- left_join(analysis, top.tens, by=c("birthday"="chartdate")) %>% 
    filter(song_id.x!=song_id.y) %>% 
    group_by(song_id = song_id.x, title = title.x, artist = artist.x, artist_id = artist_id.x, birthday) %>% 
    summarise(avg.age.top.ten = mean(weeks, na.rm=TRUE)) %>% ungroup()
  
# average age of all songs in top ten at time of release excluding that song?
  not.top.tens <- charts %>% 
    filter(peakpos > 11) %>% 
    distinct(title, artist, artist_id, song_id, chartdate, weeks) 
  
  not.top.tens <- left_join(analysis, not.top.tens, by=c("birthday"="chartdate")) %>% 
    filter(song_id.x!=song_id.y) %>% 
    group_by(song_id = song_id.x, title = title.x, artist = artist.x, artist_id = artist_id.x, birthday) %>% 
    summarise(avg.age.not.top.ten = mean(weeks, na.rm=TRUE)) %>% ungroup()

# add in the age of the top tens and not top tens
  analysis <- left_join(analysis, top.tens,     by=c("artist_id", "title", "song_id", "birthday", "artist")) 
  analysis <- left_join(analysis, not.top.tens, by=c("artist_id", "title", "song_id", "birthday", "artist")) 
  

####################################################################################
#
# combine billboard to song traits
#
####################################################################################   
analysis.traits <- left_join(analysis, traits, by=c("spotifyid"="id")) %>% 
  filter(year(birthday)>1999)  %>% 
  mutate(pred.hit.top.ten = ifelse(top.rank<11,1,0), 
         pred.top.rank = top.rank) %>% 
  select(-track_href, -type, -uri, -analysis_url, -top.rank)
  
analysis.traits %>% saveRDS('Raw Data/final.analysis.RDS')

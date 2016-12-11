setwd('/Users/bradyfowler/Documents/Fall Semester/Modeling_6021/predicting_billboard_success')
suppressMessages({
  library(dplyr)
  library(readr)
  library(lubridate)
  library(reshape2)
  library(scales) 
  library(ggplot2)
  require(stringr)
  library(ggthemes)
})
set.seed(12345)

raw <- readRDS('Raw Data/final.analysis.RDS')

############################################################
#
# echo nest audio features over time
#
############################################################
raw <- raw %>% 
  mutate(week = floor_date(birthday, unit="week")) 

long <- melt(raw %>% select(week, song_id, pred.hit.top.ten, acousticness, danceability, 
                            duration_ms, energy, instrumentalness, liveness, 
                            loudness, speechiness, tempo, valence), id.vars=c("week","song_id", "pred.hit.top.ten"))

long$variable <- str_to_title(long$variable)
long$variable<- gsub("_ms", " MS", long$variable)
long$pred.hit.top.ten <- factor(ifelse(long$pred.hit.top.ten==0,"No", "Yes"))

ggplot(long, aes(x=week, y=value, 
                 group=pred.hit.top.ten, 
                 colour=pred.hit.top.ten)) + 
  geom_smooth(alpha=.3) + ggtitle("Audio Traits Over Time") +
  facet_wrap(~variable, nrow=6, scales = "free") + 
  theme_fivethirtyeight() + labs(colour="Hit Top Ten") +
  theme(strip.text = element_text(face="bold.italic"))
ggsave("Output/1. Audio Traits Over Time.pdf", plot = last_plot(), width=11, height=8.5)

############################################################
#
# bar chart of song counts over time
#
############################################################
num_top_ten_month <- raw %>% 
  mutate(month = floor_date(birthday, unit="year")) %>% 
  select(month, pred.hit.top.ten)

ggplot(num_top_ten_month, 
       aes(x=month, 
           group=pred.hit.top.ten, 
           fill=factor(ifelse(pred.hit.top.ten==0,"No", "Yes")))) + 
  geom_bar() + labs(fill="Hit Top Ten") + ggtitle("Number of Billboard Songs by Year") +
  theme_fivethirtyeight() +
  theme(axis.title = element_text(face="italic"), 
        axis.title.x = element_text(face="italic")) + 
  xlab("Release Year") + ylab("# Tracks")
ggsave("Output/2. Number of Billboard Songs by Year.pdf", plot = last_plot(), width=11, height=8.5)

# 100% bar chart
num_top_ten_month <- raw %>% 
  mutate(month = floor_date(birthday, unit="year")) %>% 
  group_by(month) %>% 
  summarise(num_hit    = sum(ifelse(pred.hit.top.ten==1,1,0)), 
            num_no_hit = sum(ifelse(pred.hit.top.ten==0,1,0)))
meltd<- melt(num_top_ten_month, id.vars=1) 

ggplot(meltd, aes(x=month, y=value, fill=factor(ifelse(variable=="num_hit","Yes", "No")))) +
  geom_bar(stat="identity", position = "fill") +
  scale_y_continuous(labels = percent_format()) +
  theme_fivethirtyeight() + labs(fill="Hit Top Ten") +
  ggtitle("Percent of Billboard Songs by Year") +
  theme(axis.title = element_text(face="italic"), 
        axis.title.x = element_text(face="italic")) + 
  xlab("Release Year") + ylab("% Tracks")
ggsave("Output/3. Percent of Billboard Songs by Year.pdf", plot = last_plot(), width=11, height=8.5)

############################################################
#
# additional audio traits over time
#
############################################################
scatter_data <- melt(raw %>% 
  mutate(month = floor_date(birthday, unit="month")) %>% 
  select(prior.top.ten.hits, prior.not.top.ten.hits, prior.weeks.at.top.ten, 
               avg.age.top.ten, avg.age.not.top.ten, pred.hit.top.ten, weeks.at.top.ten, song_id, month), 
  id.vars=c("month","song_id", "pred.hit.top.ten"))

scatter_data$pred.hit.top.ten <- factor(ifelse(scatter_data$pred.hit.top.ten==0,"No", "Yes"))

# quick and dirty map
scatter_data$variable <- 
    ifelse(scatter_data$variable=="prior.top.ten.hits", "# Prior Top 10 Hits",
    ifelse(scatter_data$variable=="prior.not.top.ten.hits", "# Prior NOT Top 10 Hits", 
    ifelse(scatter_data$variable=="prior.weeks.at.top.ten", "# Prior Weeks at Top 10", 
    ifelse(scatter_data$variable=="avg.age.top.ten", "Average Age of Top 10 Songs", 
    ifelse(scatter_data$variable=="avg.age.not.top.ten", "Average Age of NOT Top 10 Songs", 
    ifelse(scatter_data$variable=="pred.hit.top.ten", "pred.hit.top.ten", 
    ifelse(scatter_data$variable=="weeks.at.top.ten", "# Weeks at Top 10", ""))))))) 

ggplot(scatter_data, aes(x=month, y=value, 
                         colour=pred.hit.top.ten, 
                         group = pred.hit.top.ten)) + 
  geom_smooth() + facet_wrap(~variable, nrow=3, scales = "free") + 
  labs(colour="Hit Top Ten") +
  theme(strip.text = element_text(face="bold.italic")) +
  ggtitle("Artist Traits Over Time") + theme_fivethirtyeight()
ggsave("Output/4. Artist Traits Over Time.pdf", plot = last_plot(), width=11, height=8.5)


############################################################
#
# parallel line chart
#
############################################################
# center each variable
std_dev <- long %>% 
  group_by(variable, pred.hit.top.ten) %>% 
  summarise(sd = sd(value, na.rm=TRUE))

long.sd <- left_join(long, std_dev, by=c("variable", "pred.hit.top.ten")) %>% 
  mutate(value = value-sd) %>% arrange(desc(pred.hit.top.ten), song_id) %>% 
  filter(variable!="Duration MS" & variable!="Tempo" & variable!="Loudness")

long.sd$song_id   <- as.numeric(long.sd$song_id) * ifelse(long.sd$pred.hit.top.ten=="Yes", 10000, 1)

ggplot(long.sd, aes(x = variable, y = value, group = song_id, color = pred.hit.top.ten)) + 
  geom_line(size=.1) + scale_colour_manual(values = c("#bdc3c7","#3498db")) +
  theme_fivethirtyeight() + ggtitle("Parallel Coordinates Sound Features") +
  theme(axis.title = element_text(), axis.title.x = element_text()) + 
  xlab("Sound Features") + ylab("Standard Deviations") + labs(colour="Hit Top Ten")
ggsave("Output/5. Parallel Coordinates Sound Features.pdf", plot = last_plot(), width=11, height=8.5)


############################################################
#
# violin plots by sound feature
#
############################################################
long <- melt(raw %>% select(week, song_id, pred.hit.top.ten, acousticness, danceability, 
                            duration_ms, energy, instrumentalness, liveness, 
                            loudness, speechiness, tempo, valence, prior.top.ten.hits, prior.not.top.ten.hits, 
                            prior.weeks.at.top.ten, avg.age.top.ten, avg.age.not.top.ten), id.vars=c("week","song_id", "pred.hit.top.ten")) %>% 
  mutate(variable = str_to_title(gsub("\\.", "  ", variable))) %>% 
  arrange(variable)

ggplot(long, aes(x=pred.hit.top.ten, y=value, fill=factor(pred.hit.top.ten))) + 
  facet_wrap(~variable, nrow=4, scales = "free") +
  geom_violin() + theme_fivethirtyeight() + 
  ggtitle("Sound Feature Distribution") + labs(fill="Hit Top Ten")
ggsave("Output/6. Sound Features Distributions.pdf", plot = last_plot(), width=11, height=8.5)


############################################################
#
## MISSINGNESS
#
############################################################
ggplot_missing <- function(x, title){ 
  # from http://www.njtierney.com/r/missing%20data/rbloggers/2015/12/01/ggplot-missing-data/
  x %>% is.na %>% melt %>%
    ggplot(data = ., aes(x = Var2, y = Var1)) +
    geom_raster(aes(fill = value)) +
    scale_fill_grey(name = "", labels = c("Present","Missing")) +
    theme_fivethirtyeight() + theme(axis.text.x  = element_text(angle=45, vjust=0.5), 
                                    axis.title  = element_text(), 
                                    panel.grid.major = element_blank(), panel.grid.minor = element_blank()) + 
    xlab("Variables in Dataset") + ylab( "Rows / observations") + ggtitle(title)
}

ggplot_missing(raw, "Analysis Table Missingess Survey")
ggsave("Output/8. Missingness Survey.pdf", plot = last_plot(), width=11, height=8.5)

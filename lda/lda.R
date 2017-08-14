library(data.table)

library(ggplot2)

library(stringr)

library(topicmodels)  

library(tidytext) 

library(tidyr)

library(grid)

### set variables ----

num_topic <- 6

### Load data ----

# import documents

data <- fread("~/data_path", sep = '\n', 
              header = F, col.names = "documents")

# import normal stop words document

stop_words <- fread("~/stop_words_path", sep = '\n', header = F,
                    col.names = "words")

stopwords <- as.matrix(stop_words)

### clean audio text ----

# remove null content and leave a message call

data[, documents:= tolower(documents)]

data <- data[documents != "none"]

# remove punctuation

data[, documents := str_replace_all(documents, "[,.'?!:-]", " ")]

# unnest words in transcript

data[, doc := c(1:dim(data)[1])]

words_doc <- unnest_tokens(data, word, documents) 

words_doc <- words_doc[word != "" & !(word %in% stopwords),]

words_doc <- words_doc[, .N, by = c("doc", "word")]

words_doc <- bind_tf_idf(words_doc, word, doc, N)

# filter words using tf-idf weights: threshold picked as .0005

words_doc <- words_doc[tf_idf > 0.0005, ]

# create the doc-term matrix

doc_term_matrix <- cast_dtm(words_doc, doc, word, N, weighting = tm::weightTf)


### perform lda ----

audio_lda <- LDA(doc_term_matrix, k = num_topic, control = list(seed = 1001))

audio_topics <- tidy(audio_lda)


### create table for plots: distribution of topics in each document. ----

topic_doc <- data.table(cbind(as.numeric(audio_lda@documents), as.matrix(audio_lda@gamma)))

colnames(topic_doc) <- c("doc", str_c("topic_", c(1:6)))

topic_doc = melt(topic_doc, measure.vars = c("topic_1", "topic_2", "topic_3", "topic_4", "topic_5", "topic_6"),
                 variable.name = "Topic", value.name = "Probability")

# define topics manually that you think are appropriate.

topics <- c("Topic_1", "Topic_2", "Topic_3", 
            
            "Topic_4", "Topic_5", "Topic_6")

audio_topics <- data.table(audio_topics)

audio_topics[, topics := topics[topic]]

### visulize results from lda ----

# define colors

colors <- c("darkturquoise","lightcoral", "khaki4", "green4", "hotpink", "steelblue3")

# produce plot 1: percentage of each topic over all documents

xx <- data.table(topic_doc)

xx[, max_prob := max(Probability), by = doc]

xx[, is_max_prob := Probability == max_prob]

percent <- xx[, list(percent = sum(is_max_prob)/dim(topic_doc)[1]), by = Topic]

percent[, topics := topics]

ggplot(percent) + geom_bar(aes(x = topics, y = percent, fill = topics), stat = "identity") + 
  ggtitle("Percentage of each Topic") + 
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 7)) + coord_polar()


# produce plot 2: six topics and corresponding top 15 keywords in specific topics

audio_plot <- list()

for (i in 1 : num_topic) {
  
  plot_data <- setorder(setDT(audio_topics), -beta)[, head(.SD, 15), keyby = topic]
  
  plot_data <- plot_data[topics == topics[num_topic], ]
  
  audio_plot[[i]] <- ggplot(plot_data) + 
    geom_bar(aes(y = beta, x = reorder(term, beta)), fill = colors[i],
             stat = "identity") + facet_wrap(~ topics, scale = "free") + 
    ylab("Probability") + xlab("Terms") + theme(axis.title = element_text(size = 8)) +
    coord_flip() + scale_fill_discrete(name="Topics")
  
}


## set up new page

grid.newpage()

nrow <- 2

ncol <- 3

pushViewport(viewport(layout = grid.layout(nrow, ncol)))

## define how graph of each topic distribution located in the plots

loc_col <- function(num) {
  if (num <= ncol) {
    return(num)
  }
  
  else {
    return(num - ncol)
  }
}

loc_row <- function(num) {
  if (num <= ncol) {
    return(1)
  }
  else{
    return(2)
  }
}

## now plot the 6 topics

for (i in 1:num_topic) {
  
  print(audio_plot[[i]], vp = viewport(layout.pos.row = loc_row(i),
                                       layout.pos.col = loc_col(i)))
}


# produce plot 3: distribution of topics in all documents

## store the plots

num_doc_ana <- dim(cancels)[1]

doc_analysis <- list()

for (i in 1:num_doc_ana) {
  
  plot_data <- data.table()
  
  plot_data <- topic_doc[doc == i, ]
  
  doc_analysis[[i]] <- ggplot(plot_data) + geom_bar(aes(y = Probability, x = topics, fill = topics), stat = "identity") + 
    
    ggtitle(str_c("Distribution of topic in Document ", i)) + 
    
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
}


## actually plot topic distribution in all documents and Export to computer

for (i in 1:dim(cancels)[1]) {
  
  file_name <- str_c("Document_", i, ".png")
  
  png(file = str_c("output_path", "/", file_name))
  
  print(doc_analysis[[i]])
  
  dev.off()
}

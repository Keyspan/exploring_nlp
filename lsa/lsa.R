# load packages
library(dbscan)
library(magrittr)
library(dplyr)
library(lsa)  
library(tidytext) 

# load text file to be analyzed
data <- fread("~/data.txt", sep = '\n', 
              header = F, col.names = "documents")

# load stop words document
stop_words <- fread("~/stop_words.txt", sep = '\n', header = F,
                    col.names = "words")

# documents index
data = data[, doc := c(1:dim(data)[1])]

# clean data
data[, documents := tolower(documents)] 
data[, documents := str_replace_all(documents, "[,.'?!:-]", " ")]   

# unnest words in documents
words_doc <- unnest_tokens(data, word, documents) 

words_doc <- words_doc[word != "" & !(word %in% stop_words),]

words_doc <- words_doc[, .N, by = c("doc", "word")]     

# create the doc-term matri x(transpose of term-doc matrix)
doc_term_matrix <- cast_dtm(words_doc, doc, word, N, weighting = tm::weightTf)

# perform lsa
mylsa <- lsa(doc_term_matrix)

# cluster the documents using DBSCAN
db_clust <- dbscan(mylsa$tk, 		# you can also pick part of factors
                   eps = db_eps, 	# define appropriate eps 
                   MinPts = db_minpoints)

# add the cluster of documents into original data
data[, db_cluster := as.factor(db_clust$cluster)]


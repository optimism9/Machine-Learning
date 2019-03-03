from __future__ import division             
import nltk
import math
import random
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import movie_reviews

#create feature set
print "MOVIE REVIEW CATEGORIES: ",(movie_reviews.categories())

#For better accuracy of algorithm, take a large dataset of various kinds of reviews.
docs=[]
docs = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
random.shuffle(docs)
words = []
for i in movie_reviews.words():
    words.append(i.lower())                     #summary is not based on upper/lower case, so remove it.
words = nltk.FreqDist(words)
word_features = list(words.keys())[:1000]       #take the most frequent words from FreqDist
#Function for creating feature sets
def features(files):
    sets=set(files)
    feature={}
    for i in word_features:
        feature[i]=(i in sets)          #if the word is in word_features, then True is returned 
    return feature

#Call function and create feature sets
featuresets = [(features(plot), category) for (plot, category) in docs]


print "INFORMATIVE FEATURES: "
#Create training and testing sets
trainset = featuresets[:1600]        #take a large training set for better accuracy
testset = featuresets[1600:]         #take a test set to apply the algorithm

#Import Naive Bayes Classifier and train the training set.
#Apply the algorithm on testing set and determine the accuracy
naive = nltk.NaiveBayesClassifier.train(trainset)
#print ("Accuracy of Naive Bayes Algorithm: ",(nltk.classify.accuracy(naive, testset))*100)
naive.show_most_informative_features(10)    #most common words that users used to rate the reviews
print "\n"

#Take frequently used words in a list------------------------------------------#
from nltk.corpus import stopwords
freq_words=[]
#In case the features contain stopwords, then they're not required.
#So, remove the stopwords
for i in naive.most_informative_features(10):
    freq_words.append(str(i[0]))    #using str() will remove the unicode 'u' symbol
print "FREQUENT WORDS:"
print freq_words

#Derive immediate synsets for freq_words
from nltk.corpus import wordnet

syn=[]
for i in range(0,len(freq_words)):
    syn.append(wordnet.synsets(freq_words[i]))      #synsets created

#Extract just the synonym part of the synset.
for i in range(0,len(syn)):
    for j in range(0,len(syn[i])):
        syn[i][j]=str(syn[i][j])        #convert each item to string
        syn[i][j]=word_tokenize(syn[i][j])      #tokenize the list
        syn[i][j]=syn[i][j][2].split('.')       #for synsets like doubt.n.01, split it in 'doubt','n','01'
        syn[i][j]=syn[i][j][0]              #in ['doubt','n',01], 'doubt' will be extracted

#for i in range(0,len(syn)):
#   print set(syn[i])                         #print the synonyms of each frequent word

#Some words contain underscores, indicating that they're not single words.
#Remove the underscores and update the syn[] list
for i in range(0,len(syn)):
	a=''
	for j in range(0,len(syn[i])):
		if('_' in syn[i][j]):
			syn[i][j]=syn[i][j].split('_')
			if(len(syn[i][j])):
				for k in range(0,len(syn[i][j])):
					a=a+' '+syn[i][j][k]

#now the syn[] list contains the frequent words used by users
print "Synsets of frequent words created."



#Preprocess the text-----------------------------------------------------#
#The text is a group of reviews of a movie.
#Tokenize review, stop word elimination, stemming, tf-idf, summary
from nltk.corpus import stopwords
#print "REVIEWS: "
filelist=[]

a="C:\Users\Sunitha\Desktop\Minor Project\captamerica\\"

for i in range(1,10):
    try:
        filelist.append(open(a+str(i)+'.txt','r'))
    except(IOError):
        break
fileinput=[]
for i in filelist:
    fileinput.append(i.read())      #Read the files in captamerica dataset 

print "Capt. America Movie Review Dataset imported"

filesents=[]
filewords=[]
for i in range(0,len(fileinput)):
    filesents.append(sent_tokenize(fileinput[i]))
for i in range(0,len(filesents)):
    filewords.append([])
    for j in range(0,len(filesents[i])):
        filewords[i].append(list(word_tokenize(filesents[i][j])))
stops = set(stopwords.words('english'))

#print "Files have been sent_ and word_ tokenized"

for i in range(0,len(filewords)):
    for j in range(0,len(filewords[i])):
        for k in range(0,len(filewords[i][j])):
            if filewords[i][j][k] in stops:
                filewords[i][j][k]=''         #removed stop words
for i in range(0,len(filewords)):
    for j in range(0,len(filewords[i])):
        while '' in filewords[i][j]:
            filewords[i][j].remove('')            
                   
tagged=[]
freqtagged=[]
for i in range(0,len(filewords)):
    for j in range(0,len(filewords[i])):
        tagged.append(nltk.pos_tag(filewords[i][j]))
freqtagged=nltk.pos_tag(freq_words)

from nltk.stem import WordNetLemmatizer
lem=WordNetLemmatizer()
rootp=filewords
rootf=freq_words

#basic pos tag list
pos_dict={'NN':'n','NNS':'n','NNP':'n','NNS':'n','CD':'n','DT':'n',         #n=noun
          'PRP':'pronoun','PRP$':'pronoun','WP$':'pronoun','WP':'pronoun',      
          'VB':'v','VBD':'v','VBG':'v','VBN':'v','VBP':'v','VBZ':'v',       #v=verb
          'JJ':'a','JJR':'a','JJS':'a',                     #a=adj
          'RB':'r','RBR':'r','RBS':'r','WRB':'r',               #r=adv
          'IN':'prep','TO':'prep',
          'CC':'conj',
          'UH':'interj'}        #lemmatizer uses only n, v, a, and r ---> 4 classes
#the rest aren't taken into consideration


#calculate parts of speech for each word in rootp[] & rootfp using the pos_dict above
tagged2=tagged
freqtagged2=freqtagged

#lemmatizing rootp[] according to its POS tag
for i in range(0,len(tagged2)):
    for j in range(0,len(tagged2[i])):
        tagged2[i][j]=list(tagged2[i][j])
for i in range(0,len(tagged)):
    for j in range(0,len(tagged[i])):
        if tagged2[i][j][1] not in pos_dict:
            continue
        tagged2[i][j][1]=pos_dict[tagged2[i][j][1]]

for i in range(0,len(rootp)):
    try:
        for j in range(0,len(rootp[i])):
            try:
                rootp[i][j]=lem.lemmatize(rootp[i][j],pos=tagged[i][j][1])
            except(KeyError):
                rootp[i][j]=lem.lemmatize(rootp[i][j])
            except(UnicodeDecodeError):
                rootp[i][j]=str(rootp[i][j])
            except:
                pass
    except:
        pass

#lemmatizing rootf[] according to its POS tag
for i in range(0,len(freqtagged2)):
    freqtagged2[i]=list(freqtagged2[i])
for i in range(0,len(freqtagged)):
    if freqtagged2[i][1] not in pos_dict:
        continue
    freqtagged2[i][1]=pos_dict[freqtagged2[i][1]]
for i in range(0,len(rootp)):
    try:
        for j in range(0,len(rootp[i])):
            try:
                rootf[i]=lem.lemmatize(rootf[i],pos=freqtagged[i][1])
            except(KeyError):
                rootf[i]=lem.lemmatize(rootf[i])
    except:
        pass

#Calculate TF-IDF - term frequency - inverse document frequency
#(how important a word is in a document)

#calculating TF for the freq_words in docs

l=[]                #for storing each word in each file
for i in range(0,len(filewords)):
    l.append([])
    l[i]={}
    for j in range(0,len(filewords[i])):
        for k in range(0,len(filewords[i][j])):
            l[i][filewords[i][j][k].lower()]=0

for i in range(0,len(filewords)):                                   #calculating frequency of each word in each file
    for j in range(0,len(filewords[i])):
        for k in range(0,len(filewords[i][j])):
            try:                                
                l[i][filewords[i][j][k].lower()]=l[i][filewords[i][j][k].lower()]+1
            except(KeyError):
                pass

countwords=0
tf=l
for i in range(0,len(filewords)):
    for j in range(0,len(filewords[i])):
        countwords=countwords+len(filewords[i][j])
        for k in range(0,len(filewords[i][j])):
            try:
                tf[i][filewords[i][j][k].lower()]=l[i][filewords[i][j][k].lower()]/countwords
            except(KeyError):
                pass

#now calculate idf

x=len(fileinput)            #number of documents
allwords={}
for i in range(0,len(filewords)):
    for j in range(0,len(filewords[i])):
        for k in range(0,len(filewords[i][j])):
            allwords[filewords[i][j][k].lower()]=0

m=[]            
for i in range(0,len(filewords)):                                                           #this section of code is used finding the number of documents which contain the term filewords[i][j][k]
    for j in range(0,len(filewords[i])):
       for k in range(0,len(filewords[i][j])):
            if(filewords[i][j][k].lower() not in m):
                allwords[filewords[i][j][k].lower()]=allwords[filewords[i][j][k].lower()]+1
                m.append(filewords[i][j][k].lower())
    m=[]

allwords2=allwords
for i in range(0,len(filewords)):
    for j in range(0,len(filewords[i])):
        for k in range(0,len(filewords[i][j])):
            allwords2[filewords[i][j][k].lower()]=allwords2[filewords[i][j][k].lower()]/x           #x=number of  documents

idf=allwords2
for i in range(0,len(filewords)):
    for j in range(0,len(filewords[i])):
        for k in range(0,len(filewords[i][j])):
            idf[filewords[i][j][k].lower()]=math.log1p(idf[filewords[i][j][k].lower()])         #idf calculated
                    
#for i in range(0,len(filewords(i)):

tfidf=tf
for i in range(0,len(tfidf)):
    for j in tfidf[i]:
        tfidf[i][j]=tfidf[i][j]*idf[j]
#tfidf calculated

#Set appropriate weights & return sentences with the highest weights ---> SUMMARY
#According to many research papers and in general, the first sentence of a review usually holds crucial information.
#Assuming that, for the summary, 1st sentence will be returned.
#So remove the first sentence words from tfidf.

for i in range(0,len(filewords)):
	for j in filewords[i][0]:
		if j.lower() in tfidf[i]:
			tfidf[i].pop(j.lower())
			
word=[]                         #words from each file having max tf-idf value will be stored here
r=0                                 #variable for denoting the max tf-idf value
for i in range(0,len(tfidf)):
    r=0
    for j in tfidf[i]:
        if r<tfidf[i][j]:
            r=tfidf[i][j]
            try:
                word[i]=j
            except(IndexError):
                word.append(j)
print "Words with highest tf-idf in each document:"
print word                          #list of words with the highest tf-idf in each text. len(word)=7

#SUMMARY

#cue-phrases. 
cue=['concluding', 'therefore', 'moreover','hence','furthermore','summarizing','anyway']
#since summaries are in text, there wouldn't be any differences in font types.
#so, no need to consider bold, italic, etc. If it was a URL, then you have to consider.

for i in range(0,len(filesents)):
    print "Original text %d: " %(i+1)
    print fileinput[i]
    print "Summarized text:"
    print filesents[i][0]
    for j in range(1,len(filesents[i])):
        if word[i] in filesents[i][j]:
            print filesents[i][j]
    for y in range(0,len(freq_words)):              #check freq_words list.
        if word[i] in freq_words:                           #if a word is present in the list, return the sentence
            print filesents[i][y]
    for z in range(0,len(freq_words)):              #check cue-phrase word list
        if word[i] in cue:
            print filesents[i][z]
        else:
            pass
    print "\n"                      #or instead of printing 3-4 sentences of summary, we can also print ln(n) number of sentences where n=total number of sentences

#to increase speed, pickling can be done. using pickling, training of the dataset can be 'dumped' into a separate file. this file can later be imported.
    

            
			
    



import nltk
import codecs
import pronto
import itertools
#import owlready
#from owlready import *
from nltk.tokenize import PunktSentenceTokenizer,sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer, PorterStemmer
import signal, ctypes
from pronto import Ontology
from pronto import *
import re
import pandas as pd
from pandas import DataFrame
import math
from difflib import SequenceMatcher
import sys
import collections
from collections import Counter
from collections import OrderedDict
import statistics 
from nltk.wsd import lesk
#!pip install spacy
#!pip install  gensim
import string
gensim =  frozenset({'everyone', 'however', 'toward', 'perhaps', 'regarding', 'upon', 'detail', 'with', 'former', 'everything', 'himself', 'beyond', 'ourselves', 'though', 'their', 'computer', 'twenty', 'up', 'behind', 'how', 'whenever', 'move', 'me', 'noone', 'find', 'further', 'whereas', 'none', 'thin', 'were', 'nobody', 'always', 'our', 'becomes', 'cant', 'wherever', 'else', 'and', 'found', 'say', 'last', 'couldnt', 'my', 'nor', 'so', 'during', 'various', 'thru', 'another', 'off', 'while', 'besides', 'already', 'fill', 'does', 'again', 'seems', 'first', 'or', 'there', 'kg', 'as', 'now', 'inc', 'anything', 'interest', 'into', 'alone', 'unless', 'towards', 'do', 'eleven', 'at', 'ten', 'yourself', 'take', 'few', 'by', 'anyone', 'own', 'out', 'ltd', 'through', 'still', 'becoming', 'cannot', 'could', 'didn', 'km', 'she', 'above', 'those', 'themselves', 'used', 'other', 'part', 'con', 'eg', 'hereby', 'back', 'over', 'someone', 'un', 'top', 'anyway', 'go', 'must', 'front', 'seeming', 'bill', 'herein', 'cry', 'others', 'see', 'them', 'am', 'bottom', 'yourselves', 'nine', 'mostly', 'therein', 'enough', 'name', 'herself', 'whereafter', 'his', 'doing', 'within', 'sometime', 'latterly', 'among', 'was', 'show', 'can', 'who', 'has', 'keep', 'doesn', 'every', 'down', 'everywhere', 'between', 'per', 'often', 'that', 'have', 'amoungst', 'mine', 'than', 'thereby', 'afterwards', 'also', 'please', 'somewhere', 'once', 'never', 'neither', 'for', 'around', 'serious', 'via', 'de', 'somehow', 'full', 'i', 'both', 'forty', 'system', 'mill', 'would', 'make', 'after', 'which', 'nothing', 'empty', 'seemed', 'these', 'moreover', 'beside', 'hereafter', 'anyhow', 'when', 'from', 'ever', 'will', 'under', 'to', 'each', 'amongst', 'along', 'until', 'give', 'myself', 'whence', 'if', 'such', 'about', 'sixty', 'most', 'fifty', 'latter', 'before', 'ours', 'whoever', 'nevertheless', 'many', 'without', 'across', 'quite', 'might', 'its', 'it', 'but', 'otherwise', 'being', 'namely', 'ie', 'not', 'made', 'thereafter', 'yours', 'is', 'should', 'throughout', 'eight', 'sincere', 'well', 'more', 'using', 're', 'on', 'several', 'hers', 'thereupon', 'due', 'did', 'three', 'five', 'even', 'onto', 'the', 'except', 'then', 'don', 'side', 'whither', 'next', 'him', 'something', 'below', 'sometimes', 'been', 'seem', 'same', 'in', 'you', 'because', 'her', 'either', 'they', 'hundred', 'where', 'together', 'just', 'whereupon', 'hereupon', 'itself', 'almost', 'twelve', 'your', 'done', 'thick', 'meanwhile', 'beforehand', 'hasnt', 'fifteen', 'hence', 'whereby', 'some', 'anywhere', 'he', 'had', 'this', 'amount', 'yet', 'what', 'us', 'became', 'fire', 'call', 'against', 'formerly', 'become', 'four', 'rather', 'two', 'we', 'may', 'less', 'put', 'co', 'why', 'nowhere', 'really', 'here', 'of', 'too', 'six', 'get', 'although', 'thus', 'third', 'etc', 'wherein', 'describe', 'no', 'are', 'whole', 'whatever', 'elsewhere', 'indeed', 'any', 'therefore', 'whether', 'since', 'very', 'all', 'only', 'one', 'thence', 'an', 'least', 'whom', 'whose', 'much', 'be', 'a'})
from nltk.stem import PorterStemmer
porter = PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()



ont=Ontology('PresentationOntologyV4.2Validated-Anonymised.owl'  )
oneItemParens=[]
cleanedParents1=[]
cleanedParents2=[]
cleanedChildren1=[]
cleanedChildren2=[]
cleanedFirstChildren1=[]
cleanedFirstChildren2=[]
words=[]

## Find the parents of each term in the ontology and clean them
#####################
for i in ont.terms():
   


    for j in list(i.superclasses()):

        j=str(j).split('#',1)[1]
        j=re.sub('[^A-Za-z0-9]+', '', str(j))
        cleanedParents1.append(j)
        
        


    cleanedParents2.append(cleanedParents1)
        
    cleanedParents1=[] 
    



firstword=''
words=[]
for i in cleanedParents2:
    #print('i = ',i)
    #print()
    #print('re.findall(\ [A-Z][a-z]*\ , i[0]) = ',re.findall(\ [A-Z][a-z]*\ , i[0]))
    #print()

    if re.findall('[A-Z][a-z]*', i[0])!=[] and re.findall('[A-Z][a-z]*', i[0]) != i:#Tocheck it is not the topic itself
                 
        #print(re.findall('[A-Z][a-z]*', i[0]),'!=[]=' ,re.findall('[A-Z][a-z]*', i[0]))
        #print('re.findall(\ [A-Z][a-z]* , i[0]) != i',re.findall('[A-Z][a-z]*', i[0]))
        #print('len of i[0] ',i[0],'   ',len(re.findall('[A-Z][a-z]*', i[0])), '   ',len(re.findall('[A-Z][a-z]*', i[0])[0]))
        if len(re.findall('[A-Z][a-z]*', i[0]))>1 and len(re.findall('[A-Z][a-z]*', i[0]))<=2:
            
            if len(re.findall('[A-Z][a-z]*', i[0])[0]) <=3 and str(re.findall('[A-Z][a-z]*', i[0])[0])=='Non':
                firstword=re.findall('[A-Z][a-z]*', i[0])[0]+re.findall('[A-Z][a-z]*', i[0])[1]
                words.append([[i[0],firstword],i])
            
            elif len(re.findall('[A-Z][a-z]*', i[0])[0]) <3:
                firstword=re.findall('[A-Z][a-z]*', i[0])[0]+re.findall('[A-Z][a-z]*', i[0])[1]
                words.append([[i[0],firstword],i])
            else :
                words.append([[i[0],re.findall('[A-Z][a-z]*', i[0])[0],re.findall('[A-Z][a-z]*', i[0])[1]],i])
                #print('len of i[0] ',i[0],'   ',len(re.findall('[A-Z][a-z]*', i[0])),'is i>1 and <=2  ' )
                #print()
        elif len(re.findall('[A-Z][a-z]*', i[0]))>=3:
            words1=[]
            if len(re.findall('[A-Z][a-z]*', i[0])[0]) <=3 or str(re.findall('[A-Z][a-z]*', i[0])[0])=='Non':
                firstword=re.findall('[A-Z][a-z]*', i[0])[0]+re.findall('[A-Z][a-z]*', i[0])[1]
                 
                for j in range(2,len(re.findall('[A-Z][a-z]*', i[0]))):
                    words1.append(re.findall('[A-Z][a-z]*', i[0])[j])
                words.append([[i[0],firstword,words1],i])
                words1=[]
            else:
                for j in range(len(re.findall('[A-Z][a-z]*', i[0]))):
                    words1.append(re.findall('[A-Z][a-z]*', i[0])[j])
                words.append([[i[0],words1],i])
                     
            #print('len of i[0] ',i[0],'   ',len(re.findall('[A-Z][a-z]*', i[0])),'is >=3  ')
            #print()
        elif len(re.findall('[A-Z][a-z]*', i[0]))==1:
            words.append([[i[0]],i])
            #print('len of i[0] ',i[0],'   ',len(re.findall('[A-Z][a-z]*', i[0])),'is =1  ')
            #print()
    elif re.findall('[A-Z][a-z]*', i[0])==[]:
        words.append([[i[0]],i])
        #print(' i[0] ',i[0],'   ','is =[]  ' )
        #print()
             
    else:
        words.append([[i[0]],i])
        #print('no of the above',[[i[0]],i] )
        #print()
              


for i in words:
    print(i)
    print()


#sent=['Accent','body','language','future','composed','key','principle','nonverbal','PowerPoint']

def word_Concept_Extraction(sent):
    x=0
    
    wordConcept=[]
    match=False
    while x<=len(sent)-1:
        #print('x= ',x)
        #print('sent[x]= ',sent[x])
        #print()
        match=False
        for i in words:

            if match==False and len(i[0])==2 and type(i[0][1]) is list and len(i[0][1])>1 and x+1<=len(sent)-1:
                #word=wordnet_lemmatizer.lemmatize((str(sent[x])+str(sent[x+1])).lower(),"v")
                #termWord=wordnet_lemmatizer.lemmatize((str(i[0][1][0])+str(i[0][1][1])).lower(),"v")
                #if word==termWord:
                if (porter.stem(str(sent[x]))+(porter.stem(str(sent[x+1])))).lower()==(porter.stem(str(i[0][1][0]))+porter.stem(str(i[0][1][1]))).lower():
                    wordConcept.append([sent[x],sent[x+1],i[1]])
                    #print('wordConcept = ',[sent[x],sent[x+1],i[1]])
                    #print()
                    x+=2
                    match=True
                    break

            elif match==False and len(i[0])==1 and x<=len(sent)-1:
                #word=wordnet_lemmatizer.lemmatize(str(sent[x]).lower(),"v")
                #termWord=wordnet_lemmatizer.lemmatize(str(i[0][0]).lower(),"v")
                #if word==termWord:
                if sent[x].lower()==str(i[0][0]).lower():
                    wordConcept.append([sent[x],i[1]])
                    #print('wordConcept = ',[sent[x],i[1]])
                    #print()
                    x+=1
                    match=True
                    break

            elif match==False and len(i[0])>2 and x <=len(sent)-1:
                #print('i[0]= ',i[0])
                #print()

                partConcept=[]
                counterOfMatchedWords=0
                z=x

                for j in i[0]:
                    if type(j) is list:
                        j1=j[0]
                        termWord=wordnet_lemmatizer.lemmatize(str(j1).lower(),"v")
                        if z<=len(sent)-1:
                            word=wordnet_lemmatizer.lemmatize(str(sent[z]).lower(),"v")
                            #if word==termWord:
                            if sent[z].lower()==str(j1).lower():
                                
                                partConcept.append(sent[z])
                                #print('match if j = list = ',word,'   ',termWord)
                                #print()
                                counterOfMatchedWords+=1
                                z+=1
                                match=True

                    else:
                        #x=z
                        
                        if i[0].index(j)==0:
                            if z<=len(sent)-1:
                                word=wordnet_lemmatizer.lemmatize(str(sent[z]).lower(),"v")
                                #if word==termWord:
                                if sent[z].lower()==str(j).lower():
                                    partConcept.append(sent[z])
                                    #print('match j is not list = ',word,'   ',termWord)
                                    #print()
                                    counterOfMatchedWords+=2
                                    z+=1
                                    
                        else:
                            



                            termWord=wordnet_lemmatizer.lemmatize(str(j).lower(),"v")
                            if z<=len(sent)-1:
                                word=wordnet_lemmatizer.lemmatize(str(sent[z]).lower(),"v")
                                #if word==termWord:
                                if sent[z].lower()==str(j).lower():
                                    partConcept.append(sent[z])
                                    #print('match j is not list = ',word,'   ',termWord)
                                    #print()
                                    counterOfMatchedWords+=1
                                    z+=1
                                    #match=True
                #print('counterOfMatchedWords = ',counterOfMatchedWords)
                #print()
                if counterOfMatchedWords>=2:
                    wordConcept.append([partConcept,i[1]])
                    partConcept=[]
                    counterOfMatchedWords=0
                    match=True
                    x+=z
                    z=0
                elif counterOfMatchedWords<=1:
                    partConcept=[]
                    counterOfMatchedWords=0
                    match=False
                    x=x
      
        if match==False:
            

            x+=1
    return wordConcept


from nltk.tokenize import RegexpTokenizer
     
tokenizer = RegexpTokenizer(r'\\w+')

stop = set(stopwords.words('english'))
     
stop_words = stop | set(string.punctuation) | set(gensim) 
stop_words=list(stop_words)

#stop_words=stop_words.append('one')
#print('type of stop words = ',type(stop_words))
#print(stop_words)
def simpleFilter(sentence):
     
    filtered_sent = []
    newfiltered_sent=[]
    j=[]
    lemmatizer = WordNetLemmatizer()
    #stop_words = set(stopwords.words(\ english\ ))
    words1 = word_tokenize(sentence)
    #words1 =tokenizer.tokenize(sentence)
    #print('tokenize = ',words1)

    for w in words1:
        if w not in stop_words :
            w=lemmatizer.lemmatize(w).split('-'  )
            filtered_sent.append(w[0].title())


    for i in filtered_sent:

        newfiltered_sent.append(i)
        j = re.findall('[A-Z][a-z]*-', i)
        if j:
            if j[0] not in newfiltered_sent:

                newfiltered_sent.append(lemmatizer.lemmatize(j[0]).split("-" ).title())
                     
             
     
    return newfiltered_sent
####################################################################
     
####################################################################
     
######
def findExplicitMentioned(v):
    # I have ignored if the word is repeated but for the actual characterisation of videos I need to take care of repetiotion
    # Activitate the condition of if Sim not in explicit words
    explicit_word1=[]
    explicit_word2=[]
    pathsimilarity=[]

    #for i in words:
    #    print(i)
    #    print()
    print('v = ',v)

    #for x in v:      
    #    row=x
    #    print('row =',row)
    #   print()
    
    
    filteredrow=[]

    #filteredrow1=simpleFilter(row)
    filteredrow1=word_tokenize(v)
    stopWords=['[]','.','A']
    for i in filteredrow1:
        #print('i in filtered row =',i)
        #print()
        if i not in stopWords and len(i)>2:

            filteredrow.append(i)


    #explicit_word1.append(word_Concept_Extraction(filteredrow)  )
    explicit_word1.append(word_Concept_Extraction(filteredrow)  )
        
    return explicit_word1
         




######### 1. Find Explicit mentioned words by passing df of text #################
videosnumber=1 
videoCounter=0
commentSheet=[]
allCommnetSheet=[]
lemmatizer = WordNetLemmatizer()   #lemmatizes the words
ps = PorterStemmer()    #stemmer stems the root of the word.
stop_words = set(stopwords.words('english' ))
senSimilarity=[]
explicit_word=[]
span=[]
parents=[]
fullVideoCharacterisation=[]
allSegmentsCharatcerisation=[]

topicsList=[]
topicKeys=[]
topicValues=[] 
topics=[]    
focusTopic=[]
allTopicsFocus=[]
focusConcept1=[]
concept1OfAllTopicsAreFocus1=[]
superConcepts=[]
superConceptsCount=[]

subConcepts=[]
subConceptsCount=[]

subConceptsCount=[]
topicsWithFreq=[]
segmentTopicCharacterisation=[]
superConceptsCount=[]
transAsString=' '
##### Main Program Without intervals##############


print()

dfc=pd.read_csv("Pres Segments with Full Stop as Delimiter.csv", header='infer', encoding="utf8")
segmentId=list(dfc['SegmentId'])
startTime=list(dfc['StartTime'])
endTime=list(dfc['EndTime'])
transcript=list(dfc['Transcript'])
#topicList=list(dfc['Topic'])

video=segmentId[0]
segment=[]
videoSegment=[]

sId=[]
sTime=[]
eTime=[]
tr=[]
to=[]
for i in range(0,len(segmentId)):

    if segmentId[i]==video or str(segmentId[i])==str('nan'):
        sId.append(segmentId[i])
        sTime.append(startTime[i])
        eTime.append(endTime[i])
        tr.append(transcript[i])
        #to.append(topicList[i])
        transAsString+= str(' '+str(transcript[i]))


    else:
        sId1=list(set(sId))
        sTime1=[x for x in sTime if str(x) !='nan']
        eTime1=[x for x in eTime if str(x) !='nan']

        videoSegment.append([sId1,sTime1,eTime1,tr,transAsString])

        sId=[]
        sTime=[]
        eTime=[]
        tr=[]
        to=[]
        transAsString=''
        video=segmentId[i]
        sId.append(segmentId[i])
        sTime.append(startTime[i])
        eTime.append(endTime[i])
        tr.append(transcript[i])
        #to.append(topicList[i])
        transAsString+= str(transcript[i])

#print('# of segment = ',len(videoSegment))


print('***************************************')
#ontStemedWords=ontologyStemmedWords()
#for i in ontStemedWords:
#    print('ontology stem words = ',i)
#    print()
print('*************************************')
segmentCharacterisation=[]

print('# of all video segments before matching characterisation with BERT topics= ',len( videoSegment))
print()
#for i in words:
#    print('words= ', i)
#    print()
for i in videoSegment:
    bertTopics=[]
    print('i[0]= ',i[0])
    print()
    print('i[4]= ',i[4])
    print()
    explicitWords=findExplicitMentioned(i[4])
    print('explicitWords before removing []= ',explicitWords)
    print()
    print([i[0],i[1],i[2],i[3],explicitWords])
    print()
    
    segmentCharacterisation.append([i[0],i[1],i[2],i[3],explicitWords])
    
print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
print('# of segments found= ',len(segmentCharacterisation)) 
print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
print() 


conceptsLength=[]
for j in segmentCharacterisation:
    for i in j:
        print(i)
        print()
        
    print('***********************************')
for j in segmentCharacterisation:
    
    if j[4]:
        for c in j[4]:
            if c:
                #print('c= ',c)
                if len(c)>1:
                    for cc in c:
                        
                
                        conceptsLength.append(len(cc[1]))
                else:
                    conceptsLength.append(len(c[0][1]))
print('length of concepts = ',max(conceptsLength))
    


for i in segmentCharacterisation:
    for j in i:
        print(j)
        print()

            
    print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
    print()


import xlsxwriter
workbook = xlsxwriter.Workbook('Pres Semantic Tagging-Seg With FullStop-12-7-22 .xlsx')
worksheet = workbook.add_worksheet('Sheet1')
worksheet.write('A1', 'SegmentId')
worksheet.write('B1', 'StartTime')
worksheet.write('C1', 'EndTime')
worksheet.write('D1', 'Transcript')
worksheet.write('E1', 'Concept Path')

row=1
col=0



videoName=[]
start=[]
end=[]
trans=[]
transcript=[]
for i in segmentCharacterisation:
    
    if len(i[0])>1:
        if str(i[0][0])!='nan':
            worksheet.write(row,col,i[0][0])
        else:
            worksheet.write(row,col,i[0][1])
    elif len(i[0])==1:
        worksheet.write(row,col,i[0][0])
            
    worksheet.write(row,col+1,i[1][0])
    worksheet.write(row,col+2,i[2][0])
    oldCol=col+3
    oldRow1=row
    for j in i[3]:
        
        if str(j)!='nan':
            worksheet.write(oldRow1,oldCol,j)
            oldRow1+=1
    
    oldRow2=row
    oldCol=col+4
    if i[4]:
        for j in i[4]:
            if j:
                    
                for k in j:
                    for k1 in k[-1]:
                    
                        worksheet.write(oldRow2,oldCol,k1)
                        oldCol+=1
                    oldRow2+=1
                    oldCol=col+4
                        
    if oldRow1>oldRow2:               
        row=oldRow1+1
    else:
        row=oldRow2+1
        
    col=0
workbook.close()      


# In[39]:


# Wrong code, the combining code in the combinign folder
videosnumber=1 
videoCounter=0
commentSheet=[]
allCommnetSheet=[]
lemmatizer = WordNetLemmatizer()   #lemmatizes the words
ps = PorterStemmer()    #stemmer stems the root of the word.
stop_words = set(stopwords.words('english' ))
senSimilarity=[]
explicit_word=[]
span=[]
parents=[]
fullVideoCharacterisation=[]
allSegmentsCharatcerisation=[]

topicsList=[]
topicKeys=[]
topicValues=[] 
topics=[]    
focusTopic=[]
allTopicsFocus=[]
focusConcept1=[]
concept1OfAllTopicsAreFocus1=[]
superConcepts=[]
superConceptsCount=[]

subConcepts=[]
subConceptsCount=[]

subConceptsCount=[]
topicsWithFreq=[]
segmentTopicCharacterisation=[]
superConceptsCount=[]
transAsString=''
segmentCharacterisation=[]
##### Main Program Without intervals##############


print()

dfc=pd.read_csv("PresentationBertBaseUncaseWithFullStop-Structure.csv", header='infer', encoding='cp1252')
segmentId=list(dfc['SegmentId'])
startTime=list(dfc['StartTime'])
endTime=list(dfc['EndTime'])
transcript=list(dfc['Transcript'])
topicList=list(dfc['Topic'])

video=segmentId[0]
segment=[]
videoSegment=[]

sId=[]
sTime=[]
eTime=[]
tr=[]
to=[]
for i in range(0,len(segmentId)):

    if segmentId[i]==video or str(segmentId[i])==str('nan'):
        sId.append(segmentId[i])
        sTime.append(startTime[i])
        eTime.append(endTime[i])
        tr.append(transcript[i])
        to.append(topicList[i])
        transAsString+= str(' '+str(transcript[i]))


    else:
        sId1=list(set(sId))
        sTime1=[x for x in sTime if str(x) !='nan']
        eTime1=[x for x in eTime if str(x) !='nan']

        videoSegment.append([sId1,sTime1,eTime1,tr,transAsString,to])

        sId=[]
        sTime=[]
        eTime=[]
        tr=[]
        to=[]
        transAsString=''
        video=segmentId[i]
        sId.append(segmentId[i])
        sTime.append(startTime[i])
        eTime.append(endTime[i])
        tr.append(transcript[i])
        to.append(topicList[i])
        transAsString+= str(transcript[i])

#print('# of segment = ',len(videoSegment))


print('***************************************')
#ontStemedWords=ontologyStemmedWords()
#for i in ontStemedWords:
#    print('ontology stem words = ',i)
#    print()
print('*************************************')


print('# of all video segments before matching characterisation with BERT topics= ',len( videoSegment))
print()
#for i in words:
#    print('words= ', i)
#    print()
for i in videoSegment:
    bertTopics=[]
    print('i[0]= ',i[0])
    print()
    print('i[4]= ',i[4])
    print()
    explicitWords=findExplicitMentioned(i[4])
    print('explicitWords before removing []= ',explicitWords)
    print()
    #print('Counter(explicitWords).items() = ',Counter(explicitWords).items())
    #print()
    
       

    for j in Counter(i[5]).items():
        if j[0]=='Visual Aids':
            v='VisualAid'
            bertTopics.append(v)
        else:
            bertTopics.append(j[0])


    print('bert topics= ',bertTopics)
    print()

    matchingTopics=[]

    # the implicit list contains all the topics and associated concepts within each segment
    print('explicitWords = ',explicitWords)
    print()
    if explicitWords: 
        #print('explicitWords = ',explicitWords)
        #print()


        for j in explicitWords:
            for j1 in j:
                #print('j1 of explicit words= ',j1)
                #print()
                if j1[-1][-1] in bertTopics:
                    matchingTopics.append(j1)
                elif j1[-1][-1]=='PresentationAttribute':
                    matchingTopics.append(j1)

                #else:
                #    matchingTopics.append([])



    if matchingTopics:
        segmentCharacterisation.append([i[0],i[1],i[2],i[3],matchingTopics])
    else:
        segmentCharacterisation.append([i[0],i[1],i[2],i[3],[]])
        
#else:
#    segmentCharacterisation.append([i[0],i[1],i[2],i[3],'No Characterisation'])
print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
print('# of segments found= ',len(segmentCharacterisation)) 
print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
print() 


# In[40]:


# Wrong code unneeded. The combining is in the combining folder 
import xlsxwriter
workbook = xlsxwriter.Workbook('Pres Semantic Tagginh With Bert-Seg With FullStop .xlsx')
worksheet = workbook.add_worksheet('Sheet1')
worksheet.write('A1', 'SegmentId')
worksheet.write('B1', 'StartTime')
worksheet.write('C1', 'EndTime')
worksheet.write('D1', 'Transcript')

row=1
col=0
l=0
lengthOfConcepts=[]
transMatchedWOrds=0
lengthOfTransMatchedWords=[]
print('*****************************************')
print()
print('********* Segment Characterisation length = ',len(segmentCharacterisation))
print()
for j in segmentCharacterisation:
    if j[4]:
        for c in j[4]:

            lengthOfConcepts.append(len(c[0]))
            for x in c[:-1]:
                if type (x) is list:
                    transMatchedWOrds+=(len(x))

                else:
                    transMatchedWOrds+=1
            lengthOfTransMatchedWords.append(transMatchedWOrds)
            transMatchedWOrds=0
    
    
#print('lengthOfConcepts = ',lengthOfConcepts)
#print()

topicColumn=max(lengthOfConcepts)+1
#print('topicColumn = ',topicColumn)

##  to get the max length of trans (word, noun phrase) per line that has match with ontology. This to know where to start
#writing the concept terms
transConceptColumn=max(lengthOfTransMatchedWords)
print()
worksheet.write(0,topicColumn, 'Topic')
for j in segmentCharacterisation:
    
    print('j[0]= ',j[0])
    print()
    
    print('row= ',row)
    if len(j[0])>1: # to write the ID of the segments. It sometimes comes first and nan last or vise versa
        if str(j[0][0])=='nan':
            worksheet.write(row,col,str(j[0][1]))
        else:
            worksheet.write(row,col,str(j[0][0]))
            
    else:
        worksheet.write(row,col,str(j[0][0]))
        
    worksheet.write(row,col+1,str(j[1][0]))
    worksheet.write(row,col+2,str(j[2][0]))
    l=row
    #write the transcript
    for t in j[3]:
        
        
        worksheet.write(l,col+3,str(t))
        l+=1
        
    conceptColumn=col+3
    conRow=row
    conCol=col+4
    
    # write the concepts
    #First I went through the words (single word or a phrase) from the transcript that has match with the ontology
    #and write them
    if j[4]:
        
       
        # Here I set the counters for the columns to start after the longest noun phrase found in the transcript
        #I need to start writing the concept infront of each tran word,
        #so I started from the first row I used to write the first word in the transcript=  transConceptColumn
        # I set conCol2 to write the full concept for each trans word and
        #start from the first column for the concepts to write the second concept of the second trans word
       
        conCol=(col+4)+transConceptColumn
        conRow=row
        conCol2=conCol
        for c in j[4]:
            print('c[-1]= ',c[-1])
            print()
            i=0
            for i in range(len(c[-1])-1):
                worksheet.write(conRow,conCol2,c[-1][i] )
                #print('conRow = ',conRow,'  ','c[0][i] = ',c[0][i])
                conCol2+=1
            #worksheet.write(conRow,conCol,c[1][-1])
            worksheet.write(conRow,topicColumn,c[-1][-1] )
            #print()
            #print('conRow = ',conRow,'  ','c[0][len(c[0])-1] = ',c[0][len(c[0])-1])
            #print()
            conRow+=1
            conCol2=conCol
      
               
    else:
        worksheet.write(conRow,conCol,'' )
        conRow+=1
        
        
    if l>conRow:
        row=l+1
    else :
        row=conRow+1
            
            
    
workbook.close()  








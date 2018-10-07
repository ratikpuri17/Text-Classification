
# coding: utf-8

# In[20]:


#importing the necessary libraries
import os
import nltk
import glob
from nltk.corpus import stopwords
nltk.__path__
from collections import Counter
import numpy as np
import pandas as pd
import string
import math
import operator


# In[3]:


os.getcwd()


# In[53]:


#making a dictionary with all the possible classes as its keys
path="C:\\Users\\ratik\\Anaconda3\\text_classification\\20_newsgroups_train"
l=[]
for root,dirs,files in os.walk(path):
    for name in dirs:
        #print (name)
        l.append(name)
l
newdict={}
len1=len(l) 
len1
for i in range(len1):
    newdict[l[i]]={}
newdict


# In[56]:


#reading the files in every class
g=[]
f=[]
stop_words=set(stopwords.words('english'))
dict1={}
for i in l:
    new_path="C:\\Users\\ratik\\Anaconda3\\text_classification\\20_newsgroups_train\\"+i
    print(new_path)
    for infile in glob.glob(os.path.join(new_path,'*')):
        g=[]
        file=open(infile,'r',encoding = "ISO-8859-1").read()
        g=file.split()
        list1=['Date:','date:']                                        #Filtering headers on the basis of date
        #print(g)
        flag1=0
        #g=g[50:200]
        g=[aa.lower() for aa in g]
        count1=Counter(g)
        #print(g)
        for w in g:
            if(w in list1):
                #print('j')
                flag1=1
            
            if w not in stop_words and w not in string.punctuation:   #removing stop words and punctuations
                 if(flag1==1):
                    f.append(w)
            dict1[infile]=count1
            dict1[infile]["total_counnt"]=len(g)
        #print(newdict[i])

#here we have a dictionary storing the count of every word after filtering
count=Counter(f) 


# In[57]:


#building our features which consists of a vocabulary of meaningful words
features=[]
for i,j in count.most_common(13000):
    features.append(i)

new_list = []
w=['rape','society','murder','abortion','beast','value','truth','teachings','Christ','Commandments','Golden','Dawn','Bible','directories','DoubleSpace','driver','Windows','computing','microsoft']
hello=['(i','would','like','a.', '*/',"i'm",'(or','>:','_/','15','7','>and','it,',"can't",'it.', 'x','>in','may','/*','>>>','think','writes:','also','...','3',"i've", '>i',"max>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'","i'd",'(and','----------------------------------------------------------------------',"i'll",':-)','>it','13', '32','is.','17','93','4.','is?','\\\\', '76','???', "'93"'???','--------','65','e-mail']
hello1=[ 'care', 'answer', 'opinions', 'claim', 'source', 'open', 'working', 'mac', 'including', 'institute',  'note',  'talking','private',  'single', 'play', 'provide',  'tried', 'couple', 'guess', 'rights', 'turkish', 'among', 'anybody', 'israel', 'questions', 'unless', 'check', 'price', 'likely', 'box', 'type', 'major', 'matter', 'wrong', 'sort', 'sun', 'become', 'encryption', 'speed', 'known', 'issue', 'ftp', 'home', 'include', 'ca', 'per', 'posting', 'address', 'taken', 'department', 'scsi',  'important', 'written', 'exactly', 'according', 'goes', 'black', 'san', 'western', 'states', 'usually', 'color', 'cost', 'games', 'value', 'common''hedrick@athos.rutgers.edu','--','university', 'world', 'information', 'point', 'data', 'file', 'years', 'state', 'different', 'set',  'question', 'run', 'fact', 'support', 'case',  'drive', 'call', 'hard', 'wed,', 'public', 'usa', 'second', 'real', 'mean', 'reason', 'looking', 'key', 'old', 'year', 'bit', 'free', 'general', 'group', 'list', 'news', 'line', 'version', 'research', 'david', 'post', 'possible', 'idea', 'game', 'systems', 'local', 'left', 'me.', 'so,', 'message', 'bad', 'pretty', 'true', 'already', 'wrote:', 'hope', 'john', 'running', 'current', 'place', 'original', 'access', 'card', 'word', 'large', 'mr.', 'small', 'approved:', 'others', 'change', 'files', 'buy', 'email', 'standard', 'man', 'problems', 'inc.', 'american', 'fbi', 'center', 'book', 'started', 'times',  'interested', 'bill', 'time.' '--', "that's",'apr','gmt','"only','allow', 'no,','hedrick@geneva.rutgers.edu',  "let's", 'pc', 'a.', 'talk', 'early', '*******', 'article-i.d.:', '1%', '/____/', 'baalke@kelvin.jpl.nasa.gov', '|__', 'needs','called','thanks','c','everyone', '28', 'works','nntp-posting-host:' ,'start',  '25', 'mon,','following','17', '23', 'thu,',  'fri,','tue,','gmt', 'enough','christian@aramis.rutgers.edu','distribution:', 'reply-to:', 'thu','certain','date:', 'message-id:', 'sender:', '21', 'organization:', 'references:', 'nntp-posting-host:''known', 'write', 'consider','april','26','15', 'understand','30', 'full',  'ask', 'pay', 'feel',  'went', 'mail','found','13','simply',   'says', '27', 'kind','send', 'far', 'whether', 'quite', 'work', 'article','several','32','gets','>this','up.', 'them,', "what's",'$100','-->','2)','saw', 'this?', '54','6)',  '(415)', '--if', 'it...', "it'll", '81','whoever', '>well,', '>who', '>>>>','8,', '20,','hopefully', '44',"'92",  '>>the','72',"'91",'it)', 'thoughts', '>which',   '---------','$1', 'away','2.','3.', 'so.', 'mentioned', '(as', 'j.','#>','whose', 'him.','14','11','took','supposed','too.', 'p.','say,', 'too.', 'p.', '12','24','cause','20','within','similar', 'instead', 'via',  'everything','often','well,', 'that.', 'lines:', 'me,','though','now,','making',  'came', 'perhaps', 'comes',  'means', 'based',  'heard','ever','seen','given', 'seem', 'always', 'three', 'maybe', 'less','next', 'new','nothing', 'great','>were', 'you!', '>>for', '(c)', 'thanx','=============================================================================','however.' 'article','else', 'available','copy', 'that,',  'makes','use.', '(--)',  'yes.','\\/', 'k',  '29', '1st','------------------------------------------------------------------------------',  'do.', 'knew', '>if',':)','>it', 'yet', 'remember', 'trying','93', '4.','16', '9', 'db','>as', '__', 'yes',  '>--','"i','said,', 'clear', 'difference', 'asked', 'especially', 'four', 'upon','lot', 'able', 'keep','least','let',  'long', 'tell','help','look', 'problem', 'however,', 'best', 'thing', 'rather', 'first', 'since', 'really', 'anyone', 'right', 'something','bd','people','much','part','every', 'little', 'used', 'need', 'good','no.','***', 'on,','//', '80','----------','(a)','58', 'u.', 'co', '>>is', 'cdt@vos.stratus.com', '*****','--------------------------------------------------------------------------', '73', '_/_/_/',  'm"`@("`@("`@("`@("`@("`@("`@("`@("`@("`@("`@("`@("`@("`@("`@(','>their','*******************************************************************************', 'pa', '(the',  '_/','/*', "there's","they're", 'also,', 'gave','>that', '>a', 'cut', 'not,', '1)','>what','---------------------------------------------------------------------', 'bbs', 'w.', '3)', '>>in', 'oh', 'm.', 'joe', 'became', '18','still', 'around', 'it,','although','actually','them.','almost', 'show', 'agree', 'certainly','anything', 'try','give', 'put', 'seems', 'probably','might','would','made', 'well', 'got', 'another', 'better', 'number', 'come', 'sure', 'things', 'someone','never', 'please', 'back', 'last', 'without','going', 'find', 'using','64','*not*',"he's",'want','way', 'say','>of','it?', '??','to.','----------------------------------------------------------------------------','-------------------------------------------------------------------------------', 'hp', 'lets', '55', '$3','etc.','|>|>', 'fine,', '"i\'m','>and','you.', 'you.','saying','like','8', 'whole', '6', 'told','make', 'is,','many','either', '1993','go', 'us','see','said', 'take','q','???','>:',  '>>>','six', 'r','>is','19', '22', '55.0', 'dr.', 'all,', '**','||', '(in','----','----------------------------------------------------------------------','$50','>|>',"i'll",  'you,', '>to','end', 'this.', '*/','(or',  '(i','"the','big', 'evidence', 'getting', 'day', 'thought', 'car', 'done', 'cannot', 'high', 'order', 'person', '10', 'name', '"the''----------------------------------------------------------------------','!=','--------------------------------------------------------------------------------', '7', '---',':-)','(and',':-)', ':-)',':-)' 'is,', 'one', '|>','5', '...', '>in','must', "i've",'>the','4', '>i',"i'd",  '3', "can't",'get', '>>', 'know', 'x','two', '0', 'it.', 'time', '2', "max>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'ax>'", 'think', 'writes:', 'use', 'also', 'could', 'even', '1', 'may', "i'm",]
for e in features:
    if (e not in hello1) and (e not in hello) and (e.isdigit()==False):   #removing certain more meaningless words
        new_list.append(e)
for z in w:
    if(z not in new_list):
        new_list.append(z)
features= new_list
print(features)
#we obtain the FINAL FILTERED - VOCABULARY consisting of our FEATURES
len(features)


# In[58]:


#Again traversing every file to find out the count of the occurence of every feature in the particular file 
for i in l:
    new_path="C:\\Users\\ratik\\Anaconda3\\text_classification\\20_newsgroups_train\\"+i
    print(new_path)
    for j in features:
        newdict[i][j]=0
    newdict[i]['total']=0                                   #stores total number of files in that particular class
    for infile in glob.glob(os.path.join(new_path,'*')):
        g=[]
        file=open(infile,'r',encoding = "ISO-8859-1").read()
        g=file.split()
       # g=g[50:]
        g=[aa.lower() for aa in g]
        #print(g)
        newq=[]
        for w in g:
            if(w in features):
                newq.append(w)
        count1=Counter(newq)
        newlist=list(count1)
        for j in newlist:
            newdict[i][j]=newdict[i][j]+count1[j]
        newdict[i]['total']=newdict[i]['total']+len(newq)                  


# In[59]:


#testing the files on the basis of testing data

totalcount=0
intotal=0

for u in l:
    out=[]                #consists of predicted classes of the emails to be tested
    dict1test={}  
    new_path="C:\\Users\\ratik\\Anaconda3\\text_classification\\20_newsgroups_test\\"+u
    print(new_path)
    for infile in glob.glob(os.path.join(new_path,'*')):
        g=[]
        file=open(infile,'r',encoding = "ISO-8859-1").read()
        g=file.split()
        #g=g[50:]                      #to remove headers
        g=[aa.lower() for aa in g]
        countdict1=Counter(g)
        dict1test[infile]=countdict1
        dict1test["total_count"]=len(g)
        newtest=[]
        for w in g:
            if(w in features):
                newtest.append(w)
            max1=-1*math.inf
            output='a'
        counttest=Counter(newtest)
        for i in l:
            finalval=1
            for j in features:
                temp1=newdict[i][j]+1                    #laplace correction - adding 1 to the numerator
                
                temp2=newdict[i]['total']+len(features)  #laplace correction - adding the length of the vocab to the denominator
                val=math.log(temp1)-math.log(temp2)      #calculating log probability
            
                val=(val)*(counttest[j])                 #multiplying probiblity by the occurence count of the word
            
                finalval=finalval+val
        
            if(finalval>max1):                           #picking the class with the max probability
                max1=finalval
                output=i
        
        out.append(output)
    count1=0
    count2=0
    for p in out:
        if(p==u):
            count1=count1+1
        count2=count2+1
    print("Correct Predicted: ",count1,"----","Total: ", count2)
    totalcount=totalcount+count1
    intotal=intotal+count2
    #print(totalcount)
print("accuracy=",totalcount/intotal)


# In[63]:


np.array(out).shape


# #### Predictions

# In[65]:


out


# ## Using Sklearn Implementation

# In[7]:


import string
import os


# In[11]:


import pandas as pd
import numpy as np
from collections import Counter
import nltk


# In[21]:


stop_words=stopwords.words('english')


# In[4]:


punc=string.punctuation


# In[8]:


#target
path="C:\\Users\\ratik\\Anaconda3\\text_classification\\20_newsgroups_train"
l=[]
for root, dirs, files in os.walk(path):
    for name in dirs:
        l.append(name)


# In[22]:


path="C:\\Users\\ratik\\Anaconda3\\text_classification\\20_newsgroups_train"
filenames=[]                            #storing filenames corresponding to every class
for root, dirs, files in os.walk(path):
    for name in files:
        
        filenames.append(name)
        


# In[85]:


#claculating features for vocabulary
import glob
g=[]
for fname in l:
    path="C:\\Users\\ratik\\Anaconda3\\text_classification\\mini_newsgroups_train\\"+fname
    for infile in glob.glob(os.path.join(path, '*')):
        review_file = open(infile,'r').read()
        g+=review_file.split()

g=[i.lower() for i in g]
allwords=[]
for word in g:
    #filtering the words
    if word not in stop_words and word not in punc and (word.isdigit()==False) and(word.isalnum()==True):
        allwords.append(word)
count=Counter(allwords)
vocab=count.most_common(3000)
feat=[vocab[a][0] for a in range(len(vocab))]


# In[103]:


#har document ke corresponding vocabulary mei count check karvao

di={}

for foldername in l:
    path="C:\\Users\\ratik\\Anaconda3\\text_classification\\mini_newsgroups_train\\"+foldername
    print(path)
    di["total_count"]=len(l)
    di[foldername]={}
    for infile in glob.glob(os.path.join(path, '*')):
        review_file = open(infile,'r')
        filename=review_file.name.split("\\")[-1]
        
        di[foldername][filename]={}
        
        review_file.seek(500)                          #to remove header
        fptr=review_file.read().split()
        di[foldername][filename]["total_count"]=len(fptr)
        for f in feat:
            di[foldername][filename][f]=fptr.count(f)
        
        
#print(di)


# In[120]:


#storing column names for the dataframe
col=[]
col.append("Filename")
for y in feat: 
    col.append(y)

#col.append("Class")
col


# In[121]:


#assigning columns to the dataframe
dd=pd.DataFrame(columns=col)
dd.head()


# In[104]:


#builing dictionary to store in dataframe
first_layer=di.keys()
bigdata=[]                             #stores data for rows of dataframe
#ydata=[]
for fl in first_layer:                 #classes
    if(fl=="total_count"):
        continue
    #data.append(fl)
    second_layer=di[fl].keys()
    for sl in second_layer:            #filnames
        data=[]
        data.append(sl)
        third_layer=di[fl][sl].keys()
        for tl in third_layer:         #count of respetive features
            if(tl=="total_count"):
                continue
            data.append(di[fl][sl][tl])
            #ydata.append(fl)
            
        #data.append(fl)    
        bigdata.append(data)          #stores the row


# In[109]:


ydata=[]
for i in l:
    for j in range(0,75):
        ydata.append(i)

ydata=np.array(ydata)
ydata.shape


# In[126]:


#har document ke corresponding vocabulary mei count check karvao
#making dictionary for the tesing dataframe
ditest={}

for foldername in l:
    path="C:\\Users\\ratik\\Anaconda3\\text_classification\\mini_newsgroups_test\\"+foldername
    print(path)
    ditest["total_count"]=len(l)
    ditest[foldername]={}
    for infile in glob.glob(os.path.join(path, '*')):
        review_file = open(infile,'r')
        filename=review_file.name.split("\\")[-1]           #gives filename
        
        ditest[foldername][filename]={}
        
        review_file.seek(500)                          #to remove header
        fptr=review_file.read().split()
        ditest[foldername][filename]["total_count"]=len(fptr)
        for f in feat:
            ditest[foldername][filename][f]=fptr.count(f)
        
        
#print(ditest)
dt=pd.DataFrame(columns=col)
firstlayer=ditest.keys()
bigdat=[]
#ydat=[]
for fl in firstlayer:                      #classes
    if(fl=="total_count"):
        continue
    #data.append(fl)
    secondlayer=ditest[fl].keys()
    for sl in secondlayer:                 #filenames
        dat=[]
        dat.append(sl)
        thirdlayer=ditest[fl][sl].keys()
        for tl in thirdlayer:             #count of respective feature
            if(tl=="total_count"):
                continue
            dat.append(ditest[fl][sl][tl])
            #ydat.append(fl)
            
        #data.append(fl)    
        bigdat.append(dat)                #appending the testing dile first row


# In[114]:


np.array(bigdat).shape


# In[123]:


#represents ytest-expected ouputs
ydat=[]
for i in l:
    for j in range(0,25):
        ydat.append(i)
ydat=np.array(ydat)
ydat.shape


# In[135]:


#creating testing dataframe
for i in range(len(bigdat)):
    dt.loc[i]=bigdat[i]


# In[ ]:


dt.head()


# In[ ]:


y_test=np.array(dt)


# In[133]:


x_test=np.array(bigdat)


# In[134]:


x_test.shape


# In[117]:


bigdata=np.array(bigdata)


# In[125]:


#training dataframe
for i in range(len(bigdata)):
    dd.loc[i]=bigdata[i]


# In[129]:


dd.head()


# In[130]:


x_train=np.array(dd)
x_train


# In[131]:


y_train=ydata


# In[36]:


from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()


# In[101]:


type(x_train),type(y_train)


# In[63]:


import numpy as np


# In[132]:


gnb.fit(x_train,y_train)


# In[ ]:


#x_test=np.array(a)


# In[136]:


x_test=np.array(dt)
x_test.shape


# In[137]:


ypred=gnb.predict(x_test)


# In[138]:


y_test=np.array(ydat)
y_test.shape


# In[139]:


gnb.score(x_test,y_test)


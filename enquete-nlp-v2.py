# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 10:14:18 2019

@author: Leonardo.Galler
@note: NLP research of answers to surveys at the cooperative level. 
       The v2 is aimed to develop the research without the creation of intermediate 
       datasets.
"""

def readData( file ):
    '''
    Function to read the datafile.
    input: the location and name of the file
    output: Dataframe
    '''
    import pandas as pd
    try:
        # Folder of the data
        data = pd.read_csv(file , encoding = "latin_1", delimiter = ";", header = 0 )
    except:
        print("It was not possible to read the data!\nPlease verify the file.")
    return data

#Creating a year column and adding to the dataframe
def addYearColumn( data ):
    '''
    Create a year column for better filtering of data.
    Input: Dataframe
    Output: Dataframe with an added year column    
    '''
    data["Ano"] = data['Ano-Mês Votação'].str.slice(0,4)
    return data

# Create year list
def createYearList(data):
    '''
    Create a list of year for control of processing through years.
    Input: Dataframe
    Outuput: List of unique years 
    '''
    years_list = [year for year in data["Ano"].sort_values().unique() if str(year) != 'nan']
    return data , years_list

def splitDataPerYear_unsat( data , year):
    '''
    Function to split the data per year and filter for unsatisfied and where the comments did not have "NAO SE APLICA".
    Input: a dataframe with unfiltered data
    Output: Dataset per year
    '''
    # Subsetting for unsatisfied
    ## Marking all the values that correspond to unsatisfied
    data_aux = data[data["Ano"] == str(year)].where(data['Código Resposta'] == 2, inplace = False)
    ### Removing satisfied
    data_aux.dropna(inplace = True)
    #### Marking where the comments are like 'NAO SE APLICA'
    data_aux.where(data_aux['Comentário Usuário'] != 'NAO SE APLICA', inplace = True, axis = 0)
    ##### Removing items with no comments
    data_aux.dropna(inplace = True)
    
    return data_aux
    
### Breaking the information by years and filtering just to unsatisfied users
    
def splitDataPerYear_sat( data , year ):
    '''
    Function to split the data per year and filter for satisfied and where the comments did not have "NAO SE APLICA".
    Input: a dataframe with unfiltered data
    Output: Dataset per year
    '''
    # Subsetting for unsatisfied
    ## Marking all the values that correspond to unsatisfied
    data_aux = data[data["Ano"] == str(year)].where(data['Código Resposta'] == 1, inplace = False)
    ### Removing satisfied
    data_aux.dropna(inplace = True)
    #### Marking where the comments are like 'NAO SE APLICA'
    data_aux.where(data_aux['Comentário Usuário'] != 'NAO SE APLICA', inplace = True, axis = 0)
    ##### Removing items with no comments
    data_aux.dropna(inplace = True)
    
    return data_aux

def breakYears_uns( data , year ):
    '''
    Function to generate summary information of the general dataset per year and at the end run the fuction to split the data.
    Input: Dataframe and a list of years
    Output: Dataframes broken by year and with unsatisfied users
    '''
    # Creating the datasets for years
    return splitDataPerYear_unsat(data , str(year))

def summaryView( data , year ):
    import matplotlib.pyplot as plt

    print("### Início Informações para o ano "+str(year)+". ###")
    
    # Counting unique users per year-month
    #print("Quantidade Respondentes Únicos por mês: ", data[data["Ano"] == year].groupby("Ano-Mês Votação")['Código Login'].nunique())
    def to_series( data , year , grouping_list):
        '''
        Function to create the grouped data
        Input: dataframe, year string and a list of items to group
        Output: Series with grouped data
        '''
        return data[data["Ano"] == year].groupby(grouping_list)['Código Login'].nunique()
    
    # From series to frame
    toFrame = lambda series : series.to_frame()
    
    # Reindex de df
    reindex = lambda frame : frame.reindex()
    
    # Rename columns
    renamed = lambda rename_df : rename_df.rename(columns = {'Código Login':'QTD'})
    
    # Plotting the data
    usersPerMonth_plot = lambda data : data.plot(kind = "bar" , title = "Unique users per year-month", colormap = "summer" , figsize = (10 , 10))
    
    usersPerMonth_plot( renamed( reindex( toFrame( to_series( data , year , ["Ano-Mês Votação"] ) ) ) ) )
    plt.show()
    
    # Counting unique users per satisfaction and central
    usersPerCentralPerSatisf_plot = lambda data : data.plot(kind = "bar" , stacked=True , title = "Unique users per Central and Satisfaction" , colormap = "summer" , figsize = (10 , 10))
    usersPerCentralPerSatisf_plot( renamed( reindex( toFrame( to_series( data , str(year) , ["Número Central","Código Resposta"] ) ) ) ) )
    plt.show()
    
    # Counting unique users per year-month and satisfaction 
    usersPerMonthPerSatisf_plot = lambda data : data.plot(kind = "bar" , stacked=True , title = "Unique users per year-month, Central and Satisfaction" ,  colormap = "summer" , figsize = (10 , 10))
    usersPerMonthPerSatisf_plot( renamed( reindex( toFrame( to_series( data , str(year) , ["Ano-Mês Votação","Código Resposta"] ) ) ) ) )
    plt.show()
    
    print("### Fim de Informações para "+str(year)+". ###\n")


# Creating the column with tokens from the comments      
def breakYears_sat( data , year ):
    '''
    Function to generate summary information of the general dataset per year and at the end run the fuction to split the data.
    Input: Dataframe and a list of years
    Output: Dataframes broken by year and with satisfied users
    '''    
          
    # Creating the datasets for years
    return splitDataPerYear_sat(data , str(year))

def createTokens( data ):
    '''
    Function to tokenize the comment column
    Input: A dataframe
    Output: Lower case tokens creating a new column
    '''
    data["comment_list"] = data['Comentário Usuário'].str.replace('\n',' ').str.lower().str.split(" ")
    
    return data

### Function to create the word list
def createWordList_uns(data_unsatisfied):
    '''
    Create a list of all words
    Input: Dataframe
    Output: A list with all words
    '''
    ## Creating a list with all the words
    wordsList = []
    for lista in data_unsatisfied["comment_list"]:
        for word in lista:
            wordsList.append(word)
    # Return the list
    return wordsList

def createWordList_sat(data_satisfied):
    '''
    Create a list of all words
    Input: Dataframe
    Output: A list with all words
    '''
    ## Creating a list with all the words
    wordsList = []
    for lista in data_satisfied["comment_list"]:
        for word in lista:
            wordsList.append(word)
    # Return the list
    return wordsList

def remSpeChar(word): 
    '''
    Function to remove special characters and numbers from the tokens
    Input: A string
    Ouput: A string
    '''
    symbols_list = ["\\n","/",'"',"*" , "(" , ")" , ":" , ";" ,"," , "." , "%" , "#" , "?" , 
                    ".","!","'","-","+","|","=","[","]", '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    new_word = ''
    for letter in word:
        if(letter in symbols_list):
            new_word = new_word + '' 
        else:
            new_word = new_word + letter
    
    return new_word
    ### End of function

def remSpecChar_fromList(lists): 
    '''
    Function to remove special characters and numbers from the tokens and create a list
    Input: List
    Output: List
    '''
    wordsList = []
    for word in lists:
        wordsList.append(remSpeChar(word))
    
    return wordsList

# Removing words with just 2 character length
def remTwoCharLenWords(lists): 
    '''
    Function to remove two characters length from the tokens
    Input: List
    Output: List
    '''
    wordsList = [word for word in lists if len(word) > 3]
    return wordsList

### removing spaces
def remSpaces(lists):
    '''
    Function to remove spaces from the tokens.
    Input: List
    Output: List
    '''
    wordsList = [word.strip().rstrip() for word in lists]
    return wordsList

## removing stopwords
def remStopWords(lists): 
    '''
    Function to remove stopwords from the lists.
    Input: List
    Output: List 
    '''
    from string import punctuation
    import nltk
    # Stopwords list
    stopwords = nltk.corpus.stopwords.words('portuguese') + list(punctuation) + [0,1,2,3,4,5,6,7,8,9] + ["0","1","2","3","4","5","6","7","8","9"] + ['sistema','aplica','nao']
    
    wordsList = [ word for word in lists if word not in stopwords ]
    return wordsList

### End of wordlist
    
def generateWordCloud_uns(lista , year):
    '''
    Function to generate the wordcloud charts.
    Input: A list of datasets names and a list of years
    Output: Wordcloud chart
    '''
    import nltk
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    
    # Calculating the frequency distribution
    wc = WordCloud(max_font_size = 100).generate_from_frequencies(nltk.FreqDist(lista))
    # store default colored image
    default_colors = wc.to_array()
    plt.imshow(wc.recolor(random_state=3),interpolation="bilinear")
    plt.title("Principais Reclamações - "+ year)
    plt.axis("off")
    plt.show()
    plt.savefig('charts/wordcloud_unsatisfied-'+year+'.png')
    
def generateWordCloud_sat(lista , year):
    '''
    Function to generate the wordcloud charts.
    Input: A list of datasets names and a list of years
    Output: Wordcloud chart
    '''
    import nltk
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    
    # Calculating the frequency distribution
    wc = WordCloud(max_font_size = 100).generate_from_frequencies(nltk.FreqDist(lista))
    # store default colored image
    default_colors = wc.to_array()
    plt.imshow(wc.recolor(random_state=3),interpolation="bilinear")
    plt.title("Principais Informações - "+ year)
    plt.axis("off")
    plt.show()
    plt.savefig('charts/wordcloud_satisfied-'+year+'.png')

# Preparation for the word relationship data
# Creating main nodes # Ordered list of words
def createNodes( lista ):
    '''
    Function to create the distribution frequency of words, put them in order
    Input: List
    Output: Ordered Dictionary
    '''
    import nltk , operator
    temp = dict(nltk.FreqDist(lista))
    return sorted(temp.items() , key = operator.itemgetter(1) ,reverse = True)
     
def mostRepetitiveWords( lista ):
    '''
    Slicing the data to the 10 most repeatedwords
    input: A list containing tuples of words and count number
    output: A list with the 20 bigger entries
    '''
    return lista[:10]

# Creating data with relationship between words
def createRelations( mainNodes_list , auxNodes_lists ):
    '''
    Function to map the relationship between words.
    Input: 2 lists. 1 of the main words, that will be main nodes, 2 the cleaned list of comments.
    Output: Dataframe
    '''
    import pandas as pd
    
    relations = []
    
    for mainWord in mainNodes_list:
        for auxNodes_list in auxNodes_lists:
            if( mainWord[0] in auxNodes_list ):
                for word in auxNodes_list:
                    if( mainWord[0] != word ):
                        relations.append( ( mainWord[0] , word , 1) )
                    else:
                        continue
            else:
                continue

    nlp_data = pd.DataFrame(relations, columns=['Source','Target','Weight'])
    return nlp_data.groupby(['Source','Target']).sum().reset_index()

# Create function to filter dataframe for weight bigger than 1
def filterDataFrameWeight( df ):
    '''
    Filtering the dataframe to return just the most significative words
    Input: Dataframe
    Output: Dataframe
    '''
    df.where(df['Weight'] > 5, inplace = True)
    df.dropna(inplace = True)
    
    return df

# Creating net visualization of word relation
def netView_uns( nlp_data_dataframe , year):
    '''
    Function to generate the network view.
    Input: Dataframe
    Output: Network chart
    '''
    from pyvis.network import Network
    
    nlp_net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white")
    
    # set the physics layout of the network
    nlp_net.force_atlas_2based()
    
    sources = nlp_data_dataframe.loc[:,'Source']
    targets = nlp_data_dataframe.loc[:,'Target']
    weights = nlp_data_dataframe['Weight']
    
    edge_data = zip(sources, targets, weights)
    
    for e in edge_data:
        src = e[0]
        dst = e[1]
        w = e[2]
    
        nlp_net.add_node(src, src, title=src)
        nlp_net.add_node(dst, dst, title=dst)
        nlp_net.add_edge(src, dst, value=w)
    
    neighbor_map = nlp_net.get_adj_list()
    
    # add neighbor data to node hover data
    for node in nlp_net.nodes:
        node["title"] += "<br>Neighbors:<br><style>body{font-size:18px;}</style>" + "<br>".join(neighbor_map[node["id"]])
        node["value"] = len(neighbor_map[node["id"]])
    nlp_net.toggle_stabilization(True)
    nlp_net.show_buttons(filter_=['physics'])
    nlp_net.show("NLP-Enquete"+year+".html")

def netView_sat( nlp_data_dataframe , year):
    '''
    Function to generate the network view.
    Input: Dataframe
    Output: Network chart
    '''
    from pyvis.network import Network
    
    nlp_net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white")
    
    # set the physics layout of the network
    nlp_net.force_atlas_2based()
    
    sources = nlp_data_dataframe.loc[:,'Source']
    targets = nlp_data_dataframe.loc[:,'Target']
    weights = nlp_data_dataframe['Weight']
    
    edge_data = zip(sources, targets, weights)
    
    for e in edge_data:
        src = e[0]
        dst = e[1]
        w = e[2]
    
        nlp_net.add_node(src, src, title=src)
        nlp_net.add_node(dst, dst, title=dst)
        nlp_net.add_edge(src, dst, value=w)
    
    neighbor_map = nlp_net.get_adj_list()
    
    # add neighbor data to node hover data
    for node in nlp_net.nodes:
        node["title"] += "<br>Neighbors:<br><style>body{font-size:18px;}</style>" + "<br>".join(neighbor_map[node["id"]])
        node["value"] = len(neighbor_map[node["id"]])
    nlp_net.toggle_stabilization(True)
    nlp_net.show_buttons(filter_=['physics'])
    nlp_net.show("NLP-Enquete-Satisfied"+year+".html")

def createCleanLists( series ):
    '''
    Function to remove sotpwords from all lists of comments.
    Input: pandas.Dataframes.series
    Output: list
    '''
    wordList = []
    for lista in series:
        wordList.append( remStopWords( remSpaces( remTwoCharLenWords( remSpecChar_fromList(lista) ) ) ) )
    return wordList

cleanLists = lambda nome : createCleanLists( nome["comment_list"] )

### Code execution ###
file = "C:/Users/leonardo.galler/Documents/Python35/dados/SICOOB/Enquete/enquete-NLP.csv"
# Function 1 - Reading the data and saving as a dataframe
# Function 2 - Create the "Year" column
# Function 3 - Create the list of years
# Function 4 - Break data into unsatisfied and year
data , yearsList = createYearList( addYearColumn( readData( file ) ) ) # " * " unpacking argument lists

# Running all processes by year and satisfaction
## Define the year to be analized

############################
year = '2019'             ##
############################

# 1 - Satisfied
data_aux = lambda data , year : remStopWords(
        remSpaces(
                remTwoCharLenWords(
                        remSpecChar_fromList(
                                createWordList_sat(
                                        createTokens(
                                                breakYears_sat( data , year) ) ) ) ) ) )

## Generate wordCloud
wordCloud_sat = lambda year : generateWordCloud_sat(data_aux(data , year) , year )
wordCloud_sat(year)

### Generate Network view
mainNodes_list = lambda data , year : mostRepetitiveWords( createNodes( data_aux( data , year ) ) )
auxNodes_lists = lambda data , year : cleanLists( createTokens( breakYears_sat( data , year) ) )        
netView_sat(createRelations( mainNodes_list(data , year) , auxNodes_lists(data , year) ) , year)

# 2 - Unsatisfied
data_aux_uns = lambda data , year : remStopWords(
        remSpaces(
                remTwoCharLenWords(
                        remSpecChar_fromList(
                                createWordList_uns(
                                        createTokens(
                                                breakYears_uns( data , year) ) ) ) ) ) )

## Generate wordCloud
wordCloud_uns = lambda year : generateWordCloud_uns(data_aux_uns(data , year) , year )
wordCloud_uns(year)

### Generate Network view
mainNodes_list_uns = lambda data , year : mostRepetitiveWords( createNodes( data_aux_uns( data , year ) ) )
auxNodes_lists_uns = lambda data , year : cleanLists( createTokens( breakYears_uns( data , year) ) )        
netView_uns(createRelations( mainNodes_list_uns(data , year) , auxNodes_lists_uns(data , year) ) , year)
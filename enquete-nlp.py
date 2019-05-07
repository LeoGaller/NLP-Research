# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 10:14:18 2019

@author: Leonardo.Galler
@note: NLP research of answers to surveys at the cooperative level. 
"""

def readData( file ):
    '''
    Function to read the datafile.
    input: the location and name of the file
    output: Dataframe
    '''
    import pandas as pd
    # Folder of the data
    data = pd.read_csv(file , encoding = "latin_1", delimiter = ";", header = 0 )
    # Summary statistics of the quantitative variable
    data.describe()
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
    try:
        print('Erasing used Variable to create list of years!')
        del year
    except:
        print('It was not possible to erase variable, we will continue the program!')
        
    return years_list

def createDatasetNameList( list_of_years ):
    '''
    Function to generate a list with the name of the datasets used (UNSATISFIED)
    Input: List of years
    Output: List of datasets name
    '''
    print("Generating datasets names!\n")
    return ['data_unsatisfied' + str(year) for year in years_list]

def createDatasetNameList_sat( list_of_years ):
    '''
    Function to generate a list with the name of the datasets used (SATISFIED)
    Input: List of years
    Output: List of datasets name
    '''
    print("Generating datasets names!\n")
    return ['data_satisfied' + str(year) for year in years_list]

def splitDataPerYear_unsat( data , year ):
    '''
    Function to split the data per year and filter for unsatisfied and where the comments did not have "NAO SE APLICA".
    Input: a dataframe with unfiltered data a list of years
    Output: Datasets per year
    '''
    # Subsetting for unsatisfied
    ## Marking all the values that correspond to unsatisfied
    globals()['data_unsatisfied'+str(year)] = data[data["Ano"] == str(year)].where(data['Código Resposta'] == 2, inplace = False)
    ### Removing satisfied
    globals()['data_unsatisfied'+str(year)].dropna(inplace = True)
    #### Marking where the comments are like 'NAO SE APLICA'
    globals()['data_unsatisfied'+str(year)].where(globals()['data_unsatisfied' + year ]['Comentário Usuário'] != 'NAO SE APLICA', inplace = True, axis = 0)
    ##### Removing items with no comments
    globals()['data_unsatisfied'+str(year)].dropna(inplace = True)
    
### Breaking the information by years and filtering just to unsatisfied users
    
def splitDataPerYear_sat( data , year ):
    '''
    Function to split the data per year and filter for unsatisfied and where the comments did not have "NAO SE APLICA".
    Input: a dataframe with unfiltered data a list of years
    Output: Datasets per year
    '''
    # Subsetting for unsatisfied
    ## Marking all the values that correspond to unsatisfied
    globals()['data_satisfied'+str(year)] = data[data["Ano"] == str(year)].where(data['Código Resposta'] == 1, inplace = False)
    ### Removing satisfied
    globals()['data_satisfied'+str(year)].dropna(inplace = True)
    #### Marking where the comments are like 'NAO SE APLICA'
    globals()['data_satisfied'+str(year)].where(globals()['data_unsatisfied' + year ]['Comentário Usuário'] != 'NAO SE APLICA', inplace = True, axis = 0)
    ##### Removing items with no comments
    globals()['data_satisfied'+str(year)].dropna(inplace = True)

def breakYears_unsat( data , years_list ):
    '''
    Function to generate summary information of the general dataset per year and at the end run the fuction to split the data.
    Input: Dataframe and a list of years
    Output: Dataframes broken by year and with unsatisfied users
    '''
    import matplotlib.pyplot as plt
    for year in years_list:
        if(str(year) == 'nan'):
            break
        
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
        usersPerCentralPerSatisf_plot = lambda data : data.plot(kind = "bar" , title = "Unique users per Central and Satisfaction" , colormap = "summer" , figsize = (10 , 10))
        usersPerCentralPerSatisf_plot( renamed( reindex( toFrame( to_series( data , year , ["Número Central","Código Resposta"] ) ) ) ) )
        plt.show()
        
        # Counting unique users per year-month and satisfaction 
        usersPerMonthPerSatisf_plot = lambda data : data.plot(kind = "bar" , title = "Unique users per year-month, Central and Satisfaction" ,  colormap = "summer" , figsize = (10 , 10))
        usersPerMonthPerSatisf_plot( renamed( reindex( toFrame( to_series( data , year , ["Ano-Mês Votação","Código Resposta"] ) ) ) ) )
        plt.show()
        
        print("### Fim de Informações para "+str(year)+". ###\n")
              
        # Creating the datasets for years
        splitDataPerYear_unsat(data , str(year))
    
# Creating the column with tokens from the comments
        
def breakYears_sat( data , years_list ):
    '''
    Function to generate summary information of the general dataset per year and at the end run the fuction to split the data.
    Input: Dataframe and a list of years
    Output: Dataframes broken by year and with unsatisfied users
    '''
    import matplotlib.pyplot as plt
    for year in years_list:
        if(str(year) == 'nan'):
            break
        
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
        usersPerCentralPerSatisf_plot = lambda data : data.plot(kind = "bar" , title = "Unique users per Central and Satisfaction" , colormap = "summer" , figsize = (10 , 10))
        usersPerCentralPerSatisf_plot( renamed( reindex( toFrame( to_series( data , year , ["Número Central","Código Resposta"] ) ) ) ) )
        plt.show()
        
        # Counting unique users per year-month and satisfaction 
        usersPerMonthPerSatisf_plot = lambda data : data.plot(kind = "bar" , title = "Unique users per year-month, Central and Satisfaction" ,  colormap = "summer" , figsize = (10 , 10))
        usersPerMonthPerSatisf_plot( renamed( reindex( toFrame( to_series( data , year , ["Ano-Mês Votação","Código Resposta"] ) ) ) ) )
        plt.show()
        
        print("### Fim de Informações para "+str(year)+". ###\n")
              
        # Creating the datasets for years
        splitDataPerYear_sat(data , str(year))

def createTokens( name ):
    '''
    Function to tokenize the comment column
    Input: A text that will be used as a variable
    Output: Lower case tokens creating a new column
    '''
    eval(name)["comment_list"] = eval(name)['Comentário Usuário'].str.lower().str.split(" ")
    
    return eval(name)

### Function to create the word list
def createWordList(data_unsatisfied):
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
    accents_dict = { 'ç' : 'c', 'á' : 'a', 'à' : 'a', 'â' : 'a', 'ã' : 'a', 'é' : 'e',
                     'è' : 'e', 'ê' : 'e', 'í' : 'i', 'ì' : 'i', 'î' : 'i', 'ó' : 'o',
                     'ò' : 'o', 'ô' : 'o', 'õ' : 'o', 'ú' : 'u', 'ù' : 'u', 'û' : 'u' }
    new_word = ''
    for letter in word:
        # Verify if the letter is in the list of symbols
        if(letter in symbols_list):
            new_word = new_word + '' 
        elif(letter in accents_dict): # Verify if the letter is a special character
            new_word = new_word + accents_dict[letter]
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
    stopwords = nltk.corpus.stopwords.words('portuguese') + list(punctuation) + [0,1,2,3,4,5,6,7,8,9] + ["0","1","2","3","4","5","6","7","8","9"] + ['sistema','aplica']
    
    wordsList = [ word for word in lists if word not in stopwords ]
    return wordsList
### End of wordlist
    
def generateWordCloud(lista , year):
    '''
    Function to generate the wordcloud charts.
    Input: A list of datasets names and a list of years
    Output: Wordcloud chart
    '''
    import nltk
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    
    # Calculating the frequency distribution
    globals()['fd_wordsList'] = nltk.FreqDist(lista)
    
    wc = WordCloud(max_font_size = 100).generate_from_frequencies(fd_wordsList)
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
    globals()['fd_wordsList'] = nltk.FreqDist(lista)
    
    wc = WordCloud(max_font_size = 100).generate_from_frequencies(fd_wordsList)
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

def createCleanAllWordsClean( series ):
    '''
    Function to create a list with all lists with stopwords removed.
    Input: pandas.Dataframe.series
    Output: list
    '''
    wordList = []
    for lista in series:
        wordList.append( remStopWords( remSpaces( remTwoCharLenWords( remSpecChar_fromList(lista) ) ) ) )
    
    # Create full list of words
    fullList = []
    for lista in wordList:
        for word in lista:
            fullList.append(word)
        
    return fullList

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

# Creating data with relationship between words
def createRelations( mainNodes_list , cleanLists ):
    '''
    Function to map the relationship between words.
    Input: 2 lists. 1 of the main words, that will be main nodes, 2 the cleaned list of comments.
    Output: Dataframe
    '''
    import pandas as pd
    relations = []
    for mainWord in mainNodes_list:
        for lista in cleanLists:
            if(mainWord[0] in lista):
                for word in lista:
                    if(mainWord[0] != word):
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
def netView( nlp_data_dataframe , year_list):
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

def netView_sat( nlp_data_dataframe , year_list):
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

### Code execution ###
# Step 1 - Reading the data and saving as a dataframe
try:
    data = readData("C:/Users/leonardo.galler/Documents/Python35/dados/SICOOB/Enquete/enquete-NLP.csv")
    print('Arquivo lido com sucesso!\n')
except:
    print('Não foi possível ler o arquivo! Verifique o local onde o arquivo está.\n')
    
# Step 2 - Create the year column and add to the dataframe
print("Criando a coluna de ano!\n")
try:
    data = addYearColumn(data)
    print("Coluna Criada com sucesso!")
except:
    print("Não foi possível criar a coluna de ano. Verifique os dados")

# Which years we have in the dataset
print('\nYears in the dataset:\n',data["Ano"].sort_values().unique())

# Step 3 - Creating a list of years to avoid calculating all of them again
years_list = createYearList(data)
print("Os dados abrangem os anos de: ", years_list)

# Step 4 - Defining list to save the name of the datesets created
datasetsNameList = createDatasetNameList(years_list)
datasetsNameList_sat = createDatasetNameList_sat(years_list)

# Step 5 - Breaking the information by years and filtering just to unsatisfied users
breakYears_unsat( data , years_list )
breakYears_sat( data , years_list )

# Step 6 - Create the column and tokens from comments in each dataset
for dataset in datasetsNameList:
    dataset = createTokens(dataset)

for dataset in datasetsNameList_sat:
    dataset = createTokens(dataset)

# Step 7 - Treating the comment list column and generating the chart.
listToChart = lambda lista : remStopWords( remSpaces( remTwoCharLenWords( remSpecChar_fromList( createWordList( lista ) ) ) ) )

# Step 8 - Generating the wordcloud for each year
for datasets , year in zip(datasetsNameList , years_list):
    generateWordCloud( listToChart(eval(datasets)) , year )

for datasets , year in zip(datasetsNameList_sat , years_list):
    generateWordCloud_sat( listToChart(eval(datasets)) , year )

print('\n')
print("Generating the Network Graph!")   
print('\n')

# Step 9 - Create main nodes
mainNodesAnonym = lambda nome : mostRepetitiveWords( createNodes( createCleanAllWordsClean( eval(nome)["comment_list"] ) ) )
cleanLists = lambda nome : createCleanLists( eval(nome)["comment_list"] )

# Step 10 - Creating relationship data and plotting
for datasets_name , year in zip(datasetsNameList , years_list):
    netView( filterDataFrameWeight( createRelations( mainNodesAnonym(datasets_name) , cleanLists(datasets_name) ) ) , year )

for datasets_name , year in zip(datasetsNameList_sat , years_list):
    netView_sat( createRelations( mainNodesAnonym(datasets_name) , cleanLists(datasets_name) ) , year )    
    
### EXPORTING THE DATA GENERATED IF NEEDED
#exportThis = createRelations( mainNodesAnonym('data_unsatisfied2019') , cleanLists('data_unsatisfied2019') )
#exportThis.to_csv('testCognos.csv')
#
#df = pd.DataFrame.from_dict(fd_wordsList , orient = "index")
#df.to_csv('testCognos2.csv')
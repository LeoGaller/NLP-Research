# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 10:14:18 2019

@author: Leonardo.Galler
"""

def readData( file ):
    import pandas as pd
    # Folder of the data
    data = pd.read_csv(file , encoding = "latin_1", delimiter = ";", header = 0 )
    # Summary statistics of the quantitative variable
    data.describe()
    return data

#Creating a year column and adding to the dataframe
def addYearColumn( data ):
    data["Ano"] = data['Ano-Mês Votação'].str.slice(0,4)
    return data

def splitDataPerYear_unsat( data , year ):
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
def breakYears_unsat( data , years_list ):
    for year in years_list:
        if(str(year) == 'nan'):
            break
        
        print("### Início Informações para "+str(year)+". ###")
        
        # Counting unique users per year-month
        print("Quantidade Respondentes Únicos por mês: ", data[data["Ano"] == year].groupby("Ano-Mês Votação")['Código Login'].nunique())
        
        # Counting unsatisfied users per cooperativa
        print("Quantidade Respondentes Únicos por Cooperativa : ", data[data["Ano"] == year].groupby("Número Cooperativa")['Código Login'].nunique())
        
        # Counting unique centrals
        print("Quantidade Centrais Respondentes : ", data[data["Ano"] == year]["Número Central"].nunique())
        
        # Counting unique cooperativas
        print("Quantidade de Cooperativas Respondentes : ", data[data["Ano"] == year]["Número Cooperativa"].nunique())
        
        # Counting unique users per satisfaction
        print("Quantidade Usuários Únicos Por Satisfação",data[data["Ano"] == year].groupby(['Código Resposta'])['Código Login'].nunique())
        
        # Counting unique users per satisfaction and central
        print("Quantidade Usuários Únicos Por Satisfação e Central ",data[data["Ano"] == year].groupby(['Código Resposta','Número Central'])['Código Login'].nunique())
        
        print("### Fim de Informações para "+str(year)+". ###\n")
              
        # Creating the datasets for years
        splitDataPerYear_unsat(data , str(year))
    
# Creating the column with tokens from the comments
def createTokens( name ):
    eval(name)["comment_list"] = eval(name)['Comentário Usuário'].str.lower().str.split(" ")
    return eval(name)

# Create year list
def createYearList(data):
    years_list = [year for year in data["Ano"].sort_values().unique() if str(year) != 'nan']
    return years_list

### Function to create the word list
def createWordList(data_unsatisfied):
    ## Creating a list with all the words
    wordsList = []
    
    for lista in data_unsatisfied["comment_list"]:
        for word in lista:
            wordsList.append(word)
    # Return the list
    return wordsList

def remSpeChar(word): 
    '''
        Function to remove special characters and numbers from the tokens
        
    '''
    symbols_list = ["/",'"',"*" , "(" , ")" , ":" , ";" ,"," , "." , "%" , "#" , "?" , 
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
        
    '''
    wordsList = []
    for word in lists:
        wordsList.append(remSpeChar(word))
    
    return wordsList

# Removing words with just 2 character length
def remTwoCharLenWords(lists): 
    '''
        Function to remove two characters length from the tokens
        
    '''
    wordsList = [word for word in lists if len(word) > 2]
    return wordsList

## Remove numbers (I'm already removing numbers on remSpeChar)
#wordsList = [word for word in wordsList if not word.isnumeric()]

### removing spaces
def remSpaces(lists):
    '''
        Function to remove spaces from the tokens
        
    '''
    wordsList = [word.strip().rstrip() for word in lists]
    return wordsList

## removing stopwords
def remStopWords(lists): 
    '''
        Function to remove stopwords from the lists
        
    '''
    # Stopwords list
    stopwords = ['','que','não','nao','dos','das','por','para','com','esta','está','uma','tem','sem','foi','nos','de',
             'do','da','em', 'os' , 'as' , 'um' , 'ao', 'se','ou', 'nas','pra', 'isso', 'só' , 'pelo' , 'assim' , 'meu'
             'ele', 'já' , 'vai' , 'sistema']
    wordsList = [ word for word in lists if word not in stopwords ]
    return wordsList
### End of wordlist
    
def generateWordCloud(lista , year):
    import nltk
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    
    # Calculating the frequency distribution
    fd_wordsList = nltk.FreqDist(lista)
    
    wc = WordCloud(max_font_size = 100).generate_from_frequencies(fd_wordsList)
    # store default colored image
    default_colors = wc.to_array()
    plt.imshow(wc.recolor(random_state=3),interpolation="bilinear")
    plt.title("Principais Reclamações - "+ year)
    plt.axis("off")
    plt.show()

### Code execution ###
# Step 1 - Reading the data and saving as a dataframe
try:
    data = readData("file_name")
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

# Retrieving the name of the columns and saving in other variable
#data_columns = data.columns
# ['Ano-Mês Votação', 'Código Login', 'Número Central',
#  'Número Cooperativa', 'Código Pergunta', 'Pergunta', 'Código Resposta',
#  'Resposta', 'Comentário Usuário', 'Produto Referido Enquete', 'Ano']

# Which years we have in the dataset
print('\nYears in the dataset:\n',data["Ano"].sort_values().unique())

# Step 3 - Creating a list of years to avoid calculating all of them again
years_list = createYearList(data)
print("Os dados abrangem os anos de: ", years_list)

# Step 4 - Defining list to save the name of the datesets created
datasetsNameList = ['data_unsatisfied' + str(year) for year in years_list]
print("Foram criados os datasets:")
for name in datasetsNameList:
    print(name)

# Step 5 - Breaking the information by years and filtering just to unsatisfied users
breakYears_unsat( data , years_list )

# Step 6 - Create the column and tokens from comments in each dataset
for dataset in datasetsNameList:
    dataset = createTokens(dataset)

# Step 7 - Treating the comment list column and generating the chart.
listToChart = lambda lista : remStopWords( remSpaces( remTwoCharLenWords( remSpecChar_fromList( createWordList( lista ) ) ) ) )

# Step 8 - Generating the wordcloud for each year
for datasets , year in zip(datasetsNameList , years_list):
    generateWordCloud( listToChart(eval(datasets)) , year )

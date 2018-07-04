import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import matplotlib
#import math

TITLE_SET  = ('mr', 'miss', 'mrs') 
PCLASS_SET = (1,2,3)

#sns.set(rc={'figure.figsize':(16,12)})   
sns.set(rc={'figure.figsize':(8,6)})   

def analyze_pclass(DF):
#Pclass vs. Fare    
    
    plt.figure()
    #print(DF[DF['Pclass']==1])
    sns.distplot(DF[DF['Pclass']==1]['Fare'].dropna(), hist=True, kde=True, label='Pclass 1')
    sns.distplot(DF[DF['Pclass']==2]['Fare'].dropna(), hist=True, kde=True, label='Pclass 2')
    sns.distplot(DF[DF['Pclass']==3]['Fare'].dropna(), hist=True, kde=True, label='Pclass 3')

   
    #Relation Age & Title
    plt.figure()    
    #grid = sns.FacetGrid(allDF, col='Title')
    #grid.map(sns.distplot, 'Age', kde=False)    
    for title in TITLE_SET:
        sns.distplot(DF[DF['Title']==title]['Age'].dropna(), hist=False, label=title)        
    #sns.distplot(allDF['Age'], hist=False, kde=True)
    
    #Title vs. Survived
    plt.figure()
    for pclass in PCLASS_SET:
        sns.distplot(DF[DF['Pclass']==pclass]['Age'].dropna(), hist=False, label="Pclass {0}".format(pclass))
       

def analyze_embarked(DF):
    
    #Embarked and Survised
    plt.figure()  
    
    #sns.countplot(DF['Embarked'])
    #myClass = DF.groupby(['Embarked', 'Pclass'])
    gp_Pclass = DF.groupby('Pclass')    
    gp_Embarked = DF.groupby('Embarked')    
    #print("=========")  
    print(gp_Embarked.ngroups)
    sns.countplot(DF['Embarked'], hue=DF['Survived'])
    
    """
    figList, axList =plt.subplots(1, grouped.ngroups)    
    for i, item in enumerate(grouped):        
        name, group = item         
        print("Group {0}".format(name))     
        
        sns.countplot(group['Survived'], ax=axList[i])                
        if(i!=0):
            axList[i].set_ylabel('')
    """
    
    #Embarked and Fare
    plt.figure()
    for name, group in gp_Embarked:
        print(name)
        sns.distplot(group['Fare'].dropna(), hist=False, kde=True, label=name)
        
    #Embarked and Pclass    
    print(pd.crosstab(DF['Pclass'], DF['Embarked']))
    print(pd.crosstab(DF['Pclass'], DF['Embarked']).apply(lambda r: round(r/r.sum(), 2), axis=1))
    #print(pd.crosstab(DF['Pclass'], DF['Embarked'], normalize='index'))
    
    print("=== Category count ===")
    print(gp_Pclass.size())
    print(gp_Embarked.size())
    
def analyze_ticket(DF):
    #Ticket
    print(DF)
    #TrainDF = DF.copy
    #DF.drop(DF['Survived'].isnull(), axis=0)    
    DF.dropna(subset = ['Survived'], inplace=True)
    DF.drop(columns=['Name', 
                     #'Cabin',
                     #'SibSp',
                     'Parch',
                     'Title'
                     ], inplace=True)
    #print(TrainDF)
    
    gp_Ticket = DF.groupby(['Ticket'])    
    #print(gp_Ticket.ngroup())
    #print(gp_Ticket.ngroups)    
    #print(gp_Ticket['Ticket'].count())
    
    TGroup   = pd.Series(index=DF.index)
    #DF['TGroup'] = pd.Series(np.nan, index=DF.index)
    #print(DF['TGroup'])
    singleMark = -1
    
    for name, group in gp_Ticket:            
        for row in group.index:
            #print("--- %d" %group.shape[0])
            TGroup[row] = group.shape[0]
        #gsGroup = group.groupby('Survived')        
        #if(group.shape[0]==1):
        #    for row in group['PassengerId']:                
        #        newGroup[row] = singleMark
        #else:
        #    for row in group['PassengerId']:                
        #        newGroup[row] = group.shape[0]
        #if group
        
        #pass
    #sys.exit(0)
    DF['TGroup'] = TGroup
    #sns.countplot(newGroup)
    print(DF['TGroup'])
    plt.figure()
    sns.distplot(DF[DF['Survived']==0]['TGroup'], label="S0")
    sns.distplot(DF[DF['Survived']==1]['TGroup'], label="S1")
        #if group.shape[0]>1 :
            
        #print (.ngroup)
        #print(group[['Survived', 'Sex', 'Embarked', 'Age', 'Family']])
        #print("\n")

    
    #print(gp_Ticket.size()>1)
    
    
    #for name, group in gp_Ticket:
        #print(name, gp_Ticket.ngroups)
        #sns.distplot(group['Fare'].dropna(), hist=False, kde=True, label=name)
def analyze_na(DF):
    
    #t = DF.groupby('Survived')
    #print(t.get_group(1).shape[0]/DF.shape[0])
        
    Nacount = pd.Series(index=DF.index)
    
    #for row in DF.head(10).iterrows():
    for idx, row in DF.drop(['Survived'], axis=1).iterrows(): 
        print(idx, row)        
        count = 0
        for itr in row:                        
            if pd.isnull(itr):
                count += 1
        Nacount[idx] = count
        
    #print(Nacount)
    t = Nacount.groupby(Nacount)
    print(t.size())
    
def analyze_cabin(DF):
    
    cabinDF = DF[['Cabin', 'Survived']].dropna()
    
    tt  = cabinDF.groupby('Survived')
    tt2 = DF.groupby('Survived')
    #print(tt.size())
    #print(tt2.size())
    
    
    
def analyze_fare(DF):
    plt.figure()
    sns.distplot(DF[DF['Survived']==0]['Fare'], hist=False, kde=True, label='Suvived=0')
    sns.distplot(DF[DF['Survived']==1]['Fare'], hist=False, kde=True, label='Suvived=1')
    
    plt.figure()
    sns.distplot(DF[DF['Survived']==0]['Newfare'], hist=False, kde=True, label='Suvived=0')
    sns.distplot(DF[DF['Survived']==1]['Newfare'], hist=False, kde=True, label='Suvived=1')
    
    plt.figure()
    sns.distplot(DF[DF['Pclass']==1]['Fare'], hist=False, kde=True, label='Pclass=1')
    sns.distplot(DF[DF['Pclass']==2]['Fare'], hist=False, kde=True, label='Pclass=2')
    sns.distplot(DF[DF['Pclass']==3]['Fare'], hist=False, kde=True, label='Pclass=3')
    
    plt.figure()
    sns.distplot(DF[DF['Pclass']==1]['Newfare'], hist=False, kde=True, label='Pclass=1')
    sns.distplot(DF[DF['Pclass']==2]['Newfare'], hist=False, kde=True, label='Pclass=2')
    sns.distplot(DF[DF['Pclass']==3]['Newfare'], hist=False, kde=True, label='Pclass=3')
    
    fareRange = pd.qcut(DF['Fare'],q=10)
    #DF['Frange'] = fareRange
    gp_Frange = fareRange.groupby(fareRange)
    #print(pd.concat([gp_Frange.ngroup(), DF['Frange']], axis=1))
    #DF['FCrange'] = gp_Frange.ngroup()
    
    #sns.countplot(DF['Survived'], hue=DF['FCrange'])
    plt.figure()
    sns.distplot(DF[DF['Survived']==0]['FCrange'], label='S0').set(xlim=(0,12))
    sns.distplot(DF[DF['Survived']==1]['FCrange'], label='S1').set(xlim=(0,12))
    
    
def analyze_family(DF):
    plt.figure()
    sns.countplot(DF['TGroup'], hue=DF['Survived'])
    #plt.figure()
    #sns.countplot(DF['Family'], hue=DF['Pclass'])
    
    #filter1 = ((DF['Fare']/(DF['TGroup']+1)) > 10) & (DF['Pclass']!=1)
    #print(DF[filter1][['Survived', 'Fare', 'Family', 'Pclass', 'Ticket', 'TGroup']])
    filter2  = ((DF['Family']+1) != DF['TGroup']) & (DF['Ticket'] == '1601')
    print(DF[filter2][['Survived', 'Sex', 'Fare', 'Family', 'Pclass', 'Ticket', 'TGroup', 'Age']])
    

def comp_train(DF):
    pred1DF = pd.read_csv('titanic_train.csv', index_col=0)
    pred2DF = pd.read_csv('data/train.csv', index_col=0)    
    
    pred1DF.rename(columns={'Survived':'Pred'}, inplace=True)
    pred2DF.rename(columns={'Survived':'True'}, inplace=True)
    
    diffDF = pd.concat([pred1DF, pred2DF['True'], DF], join='inner', axis=1)    
    
    filter1 = (diffDF['Pred']!= diffDF['True']) & (diffDF['Pred']==1)
    filter2 = (diffDF['Pred']!= diffDF['True']) & (diffDF['Pred']==0)
    
    print(diffDF[filter1].drop('Name', axis=1))
    print(diffDF[filter2].drop('Name', axis=1))
    
    plt.figure()    
    sns.distplot(diffDF[filter1]['Age'], kde=True, kde_kws = {'label':'pred=1'})
    sns.distplot(diffDF[filter2]['Age'], kde=True, kde_kws = {'label':'pred=0'})
    
    
    #sns.countplot(diffDF[filter1]['Pred'], hue='Sex', data=diffDF)    
    #sns.countplot(diffDF[filter1]['Pred'], hue='TGroup', data=diffDF)
    plt.figure()
    #sns.countplot(diffDF[filter2]['Pred'], hue='TGroup', data=diffDF)
    
    #sns.countplot(diffDF[filter2]['Pred'], hue='Sex', data=diffDF)
    
def comp_result(DF):
    
    pred1DF = pd.read_csv('titanic_out.csv', index_col=0)
    pred2DF = pd.read_csv('data/baseline_79904.csv', index_col=0)    
    testDF  = pd.read_csv('data/test.csv', index_col=0)
    
            
    pred1DF.rename(columns={'Survived':'Pred1'}, inplace=True)
    pred2DF.rename(columns={'Survived':'Pred2'}, inplace=True)
    #print(pred1DF)
    
    
    #print(['Pred1']+['Pred2']+testDF.columns.tolist())
    #diffDF = pd.concat([pred1DF, pred2DF, testDF], axis=1)    
    #print(DF)    
    diffDF = pd.concat([pred1DF, pred2DF, DF], join='inner', axis=1)    
    filter1 = (diffDF['Pred1']!= diffDF['Pred2']) & (diffDF['Pred1']==1)
    filter2 = (diffDF['Pred1']!= diffDF['Pred2']) & (diffDF['Pred1']==0)
    
    #testDF = DF[DF['Survived'].isna()]
    
    print(diffDF[filter1])
    print(diffDF[filter2])
    
    print(diffDF['Pred1'].value_counts())
    print(diffDF['Pred2'].value_counts())
    #plt.figure()    
    #sns.countplot(diffDF['Pred1'])
    #plt.figure()    
    #sns.countplot(diffDF['Pred2'])
    #compDF.set_index(testDF.index, tr)
    #diffDF = testDF[compDF['Survived']!= baseDF['Survived']]
    
    
    #print(compDF['Survived'] != baseDF['Survived'])

def analyze_sex(DF):    
    #print(DF['Sex'])
    filter1 = DF['Sex']=='male'
    filter2 = DF['Sex']=='female'
    
    plt.figure()
    sns.countplot(DF[filter1]['Survived'], hue='Family', data=DF)
    plt.figure()
    sns.countplot(DF[filter2]['Survived'], hue='Family', data=DF)
    
    plt.figure()
    sns.countplot(DF['Survived'], hue='Family', data=DF)
    
def analyze_age(DF):
    plt.figure()
    
    filter1 = (DF['Survived']==0) & (DF['Age']<15)
    filter2 = (DF['Survived']==1) & (DF['Age']<15)
    #filter2 = (DF['Survived']==1) & (DF['Sex']=='male')
    sns.distplot(DF[filter1]['Age'], kde=True, kde_kws = {'label':'S0'})
    sns.distplot(DF[filter2]['Age'], kde=True, kde_kws = {'label':'S1'})
    #sns.distplot(DF[(DF['Survived']==1) & (DF['Sex']=='male')]['Age'], label='S1_male')
    
    
def analyz_pclass_fare(DF):
    plt.figure()
    sns.distplot(DF[DF['Pclass']==1]['Fare'], kde=True, kde_kws={'label':'Pclass 1'}).set(xlim=(0,100))
    sns.distplot(DF[DF['Pclass']==2]['Fare'], kde=True, kde_kws={'label':'Pclass 2'}).set(xlim=(0,100))
    sns.distplot(DF[DF['Pclass']==3]['Fare'], kde=True, kde_kws={'label':'Pclass 3'}).set(xlim=(0,100))
    
    plt.figure()
    sns.distplot(DF[DF['Pclass']==1]['Newfare'], kde=True, kde_kws={'label':'Pclass 1'}).set(xlim=(0,100))
    sns.distplot(DF[DF['Pclass']==2]['Newfare'], kde=True, kde_kws={'label':'Pclass 2'}).set(xlim=(0,100))
    sns.distplot(DF[DF['Pclass']==3]['Newfare'], kde=True, kde_kws={'label':'Pclass 3'}).set(xlim=(0,100))
    
    filter3_0 = (DF['Pclass']==3) & (DF['Survived']==0)
    filter3_1 = (DF['Pclass']==3) & (DF['Survived']==1)
    filter2_0 = (DF['Pclass']==2) & (DF['Survived']==0)
    filter2_1 = (DF['Pclass']==2) & (DF['Survived']==1)
    filter1_0 = (DF['Pclass']==1) & (DF['Survived']==0)
    filter1_1 = (DF['Pclass']==1) & (DF['Survived']==1)
    
    #filter1 = (DF['Pclass']==3) & (DF['Fare']>40)
    #filter1 = (DF['Pclass']==3) & (DF['Fare']>40)
    #filter2 = (DF['Pclass']==3) & (DF['Fare']<15)
    #filter2 = (DF['Pclass']==3) & (DF['Fare']<15)
    #filter3 = (DF['Pclass']==3) & (DF['Fare']>=15) & (DF['Fare']<=40)
    
    plt.figure()
    sns.distplot(DF[filter3_0]['Newfare'], kde=True, hist=False, kde_kws={'label':'P3/S0'}).set(xlim=(0,60))
    sns.distplot(DF[filter3_1]['Newfare'], kde=True, hist=False, kde_kws={'label':'P3/S1'}).set(xlim=(0,60))
    plt.figure()
    sns.distplot(DF[filter2_0]['Newfare'], kde=True, hist=False, kde_kws={'label':'P2/S0'}).set(xlim=(0,60))
    sns.distplot(DF[filter2_1]['Newfare'], kde=True, hist=False, kde_kws={'label':'P2/S1'}).set(xlim=(0,60))
    plt.figure()
    sns.distplot(DF[filter1_0]['Newfare'], kde=True, hist=False, kde_kws={'label':'P1/S0'}).set(xlim=(0,60))
    sns.distplot(DF[filter1_1]['Newfare'], kde=True, hist=False, kde_kws={'label':'P1/S1'}).set(xlim=(0,60))
    
    
    #plt.figure()
    #sns.countplot(DF[filter2]['Survived'])
    #sns.countplot(DF['Survived'], hue=filter2)
    #plt.figure()
    #sns.countplot(DF['Survived'], hue=filter3)
    print(DF[filter1])
    
def analyze_corr(DF):
    
    plt.figure()
    corrDF = DF.corr()  
    sns.heatmap(corrDF, annot=True, annot_kws={"size": 10})
def analyz_valid_set(DF):
    validWorstIdx = pd.read_csv("tmp/validXk_worst_set.csv", index_col=0, header=None)
    
    validDF = DF.loc[validWorstIdx.index].sort_index()
    
    plt.figure()
    sns.countplot(DF['Survived'], hue=DF['Sex'])
    plt.figure()
    sns.countplot(validDF['Survived'], hue=validDF['Sex'])
    
    plt.figure()
    #sns.countplot(DF['Survived'], hue=DF['Title'], hue_order=['mr','mrs','miss','baby','girl','strong','master'])
    sns.countplot(DF['Title'], hue=DF['Survived'], order=['mr','mrs','miss','baby','child', 'girl','strong','master'])
    plt.figure()
    sns.countplot(validDF['Title'], hue=validDF['Survived'], order=['mr','mrs','miss','baby','child', 'girl','strong','master'])
    #sns.countplot(validDF['Survived'], hue=validDF['Title'], hue_order=['mr','mrs','miss','baby','girl','strong','master'])
    print(validDF)    
    
def analysis(oDF, DF):       
    
    #analyz_pclass_fare(DF)
    #analyz_valid_set(DF)
    #analyze_corr(DF)
    #comp_train(DF)
    comp_result(DF)
    #analyze_age(DF)
    #analyze_sex(DF)
    ##analyze_na(oDF)
    #analyze_pclass(DF)
    #analyze_embarked(DF)
    #analyze_cabin(DF)
    #analyze_ticket(DF)
    #analyze_fare(DF)
    #analyze_family(DF)
    #sys.exit(0)
    
    
def plot_example(DF):    
        
    #sns.stripplot(data=DF['Fare'].dropna(), hue=DF['Embarked'])
    
    
    #print(type(myClass.groups))
    #myClass.get_group('C')
    
    #DF['Embarked'].apply(pd.value_counts).plot.pie(kind='pie')
    #sns.countplot(DF['Survived'], hue=DF['Embarked'])
    #ax = sns.countplot(DF['Age']).set(xlim=30)
    #for p in ax.patches:        
    #    ax.annotate('{0}'.format(p.get_height()), (p.get_x()+0.3, p.get_height()+10))    
    
    #for label in ax.xaxis.get_ticklabels():
    #    label.set_rotation(45)
    #print(ax.xaxis.get_ticklocs())
    #print(type(axisList))
        
    #sys.exit()    
     
    """
    # Example FacetGrid
    sns.countplot(trainDF['Survived'])
    sns.countplot(trainDF['Pclass'], hue=trainDF['Survived'])
    sns.countplot(trainDF['Sex'], hue=trainDF['Survived'])    
    sns.countplot(trainDF['Embarked'], hue=trainDF['Survived'])    
    sns.countplot(trainDF['Age'], hue=trainDF['Survived'])    
    """
    
    """
    # Example FacetGrid
    grid = sns.FacetGrid(trainDF, col='Pclass')
    grid = sns.FacetGrid(trainDF, col='Survived')
    grid.map(sns.distplot, 'Age', kde=True)
    grid.map(sns.distplot, 'Fare', kde=True)
    grid.map(sns.distplot, 'SibSp', kde=False)
    grid.map(sns.distplot, 'Parch', kde=False)
    """
    
    """
    # Example countplot + filter
    survFilter0 = trainDF['Survived']==0
    survFilter1 = trainDF['Survived']==1
    
    sns.countplot(trainDF['Title'], hue=trainDF['Survived'])
    """
    
    """
    # Example Select multiple column
    print(trainDF[['Sex','Age']])       
    AgeFilter = (trainDF['Age']>17) & (trainDF['Age']<19)
    print(trainDF[AgeFilter][['Age','Survived']])
    """
    
    """
    # Example subplot
    fig = plt.figure() 
    ax1 = fig.add_subplot(121) 
    grid = sns.FacetGrid(trainDF, col='Survived', row='Sex')
    grid.map(sns.distplot, 'Age', kde=False)
    ax2 = fig.add_subplot(122) 
    grid = sns.FacetGrid(trainDF, col='Survived', row='Title')
    grid.map(sns.distplot, 'Age', kde=False)
    """
    
    """
    # Example constains
    NameFilter = trainDF['Name'].str.contains('Mr')
    print(trainDF[NameFilter]['Name'])
    """
    
    """
    # Example query
    aaa = allDF.query('Age == 18')
    print(aaa['Age'])
    """
if __name__ == '__main__':
    
    DF = pd.read_csv("data/allDF.csv", index_col=0)
    oDF = pd.read_csv("data/oallDF.csv", index_col=0)
    
    #DF.update(DF['Pclass'].astype('category'))
    #print(DF['Pclass'])
    analysis(oDF, DF)

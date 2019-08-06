import pickle
import json
import requests
import numpy as np
import re
import scipy.sparse as sparse
from PIL import Image
from io import BytesIO
from sklearn.linear_model import LogisticRegression

class APIError(Exception):
    """An API Error Exception"""

    def __init__(self, status):
        self.status = status

    def __str__(self):
        return "APIError: status={}".format(self.status)
	
def pickle_load(direc,filename):
    with open(direc + filename,'rb') as f:
        return pickle.load(f)
	
def load_all_cards(direc):
    with open(direc + 'scryfall-all-cards.json','rb') as f:
        all_cards = json.load(f)
        with open(direc + 'scryfall-all-cards-pickled.txt','wb') as f:
            pickle.dump(all_cards,f)
        return all_cards

def get_filtered_cards(direc,all_cards):
    card_filter = lambda c: c['lang']=='en' and not 'token' in c['layout'] and not 'Emblem' in c['type_line'] \
                        and not c['set_name'] in ['Throne of Eldraine','Arena New Player Experience','Invocations',
                                                  'Kaladesh Inventions','Zendikar Expeditions','Amonkhet Invocations','Legendary Cube Prize Pack',
                                                  'Mythic Edition','Game Night','Global Series Jiang Yanggu & Mu Yanling'] \
                        and not ('Gift' in c['set_name'] or 'From the Vault:' in c['set_name'] or 'Duel Decks:' in c['set_name']) \
                        and not c['set_type'] in ['planechase','archenemy','draft_innovation', 'funny', 'memorabilia', 'promo', 'token', 'vanguard'] \
                        and not c['border_color'] in ['silver','gold'] and not '//' in c['name'] and not 'ante' in c['oracle_text']
    filtered_cards = list(filter(card_filter,all_cards))
    with open(direc + 'scryfall-filtered-cards-pickled.txt','wb') as f:
        pickle.dump(filtered_cards,f)
    return filtered_cards

def get_cropped_art(c):
    response = requests.get(c['image_uris']['art_crop'])
    if response.status_code == 200:
        return np.asarray(Image.open(BytesIO(response.content)))
    else:
        raise APIError(response.status_code)
				
def get_card_by_name(cards,name):
    named_cards = [card for card in cards if card['name'].lower()==name.lower()]
    if len(named_cards)==1:
        return named_cards[0]
    else:
        return []
    
def get_card_index_by_name(cards,name):
    named_card_indices = [i for i,card in enumerate(cards) if card['name'].lower()==name.lower()]
    if len(named_card_indices)==1:
        return named_card_indices[0]
    else:
        return []

def get_words_in_oracle_text(c):
    return [word for word in [re.sub('{w}|{b}|{u}|{g}|{r}','{@}',re.sub('[—-•().,:]','',word).lower()) for word in 
                              re.split('[ \n]',''.join(re.split('[()]',c['oracle_text'])[::2]))] 
            if word !='']

def get_card_color(c):
    return ''.join(c['colors'])

def get_sorted_unique_counts(l):
    unique_items,item_counts = np.unique(l,return_counts=True)
    unique_items,item_counts = zip(*sorted(zip(unique_items,item_counts),key = lambda c: c[1],reverse=True))
    item_ind = {c : i for i,c in enumerate(unique_items)}
    return np.array(unique_items),np.array(item_counts),item_ind        

class color_model():
    def __init__(self,keys,folds,N):
        self.N = N
        self.folds = folds
        self.keys = keys
        pass

    def get_features_one_key(self,c,key):
        if key=='oracletext':
            words = get_words_in_oracle_text(c)
            return sum([['_'.join(words[i:i+n]) for i in range(len(words)+1-n)] for n in range(1,self.N+1)],[])
        elif key=='manacost':
            return ['{' + re.sub('[RGBWU]','@',m) + '}' for m in c['mana_cost'] if not m in '{}']
        elif key=='type':
            return c['type_line'].split(' — ')[0].split(' ')
        elif key=='subtype':
            type_line_split = c['type_line'].split(' — ')
            return type_line_split[1].split(' ') if len(type_line_split)>1 else []
        elif key=='name':
            return [re.sub("[,']",'',w.lower()) for w in c['name'].split(' ')]
        elif key in ['power','toughness','set_name']:
            return [c[key]] if key in c else []
        else:
            print("Unknown key:",key)
        
    def get_all_sorted_unique_counts(self):
        d = {key : get_sorted_unique_counts(list(itertools.chain.from_iterable(self.features[key]))) for key in self.keys}
        return (dict(zip(self.keys,val)) for val in zip(*d.values()))
    
    def get_features(self,cards):
        return {key: [self.get_features_one_key(c,key) for c in cards] for key in self.keys}
    
    def get_target(self,cards):
        return np.array([self.color_ind[get_card_color(c)] if get_card_color(c) in self.color_ind else -1 for c in cards])
    
    def load_cards(self,cards):
        print('loading cards')
        self.features = self.get_features(cards)
        self.card_colors = [get_card_color(c) for c in cards]
        self.unique_features,self.feature_counts,self.feature_ind = self.get_all_sorted_unique_counts()
        self.features_all_keys = list(itertools.chain.from_iterable([[(key,f) for f in self.unique_features[key]] 
                                                                  for key in self.keys]))
        self.unique_colors,self.color_counts,self.color_ind = get_sorted_unique_counts(self.card_colors)
        print('building design matrix')
        self.X = self.get_design_matrix(self.features)
        self.y = self.get_target(cards)
                
    def get_design_matrix_one_key(self,key,features):
        print('building design matrix for',key)
        x = list(itertools.chain.from_iterable([[(i,self.feature_ind[key][ff]) 
                                                 for ff in f if ff in self.feature_ind[key]] 
                                                for i,f in enumerate(features[key])]))
        if len(x)>0:
            row_ind,col_ind = zip(*x)
        else:
            row_ind = []
            col_ind = []
        return sparse.csr_matrix((np.ones(len(row_ind)),(row_ind,col_ind)),
                                      shape=[len(features[key]),len(self.unique_features[key])])
        
    def get_design_matrix(self,features):
        return sparse.hstack([self.get_design_matrix_one_key(key,features) for key in self.keys],format="csr")
        
    def train(self,cards):
        self.load_cards(cards)
        self.group = (self.folds*np.random.permutation(len(cards))/len(cards)).astype(int)
        self.regressions = [LogisticRegression(penalty='l1',solver='saga',multi_class='multinomial',
                                               verbose=10,n_jobs=-1,C=1)
                            for _ in range(self.folds)]
        print('starting fits')
        for g in range(self.folds):
            Xtrain = self.X[self.group!=g,:]
            ytrain = self.y[self.group!=g]
            self.regressions[g].fit(Xtrain,ytrain)
        self.predicted_color_dists = np.zeros([len(cards),len(self.unique_colors)])
        for g in range(self.folds):
            self.predicted_color_dists[np.ix_(self.group==g,self.regressions[g].classes_)] = self.regressions[g].predict_proba(self.X[self.group==g,:])
        self.predicted_color = np.argmax(self.predicted_color_dists,axis=1)
        self.is_correct = self.y==self.predicted_color
        self.confidence = np.max(self.predicted_color_dists,axis=1)
        
    def get_features_by_importance(self,i,color):
        feature_inds = (self.X[i,:].toarray()>0).flatten()
        coefs = self.regressions[self.group[i]].coef_[self.color_ind[color],feature_inds]
        features = [self.features_all_keys[i] for i in np.nonzero(feature_inds)[0]]
        intercept = self.regressions[self.group[i]].intercept_[self.color_ind[color]]
        return features,coefs,intercept

def classify_card_colors_kfold(cards,keys,folds=5,colors=None):
    m = color_model(keys,folds,5)
    print(colors)
    if colors!=None:
        matching_cards,inds = zip(*[(c,i) for i,c in enumerate(cards) if c['colors'] in colors])
    else:
        matching_cards,inds = cards,np.arange(len(cards))
    m.train(matching_cards)        
    return m,matching_cards,inds
	
def get_most_predictive_features(m):
    C = np.zeros([len(m.unique_colors),len(m.features_all_keys)])
    for g in range(m.folds):
        C[m.regressions[g].classes_,:] += m.regressions[g].coef_/(m.folds-1)
    ind = np.argsort(np.max(C,axis=0))[:-101:-1]
    return [(m.features_all_keys[i],dict(zip(m.unique_colors,C[:,i])),m.unique_colors[np.argmax(C[:,i])]) for i in ind]

def get_confusion_matrix(m):
    x = np.array(m.card_colors)
    y = np.array(m.unique_colors[m.predicted_color])
    colors = sorted(np.unique(m.card_colors),key = lambda c: len(c))
    return colors,np.array([[np.sum((x==c1)*(y==c2))/np.sum(x==c1) for c1 in colors] for c2 in colors])
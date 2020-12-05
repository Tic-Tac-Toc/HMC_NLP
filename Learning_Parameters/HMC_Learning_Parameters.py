import numpy as np


def HMC_Parameters_Dict(dataset):
        
    Pi = {} #Dictionnaire des Pi, une clé = un label, sa valeur le nombre de mot de ce type dans la phrase. (nombre d'occurence des labels dans le dataset)
    A = {} #Dictionnaire des proba de transitions (nombre d'occurence de l'enchaînement label1 label2 pour A[label1][label2])
    B = {} #Dictionnaire des proba d'émissions (nombre d'occurence des mots indexés par leur variables cachées, nombre de fois qu'un mot correspondait à un label)
        
    for Z in dataset: #Parcours des phrases dans le dataset Z = une phrase
        
        Z0 = Z[0] #Sélection de la première phrase
        X0 = Z0[0] #Sélection du premier label
        Y0 = Z0[1] #Sélection du premier mot
        
        if X0 not in Pi.keys(): #Si le premier label n'est pas dans le dictionnaire des Pi, alors on l'ajoute
            Pi[X0] = 0
            
        Pi[X0] = Pi[X0] + 1 #On incrémente le dictionnaire des Pi de un au niveau du label.
        
        if X0 not in B.keys(): #Si le label n'est pas dans le dictionnaire des proba d'émissions, alors on l'ajoute
            B[X0] = {}
            
        if Y0 not in B[X0].keys():  #On ajoute le premier mot au dictionnaire B indéxé par le label de ce dernier si il n'a jamais été trouvé avec ce label.
            B[X0][Y0] = 0 
        
        B[X0][Y0] = B[X0][Y0] + 1 #Incrémentation du mot, on a vu une fois de plus ce mot lié au label X0
        
        Zi_prev = Z0 #On retient le premier mot
        
        for Zi in Z[1:]: #Parcours des autres
            x = Zi[0] #Label du mot
            y = Zi[1] #Mot observé
            
            if x not in Pi.keys(): #Si le label n'est pas dans le dictionnaire de Pi on l'ajoute
                Pi[x] = 0
            
            if Zi_prev[0] not in A.keys(): #Si le label du mot précédent n'est pas dans la matrice de transitions on l'y ajoute
                A[Zi_prev[0]] = {} #On initialise à cette valeur un dictionnaire.
                
            if x not in A[Zi_prev[0]].keys(): #Si le labeldu mot actuel, n'est pas dans le dictionnaire de la matrice de transition indexé par le label du mot précédent on l'y ajoute
                A[Zi_prev[0]][x] = 0 #On initialise le dictionnaire de la valeur cachée précédente lié à la valeur caché actuelle à 0
                
            if x not in B.keys(): #Si le label n'est pas dans le dictionnaire des proba d'émissions, alors on l'ajoute
                B[x] = {}
                
            if y not in B[x].keys(): #Si le mot observé n'a jamais été observé avec le label x, on l'ajoute au dictionnaire de ce dernier
                B[x][y] = 0
                            
            Pi[x] = Pi[x] + 1 #On incrémente le nombre de fois que le label est dans le dataset
            A[Zi_prev[0]][x] = A[Zi_prev[0]][x] + 1 #On incrémente le nombre de fois que le label précédent et le label actuel s'enchâine
            B[x][y] = B[x][y] + 1 #Incrémentation du mot, on a vu une fois de plus ce mot lié au label x
            
            Zi_prev = Zi #On met à jour le mot précédent
            
    sum_Pi = np.sum(list(Pi.values())) #Somme totale du nombre de mot
    for key in Pi.keys():
        Pi[key] = Pi[key]/sum_Pi #Probabilité de l'apparition du label 
        
    for key_A_1 in A.keys():
        sum_A = np.sum(list(A[key_A_1].values())) #Somme totale du nombre de mot ayant apparu après le label key_A_1
        for key_A_2 in A[key_A_1].keys():
            A[key_A_1][key_A_2] = A[key_A_1][key_A_2]/sum_A #Probabilité de voir apparaître le label key_A_2 sachant que le précédent est key_A_1
            
    for key_B_1 in B.keys():
        sum_B = np.sum(list(B[key_B_1].values())) #Somme totale du nombre de mot associé au label key_B_1
        for key_B_2 in B[key_B_1].keys():
            B[key_B_1][key_B_2] = B[key_B_1][key_B_2]/sum_B #Probabilité de voir apparaître le mot key_B_2 sachant que le label est key_B_1
    
    
    return Pi, A, B

#On utilise des dictionnaires car selon permet une recherche plus rapide mais également car cela permet d'associer les labels à des clés uniques ! Si ce n'était pas le cas et que nous utilisions des matrices, elles seraient d'une énorme taille !
# Chaine de Markove python commands:

# Librairies to include:
import numpy as np
from numpy.linalg import matrix_power
from numpy import linalg as LA
np.set_printoptions(precision=5,suppress=True)

################################################################################################
# Matrice de transition:
P=[[0,0.3,0.4,0.3],
   [0,0.2,0,0.8],
   [0,1,0,0],
   [0,0.5,0.5,0]]
P=np.array(P)
print(P)

################################################################################################
# Pour élever la matrice a une certaine puissance. Avec P[1][3] la position P24
P7=matrix_power(P7)
print('Proba a 7 sauts:', P7[1][3])

################################################################################################
# Loi marginale de l’instant 6: A partir de Pi_0:
Pi_0=[0,0.2,0.8,0]
Pi_6=Pi_0@matrix_power(P,6)
print('Loi marginale de 6', Pi_6)

################################################################################################
# Pour démontrer qu’une CM est irréductible:
# Méthode 1:
#       Include NetworkX
#       Call isConnected()

# Méthode 2:
#       Elevation au carré
#       Voir si des 0 existent, sinon donc irréductible.

################################################################################################
# Pour trouver une qtte moyenne:
#Q=[]]
qMoy=np.dot(Q,pStat)
Print(qMoy)

################################################################################################
# Pour démontrer si une CM est périodique:

# Les puissances successives de P vont réaliser un switching dans les 0.
# Si elle est apériodique donc elle converge vers la probabilité stationnaire.
################################################################################################

#probability stationnaire M1 on prend une ligne condition: matrice non periodique

print(matrix_power(P,10000))
print(matrix_power(P,10001))

################################################################################################
#probabilite stationnaire M2 (Methode vecteur propre works with any case):
val, vec = LA.eig(np.matrix.transpose(P))
print('val',val,'\nvect: \n',vec,'\n')

pStat=vec[:,0] 
#assuming the eigen value 1 is the first value in val
pStat=pStat/np.sum(pStat)
print('Stationary probabilities by eigen values method: \n',pStat)

################################################################################################
# Fermeture transitive irreductible:

o=len(P)
Mb=np.mat(P,dtype=bool) 
I=np.eye(o,o,dtype=bool)

TC=(Mb+I)**(o-1)
print('transitive closure of P:\n', TC*1,'\n')
################################################################################################

# Periodique seulement si chaine irreductible: 
mat1=matrix_power(P,10000)
mat2=matrix_power(P,10001)
mat3=mat1-mat2

if(len(mat3[mat3>0.0000001])):
    print('periodique \n')
else:
    print('aperiodique \n')
################################################################################################
# Pour tracer tous les Pi^n:
pi_0=[1,0]

for i in range(20):
    print('pi_',i,'=',pi_0 @matrix_power(P,i))

#Pour trouver Pi1 Pi2 etc.. je regarde la valeur des composants du vecteur propore w bzidoun I think.
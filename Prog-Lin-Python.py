# Python Commands: Chapter: Linear Programming:
# Librairies to import:
import numpy as np
from numpy.linalg import matrix_power
from scipy.optimize import linprog

# Exos Typiques: Programmation Lineaire: 
# But: Maximization : donc on place le signe (-), Puisque la fonction linprog minimise.
# Ajouter le signe (-) si le but de notre exercice est de maximiser notre fonction.
# Laisser le signe + si le but est de minimiser.
c=[-4,-12]

A=[[-1,0],
   [0,-1],
   [1,0],
   [0,1],
   [1,2]]

b=[0,0,1000,500,1750]

res=linprog(c,A_ub=A,b_ub=b)
print(res)

# Explanation:
# Le vecteur colonne (c) qui contient toutes les constantes de la fonction objective. 
# La matrice A qui contient les coefficients de mes variables des contraintes d’inégalités.
# Le vecteur colonne b contient les constantes ( Syntaxe: contraintes < constante qu’on place dans la matrice B ) 

# Methode #2 de la creation de la matrice (A): Bound method:
# Bound (a,b) signifie: 1st variable >0 et <4 etc… 
print("Probleme Regime\n")
c=[2,24,13,9,20,19]
A=[[-110,-205,-160,-160,-420,-260],\
   [-4,-32,-13,-8,-4,-14],\
   [-2,-12,-54,-285,-22,-80]]
b=[-2000,-55,-800]

bound=[(0,4),(0,3),(0,2),(0,3),(0,5),(0,2)]

res = linprog(c, A_ub=A, b_ub=b, bounds=bound) 
print(res)

#It's really important to know how to read the output during the exam.
# Après résolution sur Python, on a : Output of the command:
# x=[ 4, 0, 0.41, 3,2.41, 0] 
# Le coût total de la solution minimale= fun = 88.6 USD 
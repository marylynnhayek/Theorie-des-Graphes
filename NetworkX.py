# TP NetworkX:

#############################################
# Librairies to import:
import networkx as nx
import matplotlib.pyplot as plt
import scipy as sp
from collections import deque
from numpy import *
#############################################
#graphe complet
cG=nx.complete_graph(7)
#############################################
# 3 écritures possibles:
# Ajouter le sommet A dans le graphe G.
# Ajouter le sommet 1 dans le graphe G.
# Ajouter plusieurs sommets en meme temps

G.add_node('A')
G.add_node(1)
G.add_nodes_from([2, 5, 'A', 'B'])
#############################################
# Définir G  comme un graphe
G = nx.Graph()
#############################################
# Draw the graph cG
nx.draw(cG)
#############################################
# 1-Save the graph as a png file
# 2-Show the graph (plot)
# 3-Close the plot

plt.savefig('completeGraph.png')
plt.show()
plt.close()
#############################################
#   Print l’ordre du graphe cG
print(cG.order()) 
#############################################
#	Print la liste des sommets
print(cG.nodes()) 
#############################################
# Print le degré du sommet 2 (returns a dictionary-like container)
print(cG.degree(2))
#############################################
#Dessiner le graphe avec des poids visibles.
nx.draw(G,with_labels=True)
#############################################
#Respectivement:

#1-Print les sommets du graphe G.
#2-Print l’ordre du graphe G.
#3-Print le degre du sommet A
#4-Print les voisins du sommet 6
print("Sommets de G:", list(G.nodes()))
print("Ordre du graphe :", G.order())
print("degres :", G.degree)
print("degre de A :", G.degree('A'))
print("Voisins de 6", list(G.neighbors(6)))
#############################################
# Parcours en largeur
l = list(nx.bfs_edges(G, 1))
print('largeur', l)
#############################################
# Parcours en profondeur
l = list(nx.dfs_edges(G, 1))
print('profondeur', l)
#############################################
# To see if the graphe is connected
print("The graph is connected?",nx.is_connected(G))
#############################################
# Print le nombre composantes connexes
print( "what is the nbre of connected components?",
nx.number_connected_components(G)) 
#############################################
# Voir les composantes connexes contenant le sommet 1.
print("Connected component containing vertex 1",
nx.node_connected_component(G, 1))
#############################################
# Print les composantes connexes
print('components', list(nx.connected_components(G)))
#############################################
# Create components as graphs
comp = [G.subgraph(c) for c in nx.connected_components(G)]
G1=comp[0]
print("order of G1", G1.order())
#############################################
#Création de la matrice d’adjacence. Pour élever la matrice M au cube.
M = nx.adjacency_matrix(G)
print('adjacency matrix of G:\n', M)

#Ou 

M=nx.attr_matrix(G,rc_order=[1,2,3,4,5,6,'A','B','C','D'])
print('adjacency matrix of G:', M)
print (M**3)
#############################################
# La fermeture transitive du graphe G.
n = G.order()
I = eye(n, n)
TC = (I+M)**(n-1)
TC_b = mat(TC, dtype=bool)
print(I)
print(TC)
print(TC_b)
print(TC_b*1)
#############################################
# Création d’un graphe a partir d’un fichier .txt
G=nx.read_edgelist('./graph.edges',nodetype=str)

nx.draw(G,with_labels=True)

nx.write_edgelist(G, "graph.edges_out.txt")
nx.write_graphml_lxml(G, "./graph.graphml")

plt.show()
plt.close()
#############################################
#Les algorithmes
#file implémentée par liste
#Cette fonction retourne l'ordre du parcours en largeur du graphe
def pf(G,s) :
  F=[] #file vide
  order=1 #ordre de visite
  marque={}
  for i in G.nodes() : 
    marque[i]=-1; 
  marque[s]=order
  F.append(s)
  while F :
    x=F.pop(0)    
    for y in G.neighbors(x) :
      if marque[y]==-1 :
        F.append(y)
        order=order+1
        marque[y]=order 
  return marque 
print("pf(G,1)",pf(G,'1'))
#############################################
#file implémenté plus efficacement par un deque
def pf2(G,s) :
  #F=[] #file vide
  F=deque(); #file vide
  order=1 #ordre de visite
  marque={}
  for i in G.nodes() : 
    marque[i]=-1; 
  marque[s]=order
  F.append(s)
  
  while F :
    #x=F.pop(0) 
    x=F.popleft()
    for y in G.neighbors(x) :
      if marque[y]==-1 :
        F.append(y)
        order=order+1
        marque[y]=order 
  return marque 
#############################################
#Save graph in graphml file
nx.write_graphml(G,'./gg.graphml')
#############################################
#def isCyclic(G,s) :
#F=[] #file vide
  F=deque();
  marque={}
  pred={}
  for i in G.nodes() : 
    marque[i]=-1;
    pred[1]=s
  marque[s]=1
  F.append(s)
  while F :
    #x=F.pop(0)
    x=F.popleft()
    for y in G.neighbors(x) :
      if marque[y]==-1 :
        F.append(y)
        marque[y]=1
        pred[y]=x
elif marque[y]==1 and pred[x]!=y:
        return True
  return False

print(isCyclic(G,'1'))
#############################################
# Théorème du 1st chapitre (|x| = 2|A|)
s = 0
for i in G.nodes:
s += G.degree(i)
print('somme des degres: ', s)
print(s == 2 * G.number_of_edges())
#############################################

# Graphe orienté et pondéré:
#1-Création d’un graphe orienté a partir d’un fchier .txt nommé wG.txt
wG=nx.read_edgelist('./wG.txt',create_using=nx.DiGraph(),nodetype=str,data=(('weight',int),))

print(wG['H1']['1']['weight'])

nx.draw(wG,with_labels=True)
plt.show()
plt.close()
#############################################
# Print the shortest path de H1 vers V en appliquant l’algorithme de Dijkstra.
print(nx.shortest_path(wG, source='H1', target='V',  weight='weight', method='dijkstra'))
#############################################
# Networkx algo Dijkstra
print(nx.dijkstra_path(wG, 'H1', 'V'))
print(nx.dijkstra_path_length(wG, 'H1', 'V'))
#############################################
length = nx.shortest_path_length(wG, source='H1', target='V', weight='weight')
print(length)

print(nx.dijkstra_path(wG, 'H1', 'V'))

print(nx.dijkstra_path_length(wG, 'H1', 'V'))
#############################################

#====== Analyse graphe Internet ==== Hist as in histogramme
G=nx.read_edgelist('./as_rel.txt',data=(('relation',str),))

print("Is connected?  ", nx.is_connected(G))
print("nbr of ASs is :", G.order())

G=nx.read_edgelist('./as_rel.txt',data=(('relation',str),))

print("Is connected?  ", nx.is_connected(G))
print("nbr of ASs is :", G.order())

histDeg=nx.degree_histogram(G)
print(histDeg)
print("maximum degree in the graph", len(histDeg)-1)

plt.bar(range(len(histDeg)),histDeg) 

plt.xscale('log')

plt.bar(range(len(histDeg)),histDeg) 
#############################################
# #bar chart
plt.xscale('log')
plt.bar(range(len(histDeg)),histDeg)
plt.savefig('histo.png')
#############################################
#degre moyen du graphe= (somme i*histogramme(i))/ordre
histDeg=nx.degree_histogram(G)
avg_deg=0
for i in range(len(histDeg)):
    avg_deg=avg_deg+i*histDeg[i]

avg_deg=avg_deg/G.order()
print('degre moyen du graphe:',avg_deg)
#############################################
#ordre du graphe
print(sum(histDeg)) #==G.order()
print(G.order())
#############################################
#Le poids entre 2 sommets d’un graphe
print("la ponderation entre le sommet H1 et 1 est : ",wG['H1']['1']['weight'])
#############################################
#Combien de sommets on le degré = 1? 
#combien de sommet on un degre =1 : 6672 = histDeg[1]
histDeg=nx.degree_histogram(G)
print(histDeg)
#############################################
#Si on a le degre moyen de 4.2 par exemple et il nous demande de représenter le graphe.

#pour avoir un graphe de degre moyen de 4.2, il faut avoir p=4.2/(ordre-1)

#car ordre*p=4.2 

#plus specifiquement ordre-1 car chaque sommet est connecte a ordre-1 autre sommet 

EG=nx.erdos_renyi_graph(G.order(),4.2/G.order())

histDeg2=nx.degree_histogram(EG)

plt.bar(range(len(histDeg2)),histDeg2)
#############################################
#dijkstra bande passante
def dijkstraBW(G,s):
    dist={}
    pred={}
    for i in G.nodes():
        dist[i]=0
        pred[i]=s
    dist[s]=np.inf
    Y=list(G.nodes())

    while Y:
        i=max(Y,key=dist.get)
        for j in G.neighbors(i):
            if min(dist[i],G[i][j]['weight']) > dist[j]:
                dist[j]=min(dist[i],G[i][j]['weight'])
                pred[j]= i
        Y.remove(i)
    return dist,pred

Gbw=nx.read_edgelist('./BW.txt',create_using=nx.Graph(),nodetype=str,data=(('weight',int),))
[d,p]=dijkstraBW(Gbw,'S')
print(d)
print(p)
#############################################
cyclomatique=G.number_of_edges()-G.number_of_nodes()+nx.number_connected_components(G)
#############################################
#Graph randomly generated
#by default:directed=False;
#20 sommets
#20 sommets implique 400 paires de sommets
#(400*p)/2 est le nombre moyen d'arretes pour un graphe non oriente
EG=nx.erdos_renyi_graph(20,0.4)
print(EG.number_of_edges())
nx.draw(EG,with_labels=True)
plt.show()
plt.close()
#############################################
#graphe randomly generated (Directed)
EG=nx.erdos_renyi_graph(20,0.4,directed=True)
print(EG.number_of_edges())

#by default:directed=False;
#20 sommets
#20 sommets implique 400 paires de sommets
#(400*p) est le nombre moyen d'arretes pour un graphe oriente


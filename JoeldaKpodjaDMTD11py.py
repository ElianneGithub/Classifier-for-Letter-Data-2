#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 20:41:04 2021

@author: joelda
"""


  

import matplotlib.pyplot as plt
import numpy as np
from GradDesc import *
import data_extraction

from data_extraction import *

i=0
X = []
Y = []
Z = []
with open("letter.data.txt", 'r') as infile:
    
     for line in infile:
             
         line = line.split()
         V = line[6:]
         V = [int(v) for v in V]
         V = np.array(V) 
         X.append(V) 
         Y.append(ord(line[1]) - 97) 
         V = V.reshape(16,8)

         Z.append(V)

    
X = np.array(X)
#print("Donnes de X")
#print(X)
print("X.shape : ")
print(X.shape)

Y = np.array(Y)
#print("Donnes de Y")
#print(Y)
print("Y.shape : ")
print(Y.shape)


Z = np.array(Z)
#print("Donnes de Z")
#print(Z)
#print("Z.shape : ")
#print(Z.shape)

#print(X[1234])

#print(Y[1234])

#plt.imshow(Z[1234])


S = np.column_stack((X,Y))

#print("S")
#print(S)
print("S.shape")
print(S.shape)

""" Exercice 1 Observez les données """

print("Exercice 1 ")

""" Les classes sont-elles équilibrées (i.e toutes les classes ont le même effectif) ? """

print (" En observant les données on voit bien que les classes ne sont pas équilibrées, elles n'ont pas toutes le même effectif. ")

""" Pouvez-vous expliquer pourquoi, en regardant la manière dont a été construit ce jeu de don-
nées ? """

print(" En effet elles n'ont pas toutes le même effectif car chaque lettres sont extraites des mots et certains mots sont utilisés plus souvent que d'autres ce qui explique que certaines lettres reviennent plus souvent que d'autres.Ce qui fait que les classes ont de differents effectifs ")

""" Quelle est l’effectif de la classe correspondant au ‘n’? """

print("L'effectif de la classe correspondant au n est de 5024 ")

""" En supposant qu’un classifieur réponde systématiquement ‘n’, quelle serait son taux d’erreur ? """

print(" Si un classifieur répondait systématiquement n, son taux d'erreur serait environ de 90%")


""" Exercice 2 """

print("Exercice 2")

""" Séparez les données en un ensemble d’apprentissage (90%) et un ensemble de test (10%)  """

def split_train_test(S, size = 0.1):
    S_learn = []
    S_test = []
    for s in S:
        if np.random.rand()< size:
            S_test.append(s)
        else:
            S_learn.append(s)
    return np.array(S_learn), np.array(S_test)


S_train, S_test = split_train_test(S, size = 0.1)

""" Construisez un classifieur linéaire (on pourra considérer la loss -log-vraisemblance) que vous
apprendrez sur l’ensemble d’apprentissage """

def softmax(v):
    return(((np.exp(v).T)/(np.exp(v).sum(axis = 1))).T)


def output_lin(T_vec, S):
        
    U1 = T_vec[:128*26]
    U1 = U1.reshape(128, 26)
    
    b1 = T_vec[128*26:]

    X = S[:,:-1]

    return np.array(softmax(X.dot(U1)+b1))


def loss(T_vec, S):
    
    output = output_lin(T_vec, S)
    
    Y = S[:,-1]
    score_vec = np.zeros(Y.shape, float)
    
    for i in range(len(Y)):
        score_vec[i] = output[i,int(Y[i])]
    
    return -(np.log(score_vec)).sum()


def nb_error(T_vec, S):
    
    nb_er = 0
    
    output = output_lin(T_vec, S)
    
    Y = S[:,-1]
    for i in range(len(Y)):
        v = output[i]
        classe_predite = np.argmax(v)
        classe_reelle = Y[i]
        if(classe_predite != classe_reelle): nb_er += 1
    return(nb_er)


""" Pour une solution lineaire """

T_vec = np.random.randn(129*26)*0.1

T_sol = grad_desc_stoch(loss, S_train, 129*26, 100, step = 0.00001, x_0 = T_vec)

""" Quel taux d’erreur fait ce classifieur sur l’ensemble de test? """

print(" Voici le taux d' erreurs sur l'echantillon train = ")

print(nb_error(T_sol, S_train)/S_train.shape[0])  

print(" Voici le taux d' erreurs sur l'echantillon test = ")

print(nb_error(T_sol, S_test)/S_test.shape[0])       


"""" Exercice 3 """ 

print("Exercice 3")

""" Même questions que précédement, avec un réseau de neurones à 1 couche cachée. Vous testerez les configurations suivantes:  """

""" Couche cachée à 10, 20, et 30 neurones """

Nin = 128
Nout = 26
hidden = [10,20,30]
for x in hidden:
    Nhidden = x

Nhidden = 30

""" Fonction d’activation: ReLU """

def RELU(v):
    return((1*(v>0))*v)


def activation(v):
    return(RELU(v))
    
def output1(T_vec, S):
    
    T1 = T_vec[:129*Nhidden]
    T2 = T_vec[129*Nhidden:]
    
    U1 = T1[:128*Nhidden]
    U1 = U1.reshape(128, Nhidden)
    
    b1 = T1[128*Nhidden:]
    
    U2 = T2[:Nhidden*26]
    U2 = U2.reshape(Nhidden, 26)
    
    b2 = T2[Nhidden*26:]
    
    X = S[:,:-1]

    return np.array(softmax(activation(X.dot(U1)+b1).dot(U2)+b2))

def loss1(T_vec, S):
    
    output = output1(T_vec, S)
    
    Y = S[:,-1]
    score_vec = np.zeros(Y.shape, float)
    
    for i in range(len(Y)):
        score_vec[i] = output[i,int(Y[i])]
    
    return -(np.log(score_vec)).sum()


def nb_error1(T_vec, S):
    
    nb_er = 0
    
    output = output1(T_vec, S)
    
    Y = S[:,-1]
    for i in range(len(Y)):
        v = output[i]
        classe_predite = np.argmax(v)
        classe_reelle = Y[i]
        if(classe_predite != classe_reelle): nb_er += 1
    return(nb_er)
    

""" Pour une solution non lineaire """

T_vec1 = np.random.randn(129*Nhidden+(Nhidden+1)*26)*0.01

T_sol1 = grad_desc_stoch(loss1, S_train, 129*Nhidden+(Nhidden+1)*26, 100, step = 0.00002, x_0 = T_vec1)

""" Pour ces configurations, quel taux d’erreur sur l’ensemble de test ? """

print(" Voici le taux d' erreurs sur l'echantillon train = ")

print(nb_error1(T_sol1, S_train)/S_train.shape[0])  

print(" Voici le taux d' erreurs sur l'echantillon test = ")

print(nb_error1(T_sol1, S_test)/S_test.shape[0])    

 

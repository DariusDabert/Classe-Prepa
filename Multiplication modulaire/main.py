# importation des fonctions nécessaires au programme
from tkinter import Tk, Canvas, Scale, Button
from math import sin, cos, pi, floor


def passagereperetkinter(x, y, a, b):
    """
    Entrée :
    a,b : Coordonnées du centre du repère classique
    x,y : coordonées d'un point dans le repère 'classique'
    Sortie:
    Coordonées du point dans le repère Tkinter
    """
    return(a + x, b - y)


def ProduitModuloP(k, n, p):
    """
    Entrée :
    k : k-ieme multiple de n
    n : entier dont on construit la table
    p : nombre de sommet(modulo)
    Sortie :
    La partie entiere du reste de la division de k*n par p
    """
    return floor(k * n % p)


def dessinersegment(cnv, k, n, p, ListePoints, a, b):
    """
    Entrée :
    cnv : canevas
    k : point du cercle d'indice k
    n : entier dont on construit la table
    p : nombre de sommet(modulo)
    base : liste contenant les coordonées de chaque sommet du cercle
    a,b : Coordonnées du centre du repère classique
    Sortie :
    le tracé d un segement reliant un point a son produit par n modulo p
    """
    prod = ProduitModuloP(k, n, p)
    xA, yA = ListePoints[k]
    xB, yB = ListePoints[prod]
    A = passagereperetkinter(xA, yA, a, b)
    B = passagereperetkinter(xB, yB, a, b)
    cnv.create_line(A, B, fill = 'black')


def CreerPoint(cnv, C, R=2, color='red'):
    """
    Entrée :
    cnv : canevas
    C : Coordonées du centre du cercle dans le repère de Tkinter
    R : rayon du cercle
    color : couleur du cercle (rouge par defaut)
    Sortie
    un cercle de centre C(repère classique) et de rayon R
    """
    xC, yC = C
    return cnv.create_oval(xC - R, yC - R, xC + R, yC + R, fill=color,
                           outline=color)


def dessinerpoint(cnv, ListePoints, a, b):
    """
    Entrée:
    cnv : le canevas sur lequel on dessine
    ListePoints : liste contenant les coordonnées de chaque points
    a,b : Coordonnées du centre du repère classique
    Sortie :
    Le dessin sur le canevas de l'integralité des points de la liste
    """
    p = len(ListePoints)
    for k in range(p):
        x, y = ListePoints[k]
        X, Y = passagereperetkinter(x, y, a, b)
        CreerPoint(cnv, (X, Y), R=3, color='black')


def table(value):  # fonction permettant d'actualiser le dessin lorsque la valeur de la table est modifiée
    global N
    N = float(value)
    show(cnv, N, P)


def modulo(value):  # fonction permettant d'actualiser le dessin lorsque la valeur du modulo est modifiée
    global P
    P = int(value)
    show(cnv, N, P)


def show(cnv, n, p):
    """
    Entrée:
    cnv : le canevas sur lequel on dessine
    n,p: entier dont on construit la table,modulo
    Sortie:
    Dessin du cercle, des points et des segments reliant les points
    """
    cnv.delete("all")  # on supprime le dessin précedant pour eviter la superposition
    t = 2 * pi / p  # création d'une variable contenant l'angle séparant chaque point
    listepoints = [(R * cos(k * t - pi / 2), R * sin(k * t - pi / 2)) for k in
                   range(p)]  # création d'une liste contenant les coordonnées de chaque points
    dessinerpoint(cnv, listepoints, a, b)  # création de tous les points

    for k in range(p):
        dessinersegment(cnv, k, n, p, listepoints, a,
                        b)  # création d'un segmant reliant chaque point avec son produit par n modulo p

    cnv.create_oval(360 - R, 360 - R, 360 + R, 360 + R, width=2, outline='black')


def programme_entier():
    # fonction qui gère le controle continue du tracé du cercle pour les entiers
    global N, P, cnv, R, a, b
    R = 300  # paramètre permettant de definir la taille de la fenêtre
    N = P = 2  # position de départ
    a = b = 1.2 * R  # pour avoir une marge sur les bords

    wdw.destroy()

    cnv = Canvas(master, width=2 * a, height=2 * b, bg='ivory')  # création du canevas
    cnv.pack(side="left")

    table_slider = Scale(master, label="Table de ...",  # contrôle par un curseur du nombre dont on construit la table
                         font="Arial 15 bold",
                         orient="horizontal", command=table,
                         from_=2, to=1000, resolution=1,
                         length=350)
    table_slider.pack(pady=15)  # ajout du curseur à la fenêtre

    modulo_slider = Scale(master, label="Modulo",  # contrôle par un curseur du module, c'est à dire du nombre de points
                          font="Arial 15 bold",
                          orient="horizontal", command=modulo,
                          from_=2, to=3000, length=350)
    modulo_slider.pack(pady=5)  # ajout du curseur à la fenêtre


def programmedecimaux():  # fonction qui gère le controle continue du tracé du cercle pour les décimaux
    global N, P, cnv, R, a, b
    R = 300  # paramètre permettant de definir la taille de la fenêtre
    N = P = 2  # position de départ
    a = b = 1.2 * R  # pour avoir une marge sur les bords

    wdw.destroy()

    cnv = Canvas(master, width=2 * a, height=2 * b, bg='ivory')  # création du canevas
    cnv.pack(side="left")

    table_slider = Scale(master, label="Table de ...",  # contrôle par un curseur du nombre dont on construit la table
                         font="Arial 15 bold",
                         orient="horizontal", command=table,
                         from_=2, to=150, resolution=0.01,
                         length=350)
    table_slider.pack(pady=15)  # ajout du curseur à la fenêtre

    modulo_slider = Scale(master, label="Modulo",  # contrôle par un curseur du module, c'est à dire du nombre de points
                          font="Arial 15 bold",
                          orient="horizontal", command=modulo,
                          from_=2, to=1000, length=350)
    modulo_slider.pack(pady=5)  # ajout du curseur à la fenêtre


def choixrepresentation():
    global N, P, cnv, R, a, b, master, wdw
    R = 300  # paramètre permettant de definir la taille de la fenêtre
    N = P = 2  # position de départ
    a = b = 1.2 * R  # pour avoir une marge sur les bords



    master = Tk()  # création d'une fenêtre graphique

    wdw = Canvas(master, width=1 * a, height=1 * b, bg='ivory')
    wdw.pack(side="left")

    # création d'un bouton offrant le choix de passer de manière discrète ou continue d'enter en entier
    choixentier = Button(wdw, font="Arial 15 bold", command=programme_entier, text="Table d'entier",
                         width=20)  # si cette option est choisi, la fonction
    choixentier.pack()  # programme entier est lancée

    choixdecimaux = Button(wdw, font="Arial 15 bold", command=programmedecimaux, text="Table de décimaux", width=20)
    choixdecimaux.pack()

    master.mainloop()


choixrepresentation()  # lancement du programme
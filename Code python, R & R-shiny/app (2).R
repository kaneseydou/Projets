options (encodage = 'UTF-8')

## PACKAGES UTILISES
#install.packages("DT")
#install.packages("mixtools")
#install.packages("fitdistrplus")
#install.packages("MASS")
#install.packages("xlsx")
#install.packages("openxlsx")
#install.packages("openssl")

library(shiny)
library(shinydashboard)
library(DT)
library(mixtools)
library(fitdistrplus)
library(MASS)
library(xlsx)
library(openxlsx)
library(openssl)

## FONCTION UI: POUR LA MISE EN PAGE DE L'INTERFACE
ui <-dashboardPage(
  dashboardHeader( title = "Estimation de la médiane d'un échantillon de nanoparticule",
                   titleWidth = 900),
  dashboardSidebar(
      ## GESTION DES DIFFERENTS MENUS
    sidebarMenu( 
      ## MENU PAGE D'ACCUEIL
      menuItem("PAGE D'ACCUEIL", 
               tabName = "TITRE"),
      ## MENU DATA 
      menuItem("DATA",
               tabName ="Importation"),
      ## MENU DISTRIBUTIONS DES ECHANTILLONS
      menuItem("DISTRIBUTION DES ECHANTILLONS",
               tabName = "DISTRIBUTIONS"),
      ## MENU DISTRIBUTIONS DES MEDIANES
      menuItem("DISTRIBUTION DES MEDIANES",
               tabName = "DISTRIBUTION_M"),
      ## MENU BOOSTRAP
      menuItem("BOOSTRAP", 
               tabName = "EstimationB"),
      ## MENU NORME NF ISO 16269-7
      menuItem("NORME  NF ISO 16269-7",
               tabName = "NORME")
    )
  ),
  dashboardBody(
    ## STYLE DU TITRE 
    tags$head(tags$style(HTML('
                              .main-header .logo {
                              font-family: " ", Times, "Times New Roman", serif;
                              font-weight: bold;
                              font-size: 24px;
                              }
                              ')
    )
    ),
    ## CONTENEUR DES ELEMENTS ONGLETS
    tabItems( 
      ## ONGLET POUR LA PAGE D'ACUEIL 
      tabItem("TITRE",
              ## BOITE CONTENANT LE CORP PRINCIPAL DE LA PAGE D'ACCUEIL
              
              box(title = "Cette  application  programmée avec le langage R a pour but 
                  d'estimer la médiane d'un
                  échantillon de particule en utilisant la méthode de simulation de Monte Carlo,de déterminer
                  l'incertitude-type de mesure et l'intervalle élargi associé, et deux autres méthodes
                  pour faire ces estimations ( Boostrap et la Norme NF ISO 16269-7) mais qui ne 
tiennent pas compte de l'incertitude-type associée aux differenetes mesures de l' échantillon.",
                  status = "primary",solidHeader = TRUE, collapsible = TRUE, width = 12 )
              
      ),
      ## ONGLET DATA (POUR L'IMPORTATION DES DONNEES)
      tabItem(
        tabName = "Importation",
        ## CREATION D'UN CONTROLE DE  TELECHARGEMENT DE FICHIERS (CSV ET XLSX)
        box(
        fileInput("FileInput","Choisir un fichier CSV ou XLSX",
                  multiple = FALSE,
                  accept = c("text/csv",
                             "text/comma-separated-values,text/plain",
                             ".csv",
                             ".xlsx")
        )),
        
        ## BOITE CONTENANT LE CORP DE L'ONGLET DATA
        column( width = 12,
        box( collapsible = TRUE,
             collapsed = F,title = "
             Les boutons ci-dessous permettent d'afficher l'aperçu des données et 
             un resumé de quelques résultats statistiques de l'échantillon",
             
             ## CREATION D'UN BOUTTON POUR VOIR L'APPERCU DES DONNEES
             div(style = 'overflow-y:scroll;height:376px;',
             radioButtons("view", " ",
                          c("Vide","preview","Summary"), 
                          inline = T),
                          verbatimTextOutput("sum"),
                          tableOutput("preview") ),width = 12,status = "primary" ))
             
        
        
        
        
        
       
        
        
        
        
        
        
        
        
        
             
      ),
      ## ONGLET DISTRIBUTIONS DES ECHANTILLONS
      tabItem("DISTRIBUTIONS",
              ## BOITES CONTENANT LE CORP DE L'ONGLET
              box( title="La meilleure distribution de probabilité
                   parmi la loi normale, log-normale, mélange de deux lois normales 
                   et mélange de deux lois log-normales de l'échantillon est donnée ici à partir
du critère BIC et AIC ainsi que la
                   distribution en histogramme avec 
                   une estimation de la loi choisie"
                   ,uiOutput("ech"),
                   
              box(title = "Tableau des paramétres estimés  et le résultat du 
                  critère AIC et BIC pour la selection du modele, en fonction des lois", 
                  status = "primary",
                  div(style = 'overflow-y:scroll;height:376px;',
                  tableOutput("m")),
                  sliderInput("bins","nb of breaks",1,35,20),
                  solidHeader = T, collapsible = T),
                   
              box(title= "Visualisation de la distribution de l'échantillon 
                  avec une estimation de la loi choisie",
                  ## BOUTTON DE SELECTION DE MODELE 
                  
                  radioButtons('Selection', 'selection du modèle',c('Normale','LogNormale',
                                      'Melange de deux lois Normales',
                                      'Melange de deux lois logNormales'),
                                      inline = T),
                                      status = "primary",
                  div(style = 'overflow-y:scroll;height:392px;',
                                      plotOutput("histograme")),
                                      solidHeader = T,
                                      collapsible = T),
                                status = "primary",solidHeader = T,
                                collapsible = TRUE, width = 12)
      ),
      ## ONGLET METHODE DE MONTE CARLO
      tabItem("DISTRIBUTION_M",
              ## BOITE CONTENANT LE CORP DE L'ONGLET
              box(title = "Une fois choisi l'échantillon dont on souhaite estimer la médiane,
                  nous considerons que Chaque représentant de l'échantillon
                  est une variable aléatoire qui suit une loi normale de moyenne la valeur mesurée
                  et d'incertitude-type 2.01 nm (modifiable par l'utilisateur).
                  Nous simulons une réalisation  de chacune des variables aléatoires 
                  de l'échantillon et calculons la médiane des simulations réalisées
                  .On repète cela un certain nombre de fois (nombre de simulation),
                  la moyenne de ces médianes sera notre estimation et 
                  l'incertitude-type équivaut à l'écart-type, 
                  nous determinons également un intervalle élargi à 95%  à partir
                  des quantiles de la simulation.",
                  uiOutput("echan"),
                  
                  box (title = "Choisir le nombre de simulation ",
                       selectInput("nb_Sim","  ",c(100,1e4,1e5)) ),
                  box (title = " Choisir l'incertitude-type ",  
                       numericInput("ecart"," ",value = 2.01)),
                  verbatimTextOutput("e"),
                  box (title = "Résultats de la simulation de Monte Carlo ",
                       div(style = 'overflow-y:scroll;height:356px;',
                       tableOutput("Estim")), 
                       sliderInput("bin","nb of breaks",1,35,20),
                       status = "primary",solidHeader = T, collapsible = TRUE),
                  box (title = "  La distribution des médianes ",
                       div(style = 'overflow-y:scroll;height:472px;',
                       plotOutput("distribution")),
                       status = "primary",solidHeader = T, collapsible = TRUE)
                  ,status = "primary",solidHeader = T, collapsible = TRUE, width = 12)
    ), 
    ## ONGLET BOOSTRAP
    tabItem(tabName = "EstimationB",
            box(title = " Une fois choisi l échantillon et le nombre de 
                replique boostrap, nous obtenons une estimation de la médiane 
                en utilisant la méthode de Boostrap, qui consiste à 
                reéchantillonner avec remise, pour obtenir des échantillons simulés 
               de meme taille que l’échantillon initial. Nous faisons cela avec 
              un nombre d'échantillons bootstrap (Nombre de replicat: B ) qui nous donneront 
              donc  B valeurs de la médiane et nous résumons en prenant la moyenne et l’écart-type de 
                toutes ces médianes.", 
                uiOutput("echant"),
                div(style = 'overflow-y:scroll;height:176px;',
                selectInput("N","Nombre de replicat",c(100,1000,1e4,1e5)),
                tableOutput("Estim.Boot")),status = "primary",solidHeader = T, collapsible = TRUE, width = 12)),
    ## ONGLET NORME 
    tabItem(tabName = "NORME", 
            box(title = "Estimation ponctuelle de la médiane(avec Intervalle de 
                     confiance) selon la la Norme NF ISO 16269-7",
                uiOutput("Echant"),
                box(
                uiOutput("EEchant")), verbatimTextOutput("N_Estimation"),
                selectInput("NV"," Choisir le niveau de confiance",
                            c(80,90,95,98,99,99.5,99.8,99.9)),
                box(title = "Intervalle unilateral",
                    radioButtons("Unilateral","Limite",c("inf","sup"),inline = T),
                    verbatimTextOutput("I_C"),status = "primary",solidHeader = T,
                    collapsible = TRUE),
                box(title = "Intervalle bilateral",
                    verbatimTextOutput("I_CP"),
                    status = "primary",solidHeader = T, 
                    collapsible = TRUE),status = "primary",solidHeader = T, collapsible = TRUE, width = 12 )
    )
    
  ),
 
 
  DT::dataTableOutput("inf"),
  DT::dataTableOutput("sup"),
  DT::dataTableOutput("unilateral"),
  DT::dataTableOutput("bilateral")
  )
    )

  
  

#### METHODE DE MONTE CARLO

Simulation<-function(df,M,e) {
  
  ## CETTE FONCTION NOUS DONNE LES M MEDIANES APRES SIMULATION
  ## ARGUMENTS:
    # df: ECHANTILLON(UN VECTEUR CONTENANT LES DONNEES)
    # M: NOMBRE DE SIMULATION
    # e: INCERTITUDE-TYPE ASSOCIEE A LA MESURE
  
  df<-subset(df,!is.na(df)) 
  X_i<-NULL 
  y<-NULL
  Realisation<-function(){
    # UNE SOUS FONCTION : ELLE TIRE DE MANIERE ALEATOIRE
    # UNE REALISATION DE CHACUNE DES MESURES DE L'ECHANTILLON
    # ELLE RETURN UN VECTEUR DE MEME TAILLE QUE L'ECHANTILLON 
    # CHAQUE MESURE DE L'ECHANTILLON EST UNE VARIABLE ALEATOIRE
    # DE MOYENNE LA VALEURE MESUREE ET D'ECART-TYPE L'INCERTITUDE DE LA MESURE  
    
    for (i in 1:length(df) ) { 
      X_i[i] <- rnorm (1,
                       df[i],
                       e)
    }
    return (X_i)
  }
  # NOUS ALLONS REPETER M FOIS LA SIMULATION
  Y<-replicate(M,Realisation())
  # NOUS RECUPERONS LA MATRICE DES M REALISATIONS
  # CHAQUE COLONNE CONTIENT UN VECTEUR DE REALISATION
  M_<-data.frame (Y)
  # CALCUL DE LA MEDIANE DE CHAQUE COLONNE DE LA MATRICE
  y<-apply(M_,2,median)
  return (y)
}


Estimation_<-function(x){
  ## CETTE FONCTION RECUPERE LE RESULTAT DE LA FONCTION(Simulation) 
  ## i.e PREND LES M MEDIANES OBTENUES ET RETURNE LA MOYENNE DE CES MEDIANES QUI SERA L'ESTIMATION
  ## L'INCERTITUDE-TYPE (ECART-TYPE) ET LES DEUX BORNES DE L'INTERVALLE ELARGI
  ## ARGUMENTS:
  # df: ECHANTILLON(UN VECTEUR CONTENANT LES DONNEES)
  # M: NOMBRE DE SIMULATION
  # e: INCERTITUDE-TYPE ASSOCIEE A LA MESURE
  ## LES RESULTATS SONT ARRONDIS A UN CHIFFRE APRES LA FIRGULE
  #x<- Simulation(X,M,e)
  moyenne_<-mean(x)
  e_c<-sqrt(var(x))
  born_inf<-round (quantile(x,probs = 0.025),1)
  born_sup<-round(quantile(x,probs = 0.975),1)
  # M<-matrix(c(round(moyenne_,1),
  #             round(e_c,1),
  #             round( born_inf,1),
  #             round(born_sup,1)))
  # 
  resultat<-data.frame(Nom = c("Estimation de la mediane",
                         "Incertitude-type",
                         "Borne inferieure", 
                         "Borne superieure"),
                       Valeur = c(round(moyenne_,1),
                         round(e_c,1),
                         round( born_inf,1),
                         round(born_sup,1))   
                       )
  # colnames(M)<-"Resultat"
  # rownames(M)<-c("Estimation de la mediane",
  #                "Incertitude-type",
  #                "Borne inferieure", 
  #                "Borne superieure")
  # Me1<-data.frame(M)
  return (resultat)
}

## ESIMATION PAR BOOSTRAP

# NoUS REECHANTYILLONNONS AVEC REMIOSE, POUR OBTENIR DES ECHANTILLONS SIMULES DE MEME TAILLE
# QUE L'ECHANTILLON INITIAL.NOUS FAISONS CELA AVEC UN NOMBRE D'ECHANTILLON BOOSTRAP ET NOUS 
# OBTENONS CE NOMBRE DE VALEURS DE LA MEDIANE
# AINSI NOUS RESUMONS EN PRENANT LA MOYENNE ET L'ECART-TYPE DE TOUTES CES MEDIANES.

Estim.Boot<-function(X,B) {
  ## CETTE FONCTION NOUS DONNE L'ESTIMATION DE LA MEDIANE ET L'INCERTITUDE-TYPE ASSOCIEE
  nb.obs<-length(X) # RECUPERATION DE LA TAILLE DE L'ECHANTILLON
  # ELLE TIRE DE MANIERE ALEATOIRE nb.obs (LA TAILLE DE L'ECHANTILLON) DE MESURES
  # DANS L'ECHANTILLON AVEC REMISE.
  # X: Echantillon
  # B: Nombre de replique boostrap
  X<-subset(X,!is.na(X))
  med<-rep(0,B) # VECTEUR VIDE
  for (i in 1:B){
    ech.boot<-sample(X,nb.obs,replace = TRUE) # TIRAGE ALEATOIRE AVEC REMISE DE nb.obs DANS X
    med[i]<-median(ech.boot) # CALCUL DE LA MEDIANE DE LA ieme REPETITION DE DU TIRAGE 
  }
  Mediane_moyen<-mean(med) # MOYENNE DES MEDIANES
  e_c1<-sqrt(var(med))
  #e_c<-sqrt(var(estim.med)/nb.obs)
  #result<-matrix(c(round(Mediane_moyen,1),round(e_c1,1)))
  resultat<-data.frame(Nom = c("Estimation de la mediane","Incertitude-type"),
                       valeur = c(round(Mediane_moyen,1),round(e_c1,1)) )
 # colnames(result)<-"Resultat"
  #rownames(result)<-c("Estimation de la mediane","Incertitude-type")
  return (resultat)
  
}

## RECHECHE DU MEILLEURE MODELE 

BestModel<-function(dataNano,Name){
  #Alexandre Allard, LNE, December 2016
  #This function aims at the choice of the best probability distribution for the size of nanoparticles samples. 
  #However, it could be applied to any kind of data.
  # 4 models are considered : 
  # 1) A gaussian distribution
  # 2) A mixture of 2 Gaussian distributions
  # 3) A log-normal distritbuion
  # 4) A mixture of 2 log-normal distributions
  # Inputs : 
  #     - dataNano : A n-by-1 vector containing the data to be analyzed
  #     - Name : A string containing the label for the X-axis of the histogram
  # Output : 
  #     - SummaryFit : A data.frame with four lines corresponding to the four models
  #       Model : The considered model
  #       LL : The log-likelihood
  #       Mu1 : The parameter mu of the first component of the mixture or of the probability distribution
  #       Sigma1 : The parameter sigma of the first component of the mixture or of the probability distribution
  #       Lambda1 : The weight of the first component (1 if the model is not a mixture)
  #       Mu2 : The parameter mu of the second component of the mixture (0 if the model is not a mixture)
  #       Sigma2 : The parameter sigma of the second component of the mixture (0 if the model is not a mixture)
  #       Lambda2 : The weight of the second component (0 if the model is not a mixture)
  #       AIC : Akaike Information Criterion
  #       BIC : Bayesian Information Criterion
  dataNano<-subset(dataNano,!is.na(dataNano))
  #Fit of a Gaussian distribution
  fitGaussian<-fitdist(dataNano,"norm")
  #Fit of a Mixture Gaussian distribution
  modelMixGaussian<-normalmixEM(dataNano,lambda = 0.4, mu=c(mean(dataNano)-sd(dataNano),mean(dataNano)+sd(dataNano)), sigma = c(sd(dataNano),sd(dataNano)), maxit=10000)
  #Fit of a LogNormal distribution
  fitLogN<-fitdist(dataNano,"lnorm")
  #Fit of a mixture of LogNormal distribution
  LData<-log(dataNano)
  modelMixLN<-normalmixEM(LData,lambda = 0.4, mu=c(log(modelMixGaussian$mu[1]),log(modelMixGaussian$mu[2])), sigma = c(modelMixGaussian$sigma[1]/modelMixGaussian$mu[1],modelMixGaussian$sigma[2]/modelMixGaussian$mu[2]),maxit=10000)
  #Create the data.frame containing the results
  SummaryFit=data.frame(
    Modele=c("Normale","Melange de deux lois Normales","LogNormale","Melange de deux lois logNormales"),
    LL=c(fitGaussian$loglik,modelMixGaussian$loglik,fitLogN$loglik,modelMixLN$loglik-sum(log(dataNano))),
    Mu1=c(fitGaussian$estimate[1],modelMixGaussian$mu[1],fitLogN$estimate[1],modelMixLN$mu[1]),
    Sigma1=c(fitGaussian$estimate[2],modelMixGaussian$sigma[1],fitLogN$estimate[2],modelMixLN$sigma[1]),
    Lambda1=c(1,modelMixGaussian$lambda[1],1,modelMixLN$lambda[1]),
    Mode1=c(fitGaussian$estimate[1],modelMixGaussian$mu[1],exp(fitLogN$estimate[1]-fitLogN$estimate[2]^2),exp(modelMixLN$mu[1]-modelMixLN$sigma[1]^2)),
    Median1=c(fitGaussian$estimate[1],modelMixGaussian$mu[1],exp(fitLogN$estimate[1]),exp(modelMixLN$mu[1])),
    Mu2=c(0,modelMixGaussian$mu[2],0,modelMixLN$mu[2]),
    Sigma2=c(0,modelMixGaussian$sigma[2],0,modelMixLN$sigma[2]),
    Lambda2=c(0,modelMixGaussian$lambda[2],0,modelMixLN$lambda[2]),
    Mode2=c(0,modelMixGaussian$mu[2],0,exp(modelMixLN$mu[2] - modelMixLN$sigma[2]^2)),
    Median2=c(0,modelMixGaussian$mu[2],0,exp(modelMixLN$mu[2])),
    AIC=c(round(fitGaussian$aic,1),round(10-2*modelMixGaussian$loglik,1),round(fitLogN$aic,1),round(10-2*(modelMixLN$loglik-sum(log(dataNano))),1)),
    BIC=c(round(fitGaussian$bic,1),round(-2*modelMixGaussian$loglik+log(length(dataNano))*5,1),round(fitLogN$bic,1),round(-2*(modelMixLN$loglik-sum(log(dataNano)))+log(length(dataNano))*5,1))
  )
  return(SummaryFit)
  
}

### APPLICATION DE LA NORME/
## TABLEAU 1
# Echantillon de taille petite
n<-5:100
# Les valeursd de k en fonction du niveau de confiance
k_80<-c(2,2,2,3,3,4,4,5,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,
        13,13,14,14,15,15,15,16,16,17,17,18,18,19,19,20,20,21,21,
        22,22,22,23,23,24,24,25,25,25,26,26,27,27,28,28,29,29,30,30,
        31,31,31,32,32,33,33,34,34,35,35,36,36,37,37,38,38,39,39,40,40,
        41,41,41,42,42,43,43,44,44,45,45,46)
k_90<- c(1,1,2,2,3,3,3,4,4,5,5,5,6,6,7,7,8,8,8,9,9,10,10,11,11,
         11,12,12,13,13,14,14,15,15,16,16,16,17,17,18,18,19,19,20,
         20,20,21,21,22,22,23,23,24,24,25,25,25,26,26,27,27,28,28,29,
         29,30,30,31,31,31,32,32,33,33,34,34,35,35,36,36,37,37,38,38,38,
         39,39,40,40,41,41,42,42,43,43,44)
k_95  <-c(1,1,1,2,2,2,3,3,4,4,4,5,5,6,6,6,7,7,8,8,8,9,9,10,10,11,11,11,12,12,
        13,13,14,14,14,15,15,16,16,17,17,17,18,18,19,19,20,20,21,21,21,22,22,
        23,23,24,24,25,25,25,26,26,27,27,28,28,29,29,29,30,30,31,31,32,32,33,33,
        34,34,34,35,35,36,36,37,37,38,38,39,39,39,40,40,41,41,42)
k_98 <-c(0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,7,7,7,8,8,9,9,9,10,10,11,11,11,12,12,
        13,13,14,14,14,15,15,16,16,17,17,17,18,18,19,19,19,20,20,21,21,22,22,23,23,
        23,24,24,25,25,26,26,26,27,27,28,28,29,29,30,30,30,31,31,32,32,33,33,34,34,34,
        35,35,36,36,37,37,38,38,38,39,39,40)
k_99 <-c(0,0,1,1,1,1,2,2,2,3,3,3,4,4,5,5,5,6,6,6,7,7,8,8,8,9,9,9,10,10,11,11,11,
         12,12,13,13,14,14,14,15,15,16,16,16,17,17,18,18,19,19,19,20,20,21,21,
         21,22,22,23,23,24,24,24,25,25,26,26,27,27,27,28,28,29,29,30,30,31,
         31,31,32,32,33,33,34,34,34,35,35,36,36,37,37,38,38,38)
k_99.5 <-c(0,0,0,1,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,7,7,7,8,8,8,9,
           9,10,10,10,11,11,12,12,12,13,13,14,14,14,15,15,16,16,16,17,
           17,18,18,18,19,19,20,20,21,21,21,22,22,23,23,23,24,24,25,25,26,
           26,26,27,27,28,28,29,29,29,30,30,31,31,32,32,32,33,33,34,34,
           35,35,35,36,36,37,37)
k_99.8 <- c(0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,4,4,4,5,5,5,6,6,7,7,7,8,
            8,8,9,9,10,10,10,11,11,11,12,12,13,13,13,14,14,15,15,15,
            16,16,17,17,17,18,18,19,19,19,20,20,21,21,21,22,22,23,23,23,
            24,24,25,25,26,26,26,27,27,28,28,28,29,29,30,30,31,31,31,32,
            32,33,33,34,34,34,35,35,36)
k_99.9 <- c(0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,7,7,7,8,8,9,
           9,9,10,10,10,11,11,12,12,12,13,13,13,14,14,15,15,15,16,16,17,
           17,17,18,18,19,19,19,20,20,21,21,21,22,22,23,23,23,24,24,25,
           25,25,26,26,27,27,28,28,28,29,29,30,30,30,31,31,32,32,33,33,33,34,34,35)
Tableau1<-cbind(n,k_80,k_90,k_95,k_98,k_99,k_99.5,k_99.8,k_99.9)
Tableau1<-data.frame(Tableau1)
head(Tableau1)

## TABLEAU 2
# Echantillon de taille petite
#  Les valeursd de k en fonction du niveau de confiance
n<-5:100
k_p_80 <-c(1,1,2,2,3,3,3,4,4,5,5,5,6,6,7,7,8,8,8,9,9,10,10,11,11,11,
          12,12,13,13,14,14,15,15,16,16,16,17,17,18,18,19,19,20,20,20,
          21,21,22,22,23,23,24,24,25,25,25,26,26,27,27,28,28,29,29,30,30,
          31,31,31,32,32,33,33,34,34,35,35,36,36,37,37,38,38,38,39,39,
          40,40,41,41,42,42,43,43,44)
k_p_90<- c(1,1,1,2,2,2,3,3,4,4,4,5,5,6,6,6,7,7,8,8,8,9,9,10,10,11,11,
           11,12,12,13,13,14,14,14,15,15,16,16,17,17,17,18,18,19,19,20,
           20,21,21,21,22,22,23,23,24,24,25,25,25,26,26,27,27,28,28,29,29,
           29,30,30,31,31,32,32,33,33,34,34,34,35,35,36,36,37,37,38,38,39,39,
           39,40,40,41,41,42)
k_p_95<- c(0,1,1,1,2,2,2,3,3,3,4,4,5,5,5,6,6,6,7,7,8,8,8,9,9,10,10,10,11,11,
          12,12,13,13,13,14,14,15,15,16,16,16,17,17,18,18,19,19,19,20,20,21,
          21,22,22,22,23,23,24,24,25,25,26,26,26,27,27,28,28,29,29,29,30,30,31,
          31,32,32,33,33,33,34,34,35,35,36,36,37,37,38,38,38,39,39,40,40)
k_p_98<- c(0,0,1,1,1,1,2,2,2,3,3,3,4,4,5,5,5,6,6,6,7,7,8,8,8,9,9,9,10,10,11,11,
           11,12,12,13,13,14,14,14,15,15,16,16,16,17,17,18,18,19,19,19,20,20,21,
           21,21,22,22,23,23,24,24,24,25,25,26,26,27,27,27,28,28,29,29,30,30,31,31,
           31,32,32,33,33,34,34,34,35,35,36,36,37,37,38,38,38)
k_p_99<- c(0,0,0,1,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,7,7,7,8,8,8,9,9,10,10,10,
           11,11,12,12,12,13,13,14,14,14,15,15,16,16,16,17,17,18,18,18,19,19,20,
           20,21,21,21,22,22,23,23,23,24,24,25,25,26,26,26,27,27,28,28,29,29,29,30,
           30,31,31,32,32,32,33,33,34,34,35,35,35,36,36,37,37)
k_p_99.5 <- c(0,0,0,0,1,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,7,7,7,8,8,9,9,9,10,10,10,
              11,11,12,12,12,13,13,14,14,14,15,15,16,16,16,17,17,18,18,18,19,19,20,20,
              20,21,21,22,22,23,23,23,24,24,25,25,25,26,26,27,27,28,28,28,29,29,30,30,30,
              31,31,32,32,33,33,33,34,34,35,35,36,36)
k_p_99.8<- c(0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,7,7,7,8,8,9,9,9,10,10,10,
             11,11,12,12,12,13,13,13,14,14,15,15,15,16,16,17,17,17,18,18,19,19,19,20,20,
             21,21,21,22,22,23,23,23,24,24,25,25,25,26,26,27,27,28,28,28,29,29,30,30,30,
             31,31,32,32,33,33,33,34,34,35)
k_p_99.9<- c(0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,7,7,7,8,8,8,9,9,
             9,10,10,11,11,11,12,12,12,13,13,14,14,14,15,15,15,16,16,17,17,17,18,18,
             19,19,19,20,20,21,21,21,22,22,23,23,23,24,24,25,25,25,26,26,27,27,27,28,
             28,29,29,30,30,30,31,31,32,32,32,33,33,34)
Tableau2<-cbind(n,k_p_80,k_p_90,k_p_95,k_p_98,k_p_99,k_p_99.8,k_p_99.9)
Tableau2<-data.frame(Tableau2)
#head(Tableau2)
# Echantillon de taille superieure a 100 
## TABLEAU 3
# NIVEAU DE CONFIANCE
N_V<- c (80,90,95,98,99,99.5,99.8,99.9)
u<- c(0.84162122,1.28155156,1.64485364,2.05374892,2.32634788,2.57582930,2.87816173,3.09023229)
c <- c( 0.75,0.903,1.087,1.3375,1.536,1.74,2.014,2.222)
Tableau3<-cbind(N_V,u,c)
Tableau3<-data.frame(Tableau3)
#Tableau3
## TABLEAU 4
# NIVEAUN DE CONFIANCE
N_V<- c (80,90,95,98,99,99.5,99.8,99.9)
u<- c(1.28155156,1.64485364,1.95996400,2.32634788,2.57582930,2.80703376,3.09023229,3.29052672)
c <- c( 0.903,1.087,1.274,1.536,1.74,1.945,2.222,2.437)
Tableau4<-cbind(N_V,u,c)
Tableau4<-data.frame(Tableau4)
#Tableau4
##  CALCUL DE L'INTERVALLE DE CONFIANCE
# CAS UNILATERAL
I_C<-function(X,Name,C,Ip){
  X<-subset(X, !is.na(X))
  #C<-as.numeric (readline(prompt = " Quel niveau de confiance parmi #[80,90,95,98,99,99.5,99.8,99.9] ? "))
  # I<-readline(prompt = "Unilateral ou bilateral ? : ")
  X<-sort(X)
  Me<-median(X)
  k<-NULL
  n<-length(X)
  Me<-median(X)
  Tab<-NULL
  ## CAS UNILATERAL
  if( Name =="unilateral") {
    ## SI L'ECHANTILLON EST DE TAILLE PETITE
    if ( n <= 100 ) {
      Tab <- Tableau1
      ## GESTION EN FONCTION DU NIVEAU DE CONFIANCE
      if (C==80) { k <- Tab[,2][n-4] }
      else if ( C==90) { k<-Tab[,3][n-4] }
      else if ( C==95) { k<-Tab[,4][n-4] }
      else if ( C==98 && n >5 ) { k<-Tab[,5][n-4] }
      else if ( C==99 && n > 6) { k<-Tab[,6][n-4] }
      else if ( C==99.5 && n > 7 ) { k<-Tab[,7][n-4] }
      else if ( C==99.8 && n > 8 ) { k<-Tab[,8][n-4] }
      else if ( C==99.9 && n > 9 ) { k<-Tab[,9][n-4] }
      m <- n- k+1
      T1 <- X[k]
      T2 <- X[m]
    }
    ## SI L'ECHANTILLON EST DE GRANDE TAILLE  
    else {
      Tab<-Tableau3
      ## GESTION EN FONCTION DU NIVEAU DE CONFIANCE
      if ( C==80 ) { 
        u <- Tab[,2][1]
        c<- Tab[,3][1]
      }
      else if ( C==90) {
        u <- Tab[,2][2]
        c<- Tab[,3][2] 
      }
      else if ( C==95) {
        u <- Tab[,2][3]
        c<- Tab[,3][3] 
      }
      else if ( C==98) {
        u <- Tab[,2][4]
        c<- Tab[,3][4] 
      }
      else if ( C==99) {
        u <- Tab[,2][5]
        c<- Tab[,3][5]  
      }
      else if ( C==99.5) {
        u <- Tab[,2][6]
        c<- Tab[,2][6] 
      }
      else if ( C==99.8) {
        u <- Tab[,2][7]
        c<- Tab[,3][7]  
      }
      else if ( C==99.9) {
        u <- Tab[,2][8]
        c<- Tab[,3][8]  
      }
      ## CALCUL DE L'EQUATION DONNEE PAR LA NORME
      y<-function ()  { 
        return (0.5*(n+1-u*(1+0.4/n)*sqrt(n-c)))
      }
      
      k <-floor(y())
      T1 <- X[k]
      m<-n-k+1
      T2 <- X[m]
    } 
    # M<-matrix(c(T1,Me,T2))
    # rownames(M)<- c("Limite inferieure  "," Estimation ponctuelle de la médiane  "," Limite superieur")
    #M<-data.frame(M)
    # print ("Limite de confiance inferieure (inf) ou superieur (sup)")
    #Ip<-readline(prompt = "inf ou sup ? : ")
    if (Ip=="inf") {
      M<-matrix(c(round(T1,1),round(Me,1)))
      colnames(M)<-"Resultat"
      rownames(M)<- c("Limite inferieure  "," Estimation ponctuelle de la mediane  ")
      M<-data.frame(M)
      return (M)}
    else if (Ip=="sup") {
      
      M<-matrix(c(round(Me,1),round(T2,1)))
      colnames(M)<-"Resultat"
      rownames(M)<- c(" Estimation ponctuelle de la mediane  "," Limite superieure")
      M<-data.frame(M)
      return ( M)}
    # else { return  ("Ecrivez bien inf ou sup")}
    return (M)}
  #else if (Name!="unilateral" && Name!="bilateral") { return ( "Intervalle Ã  revoir")}
}
## INTERVALLE BILATERAL
I_CP<-function(X,Name,C){
  X<-subset(X, !is.na(X))
  #C<-as.numeric (readline(prompt = " Quel niveau de confiance parmi #[80,90,95,98,99,99.5,99.8,99.9] ? "))
  # I<-readline(prompt = "Unilateral ou bilateral ? : ")
  X<-sort(X)
  Me<-median(X)
  k<-NULL
  n<-length(X)
  Me<-median(X)
  Tab<-NULL 
  if (Name == "bilateral"){
    if ( n <= 100 ) {
      Tab <- Tableau2
      if ( C==80) { k <- Tab[,2][n-4] }
      else if ( C==90) { k<-Tab[,3][n-4] }
      else if ( C==95 && n >5) { k<-Tab[,4][n-4] }
      else if ( C==98 && n >6 ) { k<-Tab[,5][n-4] }
      else if ( C==99 && n > 7) { k<-Tab[,6][n-4] }
      else if ( C==99.5 && n > 8 ) { k<-Tab[,7][n-4] }
      else if ( C==99.8 && n > 9 ) { k<-Tab[,8][n-4] }
      else if ( C==99.9 && n > 10 ) { k<-Tab[,9][n-4] }
      m <- n- k+1
      T1 <- X[k]
      T2 <- X[m]
    }
    else {
      Tab<-Tableau4
      if ( C==80 ) {
        u <- Tab[,2][1]
        c<- Tab[,3][1]
      }
      else if ( C==90) {
        u <- Tab[,2][2]
        c<- Tab[,3][2] 
      }
      else if ( C==95) {
        u <- Tab[,2][3]
        c<- Tab[,3][3] 
      }
      else if ( C==98) {
        u <- Tab[,2][4]
        c<- Tab[,3][4] 
      }
      else if ( C==99) {
        u <- Tab[,2][5]
        c<- Tab[,3][5]  
      }
      else if ( C==99.5) {
        u <- Tab[,2][6]
        c<- Tab[,2][6] 
      }
      else if ( C==99.8) {
        u <- Tab[,2][7]
        c<- Tab[,3][7]  
      }
      else if ( C==99.9) {
        u <- Tab[,2][8]
        c<- Tab[,3][8]  
      }
      y<-function ()  {
        return (0.5*(n+1-u*(1+0.4/n)*sqrt(n-c)))
      }
      k <-floor(y())
      m <- n- k+1
      T1 <- X[k]
      T2 <- X[m]
      
    }
    M<-matrix(c(round(T1,1),round(Me,1),round(T2,1)))
    colnames(M)<-"Resultat"
    rownames(M)<- c("Limite inferieure    "," Estimation ponctuelle    "," Limite superieure")
  }
  return (data.frame(M))
}

ici<-tags$a(href='https://iopscience.iop.org/article/10.1088/1361-6501/ab1495',
            tags$img(src='logo_lne.jpg',height='60',width='200'))


## FONCTION SERVER 

server <- function(input, output) {
  ## LECTURE DES DONNEES (CSV ET XLSX)
  df<-reactive({
    infile <- input$FileInput
    ext <- tools::file_ext(infile$datapath)
    # req(infile)
    if (is.null(infile))
      return(NULL)
    else if(ext =="csv") {
      X<- read.csv(infile$datapath,
                   header = TRUE, sep = ';')}
    else if (ext == "xlsx") {
      X<- read.xlsx( infile$datapath,1)
    }
    return(X)})
  ## CHOIX D'ECHANTILLON
  output$choix <- renderUI({
    selectInput("ys",label = "CHOISIR UN ECHANTILLON",
    choices = names(df()) ,multiple =F)
  })
  output$ech<- renderUI({
    radioButtons("Echantillons", " ", choices = names(df()), 
                 inline = T)
  })
  output$echan<- renderUI({
    radioButtons("Echant"," ", choices = names(df()), 
                 inline = T)
  })
  output$echant<-  renderUI({
    radioButtons("Echanti", " ", choices = names(df()), 
                 inline = T)
  })
  output$Echant<-  renderUI({
    radioButtons("Echantillon", " ", choices = names(df()), 
                 inline = T)
  })
  
## FONCTIONS POUR RENDRE REACTIVE L'ECHANTILLON APPELE
  dis<-reactive({ 
    d<-df()
    v<-input$Echantillons
    d<-d[,v]
    return (d)
  })
  dist<-reactive({ 
    d<-df()
    v<-input$Echant
    d<-d[,v]
    return (d)
  })
  dist1<-reactive({ 
    d<-df()
    v<-input$Echanti
    d<-d[,v]
    return (d)
  })
  dist2<-reactive({ 
    d<-df()
    v<-input$Echantillon
    d<-d[,v]
    return (d)
  })
  Data<-reactive({
    B=df()
    d <- df()
    #d<-subset(d,is.na(d))
    if(input$selectvar=='All'){
      return(print(df()))
    }
    if(input$selectvar=='select'){
      variables=input$ys
      d=d[,variables]
      return(d)}
  })
  ## Lecture PRELMIERES LIGNES 
  output$preview<- renderTable({
    if(input$view == "preview")
      return(head(df()))
  })
  ## SUMMARY 
  output$sum<- renderPrint({
    if(input$view == "Summary")
      return (summary(df()))
  })
  ## RECHERCHE DU MEILLEURE MODELE
  output$m<-renderTable({
    return ( BestModel(dis(),names(dis())))
  })
  ## DISTRIBUTIONS DES ECHANTILLONS ET ESTIMATIONS DES DISTRIBUTIONS PAR LES QUATRES LOIS
    # Normale
    # LogNormale
    # Melange de deux lois Normales
    # Melange de deux lois logNormales
  output$histograme<-renderPlot({
    dataNano<-subset(dis(),!is.na(dis()))
    xS<-sort(dataNano)
    if(input$Selection=="Normale"){
      x<-fitdist(dataNano,"norm")
      hist(dataNano,probability = TRUE, main=names(dis()),breaks=input$bins,
           xlab=expression(Projected~~area-equivalent~~diameter~~D[area-eq]),
           ylab="Probability",ylim=c(0,1.25*max(dnorm(xS,x$estimate[1],x$estimate[2]))))
      lines(xS,dnorm(xS,x$estimate[1],x$estimate[2]),lwd=2, col="red") 
      legend("topleft", legend = "Distribution Normale", fill = "red")}
    else if(input$Selection=="LogNormale") {
      fitLogN<-fitdist(dataNano,"lnorm")     
      hist(dataNano,probability = TRUE, main=names(dis()),breaks=input$bins,
           xlab=expression(Projected~~area-equivalent~~diameter~~D[area-eq]),
           ylab="Probability",ylim=c(0,1.25*max(dnorm(log(xS),
                              fitLogN$estimate[1],fitLogN$estimate[2])/xS)))
      lines(xS,dnorm(log(xS),fitLogN$estimate[1],fitLogN$estimate[2])/xS, lwd=2, col="red")
      legend("topleft", legend = "Distribution LogNormale", fill = "red")
    }
    else if(input$Selection=='Melange de deux lois Normales'){
      modelMixGaussian<-normalmixEM(dataNano,lambda = 0.4,mu=c(mean(dataNano)-sd(dataNano),mean(dataNano)+sd(dataNano)),sigma = c(sd(dataNano),sd(dataNano)), maxit=10000)
      hist(dataNano,probability = TRUE, main=names(dis()),
           breaks=input$bins, xlab=expression(Projected~~area-equivalent~~diameter~~D[area-eq]),
           ylab="Probability",
           ylim=c(0,1.1*max((modelMixGaussian$lambda[1]*dnorm(xS,modelMixGaussian$mu[1],
           modelMixGaussian$sigma[1])+modelMixGaussian$lambda[2]*dnorm(xS,modelMixGaussian$mu[2],
                                                        modelMixGaussian$sigma[2])))))
     # plot(modelMixGaussian,whichplots=2, cex.axis=1.4, cex.lab=1.5, cex.main=1.5, 
          # main2="Mixture Gaussian model",marginal=TRUE)
      lines(xS,(modelMixGaussian$lambda[1]*dnorm(xS,modelMixGaussian$mu[1],modelMixGaussian$sigma[1])+modelMixGaussian$lambda[2]*dnorm(xS,modelMixGaussian$mu[2],modelMixGaussian$sigma[2])), lwd=2, col="blue", xlab=expression(Projected~~area-equivalent~~diameter~~D[area-eq]))
      lines(xS,(modelMixGaussian$lambda[1]*dnorm(xS,modelMixGaussian$mu[1],modelMixGaussian$sigma[1])), lwd=2, col="green", xlab=expression(Projected~~area-equivalent~~diameter~~D[area-eq]))
      lines(xS,(modelMixGaussian$lambda[2]*dnorm(xS,modelMixGaussian$mu[2],modelMixGaussian$sigma[2])), lwd=2, col="red", xlab=expression(Projected~~area-equivalent~~diameter~~D[area-eq]))
      legend(x=150,y=0.01, legend = c("Distribution Normale1","Distribution Normale2","Distribution de deux normales"), fill = c("green","blue","red"),cex = 0.8)
    }
    else if(input$Selection=='Melange de deux lois logNormales'){
      modelMixGaussian<-normalmixEM(dataNano,lambda = 0.4,mu=c(mean(dataNano)-sd(dataNano),mean(dataNano)+sd(dataNano)),sigma = c(sd(dataNano),sd(dataNano)), maxit=10000)
      LData<-log(dataNano)
      modelMixLN<-normalmixEM(LData,lambda = 0.4, mu=c(log(modelMixGaussian$mu[1]),log(modelMixGaussian$mu[2])), sigma = c(modelMixGaussian$sigma[1]/modelMixGaussian$mu[1],modelMixGaussian$sigma[2]/modelMixGaussian$mu[2]),maxit=10000)
      hist(dataNano, probability = TRUE, main=names(dis()),breaks=input$bins, xlab=expression(Projected~~area-equivalent~~diameter~~D[area-eq]),ylab="Probability",ylim=c(0,1.25*max(1/xS*(modelMixLN$lambda[1]*dnorm(log(xS),modelMixLN$mu[1],modelMixLN$sigma[1])+modelMixLN$lambda[2]*dnorm(log(xS),modelMixLN$mu[2],modelMixLN$sigma[2])))))
      lines(xS,1/xS*modelMixLN$lambda[1]*dnorm(log(xS),modelMixLN$mu[1],modelMixLN$sigma[1]), lwd=2, col="green")
      lines(xS,1/xS*modelMixLN$lambda[2]*dnorm(log(xS),modelMixLN$mu[2],modelMixLN$sigma[2]), lwd=2, col="red")
      lines(xS,1/xS*(modelMixLN$lambda[1]*dnorm(log(xS),modelMixLN$mu[1],modelMixLN$sigma[1])+modelMixLN$lambda[2]*dnorm(log(xS),modelMixLN$mu[2],modelMixLN$sigma[2])), lwd=2, col="blue")
      legend(x=150,y=0.01, legend = c("Distribution LogNormale1","Distribution LogNormale2","Distribution de deux Lognormales"), fill = c("green","blue","red"),cex = 0.8)
    }
  }
)
  ## FONCTION POUR RENDRE REACTIVE LA FONCTION SIMULATION DEFINI A L'EXTERIEUR DU SERVER 
  Simu<-reactive({
    return (Simulation(dist(),input$nb_Sim,e()))
  })
  ## FONCTION POUR RENDRE REACTIVE LA FONCTION ESTIMATION_ DEFINI A L'EXTERIEUR DU SERVER 
  
  Estimation<-reactive({
    return (Estimation_(Simu()))
  })
  ## APPEL DE LA FONCTION ESTIMATION POUR LA SIMULATION DE MONTE CARLO
  output$Estim<-renderTable({
    Estimation()
  })
  
  ## ESTIMATION PAR BOOSTRAP
  output$Estim.Boot<-renderTable({
    Estim.Boot(dist1(),input$N)
  }
  )
  ## UNE FONCTION (NOR) REACTIVE POUR UNE ESTIMATION PONCTUELLE DE LA MEDIANE
  Nor<-reactive({
    m<-median(dist2(),na.rm=T)
    MEDIANE<-data.frame(round(m,1))
    colnames(MEDIANE)<-"Resultat"
    rownames(MEDIANE)<-"Estimation ponctuelle de la mediane"
    return (MEDIANE)
  }
  )
  ## ESTIMATION SELON LA NORME 
  output$N_Estimation<-renderPrint ({
    return (Nor())
  }
  )
  
  ## DISTRIBUTION DES MEDIANES
  ## TEMPS D'ATTENTE APRES AVOIR CHOISIR L'INCERTITUDE
  e <- reactive(input$ecart) %>% debounce(1500)
  ## DISTRIBUTION DES MEDIANES SIMULEES
  output$distribution<-renderPlot({
    hist(Simu(),probability = T,breaks = input$bin,col= ' gray ',xlab = 'médiane (nm)', main = " ")
    abline( v= Estimation()[1,2] , col= "red")
    abline( v=Estimation()[3,2] , col= "blue")
    abline( v= Estimation()[4,2], col= "blue")
    legend("topleft", legend = c("y_inf", "Estimation", "y_sup"), fill = c("blue", "red", "blue")
    )
  }
)
  
  ## INTERVALLE DE CONFIANCE (UNILATERAL)
  output$EEchant<-renderUI({checkboxGroupInput("show_vars", "Columns in diamonds to show:",
                                                 names(df()), selected = names(df()))})
  dist2<-reactive({ 
    d<-df()
    v<-input$Echantillon
    d<-d[,v]
    return (d)
  })
  X<-reactive({checkboxGroupInput("show_vars", "Columns in diamonds to show:",
                                  names(df()), selected = names(df()))})
  
  output$I_C<-renderPrint({ 
  I_C(dist2(),"unilateral",input$NV,input$Unilateral)
    
  }
)
  ## INTERVALLE DE CONFIANCE (BILATERAL)

  output$I_CP<-renderPrint({ 
     I_CP(dist2(),"bilateral",input$NV)
  }
  )
}
shinyApp(ui = ui, server = server)


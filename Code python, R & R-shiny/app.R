library(shiny)
library(shinydashboard)

ui <- dashboardPage(
  dashboardHeader( title = "Exemple"),
  dashboardSidebar(
    menuItem("DATA",
             tabName ="Importation")
  ),
  dashboardBody(
    
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
              
        ),
        div(style = 'overflow-y:scroll;height:376px;',
            radioButtons("view", " ",
                         c("Vide","preview","Summary"), 
                         inline = T),
            verbatimTextOutput("sum"),
            tableOutput("preview") )
        
        )
      
      )
    
  )
)

server <- function(input, output) { 
  
  df<-reactive({
    infile <- input$FileInput
    ext <- tools::file_ext(infile$datapath)
    if (is.null(infile))
      return(NULL)
    else if(ext =="csv") {
      X<- read.csv(infile$datapath,
                   header = TRUE, sep = ';')}
    else if (ext == "xlsx") {
      X<- read.xlsx( infile$datapath,1)
    }
    return(X)})
  
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
  
  
  
  
  }

shinyApp(ui, server)
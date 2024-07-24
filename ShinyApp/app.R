library(gtools)
library(shiny)
library(shinydashboard)

# Paths
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
print(getwd())

# Theme
library(bslib)
theme = bs_theme(bootswatch = "morph",
                 version = 5,
                 success = "#86C7ED",
                 "table-color" = "#86C7ED",
                 base_font = font_google("Lato"))

# UI
ui <- navbarPage(
  "Splendor AI",
  theme = theme,
  tags$head(
    tags$style(HTML("
      .container-fluid {
        max-width: 1000px; 
        margin: 0 auto; 
      }
    "))
  ), 
  tabPanel("Introduction", 
           fluidPage(
             fluidRow(
               column(8,
                      img(src = "introduction_banner.png", height = "200px", 
                          style = "margin-bottom: 20px; border-radius: 6px;"),
               ),
               column(4,
                      div(class = "card text-white bg-warning mb-3", 
                          style = "max-width: 20rem; height: 200px",
                          div(class = "card-header", "Alert!"),
                          div(class = "card-body",
                              h4(class = "card-title", "If you just want to play the game, please go to ", a(href = "https://example.com", "gamehost.com")),
                              p(class = "card-text", "Access is completely free.")
                          )
                      )
               )
             ), 
             h3("Welcome to Splendor AI!"),
             p("This project showcases my work in building a Splendor AI trained through self-play. In the navigation bar up top you can learn more about the model, training process, and even see the AI in action."), 
             h3("Why did I make this?"), 
             p("This is actually my first reinforcement learning project.  While my heart lies with language processing, a lot of people I look up to consistently mention how important it is to know the foundations of everything we've discovered in machine learning.  RLHF has been driving our ability to actually interface with large models, and I wanted to build my way towards integrating RLHF into my own language model (53m parameters, hosted at _).")
           )
  ),
  tabPanel("Code Walkthrough",
           fluidPage(
             tabsetPanel(
               tabPanel("Environment",
                        br(),
                        h2("Physical overview"),
                        p("Here's what the environment looks like:"), 
                        fluidRow(
                          column(6,
                                 img(src = "game_vis/example_game/state_26.jpg", width = "100%", 
                                     style = "margin-bottom: 20px; border-radius: 6px;")
                          ),
                          column(3,
                                 h5("The game board:"), 
                                 img(src = "game_vis/board.jpg", width = "100%", 
                                     style = "margin-bottom: 20px; border-radius: 6px;")
                          ),
                          column(3,
                                 h5("One of two players:"),
                                 img(src = "game_vis/player.jpg", width = "100%", 
                                     style = "margin-bottom: 20px; border-radius: 6px;")
                          )
                        ),
                        tags$ul(
                          tags$li("**Nobles**: Points automatically awarded for reaching their requirements"),
                          tags$li("**Tokens**: Players can choose 3 unique or 2 of the same kind per turn, as their move for that turn."),
                          tags$li("**Cards**: Cards that yield a reusable token when purchased, and possibly points.")
                        ), 
                        br(), 
                        h2("State representation"), 
                        p(style="text-align: justify;",
                          strong("The final state vector has 242 dimensions, which is made up of 150 from the board and 42 for each player. "),
                          "The reason for the large size is the one-hot representations of gems, meaning that instead of using", 
                          "one dimension from 0-6 or -3-3, I wanted to keep the representation orthogonal and use 6 dimensions.", 
                          "The board is made up of 3 tiers of 4 cards each, meaning that there are 12 cards plus 3 nobles on the board. ", 
                          "Each card is represented as a one-hot of its gem type, so length 5, plus another vector of", 
                          "length 5 for its cost and one for the points it rewards. ", 
                          strong("This means that the card shop makes up 3*4*(5+5+1) = 12*11 = 132 dimensions. "), 
                          strong("The nobles make up 3*(5+1) = 18, and the board gems make up just 6."), 
                          "Each player's portion of the state has 6 dimensions representing their gem inventory, 5 representing their", 
                          "purchased resources, and 33 for their (up to) 3 reserved cards of length 11."), 
                        img(src = "game_vis/full state vectorized.jpg", height = "600px"), 
                        p("")
               ),
               tabPanel("Model",
                        br(),
                        h2("The Model Design"),
                        p("Description of the model architecture."),
                        fluidRow(
                          column(6, 
                                 img(src = "splendor_tensorboard/action_weights.png", width = "100%", 
                                     style = "margin-bottom: 20px; border-radius: 6px;")
                          ),
                          column(6, 
                                 img(src = "splendor_tensorboard/Dense1_weights.png", width = "100%", 
                                     style = "margin-bottom: 20px; border-radius: 6px;")
                          )
                        ),
                        p(style = "text-align: justify;",
                          "A lot of model architectures yeilded promising results.  This section will be expanded on after completion.")
               ),
               tabPanel("Training",
                        br(),
                        h2("Training Process"),
                        p(style = "text-align: justify;", "Something hugely useful about reinforcement learning is its inherit ability to not overfit.  We're generating examples from a nearly infinite set of states in which the cards and player information are shuffling around and changing.  The remaining problem is the correlation of states, but within my code, I've shuffled the data to avoid this using random.sample.  The only necessary requirements of the model are to not have an excessively large model or learning rate."), 
                        img(src = "random.jpg"), 
                        p("You can see my training dynamics below along with my exponentially decaying learning rate."), 
                        fluidRow(
                          column(6, 
                                 img(src = "splendor_tensorboard/learning_rate.png", width = "100%", 
                                     style = "margin-bottom: 20px; border-radius: 6px;")
                          ),
                          column(6, 
                                 img(src = "splendor_tensorboard/epsilon.png", width = "100%", 
                                     style = "margin-bottom: 20px; border-radius: 6px;")
                          )
                        ),
                        fluidRow(
                          column(6, 
                                 img(src = "splendor_tensorboard/batch_loss.png", width = "100%", 
                                     style = "margin-bottom: 20px; border-radius: 6px;")
                          ),
                          column(6, 
                                 img(src = "splendor_tensorboard/avg_q.png", width = "100%", 
                                     style = "margin-bottom: 20px; border-radius: 6px;")
                          )
                        ),
                        br(),
                        h3("Training Dynamics"),
                        p(style = "text-align: justify;", "The dynamics of the training process and how to interpret the results."),
                        fluidRow(
                          column(6, 
                                 img(src = "splendor_tensorboard/avg_reward.png", width = "100%", 
                                     style = "margin-bottom: 20px; border-radius: 6px;")
                          ),
                          column(6, 
                                 img(src = "splendor_tensorboard/action_hist.png", width = "100%", 
                                     style = "margin-bottom: 20px; border-radius: 6px;")
                          )
                        )
               )
             ),
             tags$style(HTML("
               .nav.nav-tabs {
                 display: flex;
                 justify-content: center;
               }
               .nav.nav-tabs li {
                 flex: 1;
                 text-align: center;
               }
             "))
           )
  ),
  tabPanel("Model Showcase", 
           fluidPage(
             h2("Model Showcase"),
             p("This highlights performance graphs (vs. human and self) and shows example games."), 
             img(src = "temp.jpg", height = "300px", style="margin-bottom: 20px;"),
             br(),
             h3("Game Viewer"),
             p("This are some of the highlight games I took from the model training process.  Not all games are at its highest performance, but rather show behaviors I find interesting and very dynamic."), 
             fluidRow(
               column(3,
                      selectInput("scenario", "Select Game Scenario:", choices = c("Not Buying Tier 1 Cards", "Resource Hogging")),
                      actionButton("prevBtn", "Previous"),
                      actionButton("nextBtn", "Next")
               ),
               column(9,
                      imageOutput("gameImage", width = "100%")
               )
             )
           )
  )
)

# Server
server <- function(input, output, session) {
  # Load scenarios and directories
  image_dirs <- list(
    "Not Buying Tier 1 Cards" = "game_vis/example_game/straight_shot/",
    "Resource Hogging" = "game_vis/example_game/resource_hogging/"
  )
  
  current_index <- reactiveVal(1)
  
  images <- reactive({
    scenario <- input$scenario
    dir_path <- file.path("www", image_dirs[[scenario]])
    mixedsort(list.files(path = dir_path, pattern = "state_\\d+\\.jpg", full.names = TRUE))
  })
  
  # Update index upon input
  observeEvent(input$scenario, {
    current_index(1)
  })
  
  # Navigate to the previous image
  observeEvent(input$prevBtn, {
    new_index <- max(current_index() - 1, 1)
    current_index(new_index)
  })
  
  # Navigate to the next image
  observeEvent(input$nextBtn, {
    new_index <- min(current_index() + 1, length(images()))
    current_index(new_index)
  })
  
  # Display the current image
  output$gameImage <- renderImage({
    img_list <- images()
    list(
      src = img_list[current_index()],
      contentType = "image/jpeg",
      alt = "Game State Image",
      style = "max-width: 100%; max-height: 500px;"
    )
  }, deleteFile = FALSE)
}

# Run
shinyApp(ui = ui, server = server)

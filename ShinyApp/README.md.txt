# Splendor AI Shiny App

This repository contains a Shiny app for visualizing and interacting with the Splendor AI project.

## Running the App

To run the app locally, follow these steps:

1. **Clone this repository:**
   ```sh
   git clone https://github.com/BreckEmert/Splendor-AI.git
   cd Splendor-AI/ShinyApp

2. **Install required R packages**
    ```{r}
    install.packages(c("shiny", "shinydashboard", "gtools"))
    ```
3. **Run the app**
    ```{r}
    shiny::runApp("app.R")
    ```

## Overview

### Introduction Tab
- Overview of the project and its objectives

### Code Walkthrough Tab
- Detailed description of the environment setup and state representation.
- Information about the model and training process.

### Model Showcase Tab
- Allows users to select various games that I found interesting and navigate through a realistic visualization of the game states.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
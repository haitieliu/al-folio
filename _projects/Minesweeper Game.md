---
layout: page
title: Minesweeper Game
description: AI-Powered Minesweeper Game
img: /assets/img/minesweeper/minelogo.png
importance: 4
category: fun
---

## Introduction

In this project, I developed an interactive Minesweeper game using Python and the Pygame library. The game not only allows users to play Minesweeper with a graphical interface but also features an Artificial Intelligence (AI) agent capable of making moves based on logical deductions.


![Additional Screenshot](/assets/img/minesweeper/2024-01-06%20205328.png)

## Technologies Used

- **Python**: The core programming language for the project.
- **Pygame**: A Python library used for game development, providing the tools necessary for rendering graphics, handling events, and more.
- **Minesweeper and AI Logic**: Custom Python classes to handle the game logic and AI decision-making.

## Game Features

- **Standard Minesweeper Gameplay**: Users can click to reveal cells and right-click to place flags on cells where they suspect mines are located.
- **AI Integration**: At any point during the game, players can invoke an AI agent to make a safe move or a random move if no obvious safe moves are available.
- **Customizable Difficulty**: The game's grid size and the number of mines can be adjusted to change the difficulty level.
- **Graphical Interface**: The game presents a visually appealing interface with clear indications of mines, flags, and numbers indicating nearby mines.

## Development Process

1. **Setting Up Pygame**: The first step involved setting up Pygame and creating a basic window where the game would be played.
2. **Designing the Game Board**: I defined the game board using a grid system. Each cell in the grid could either contain a mine or a number indicating the count of adjacent mines.
3. **Implementing Game Logic**: The core logic of Minesweeper, including mine placement and game rules (revealing cells, game win/loss conditions), was implemented.
4. **AI Agent Development**: I developed an AI agent using logical reasoning. The AI uses knowledge about safe and mined cells to make informed decisions.
5. **User Interface Design**: I added a user-friendly interface, including buttons for user interactions like resetting the game or asking the AI for help.
6. **Testing and Refinement**: The final stage involved thorough testing and refining the game's mechanics and AI's decision-making process.

## Challenges and Learnings

- **AI Implementation**: Developing an AI agent that makes decisions based on the current state of the game was challenging and required a good understanding of logical deduction and Minesweeper's rules.
- **Pygame Mechanics**: Learning and effectively using Pygame for rendering the game's interface was a key part of the project.

## Conclusion

This project was a rewarding experience that combined game development with artificial intelligence. It not only solidified my skills in Python programming but also in applying logical problem-solving to create an intelligent game agent.

## Screenshots

![Minesweeper Game Screenshot](/assets/img/minesweeper/Screenshot%202024-01-06%20205523.png)
![Additional Screenshot](/assets/img/minesweeper/010602.png)


## Check Out the Code

The complete implementation of my Minesweeper game, including the AI agent, is available on GitHub. Feel free to explore the code, try it out, and contribute if you're interested!

[View the Minesweeper Project on GitHub](https://github.com/haitieliu)

In this repository, you'll find detailed code for the game logic, AI algorithms, and graphical interface using Pygame. The README file in the repository provides more information on how to set up and run the game.



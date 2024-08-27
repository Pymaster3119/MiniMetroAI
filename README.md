# Mini Metro AI

I recreated the game Mini Metro and created an AI model to play it!

## Introduction

Mini Metro AI is a recreation of the popular game Mini Metro, where players design subway maps by connecting different stations. This project includes an AI model that learns how to play the game autonomously, managing the metro system efficiently.

## Installation

To run the project locally, follow these steps:

1. Clone the repository:

git clone https://github.com/yourusername/minimetro-ai.git

3. Start the game and AI:

python run_game.py

## Usage

- gameinterminal.py: Starts the Mini Metro game and the AI that plays it.
- Customize the game's parameters or train the AI with different settings by modifying other files.

## AI Model

The AI model is trained using Deep Q-Learning (DQN) with a custom environment simulating the game. It takes the current state of the game (stations, lines, and trains) and predicts the optimal actions.

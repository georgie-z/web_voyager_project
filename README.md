# Web Voyager

This project is based on [WebVoyager](https://arxiv.org/abs/2401.13919) by He, et. al., a vision-enabled web-browsing agent capable of controlling the mouse and keyboard. The agent works by viewing annotated browser screenshots for each turn, then choosing the next step to take. The agent architecture is a basic reasoning and action (ReAct) loop.

The original code was cloned from https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/web-navigation/web_voyager.ipynb and was adapted for this task.

# AI Web Page Navigator for Bunnings

## Project Overview

This project involves developing an AI-driven web page navigator for [Bunnings Warehouse](http://Bunnings.com.au). The core functionality requires the AI to autonomously search for a product, add it to the cart, and navigate to the checkout review page while handling minor UI variations.  

Beyond the core task, the additional feature chosen for implementation is:
* Trajectory Maintenance – Ensures the AI can recover from navigation failures and continue toward its objective.

## Summary of technologies used that helped achieved the task:

1. LangChain & LangGraph (AI Reasoning and Decision-Making)
- LangChain is used to structure the AI’s decision-making process and interactions with the web.
- LangGraph is a graph-based execution framework that organizes the AI's actions into a structured workflow.
- These technologies help the AI agent process user requests, track its state, and make sequential decisions while browsing.

2. GPT-4V (Multi-Modal Model for Vision and Text)
- The agent leverages GPT-4V, a vision-enabled AI model that can analyse web page screenshots.
- This allows the AI to “see” web elements and interact with them intelligently (e.g., clicking buttons, scroll, type, etc).
- It ensures the AI can navigate dynamic web pages without relying on pre-defined HTML structures.

3. Playwright (Automated Browser Interaction)
- Playwright is used to control a web browser programmatically.
- The AI can simulate user actions such as clicking, typing, scrolling, and navigating between pages.
- This enables real-time interaction with websites like Bunnings.

4. Bounding Box Annotations (Set-of-Marks UI Affordances)
- The AI uses JavaScript to mark web elements (buttons, text fields, etc.) with bounding boxes.
- These numbered labels allow GPT-4V to reference elements visually, making its actions more accurate.

##  Explanation of how the AI agent makes decisions and handles edge cases

The AI agent uses the **ReAct Loop** (Reasoning and Acting Loop) to make decisions in a structured, iterative process:

1. **Reasoning**: The agent first examines the environment (the state - page, user input, screenshot image and bounding boxes) and is guided by the provided prompt for context, task instructions, and constraints, helping the agent determine the next logical step. 

2. **Action**: After reasoning, the agent takes action, such as clicking a button, typing text, scrolling or go back to original URL etc.

3. **Observation**: After the action, the agent observes the changes in the environment (e.g., the page’s new state). It uses the scratchpad to record its previous reasoning, actions and observations, reassesses the situation, and decides whether further action is needed.

4. **Iteration**: The loop repeats—reasoning about the new state, taking actions, and observing outcomes—until the task is completed. This cycle of reasoning, acting, and observing helps the agent adapt to dynamic situations and make informed decisions. 

## Demo Video

<a href="https://drive.google.com/file/d/17NRo0UA5qV3mxJW9P0stG4rfNWsqbzBR/view?usp=drive_link">
  <img src="https://github.com/georgie-z/web_voyager_project/blob/main/screenshot.png" width="900" />
</a>

## Instructions to run

### Set up prerequisites

- Python 3.9 or higher
- [pip](https://pip.pypa.io/en/stable/installation/)
- [Node.js](https://nodejs.org/) (required for Playwright)
- OpenAI API key
- Langchain API key (optional)

### Clone repo
git clone `https://github.com/georgie-z/web_voyager_project.git`  
cd web_voyager_project

### Create an environment and install dependencies

python -m venv web_voyager_env  
web_voyager_env\Scripts\activate  
pip install -r requirements.txt  

### Set Up Environment Variables:
Replace 'OPENAI_API_KEY' and 'LANGCHAIN_API_KEY' with your actual API keys in the .env file

### Instructions to run the AI agent

Optional: To edit the product you want to order go to the end (line 505) of web_voyager_bunnings.py script and edit the question. Example is already provided in the code. 

In terminal or command prompt:  
python web_voyager_bunnings.py



  




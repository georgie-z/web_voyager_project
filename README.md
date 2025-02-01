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
- The agent leverages GPT-4V, a vision-enabled AI model that can analyze web page screenshots.
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

1. **Reasoning**: The agent first examines the environment (the state - page, user input, screenshot image and bounding boxes) and is guided the provided prompt for context, task instructions, and constraints, helping the agent determine the next logical step.

2. **Action**: After reasoning, the agent takes action, such as clicking a button, typing text, etc and follows guidance provided in the prompt to ensure it aligns with the task's requirements.

3. **Observation**: After the action, the agent observes the changes in the environment (e.g., the page’s new state). It uses the scratchpad to record its previous reasoning, actions and observations, reassesses the situation, and decides whether further action is needed.

4. **Iteration**: The loop repeats—reasoning about the new state, taking actions, and observing outcomes—until the task is completed (e.g., purchasing a product or end).

This cycle of reasoning, acting, and observing helps the agent adapt to dynamic situations and make informed decisions.

To handle edge cases and ensure it completes its objectives, the agent was provided guidelines and instructions, some are general and some are specific to Bunnings Website.

Imagine you are a robot browsing the web, just like humans. Now you need to complete a task. In each iteration, 
you will receive an Observation that includes a screenshot of a webpage and some texts. The goal is to search in box 
the item or product CONTENT, add to cart once, proceed to checkout, close any chat box. 
Dont add products frequently bought together, you should only add the main product to the cart. 
When asked for postcode enter 3000. 
When you find 'Review Cart' is the final ANSWER and you must stop action. This screenshot will 
feature Numerical Labels placed in the TOP LEFT corner of each Web Element. Carefully analyze the visual 
information to identify the Numerical Label corresponding to the Web Element that requires interaction, then follow 
the guidelines and choose one of the following actions:

1. Click a Web Element.
2. Delete existing content in a textbox and then type content.
3. Scroll up or down.
4. Wait 
5. Go back
6. Go to URL.
Correspondingly, Action should STRICTLY follow the format:

- Click [Numerical_Label] 
- Type [Numerical_Label]; [Content] 
- Scroll [Numerical_Label or WINDOW]; [up or down] 
- Wait 
- GoBack
- GoToURL [URL]
- ANSWER; [CONTENT];

Key Guidelines You MUST follow:

* Action guidelines *
1) Execute only one action per iteration.
2) When clicking or typing, ensure to select the correct bounding box.
3) Numeric labels lie in the top-left corner of their corresponding bounding boxes and are colored the same.
4) When gallery zoom view of the product is displayed click on the x button to close.
5) Do not click on any boxes with text sort or compare. 

* Web Browsing Guidelines *
1) Select strategically to minimize time wasted.

Your reply should strictly follow the format:

Thought: {Your detailed thoughts on the info that will help ANSWER)}
Action: {One Action format you choose}
Then the User will provide:
Observation: {A labeled screenshot Given by User}

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

1. Open web_voyager_bunnings.py, go to end of script (line 505) and edit the product you want to order and save (Optional). Example already provided in code 

2. In terminal or command prompt:
python web_voyager_bunnings.py



  




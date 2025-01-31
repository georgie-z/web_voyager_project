# %% 
# ### Web Voyager Bunnings

# The original code was cloned from https://github.com/langchain-ai/langgraph/blob/main/examples/web-navigation/web_voyager.ipynb and modified for the purpose of this project.

# WebVoyager (https://arxiv.org/abs/2401.13919) by He, et. al., is a vision-enabled web-browsing agent capable of controlling the mouse and keyboard.
# 
# It works by viewing annotated browser screenshots for each turn, then choosing the next step to take. The agent architecture is a basic reasoning and action (ReAct) loop. 
# The unique aspects of this agent are:
# - It's usage of [Set-of-Marks](https://som-gpt4v.github.io/)-like image annotations to serve as UI affordances for the agent
# - It's application in the browser by using tools to control both the mouse and keyboard

# %% Set up 
# Install required packages - uncomment to install
# %pip install -U --quiet langgraph langsmith langchain_openai
# %pip install --upgrade --quiet  playwright > /dev/null
# !playwright install

# %%
# Load environment variables from the .env file
from dotenv import load_dotenv
load_dotenv()

# %%
import nest_asyncio

# This is just required for running async playwright in a Jupyter notebook
nest_asyncio.apply()

# %% 
# ### Define graph
# 
# ### Define graph state
# 
# The state provides the inputs to each node in the graph.
# 
# In our case, the agent will track the webpage object (within the browser), annotated images + bounding boxes, the user's initial request, and the messages containing the agent scratchpad, system prompt, and other information.
# 

# %%
from typing import List, Optional
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage, SystemMessage
from playwright.async_api import Page


class BBox(TypedDict):
    x: float
    y: float
    text: str
    type: str
    ariaLabel: str


class Prediction(TypedDict):
    action: str
    args: Optional[List[str]]


# This represents the state of the agent
# as it proceeds through execution
class AgentState(TypedDict):
    page: Page  # The Playwright web page lets us interact with the web environment
    input: str  # User request
    img: str  # b64 encoded screenshot
    bboxes: List[BBox]  # The bounding boxes from the browser annotation function
    prediction: Prediction  # The Agent's output
    # A system message (or messages) containing the intermediate steps
    scratchpad: List[BaseMessage]
    observation: str  # The most recent response from a tool

# %%
# ### Define tools
# 
# The agent has 6 simple tools:
# 
# 1. Click (at labeled box)
# 2. Type
# 3. Scroll
# 4. Wait
# 5. Go back
# 6. Go to search engine (Google)
# 
# 
# We define them below here as functions:

# %%
import asyncio
import platform

async def click(state: AgentState):
    # - Click [Numerical_Label]
    page = state["page"]
    click_args = state["prediction"]["args"]
    if click_args is None or len(click_args) != 1:
        return f"Failed to click bounding box labeled as number {click_args}"
    bbox_id = click_args[0]
    bbox_id = int(bbox_id)
    try:
        bbox = state["bboxes"][bbox_id]
    except Exception:
        return f"Error: no bbox for : {bbox_id}"
    x, y = bbox["x"], bbox["y"]
    await page.mouse.click(x, y)
    # TODO: In the paper, they automatically parse any downloaded PDFs
    # We could add something similar here as well and generally
    # improve response format.
    return f"Clicked {bbox_id}"


async def type_text(state: AgentState):
    page = state["page"]
    type_args = state["prediction"]["args"]
    if type_args is None or len(type_args) != 2:
        return (
            f"Failed to type in element from bounding box labeled as number {type_args}"
        )
    bbox_id = type_args[0]
    bbox_id = int(bbox_id)
    bbox = state["bboxes"][bbox_id]
    x, y = bbox["x"], bbox["y"]
    text_content = type_args[1]
    await page.mouse.click(x, y)
    # Check if MacOS
    select_all = "Meta+A" if platform.system() == "Darwin" else "Control+A"
    await page.keyboard.press(select_all)
    await page.keyboard.press("Backspace")
    await page.keyboard.type(text_content)
    await page.keyboard.press("Enter")
    return f"Typed {text_content} and submitted"


async def scroll(state: AgentState):
    page = state["page"]
    scroll_args = state["prediction"]["args"]
    if scroll_args is None or len(scroll_args) != 2:
        return "Failed to scroll due to incorrect arguments."

    target, direction = scroll_args

    if target.upper() == "WINDOW":
        # Not sure the best value for this:
        scroll_amount = 500
        scroll_direction = (
            -scroll_amount if direction.lower() == "up" else scroll_amount
        )
        await page.evaluate(f"window.scrollBy(0, {scroll_direction})")
    else:
        # Scrolling within a specific element
        scroll_amount = 200
        target_id = int(target)
        bbox = state["bboxes"][target_id]
        x, y = bbox["x"], bbox["y"]
        scroll_direction = (
            -scroll_amount if direction.lower() == "up" else scroll_amount
        )
        await page.mouse.move(x, y)
        await page.mouse.wheel(0, scroll_direction)

    return f"Scrolled {direction} in {'window' if target.upper() == 'WINDOW' else 'element'}"


async def wait(state: AgentState):
    sleep_time = 5
    await asyncio.sleep(sleep_time)
    await asyncio.sleep(sleep_time)
    return f"Waited for {sleep_time}s."


async def go_back(state: AgentState):
    page = state["page"]
    await page.go_back()
    return f"Navigated back a page to {page.url}."


async def go_to_url(state: AgentState):
    page = state["page"]
    go_to_url_args = state["prediction"]["args"]
    if go_to_url_args is None or len(go_to_url_args) != 1:
        return f"Failed to go to {go_to_url_args}"
    await page.goto(go_to_url_args[0])
    return f"Navigated to {go_to_url_args}"

# %% 
# ### Define Agent
# 
# The agent is driven by a multi-modal model and decides the action to take for each step. It is composed of a few runnable objects:
# 
# 1. A `mark_page` function to annotate the current page with bounding boxes
# 2. A prompt to hold the user question, annotated image, and agent scratchpad
# 3. GPT-4V to decide the next steps
# 4. Parsing logic to extract the action
# 
# 
# Let's first define the annotation step:
# #### Browser Annotations
# 
# This function annotates all buttons, inputs, text areas, etc. with numbered bounding boxes. GPT-4V then just has to refer to a bounding box
# when taking actions, reducing the complexity of the overall task.

# %%
import base64

from langchain_core.runnables import chain as chain_decorator

# Some javascript we will run on each step
# to take a screenshot of the page, select the
# elements to annotate, and add bounding boxes
with open("mark_page.js") as f:
    mark_page_script = f.read()


@chain_decorator
async def mark_page(page):
    await page.evaluate(mark_page_script)
    for _ in range(10):
        try:
            bboxes = await page.evaluate("markPage()")
            break
        except Exception:
            # May be loading...
            asyncio.sleep(3)
    screenshot = await page.screenshot()
    # Ensure the bboxes don't follow us around
    await page.evaluate("unmarkPage()")
    return {
        "img": base64.b64encode(screenshot).decode(),
        "bboxes": bboxes,
    }

# %% 
# #### Agent definition
# 
# Now we'll compose this function with the prompt, llm and output parser to complete our agent.

# %%
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI


async def annotate(state):
    marked_page = await mark_page.with_retry().ainvoke(state["page"])
    return {**state, **marked_page}


def format_descriptions(state):
    labels = []
    for i, bbox in enumerate(state["bboxes"]):
        text = bbox.get("ariaLabel") or ""
        if not text.strip():
            text = bbox["text"]
        el_type = bbox.get("type")
        labels.append(f'{i} (<{el_type}/>): "{text}"')
    bbox_descriptions = "\nValid Bounding Boxes:\n" + "\n".join(labels)
    return {**state, "bbox_descriptions": bbox_descriptions}


def parse(text: str) -> dict:
    action_prefix = "Action: "
    if not text.strip().split("\n")[-1].startswith(action_prefix):
        return {"action": "retry", "args": f"Could not parse LLM Output: {text}"}
    action_block = text.strip().split("\n")[-1]

    action_str = action_block[len(action_prefix) :]
    split_output = action_str.split(" ", 1)
    if len(split_output) == 1:
        action, action_input = split_output[0], None
    else:
        action, action_input = split_output
    action = action.strip()
    if action_input is not None:
        action_input = [
            inp.strip().strip("[]") for inp in action_input.strip().split(";")
        ]
    return {"action": action, "args": action_input}


# Will need a later version of langchain to pull
# this image prompt template

# %%
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate
from langchain.prompts.chat import MessagesPlaceholder
from langchain.prompts.chat import HumanMessagePromptTemplate
from langchain_core.prompts.image import ImagePromptTemplate

prompt = ChatPromptTemplate(
  messages=[
    SystemMessagePromptTemplate(
    prompt=[
        PromptTemplate.from_template(
            "Imagine you are a robot browsing the web, just like humans. Now you need to complete a task. In each iteration, \n"
            "you will receive an Observation that includes a screenshot of a webpage and some texts. The goal is to search in box \n"
            "the item or product CONTENT, add to cart once, proceed to checkout, close any chat box. \n"
            "Dont add products frequently bought together, you should only add the main product to the cart. \n"
            "When asked for postcode enter 3000. \n"
            "When you find 'Review Cart' is the final ANSWER and you must stop action. This screenshot will \n"
            "feature Numerical Labels placed in the TOP LEFT corner of each Web Element. Carefully analyze the visual \n"
            "information to identify the Numerical Label corresponding to the Web Element that requires interaction, then follow \n"
            "the guidelines and choose one of the following actions:\n\n"
            "1. Click a Web Element.\n"
            "2. Delete existing content in a textbox and then type content.\n"
            "3. Scroll up or down.\n"
            "4. Wait \n"
            "5. Go back\n"
            "6. Go to URL.\n"
            "Correspondingly, Action should STRICTLY follow the format:\n\n"
            "- Click [Numerical_Label] \n"
            "- Type [Numerical_Label]; [Content] \n"
            "- Scroll [Numerical_Label or WINDOW]; [up or down] \n"
            "- Wait \n"
            "- GoBack\n"
            "- GoToURL [URL]\n"
            "- ANSWER; [CONTENT];\n\n"
            "Key Guidelines You MUST follow:\n\n"
            "* Action guidelines *\n"
            "1) Execute only one action per iteration.\n"
            "2) When clicking or typing, ensure to select the correct bounding box.\n"
            "3) Numeric labels lie in the top-left corner of their corresponding bounding boxes and are colored the same.\n"
            "4) When gallery zoom view of the product is displayed click on the x button to close.\n"
            "5) Do not click on any boxes with text sort or compare. \n\n"
            "* Web Browsing Guidelines *\n"
            "1) Select strategically to minimize time wasted.\n\n"
            "Your reply should strictly follow the format:\n\n"
            "Thought: {{Your detailed thoughts on the info that will help ANSWER)}}\n"
            "Action: {{One Action format you choose}}\n"
            "Then the User will provide:\n"
            "Observation: {{A labeled screenshot Given by User}}\n"
        ),
    ],
),
    MessagesPlaceholder(
      optional=True,
      variable_name="scratchpad",
    ),
    HumanMessagePromptTemplate(
      prompt=[
        ImagePromptTemplate(
          template={"url":"data:image/png;base64,{img}"},
          input_variables=[
            "img",
          ],
        ),
        PromptTemplate.from_template("{bbox_descriptions}"),
        PromptTemplate.from_template("{input}"),
      ],
    ),
  ],
  input_variables=[
    "bbox_descriptions",
    "img",
    "input",
  ],
  partial_variables={"scratchpad":[]},
)

# prompt = hub.pull("wfh/web-voyager")
print(prompt)
# %%
# Initialise a gpt-4o model
llm = ChatOpenAI(model="gpt-4o", max_tokens=4096)

# Define the agent by chaining components
agent = annotate | RunnablePassthrough.assign(
    prediction=format_descriptions | prompt | llm | StrOutputParser() | parse
)

# %% 
# ## Compile the graph
# 
# We've created most of the important logic. We have one more function to define that will help us update the graph state after a tool is called.

# %%
import re

def update_scratchpad(state: AgentState):
    """After a tool is invoked, we want to update
    the scratchpad so the agent is aware of its previous steps"""
    old = state.get("scratchpad")
    if old:
        txt = old[0].content
        last_line = txt.rsplit("\n", 1)[-1]
        step = int(re.match(r"\d+", last_line).group()) + 1
    else:
        txt = "Previous action observations:\n"
        step = 1
    txt += f"\n{step}. {state['observation']}"

    return {**state, "scratchpad": [SystemMessage(content=txt)]}

# %% 
# Now we can compose everything into a graph:

# %%
from langchain_core.runnables import RunnableLambda

from langgraph.graph import END, START, StateGraph

graph_builder = StateGraph(AgentState)


graph_builder.add_node("agent", agent)
graph_builder.add_edge(START, "agent")

graph_builder.add_node("update_scratchpad", update_scratchpad)
graph_builder.add_edge("update_scratchpad", "agent")

tools = {
    "Click": click,
    "Type": type_text,
    "Scroll": scroll,
    "Wait": wait,
    "GoBack": go_back,
    "GoToURL": go_to_url
}

for node_name, tool in tools.items():
    graph_builder.add_node(
        node_name,
        # The lambda ensures the function's string output is mapped to the "observation"
        # key in the AgentState
        RunnableLambda(tool) | (lambda observation: {"observation": observation}),
    )
    # Always return to the agent (by means of the update-scratchpad node)
    graph_builder.add_edge(node_name, "update_scratchpad")

def select_tool(state: AgentState):
    # Any time the agent completes, this function
    # is called to route the output to a tool or
    # to the end user.
    action = state["prediction"]["action"]
    if action == ["ANSWER"]:
        return END
    if action == "retry":
        return "agent"
    print(f"Routing action: {action}")
    return action

graph_builder.add_conditional_edges("agent", select_tool)

graph = graph_builder.compile()

# %% 
# ## Use the graph
# 
# Now that we've created the whole agent executor, we can run it on a few questions! We'll start our browser at "https://www.bunnings.com.au" and then let it control the rest.
# 
# Below is a helper function to help print out the steps to the notebook (and display the intermediate screenshots).

# %%
from IPython import display
from playwright.async_api import async_playwright

async def setup_browser():
    browser = await async_playwright().start()
    # We will set headless=False so we can watch the agent navigate the web.
    browser = await browser.chromium.launch(headless=False, args=None)
    page = await browser.new_page()
    return page

async def call_agent(question: str, page, max_steps: int = 150):
    event_stream = graph.astream(
        {
            "page": page,
            "input": question,
            "scratchpad": [],
        },
        {
            "recursion_limit": max_steps,
        },
    )
    final_answer = None
    steps = []
    async for event in event_stream:
        # We'll display an event stream here
        if "agent" not in event:
            continue
        pred = event["agent"].get("prediction") or {}
        action = pred.get("action")
        action_input = pred.get("args")
        display.clear_output(wait=False)
        steps.append(f"{len(steps) + 1}. {action}: {action_input}")
        print("\n".join(steps))
        display.display(display.Image(base64.b64decode(event["agent"]["img"])))
        if "ANSWER" in action:
            final_answer = action_input[0]
            print(final_answer)
            break
    return final_answer

# %%

async def main(question):
    page = await setup_browser()
    res = await call_agent(question, page)
    print(f"Final response: {res}")
    return

if __name__ == "__main__":
    question = "Go to https://www.bunnings.com.au, close the chat box and buy small garden hose"
    asyncio.run(main(question))



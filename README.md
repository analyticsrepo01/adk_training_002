# ADK 101 Training - 30min

This README provides a quick start guide to using the Agent Development Kit (ADK) based on the "ADK_Training_30min.ipynb" notebook.

## Prerequisites

-   A Google Cloud project with the Vertex AI API enabled.
-   `gcloud` CLI installed and configured.
-   Python 3.10 or higher.
-   Familiarity with Jupyter notebooks.

## Setup

1.  **Environment Variables:**

    -   Set the `PROJECT_ID` and `LOCATION` environment variables. Replace `"my-project-0004-346516"` with your Google Cloud project ID and `"us-central1"` with your desired location.

    ```python
    import os

    PROJECT_ID = "my-project-0004-346516"  # Replace with your project ID

    if not PROJECT_ID:
        PROJECT_ID = str(os.environ.get("GOOGLE_CLOUD_PROJECT"))

    LOCATION = "us-central1"  # @param {type:"string"}

    os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID
    os.environ["GOOGLE_CLOUD_LOCATION"] = LOCATION
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "TRUE"  # Use Vertex AI API
    ```

2.  **Install ADK:**

    -   Download the ADK wheel file from the specified Google Cloud Storage bucket.

    ```bash
    !gcloud storage ls gs://adk_training/sdk
    !gcloud storage cp gs://adk_training/sdk/google_adk-0.0.2.dev20250324+739344859-py3-none-any.whl .
    ```

    -   Install the downloaded wheel file using pip.

    ```bash
    !pip3 install google_adk-0.0.2.dev20250324+739344859-py3-none-any.whl
    ```

## Training the Models

The notebook demonstrates training and usage of different agents. Here's a breakdown:

### 1. Hello World Agent

-   This agent simply outputs "hello world" in a random language.
-   The agent is defined with an instruction to always say "hello world" and to output it in a random language, with the language in brackets.

    ```python
    from google.genai import types
    from google.adk.agents import Agent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService

    MODEL = "gemini-2.0-flash-001"

    hello_world_agent = Agent(
        model=MODEL,
        name="hello_world_agent",
        description="An agent that says 'hello world'",
        instruction="""You always say 'hello world' to the user, and nothing else.
        Output 'hello world' in a random language.
        Put the language in brackets.

        Example Output 1:
        hello world (English)

        Example Output 2:
        你好，世界 (Chinese)
        """,
        generate_content_config=types.GenerateContentConfig(
            max_output_tokens=100,
        ),
    )

    # Session and Runner
    session_service = InMemorySessionService()
    session = session_service.create_session(app_name="hello_world_example", user_id="user12345", session_id="session12345")
    runner = Runner(agent=hello_world_agent, app_name="hello_world_example", session_service=session_service)

    # Agent Interaction
    def call_agent(runner, query):
      content = types.Content(role='user', parts=[types.Part(text=query)])
      events = runner.run(user_id="user12345", session_id="session12345", new_message=content)
      return events

    events = call_agent(runner, "hello")
    ```

### 2. Hello Name Agent (Multi-turn Conversation)

-   This agent engages in a multi-turn conversation to learn the user's name and then greets them.
-   The agent first asks for the user's name and persists in trying to get the name. Once the name is provided, it greets the user.

    ```python
    # Agent
    hello_name_agent = Agent(
        model=MODEL,
        name="hello_name_agent",
        description="An agent that says 'hello USERNAME'",
        instruction="""
        You need to first ask the user's name.
        Try best to convince the user to give you a name, let it be first name, last name, or nick name.

        Once you get the user's name, say 'hello USERNAME'.
        """,
        generate_content_config=types.GenerateContentConfig(
            max_output_tokens=100,
        ),
    )

    # Session and Runner
    session_service = InMemorySessionService()
    session = session_service.create_session(app_name="hello_name_example", user_id="user12345", session_id="session12345")
    runner = Runner(agent=hello_name_agent, app_name="hello_name_example", session_service=session_service)

    # Agent Interaction
    def call_agent(runner, session, query):
      content = types.Content(role='user', parts=[types.Part(text=query)])
      events = runner.run(user_id=session.user_id, session_id=session.id, new_message=content)
      return events

    events = call_agent(runner, session, "hello")
    ```

### 3. Simple Math Agent (Using Tools)

-   This agent uses Python functions as tools to perform basic arithmetic operations.
-   The agent is equipped with `add`, `subtract`, `multiply`, and `divide` functions.

    ```python
    def add(numbers: list[int]) -> int:
      """Calculates the sum of a list of integers."""
      return sum(numbers)

    def subtract(numbers: list[int]) -> int:
        """Subtracts numbers in a list sequentially from left to right."""
        if not numbers:
            return 0  # Handle empty list
        result = numbers[0]
        for num in numbers[1:]:
            result -= num
        return result

    def multiply(numbers: list[int]) -> int:
      """Calculates the product of a list of integers."""
      product = 1
      for num in numbers:
        product *= num
      return product

    def divide(numbers: list[int]) -> float:  # Use float for division
        """Divides numbers in a list sequentially from left to right."""
        if not numbers:
            return 0.0 # Handle empty list
        if 0 in numbers[1:]: # Check for division by zero
            raise ZeroDivisionError("Cannot divide by zero.")
        result = numbers[0]
        for num in numbers[1:]:
            result /= num
        return result

    simple_math_agent = Agent(
        model=MODEL,
        name="simple_math_agent",
        description="This agent performs basic arithmetic operations (addition, subtraction, multiplication, and division) on user-provided numbers, including ranges.",
        instruction="""
          I can perform addition, subtraction, multiplication, and division operations on numbers you provide.
          Tell me the numbers you want to operate on.
          For example, you can say 'add 3 5', 'multiply 2, 4 and 3', 'Subtract 10 from 20', 'Divide 10 by 2'.
          You can also provide a range: 'Multiply the numbers between 1 and 10'.
        """,
        generate_content_config=types.GenerateContentConfig(temperature=0.2),
        tools=[add, subtract, multiply, divide],
    )
    ```

### 4. Advanced Math Agent (Agent as Tool)

-   This agent uses the `simple_math_agent` as a tool to solve complex math problems.
-   It breaks down complex computations into simpler operations and delegates them to the `simple_math_agent`.

    ```python
    agent_math_advanced_instruction = '''
    I am an advanced math agent. I handle user query in the below steps:

    1. I shall analyse the chat log to understand current question and make a math formula for it.
    2. Break down a complex compuation based on arithmetic priority and hand over to simple_math_agent for the calculation.
    3. Note that simple_math_agent can only understand numbers, so I need to convert natural language expression of numbers into digits.

    <example>
    <input> alice gives us 3 apples, bob gives us 5 apples. They do this seven times. Then we eat four apples. How many apples do we have now? </input>
    <think> what is (3+5) * 7 -4 </think>
    <think>I need to first calculate (3+5) as the highest priority operation.</think>
    <call_tool> pass (3+5) to simple_math_agent </call_tool>
    <tool_response>8</tool_response>
    <think> The question now becomes 8 * 7 - 4, and next highest operation is 8 * 7</think>
    <call_tool> pass 8 * 7 to simple_math_agent </call_tool>
    <tool_response>56</tool_response>
    <think> The question now becomes 56 - 4, and next highest operation is 56 - 4</think>
    <call_tool> pass 56 - 4 to simple_math_agent </call_tool>
    <tool_response>52</tool_response>
    <think>There is a single number, so it is the final answer.</think>
    <output>The result of "(3+5) * 7 - 4" is 52</output>
    </example>
    '''

    agent_math_advanced = Agent(
        model=MODEL,
        name="agent_math_advanced",
        description="The advanced math agent can break down a complex computation into multiple simple operations and use math_agent to solve them.",
        instruction=agent_math_advanced_instruction,
        tools=[AgentTool(agent=simple_math_agent)],
        generate_content_config=types.GenerateContentConfig(temperature=0.2),
    )
    ```

### 5. Grammar Correction Agent (Input/Output Format Control)

-   This agent corrects grammar mistakes in text, explains the errors, and returns both the corrected text and the explanations in JSON format.
-   It uses Pydantic schemas to ensure data consistency and validity.

    ```python
    from typing import List
    from pydantic import BaseModel, Field

    class OutputSchema(BaseModel):
        original_query: str = Field(description="The original text from user.")
        corrected_text: str = Field(description="The corrected text.")
        errors: List[str] = Field(description="An array of descriptions of each error.")
        explanations: List[str] = Field(description="An array of explanations for each correction.")

    json_schema = OutputSchema.model_json_schema()

    agent_grammar = Agent(
        model=MODEL,
        name='agent_grammar',
        description="This agent corrects grammar mistakes in text provided by children, explains the errors in simple terms, and returns both the corrected text and the explanations.",
        instruction=f"""
            You are a friendly grammar helper for kids.  Analyze the following text,
            correct any grammar mistakes, and explain the errors in a way that a
            child can easily understand.  Don't just list the errors; explain them
            in a paragraph using simple but concise language.

            Output in a JSON object with the below schema:
            {json_schema}
        """,
        output_schema=OutputSchema,
        generate_content_config=types.GenerateContentConfig(response_mime_type="application/json"),
        disallow_transfer_to_parent = True,
        disallow_transfer_to_peers=True
    )
    ```

## Running the Agents

-   The notebook provides example code for running each agent and interacting with them.
-   Use the `call_agent` function to send queries to the agents and the `pprint_events` function to display the agent's responses.

## Additional Resources

-   [Agent Development Kit (ADK) Documentation](https://cloud.google.com/agent-development-kit)
-   [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
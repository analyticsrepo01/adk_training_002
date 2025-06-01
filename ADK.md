pip install google-adk toolbox-core

adk

adk create hotel-agent-app

    from google.adk.agents import Agent

    root_agent = Agent(
        model='gemini-2.0-flash-001',
        name='hotel_agent',
        description='A helpful assistant that answers questions about a specific city.',
        instruction='Answer user questions about a specific city to the best of your knowledge. Do not answer questions outside of this.',
    )

adk web

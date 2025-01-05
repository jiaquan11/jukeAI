from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)
from langchain_openai import ChatOpenAI

messages = [
    SystemMessage("you're a good assistant, you always respond with a joke."),
    HumanMessage("i wonder why it's called langchain"),
    AIMessage(
        'Well, I guess they thought "WordRope" and "SentenceString" just didn\'t have the same ring to it!'
    ),
    HumanMessage("and who is harrison chasing anyways"),
    AIMessage(
        "Hmmm let me think.\n\nWhy, he's probably chasing after the last cup of coffee in the office!"
    ),
    HumanMessage("what do you call a speechless parrot"),
]

messages_last = trim_messages(
    messages,
    max_tokens=45,
    strategy="last", #截取策略，保留最后一条对话消息
    token_counter=ChatOpenAI(model="gpt-4o-mini"),
)
print(messages_last)
print("===============")

# 保留 system prompt
messages_system = trim_messages(
    messages,
    max_tokens=45,
    strategy="last",
    token_counter=ChatOpenAI(model="gpt-4o-mini"),
    include_system=True,
    allow_partial=True,
)
print(messages_system)
print("===============")

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    filter_messages,
)

messages = [
    SystemMessage("you are a good assistant", id="1"),
    HumanMessage("example input", id="2", name="example_user"),
    AIMessage("example output", id="3", name="example_assistant"),
    HumanMessage("real input", id="4", name="bob"),
    AIMessage("real output", id="5", name="alice"),
]

messages_filter = filter_messages(messages, include_types="human")
print(messages_filter)
print("===============")

messages_filter2 = filter_messages(messages, exclude_names=["example_user", "example_assistant"])
print(messages_filter2)
print("===============")

messages_filter3 = filter_messages(messages, include_types=[HumanMessage, AIMessage], exclude_ids=["3"])
print(messages_filter3)
print("===============")
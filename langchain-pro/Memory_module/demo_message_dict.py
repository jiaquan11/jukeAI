from langchain.memory import ChatMessageHistory
from langchain.schema import messages_from_dict, messages_to_dict

# 实例化对象
history = ChatMessageHistory()

# 添加历史信息
history.add_user_message("hi")
history.add_ai_message("what is up?")
# 保存历史信息到字典里

dicts = messages_to_dict(history.messages)
print(dicts)

# 重加载历史信息
new_message = messages_from_dict(dicts)
print(new_message)
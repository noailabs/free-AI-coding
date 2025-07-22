# Free AI coding
Future of coding is writing specifications (AGENTS.md)

There are many AI coding agents and tools available (free of commercial):  
- [claude-code](https://github.com/anthropics/claude-code)
- [gemini-cli](https://github.com/google-gemini/gemini-cli)
- [codex](https://github.com/openai/codex)
- [opencode](https://github.com/opencode-ai/opencode)
- [kilocode](https://github.com/Kilo-Org/kilocode)
- [Roo-Code](https://github.com/RooCodeInc/Roo-Code)
- [Cline](https://github.com/cline/cline)
- [Cursor](https://cursor.com/)
- [Windsurf](https://windsurf.com/)
- ...
  
But you can do AI coding absolutely free, without any IDEs, editors, or agents. You just need access to the latest, most capable models via the web or an API.

### Minimal Openai compatible (Legacy) API script 

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url=os.environ["OPENAI_BASE_URL"],
    api_key=os.environ["OPENAI_API_KEY"]
)

#system_prompt="""You are helpfull assistant"""
#messages=[{"role": "system","content": system_prompt}]
settings={
    "temperature":1.0
}

model=os.environ["OPENAI_MODEL"]

def llm(messages):
  stream = client.chat.completions.create(
    messages=messages,
    model=model,
    stream=True,
    stream_options={"include_usage": True},
    **settings
  )
  responses=[]
  usage = None
  for chunk in stream:
    content=chunk.choices[0].delta.content
    if content is not None:
      print(content, end="")
      responses.append(content)
    if chunk.usage:
      usage = chunk.usage
  print()
#  print("usage:",usage)
  return {
      "response": "".join(responses),
      "usage": usage
  }

def llmp(prompt):
  messages=[{"role": "user","content": prompt}]
  return llm(messages)
```

### Generate llm-cli.py spec
```python
SPEC="""# llm-cli.py specification file
llm-cli.py – A tiny, robust CLI wrapper for OpenAI-compatible chat endpoints.

Features:
- Reads prompts from command line, file, or stdin: line + stdin + file.
- Manages conversation history in a JSON file.
- Configurable via environment variables or a .env file (OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL).
- Streams responses for immediate feedback.
- Saves responses to a file.
- Control over model settings like temperature.

Usage Examples
--------------
# 1) Simple prompt from command line
llm-cli.py --prompt "Explain quantum entanglement"

# 2) Pipe content from a file as the prompt
cat prompt.md | llm-cli.py --system_file system.md --response_file answer.md

# 3) Continue an existing conversation
llm-cli.py --messages_file history.json --prompt "Give me a Python example"

# 4) Use a different model and temperature for more creative output
llm-cli.py -p "Write a haiku about a robot" --model gpt-4 --temperature 1.2

# 5) Debug the payload being sent to the API
cat code.py | llm-cli.py -p "Review this code" -v

# 6) Dry run, print context before send to llm and exit (do nothig)
llm-cli.py -p "Say a joke about cats" --dry
"""
model="gemini-2.5-pro"
llmp(f"Generate python code based on this specification:{SPEC}")
print()

```

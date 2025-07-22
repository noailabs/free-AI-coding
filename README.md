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

### Write minimal llm-cli.py (llm cli tool) spec and generate code
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

### llm-cli.py
```python
%%writefile llm-cli.py
import argparse
import os
import sys
import json
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

# Load environment variables from .env file (e.g., OPENAI_API_KEY, LLM_MODEL)
load_dotenv()

# --- Utility Functions ---

def _read_content_from_file(filepath: str) -> str | None:
    """Reads content from a given file path."""
    if filepath and os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except IOError as e:
            print(f"Error reading file {filepath}: {e}", file=sys.stderr)
            sys.exit(1)
    return None

def _read_content_from_stdin() -> str | None:
    """Reads content from stdin if available."""
    # Check if stdin is being piped into (not an interactive terminal)
    if not sys.stdin.isatty() and not sys.stdin.closed:
        try:
            content = sys.stdin.read().strip()
            return content if content else None
        except Exception as e:
            print(f"Error reading from stdin: {e}", file=sys.stderr)
            sys.exit(1)
    return None

def load_messages(filepath: str) -> list[dict]:
    """Loads conversation history from a JSON file."""
    if filepath and os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                messages = json.load(f)
                if not isinstance(messages, list):
                    raise ValueError("Messages file must contain a JSON array.")
                return messages
        except (IOError, json.JSONDecodeError, ValueError) as e:
            print(f"Error loading messages from {filepath}: {e}", file=sys.stderr)
            sys.exit(1)
    return []

def save_messages(filepath: str, messages: list[dict]):
    """Saves conversation history to a JSON file."""
    if filepath:
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(messages, f, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"Error saving messages to {filepath}: {e}", file=sys.stderr)
            sys.exit(1)

# --- Main Logic ---

def main():
    parser = argparse.ArgumentParser(
        description="A tiny, robust CLI wrapper for OpenAI-compatible chat endpoints.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""\
Usage Examples:
  1) Simple prompt from command line:
     llm-cli.py --prompt "Explain quantum entanglement"

  2) Pipe content from a file as the prompt (stdin used as main prompt):
     cat prompt.md | llm-cli.py --system_file system.md --response_file answer.md

  3) Continue an existing conversation:
     llm-cli.py --messages_file history.json --prompt "Give me a Python example"

  4) Use a different model and temperature for more creative output:
     llm-cli.py -p "Write a haiku about a robot" --model gpt-4 --temperature 1.2

  5) Debug the payload being sent to the API (combines explicit prompt with stdin):
     cat code.py | llm-cli.py -p "Review this code" -v

  6) Dry run, print context before send to llm and exit (do nothig):
     llm-cli.py -p "Say a joke about cats" --dry
"""
    )

    # Input/Output options
    parser.add_argument(
        '-p', '--prompt', type=str,
        help="The prompt string to send to the LLM. If combined with stdin, it acts as a prefix."
    )
    parser.add_argument(
        '--prompt_file', type=str,
        help="Path to a file containing the prompt. Takes precedence over --prompt. If combined with stdin, it acts as a prefix."
    )
    parser.add_argument(
        '--system', type=str,
        help="An optional system message to guide the LLM's behavior."
    )
    parser.add_argument(
        '--system_file', type=str,
        help="Path to a file containing the system message. Takes precedence over --system."
    )
    parser.add_argument(
        '--messages_file', type=str,
        help="Path to a JSON file for reading/writing conversation history."
    )
    parser.add_argument(
        '--response_file', type=str,
        help="Path to a file to save the complete LLM response."
    )

    # Model parameters
    parser.add_argument(
        '--model', type=str,
        help=f"The model to use. Defaults to LLM_MODEL env var or 'gpt-3.5-turbo'. (Current default: {os.getenv('LLM_MODEL', 'gpt-3.5-turbo')})"
    )
    parser.add_argument(
        '--temperature', type=float, default=1.0,
        help="Sampling temperature (0.0 to 2.0). Higher values make output more random."
    )
    parser.add_argument(
        '--max_tokens', type=int,
        help="The maximum number of tokens to generate in the chat completion."
    )

    # Control/Debug options
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help="Enable verbose output, including the full payload sent to the API."
    )
    parser.add_argument(
        '--dry', action='store_true',
        help="Perform a dry run: print the conversation context and exit without calling the API."
    )

    args = parser.parse_args()

    # --- Setup API Client ---
    api_key = os.getenv('OPENAI_API_KEY') or os.getenv('LLM_API_KEY')
    api_base = os.getenv('OPENAI_BASE_URL') or os.getenv('LLM_BASE_URL')
    model = args.model or os.getenv('OPENAI_MODEL')

    if not api_key and not args.dry: # API key is not needed for dry run
        print("Error: OPENAI_API_KEY or LLM_API_KEY environment variable not set.", file=sys.stderr)
        print("Please set your API key or use '--dry' for a dry run.", file=sys.stderr)
        sys.exit(1)

    client_params = {}
    if api_key:
        client_params['api_key'] = api_key
    if api_base: # Only set base_url if a custom base URL is provided
        client_params['base_url'] = api_base
    # If api_base is not set, OpenAI client uses its default (https://api.openai.com/v1)

    try:
        client = OpenAI(**client_params)
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}", file=sys.stderr)
        sys.exit(1)


    # --- Prepare Messages ---
    messages = []

    # Load existing messages if --messages_file is provided
    if args.messages_file:
        messages = load_messages(args.messages_file)

    # Add system message (priority: --system_file > --system)
    system_content = _read_content_from_file(args.system_file)
    if not system_content:
        system_content = args.system

    if system_content:
        # Check if system message already exists as the first message
        # If not, or if different, prepend it. If different, replace it.
        if messages and messages[0].get('role') == 'system':
            if messages[0].get('content') != system_content:
                messages[0] = {'role': 'system', 'content': system_content}
        else:
            messages.insert(0, {'role': 'system', 'content': system_content})


    # Get prompt content (priority: --prompt_file > --prompt. Then combine with stdin)
    primary_prompt_content = None
    if args.prompt_file:
      primary_prompt_content = _read_content_from_file(args.prompt_file)
    if args.prompt:
      primary_prompt_content = f"{args.prompt.strip()}\n\n{primary_prompt_content.strip()}"

    stdin_content = _read_content_from_stdin()
    if stdin_content:
      primary_prompt_content = f"{stdin_content.strip()}\n\n{primary_prompt_content.strip()}"

    if not primary_prompt_content:
        print("Error: No prompt provided. Use --prompt, --prompt_file, or pipe content to stdin.", file=sys.stderr)
        sys.exit(1)

    prompt_content = primary_prompt_content
    # Add the current user prompt
    messages.append({'role': 'user', 'content': prompt_content})

    # --- Prepare API Payload ---
    api_payload = {
        "model": model,
        "messages": messages,
        "temperature": args.temperature,
        "stream": True # Always stream as per specification
    }
    if args.max_tokens is not None:
        api_payload['max_tokens'] = args.max_tokens

    # --- Dry Run ---
    if args.dry:
        print("\n--- Dry Run: API Payload ---")
        print(json.dumps(api_payload, indent=2, ensure_ascii=False))
        print("-----------------------------\n")
        sys.exit(0)

    # --- Verbose Output ---
    if args.verbose:
        print("\n--- Sending Payload ---")
        print(json.dumps(api_payload, indent=2, ensure_ascii=False))
        print("-----------------------\n")

    # --- Make API Call and Stream Response ---
    full_response_content = ""
    try:
        # For a clean display when streaming
        if not args.verbose:
            print("\n--- LLM Response ---")

        stream = client.chat.completions.create(**api_payload)

        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                print(content, end='', flush=True)
                full_response_content += content
        print("\n--------------------\n") # End of streamed response

    except OpenAIError as e:
        print(f"\nError calling OpenAI API: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Save Response and Messages ---
    if full_response_content:
        # Add the assistant's response to messages
        messages.append({'role': 'assistant', 'content': full_response_content.strip()})

        # Save complete response to file
        if args.response_file:
            try:
                with open(args.response_file, 'w', encoding='utf-8') as f:
                    f.write(full_response_content.strip())
                if args.verbose:
                    print(f"Response saved to {args.response_file}")
            except IOError as e:
                print(f"Error saving response to {args.response_file}: {e}", file=sys.stderr)

        # Save updated conversation history
        if args.messages_file:
            save_messages(args.messages_file, messages)
            if args.verbose:
                print(f"Conversation history saved to {args.messages_file}")

if __name__ == "__main__":
    main()
```

### Analyze llm-cli.py itself
```bash
!python llm-cli.py -p "analyze this code, suggest improvements:" -v --prompt_file llm-cli.py --dry
```


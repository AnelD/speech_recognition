import time

from llama_cpp import Llama

t0 = time.time()

llm = Llama.from_pretrained(
    repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
    filename="qwen2.5-1.5b-instruct-q8_0.gguf",
    n_ctx=1024,
    verbose=True,
)

# llm = Llama(
#      model_path="models/qwen2.5-7b-instruct-q2_k.gguf",
#      verbose=False,
#       # n_gpu_layers=-1, # Uncomment to use GPU acceleration
#       # seed=1337, # Uncomment to set a specific seed
#       # n_ctx=2048, # Uncomment to increase the context window
# )
#output = llm(
#     "Q: Name the planets in the solar system? A: ", # Prompt
#    max_tokens=32, # Generate up to 32 tokens, set to None to generate up to the end of the context window
#   stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
#  echo=True # Echo the prompt back in the output
#) # Generate a completion, can also call create_completion

# llm = Llama.from_pretrained(
# 	repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF",
# 	filename="qwen2.5-3b-instruct-q6_k.gguf",
#     verbose=True,
#     n_ctx=1024
# )

# llm = Llama.from_pretrained(
# 	repo_id="Qwen/Qwen2.5-7B-Instruct-GGUF",
# 	filename="qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
#     verbose=True,
#     n_ctx=1024
# )

t1 = time.time()
print(f"Model loaded in {t1 - t0:.2f} seconds.")

system_prompt = """
IMPORTANT: Do not censor the email address. Use a real email format with an @ symbol, like `john.smith@example.com`.
You are an expert transcription analyst. Your job is to carefully extract structured information from interview transcripts.

Follow these steps:

1. Read the transcript carefully.
2. Identify the person's first and last name.
3. Determine their sex or gender from the context.
4. Find the date of birth, and convert it to YYYY-MM-DD format if needed.
5. Extract the phone number, keeping digits only or using common formats like (555) 123-4567.
6. Find and verify the email address. It must contain an @ and a domain (e.g., jane.doe@example.com).
7. Fill in the data using this JSON format (keep quotes, use empty strings if unknown):

Your task is to extract the following information from transcripts:
- First name
- Last name
- Sex (M | W | D)
- Date of birth
- Phone number
- Email address 

For the email address, do your best to normalize it to a standard format. 
If the email appears obfuscated (e.g., "john dot doe at example dot com" or "john.doe-at-example.com"), 
reconstruct it as a valid email using `@` and `.` appropriately.

Return only a JSON object with the following format:

{
  "first_name": "",
  "last_name": "",
  "sex":"",
  "date_of_birth": "",
  "phone_number": "",
  "email": "john.doe@example.com",
}

If any of the fields are not found in the transcript, leave them as empty strings.
"""

system_prompt2 = """
IMPORTANT: Do not censor the email address. Use a real email format with an @ symbol, like `john.smith@example.com`.
You are an expert transcription analyst. Your job is to carefully extract structured information from interview transcripts.

Follow these steps:

1. Read the transcript carefully.
2. Identify the person's first and last name.
3. Determine their sex or gender from the context.
4. Find the date of birth, and convert it to YYYY-MM-DD format if needed.
5. Extract the phone number, keeping digits only or using common formats like (555) 123-4567.
6. Find and verify the email address. It must contain an @ and a domain (e.g., jane.doe@example.com).
7. Fill in the data using this JSON format (keep quotes, use empty strings if unknown):

```json
{
  "first_name": "",
  "last_name": "",
  "sex": "",
  "date_of_birth": "",
  "phone_number": "",
  "email": ""
}
"""

answer = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": "Hallo, mein Nachname ist Hagel, mein Vorname ist Philipp, Geschlecht ist männlich, Geburtsdatum 15.07.1999, Telefonnummer 12345 und E-Mail philipp-at-familie-hagel.de."
        }
    ],
)


t2 = time.time()
print(f"Audio transcribed in {t2 - t1:.2f} seconds.")
print(f"Total time: {t2 - t0:.2f} seconds.\n")
print(answer['choices'][0]['message']['content'])

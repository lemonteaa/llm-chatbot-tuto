from llama_cpp import Llama, LlamaDiskCache, LlamaGrammar, LlamaTokenizer

import json

from llm_provider import llm

from prompt_templates import *
from function_call import *

web_question1 = "What is the latest news on the war in Ukraine? I'm interested in the impact on financial markets."
offline_question1 = "Hi there! How's your day going?"

init_sys = { "role": "system", "content": system_inst(system_prompt, [web_search_synthesis, code_interpreter], eg) }

message_history = [init_sys]

tokenizer = LlamaTokenizer(llm)
function_token = str(tokenizer.encode("function", add_bos=False)[0])
assistant_token = str(tokenizer.encode("assistant", add_bos=False)[0])

fn_call_schema = LlamaGrammar.from_json_schema(fn_schema)

while True:
    user_q = input("User> ")
    message_history.append({ "role": "user", "content": user_q })
    p1 = chatml_template(message_history, nudge_role = "")
    o1 = llm(prompt=p1, max_tokens=1, logit_bias={ function_token: 50.0, assistant_token: 50.0 })
    role_pick = o1["choices"][0]["text"]
    print("Role picked: " + role_pick)
    if role_pick == "function":
        o2 = llm(prompt=p1 + "function\n", max_tokens=500, grammar=fn_call_schema, stop=["<|im_end|>"])
        print(o2)
        message_history.append({ "role": "function", "content": o2["choices"][0]["text"] })
        fn_obj = json.loads(o2["choices"][0]["text"])
        args = fn_obj["params"]
        fn_result = function_registry[fn_obj["function"]](**args)
        #message_history.append({ "role": "system", "content": "We are in development mode. Functions are not yet implemented, so please simulate the result as if the function is called successfully."})
        message_history.append({ "role": "system", "content": f"{fn_result['further_instruct']}\n\n# Function call result:\n{fn_result['answer']}"})
        p2 = chatml_template(message_history, nudge_role="assistant")
        o2_stream = llm(prompt=p2, max_tokens=500, stop=["<|im_end|>"], stream=True)
        response = ""
        for t in o2_stream:
            u = t["choices"][0]["text"]
            response = response + u
            print(u, end='', flush=True)
            #break
        message_history.append({ "role": "assistant", "content": response })
    elif role_pick == "assistant":
        llm_stream = llm(prompt = p1 + "assistant\n", max_tokens=800, stop=["<|im_end|>"], stream=True)
        response = ""
        for t in llm_stream:
            u = t["choices"][0]["text"]
            response = response + u
            print(u, end='', flush=True)
            #break
        message_history.append({ "role": "assistant", "content": response })
    else:
        raise ValueError("Unknown role:" + role_pick)




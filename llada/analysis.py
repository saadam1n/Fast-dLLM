import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoTokenizer

"""
Analyzes how token predictions change across decoding iterations.
Compares current predictions with previous iterations to track stability.
"""

# Configuration
K_RUNNER_UPS = 3  # Number of runner-up tokens to display

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

# Load configuration
with open("data/ntis/config.txt") as f:
    num_ift_iters = int(next(f))
    response_start = int(next(f))

print(f"Num IFT iters: {num_ift_iters}")
print(f"Response start: {response_start}\n")

prev_tokens = None
prev_logits = None
prev_ntprob = None
prev_atprob = None

for i in range(4 * num_ift_iters): 
    block_idx = i // num_ift_iters
    cbs = response_start + block_idx * 32  # current block start
    cbe = cbs + 32  # current block end

    # Load current iteration data
    cur_tokens = torch.load(f"data/ntis/decoded-{i}.pt").squeeze(0).cpu()
    cur_logits = torch.load(f"data/ntis/logits-{i}.pt").squeeze(0).cpu()
    cur_atprob = F.softmax(cur_logits, dim=-1)

    # Get probabilities for selected tokens
    prob_index = cur_tokens if i == 0 else cur_tokens[cbs:cbe]
    cur_ntprob = torch.gather(cur_atprob, dim=-1, index=prob_index.unsqueeze(-1))

    # Fix size for first tensor
    if i == 0:
        cur_atprob = cur_atprob[cbs:cbe]
        cur_ntprob = cur_ntprob[cbs:cbe]

    print(f"Iter {i}: {tokenizer.batch_decode(cur_tokens, skip_special_tokens=False)}\n")

    # Build table data
    rows = []
    for j in range(cbs, cbe):
        cpos = cur_tokens[j].item()
        
        row = {
            'Pos': j - cbs,
            'Cur Token': tokenizer.decode(cur_tokens[j]),
            'Cur Prob': f"{cur_ntprob[j - cbs].item():.3f}",
        }
        
        # Add current runner-ups
        cur_runner_ups = torch.sort(cur_atprob[j - cbs], descending=True)
        for k in range(0, K_RUNNER_UPS + 1):
            row[f'Cur R{k}'] = tokenizer.decode(cur_runner_ups.indices[k])
            row[f'Cur R{k} Prob'] = f"{cur_runner_ups.values[k].item():.3f}"
        
        # Add previous iteration data if available
        if i > 0:
            ppos = prev_tokens[j].item()
            
            # Determine change type
            if cpos != ppos:
                if cpos != 126336 and ppos != 126336:
                    change_type = "!!!"
                elif ppos == 126336:
                    change_type = "DEC"
                elif cpos == 126336:
                    change_type = "REV"
                else:
                    change_type = "???"
            else:
                change_type = ""
            
            row['Type'] = change_type
            row['Prev Token'] = tokenizer.decode(prev_tokens[j])
            row['Prev Prob'] = f"{prev_ntprob[j - cbs].item():.3f}"
            
            # Add previous runner-ups
            prev_runner_ups = torch.sort(prev_atprob[j - cbs], descending=True)
            for k in range(1, K_RUNNER_UPS + 1):
                row[f'Prev R{k}'] = tokenizer.decode(prev_runner_ups.indices[k])
                row[f'Prev R{k} Prob'] = f"{prev_runner_ups.values[k].item():.3f}"
        
        rows.append(row)

    # Create and display DataFrame
    df = pd.DataFrame(rows)
    
    # Reorder columns for better readability
    if i == 0:
        col_order = ['Pos', 'Cur Token', 'Cur Prob']
        for k in range(0, K_RUNNER_UPS + 1):
            col_order.extend([f'Cur R{k}', f'Cur R{k} Prob'])
    else:
        col_order = ['Type', 'Pos', 'Prev Token', 'Prev Prob', 'Cur Token', 'Cur Prob']
        for k in range(0, K_RUNNER_UPS + 1):
            col_order.extend([f'Cur R{k}', f'Cur R{k} Prob'])
        for k in range(1, K_RUNNER_UPS + 1):
            col_order.extend([f'Prev R{k}', f'Prev R{k} Prob'])
    
    df = df[col_order]
    
    if i == 0:
        print("Current iteration tokens:")
    else:
        print("Changes since last iteration:")
    print(df.to_string(index=False, col_space=16))

    # Store for next iteration
    prev_tokens = cur_tokens
    prev_logits = cur_logits
    prev_ntprob = cur_ntprob
    prev_atprob = cur_atprob

    print("\n" + "="*120 + "\n")

exit()
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer

"""
how often does our original prediction for this token line up with our prediction two steps ago?

e.g. step 1: we are decoding token 1 with token 0 fixed
step 0, we zero clue about token 0

""" 

tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

with open("data/ntis/config.txt") as f:
    num_ift_iters = int(next(f))
    response_start = int(next(f))

print("Num IFT iters")
print(num_ift_iters)
print(response_start)

for i in range(4): # 4 * num_ift_iters
    block_idx = i // num_ift_iters
    # short for current block start
    cbs = response_start + block_idx * 32
    # short for current end
    cbe = cbs + 32

    cur_tokens = torch.load(f"data/ntis/decoded-{i}.pt").squeeze(0).cpu()
    cur_logits = torch.load(f"data/ntis/logits-{i}.pt").squeeze(0).cpu()
    cur_atprob = F.softmax(cur_logits, dim=-1)

    prob_index = cur_tokens if i == 0 else cur_tokens[cbs:cbe]
    cur_ntprob = torch.gather(cur_atprob, dim=-1, index=prob_index.unsqueeze(-1))

    # fix size funkyness with the first tensor
    if i == 0:
        cur_ntprob = cur_ntprob[cbs:cbe]

    #print(cur_tokens)
    #print(cur_ntprob)
    print(f"Iter {i}:\t{tokenizer.batch_decode(cur_tokens, skip_special_tokens=False)}")

    if i > 0:
        print(f"Changes since last iteration:")

        for j in range(cbs, cbe):
            cpos = cur_tokens[j].item()
            ppos = prev_tokens[j].item()

            if cpos != ppos:
                type = "REG\t"

                if (j - cbs) % num_ift_iters == i % num_ift_iters and False:
                    type = "EXP\t"
                elif cpos != 126336 and ppos != 126336:
                    type = "!!!\t"
            else:
                type = "\t"

            # print runner up tokens
            prev_runner_ups = torch.sort(prev_atprob, descending=True)
            cur_runner_ups = torch.sort(cur_atprob, descending=True)

            # claude: ideally we want to be able to print the k-most probably runner ups

            print(f"\t{type} block pos {j - cbs}: {tokenizer.decode(prev_tokens[j])}\t({prev_ntprob[j - cbs].item():.3f})\t-> {tokenizer.decode(cur_tokens[j])}\t({cur_ntprob[j - cbs].item():.3f})"
                  f"\t{tokenizer.decode(cur_runner_ups.indices[j - cbs, 1])}\t({cur_runner_ups.values[j - cbs, 1].item():.3f})"
                  f"\t{tokenizer.decode(prev_runner_ups.indices[j - cbs, 1])}\t({prev_runner_ups.values[j - cbs, 1].item():.3f})"
            )

    prev_tokens = cur_tokens
    prev_logits = cur_logits
    prev_ntprob = cur_ntprob
    prev_atprob = cur_atprob

    print("\n\n\n")

exit()

for delay in range(1, 8):
    break

    total_cnt = 0
    mismatch_cnt = 0

    for i in range(delay, 128):
        prev_logits = torch.load(f"data/step_{i - delay}.pt")

        if i == 1:
            prev_logits = prev_logits[:, -32:, :]

        cur_logits = torch.load(f"data/step_{i}.pt")

        prev_prob = F.softmax(prev_logits, dim=2).cpu()
        cur_prob = F.softmax(cur_logits, dim=2).cpu()

        token_idx = i % 32

        prev_most_likely = torch.max(prev_logits, dim=-1)[1][:, token_idx].item()
        cur_most_likely = torch.max(cur_logits, dim=-1)[1][:, token_idx].item()

        picked = tokenizer.decode(cur_most_likely)

        matched = (prev_most_likely == cur_most_likely)

        if not matched:
            mismatch_cnt += 1
            header_str = "MISMATCH"
        else:
            header_str = "Match!!!"
        total_cnt += 1

        print(f"{'Match!!!' if prev_most_likely == cur_most_likely else 'MISMATCH'}\t{prev_prob[:, token_idx, prev_most_likely].item()}\t{cur_prob[:, token_idx, cur_most_likely].item()}\t{tokenizer.decode(prev_most_likely)}\t{picked}")



        cur_tokens = torch.load(f"data/decoded-{i - 1}.pt").cpu()
        #print(tokenizer.batch_decode(cur_tokens, skip_special_tokens=True))

        #print("\n\n\n")

        if "endoftext" in picked:
            break

    print(f"{delay}: {mismatch_cnt / total_cnt:.2%}")
    
    break

y_axis = []
x_axis = []

for i in range(1, 128):
    prev_logits = torch.load(f"data/step_{i - 1}.pt")

    if i == 1:
        prev_logits = prev_logits[:, -32:, :]

    cur_logits = torch.load(f"data/step_{i}.pt")

    prev_logits = F.softmax(prev_logits, dim=2).cpu()
    cur_logits = F.softmax(cur_logits, dim=2).cpu()

    cur_tokens = torch.load(f"data/decoded-{i}.pt").cpu()

    most_likely = torch.argmax(prev_logits, dim=-1)
    x0_p = torch.squeeze(torch.gather(prev_logits, dim=-1, index=torch.unsqueeze(most_likely, -1)), -1) # b, l


    prev_idx = (i + 31) % 32
    token_idx = i % 32
    cur_token = torch.argmax(cur_logits, dim=-1)[:, token_idx]

    # raw prediction of this token 1 decoding step ago
    original_prob = prev_logits[:, token_idx, cur_token].item()

    # take previous sampled token and modulate it
    conditional_prob = (prev_logits[:, prev_idx].max() * cur_logits[:, token_idx, cur_token]).item()

    y_axis.append(original_prob)
    x_axis.append(conditional_prob)

    print(f"{original_prob}\t{conditional_prob}")

# Create the plot
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.scatter(x_axis, y_axis, alpha=0.6, s=20)
plt.xlabel('Conditional Probability', fontsize=12)
plt.ylabel('Original Probability', fontsize=12)
plt.title('Original vs Conditional Probabilities', fontsize=14)
plt.grid(True, alpha=0.3)

# Add diagonal line for reference
max_val = max(max(x_axis), max(y_axis))
min_val = min(min(x_axis), min(y_axis))
plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='y=x')
plt.legend()

# Save as PNG
plt.tight_layout()
plt.savefig('logits_comparison.png', dpi=150, bbox_inches='tight')
print("\nPlot saved as 'logits_comparison.png'")
plt.close()
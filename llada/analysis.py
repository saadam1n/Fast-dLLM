import torch
import torch.nn.functional as F

from transformers import AutoTokenizer

"""
how often does our original prediction for this token line up with our prediction two steps ago?

e.g. step 1: we are decoding token 1 with token 0 fixed
step 0, we zero clue about token 0

""" 

tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

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
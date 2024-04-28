import torch
import argparse
import time
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig, LlamaForCausalLM
from tree_decoding import TreeDecoding

# TODO: hard code for now. load from checkpoint later
tree_paths = [
        # (root)
        [0], [1], [2], [3], # branches of root (top 4 append)
        # (node 0)
        [0, 0], # branches of node 0 (top 3 append)
        [0, 1],
        [0, 2],
        # (node 1)
        [1, 0], # branches of node 1 (top 2 append)
        [1, 1],
        # (node 2)
        [2, 0], # branches of node 2 (top 2 append)
        [2, 1],
        # (node 3)
        [3, 0], # branches of node 3 (top 1 append)

        # [0, 0, 0],
        # [0, 0, 1],
        # [0, 0, 2],
        # [0, 1, 0],
        # [0, 1, 1],
        # [0, 2, 0],
        # [0, 2, 1],
        # [1, 0, 0],
        # [0, 0, 0, 0],
        # [0, 0, 0, 1],
        # [0, 0, 0, 2],
        # [0, 0, 0, 0, 0],
        # [0, 0, 0, 0, 1]
    ]


class TreeAttention:
    def __init__(self, tree_paths):
        self.tree_paths = tree_paths

        self.tree_attn_mask = None
        self.position_ids = None
        self.init_tree_attention()

        assert self.tree_attn_mask is not None, "tree tree_attn_mask should have been init"
        assert self.position_ids is not None, "tree position_ids should have been init"

    def init_tree_attention(self):
        tree_paths = self.tree_paths

        sorted_tree_paths = sorted(tree_paths, key=lambda x: (len(x), x))
        tree_len = len(sorted_tree_paths) + 1

        depth_count = []
        prev_depth = 0
        for path in sorted_tree_paths:
            depth = len(path)
            if depth != prev_depth:
                depth_count.append(0)
            depth_count[depth - 1] += 1
            prev_depth = depth

        # prepare attention mask for tree attention
        self.tree_attn_mask = torch.eye(tree_len, tree_len)
        self.tree_attn_mask[:, 0] = 1
        root_idx = 0
        # TODO: abstruct as a array base tree?
        for cur_depth_cnt in depth_count:
            # iter over all branches in the current depth
            for branch_idx in range(cur_depth_cnt):
                current_path = sorted_tree_paths[root_idx + branch_idx]
                if len(current_path) == 1:
                    continue

                # find the ancestor(prefix)
                ancestor_idx = []
                for c in range(len(current_path) - 1):
                    prefix_path = current_path[:c + 1]
                    ancestor_idx.append(sorted_tree_paths.index(prefix_path) + 1)

                # mask visitable nodes in the current path
                # +1 to offset root
                self.tree_attn_mask[root_idx + branch_idx + 1, ancestor_idx] = 1
            root_idx += cur_depth_cnt

        # init position ids
        root_idx = 0
        self.position_ids = torch.zeros(tree_len, dtype=torch.long)
        for dep, cur_depth_cnt in enumerate(depth_count):
            self.position_ids[root_idx + 1:root_idx + 1 + cur_depth_cnt] = dep + 1
            root_idx += cur_depth_cnt

    def top_k_generation(self):
        pass

def test_tree_attn_mask():
    _tree_paths = [
        [0],
        [1],
        [0, 0],
        [0, 1],
        [0, 2],
        [1, 0],
        [1, 1],
        [1, 2],
    ]
    tree = TreeAttention(_tree_paths)
    # TODO: 一种快速构建法? 一列下来
    result = torch.tensor([
        [1.,0.,0.,0.,0.,0.,0.,0.,0.],
        [1.,1.,0.,0.,0.,0.,0.,0.,0.],
        [1.,0.,1.,0.,0.,0.,0.,0.,0.],
        [1.,1.,0.,1.,0.,0.,0.,0.,0.],
        [1.,1.,0.,0.,1.,0.,0.,0.,0.],
        [1.,1.,0.,0.,0.,1.,0.,0.,0.],
        [1.,0.,1.,0.,0.,0.,1.,0.,0.],
        [1.,0.,1.,0.,0.,0.,0.,1.,0.],
        [1.,0.,1.,0.,0.,0.,0.,0.,1.],
        ])
    print(tree.tree_attn_mask)
    assert torch.equal(tree.tree_attn_mask, result)

    print(tree.position_ids)

@torch.no_grad()
def stream_inference(model, tokenizer, input_ids, past_key_values, max_gen_len):
    prefill_time = 0
    decode_time = []

    start = time.time()
    outputs = model(
        input_ids=input_ids,
        past_key_values=past_key_values,
        use_cache=True,
    )
    torch.cuda.synchronize()
    end = time.time()
    prefill_time = (end - start)

    past_key_values = outputs.past_key_values
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    generated_ids = [pred_token_idx.item()]
    pos = 0
    for _ in range(max_gen_len - 1):
        start = time.time()
        outputs = model(
            input_ids=pred_token_idx,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        generated_ids.append(pred_token_idx.item())
        generated_text = (
            tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
                spaces_between_special_tokens=False,
            )
            .strip()
            .split(" ")
        )

        now = len(generated_text) - 1
        if now > pos:
            print(" ".join(generated_text[pos:now]), end=" ", flush=True)
            pos = now

        torch.cuda.synchronize()
        end = time.time()
        decode_time.append(end - start)

        if pred_token_idx == tokenizer.eos_token_id:
            break
    print(" ".join(generated_text[pos:]), flush=True)
    return past_key_values, prefill_time, decode_time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="nickypro/tinyllama-110M")
    parser.add_argument("--draft_model", type=str, default=None)
    args = parser.parse_args()

    # TODO: hard code
    args.model = "/home/ring/Documents/workspace/modules/tinyllama-110M"

    MAX_LENGTH = 128
    prompt = "Tell a story begin with: One day, Lily met a Shoggoth."

    batch_size = len(prompt)
    model_name_or_path = args.model

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16, device_map="cuda")

    model = TreeDecoding(model)

    input_ids = tokenizer(prompt, return_tensors="pt", padding=True)["input_ids"]
    input_ids = input_ids.to(model.device)

    past_key_values = None

    past_key_values, prefill_time, decode_time = stream_inference(
        model, tokenizer, input_ids, past_key_values, max_gen_len=MAX_LENGTH
    )

    # number of tokens in context / time for processing context * batch size
    prefill_tokens_per_second = input_ids.shape[1] / prefill_time * batch_size
    # 1 second / median time per token in seconds * batch size
    decode_tokens_per_second = 1 / np.median(decode_time) * batch_size

    device = next(model.parameters()).device
    memory_used = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    memory_pct = memory_used / (torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)) * 100

    print(f" ** Speed (Prefill): {prefill_tokens_per_second:.2f} tokens/second")
    print(f" ** Speed (Decode): {decode_tokens_per_second:.2f} tokens/second")
    print(f" ** Max Memory (VRAM): {memory_used:.2f} GB ({memory_pct:.2f}%)")




if __name__ == "__main__":
    main()







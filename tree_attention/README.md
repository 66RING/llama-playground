# Tree Attention

> 本质基于list的树结构 + 不固定分叉

算法:

1.  

## Tree attention mask construction

> Medusa/Eagle style

- each row represent a branch
    * each value masked with 1 is the **ancestor**

1. eye mask

```
1 0 0 0 0
0 1 0 0 0
0 0 1 0 0
0 0 0 1 0
0 0 0 0 1
```

2. root mask

```
1 0 0 0 0
1 1 0 0 0
1 0 1 0 0
1 0 0 1 0
1 0 0 0 1
```

3. each depth mask construction

```python
# depth_counts = [4, 8, 8, 3, 2], depth=0 have 4 nodes. depth=1 have 8 nodes. etc
# sorted_tree_choices: subtree that need to extend, not fixed size
#   00 list = [0]
#   01 list = [1]
#   02 list = [2]
#   03 list = [3]
#   04 list = [0, 0]
#   05 list = [0, 1]
#   06 list = [0, 2]


# tree_attn_mask = step2 above
start = 0
for i in range(len(depth_counts)):
    # NOTE: j: node idx
    for j in range(depth_counts[i]):
        # NOTE: pick up a subtree(path)
        # e.g. [0, 0, 1] root[0] -> root[0][0] -> root[0][0][1]
        cur_tree_choice = sorted_tree_choices[start + j]
        # retrieve ancestor position
        if len(cur_tree_choice) == 1:
            continue
        ancestor_idx = []
        for c in range(len(cur_tree_choice) - 1):
            # NOTE: index(item) return the idx of item
            # find ancestor idx
            ancestor_idx.append(sorted_tree_choices.index(cur_tree_choice[:c + 1]) + 1)
        tree_attn_mask[j + start + 1, ancestor_idx] = 1
    # NOTE: offset to next depth
    start += depth_counts[i]
```

1. **position id construction**
    - pos = dep + 1
    - `[0, 1, 1, 1, 1, 2, 2, 2, 2 ....]`

- TODO: 如何应用pos id


6. 结果取回







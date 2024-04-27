import torch
# TODO: hard code for now. load from checkpoint later
# relative path
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
        self.init_tree_attention()
        assert self.tree_attn_mask is not None, "tree tree_attn_mask should have been init"

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

        # TODO: position id

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

def main():
    # tree = TreeAttention(tree_paths)
    # print(tree.tree_attn_mask)

    test_tree_attn_mask()



if __name__ == "__main__":
    main()







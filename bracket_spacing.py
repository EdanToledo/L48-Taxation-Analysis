import numpy as np 

top_bracket_cutoff = 89.5
n_brackets = 3

linear = np.round(np.linspace(
                0, top_bracket_cutoff, n_brackets
            ),2)


b0_max = top_bracket_cutoff / (2 ** (n_brackets - 2))
log = np.round(np.concatenate(
                [
                    [0],
                    2
                    ** np.linspace(
                        np.log2(b0_max),
                        np.log2(top_bracket_cutoff),
                        n_brackets - 1,
                    ),
                ]
            ),2)


print("Linear",linear)
print("Log",log)
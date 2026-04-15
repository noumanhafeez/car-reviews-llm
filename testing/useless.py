import numpy as np

def tree_to_json(tree):
    """
    Convert decision tree into JSON-safe nested dictionary.
    Handles numpy types + recursion safely.
    """

    if not isinstance(tree, dict):
        return convert_value(tree)

    result = {}

    for feature, branches in tree.items():

        feature = convert_value(feature)
        result[feature] = {}

        for value, subtree in branches.items():

            value = convert_value(value)

            result[feature][value] = tree_to_json(subtree)

    return result


def convert_value(obj):
    """
    Convert numpy / python objects to JSON-safe types
    """

    # numpy integers → int
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)

    # numpy floats → float
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)

    # everything else → string-safe
    if obj is None:
        return None

    return obj if isinstance(obj, (str, int, float, bool)) else str(obj)
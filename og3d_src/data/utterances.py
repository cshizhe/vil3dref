
def is_explicitly_view_dependent(tokens):
    """
    :return: a boolean mask
    """
    target_words = {'front', 'behind', 'back', 'right', 'left', 'facing', 'leftmost', 'rightmost',
                    'looking', 'across'}
    for token in tokens:
        if token in target_words:
            return True
    return False
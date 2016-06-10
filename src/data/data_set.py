# -*- coding: utf-8 -*-


class DataSet(object):
    """
    Representing train, valid or test sets

    Parameters
    ----------
    data : list
    oneHot : bool
        If this flag is set, then all labels which are not `targetDigit` will
        be transformed to False and `targetDigit` bill be transformed to True.
    targetDigit : string
        Label of the dataset, e.g. '7'.

    Attributes
    ----------
    input : list
    label : list
        A labels for the data given in `input`.
    one_hot : bool
    target_digit : string
    """

    def __init__(self, data, one_hot=True, target_digit='7'):

        # The label of the digits is always the first fields
        self.input = 1.0 * data[:, 1:]/255
        self.label = data[:, 0]
        self.one_hot = one_hot
        self.target_digit = target_digit

        # Transform all labels which is not the target_digit to False,
        # The label of target_digit will be True,
        if one_hot:
            self.label = list(map(lambda a: 1 if str(a) == target_digit else 0,
                                  self.label))

    def __iter__(self):
        return self.input.__iter__()

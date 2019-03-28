class setter_property(object):
    '''
    Note: setter_property incorrectly triggers method-hidden in pylint.

    '''
    def __init__(self, fn, doc=None):
        self.fn = fn
        self.__name__ = fn.__name__
        self.__doc__ = doc if doc is not None else fn.__doc__
    def __get__(self, obj, cls):
        if self.__name__ in obj.__dict__:
            return obj.__dict__[self.__name__]
        else:
            return None
    def __set__(self, obj, value):
        return self.fn(obj, value)
    def __delete__(self, obj):
        return self.fn(obj, None)

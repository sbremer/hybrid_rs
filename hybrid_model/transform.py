"""
Transformations intended to be used on ratings from 1-5 to scale down to range 0.0 - 1.0
"""

class Transformation:
    @staticmethod
    def transform(x):
        raise NotImplementedError

    @staticmethod
    def invtransform(x):
        raise NotImplementedError


class TransformationNone(Transformation):
    @staticmethod
    def transform(x):
        return x

    @staticmethod
    def invtransform(x):
        return x


class TransformationLinear(Transformation):
    @staticmethod
    def transform(x):
        return (x - 0.5) / 5

    @staticmethod
    def invtransform(x):
        return x * 5 + 0.5


class TransformationLinearShift(Transformation):
    @staticmethod
    def transform(x):
        return (x - 3) / 5

    @staticmethod
    def invtransform(x):
        return x * 5 + 3


class TransformationQuad(Transformation):
    @staticmethod
    def transform(x):
        return x ** 2 / 26

    @staticmethod
    def invtransform(x):
        return (x * 26) ** 0.5

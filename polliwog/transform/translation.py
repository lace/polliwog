import numpy as np

def translation(points, translation_factor=None):
    '''
    Args:

        - points:
            an np.ndarray of 1 x N vectors
        - translation_factor (optional, default is the mean vector of 'points'):
            a 1 x N np.ndarray

    Returns a tuple (image, translation_factor):

        - image:
                the image of 'points' under the translation, with its order preserved.
        - translation_factor:
                the translation_factor. this is useful when translation_factor
                is initially None, and the caller wants to apply the inverse
    '''
    if translation_factor is None:
        translation_factor = -np.mean(points, axis=0)

    image = points + translation_factor

    return (image, translation_factor)

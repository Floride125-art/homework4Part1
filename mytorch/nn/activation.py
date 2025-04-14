import numpy as np


class Softmax:
    """
    A generic Softmax activation function that can be used for any dimension.
    """
    def __init__(self, dim=-1):
        """
        :param dim: Dimension along which to compute softmax (default: -1, last dimension)
        DO NOT MODIFY
        """
        self.dim = dim

    def forward(self, Z):
        """
        :param Z: Data Z (*) to apply activation function to input Z.
        :return: Output returns the computed output A (*).
        """
        if self.dim > len(Z.shape) or self.dim < -len(Z.shape):
            raise ValueError("Dimension to apply softmax to is greater than the number of dimensions in Z")
        
        # TODO: Implement forward pass
        # Compute the softmax in a numerically stable way
        # Apply it to the dimension specified by the `dim` parameter
        Z_shifted = Z - np.max(Z, axis=self.dim, keepdims=True)
        exp_Z = np.exp(Z_shifted)
        self.A = exp_Z / np.sum(exp_Z, axis=self.dim, keepdims=True)
        return self.A

    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt output
        :return: Gradient of loss with respect to activation input
        """
        # TODO: Implement backward pass
        
        # Get the shape of the input
        shape = self.A.shape
        # Find the dimension along which softmax was applied
        C = shape[self.dim]
        A_original = self.A.copy()
           
        # Reshape input to 2D
        if len(shape) > 2:
            A_moved = np.moveaxis(self.A, self.dim, -1)
            dLdA_moved = np.moveaxis(dLdA, self.dim, -1)
            moved_shape = A_moved.shape
            self.A = A_moved.reshape(-1, C)
            dLdA = dLdA_moved.reshape(-1, C)

        dot_product = np.sum(self.A * dLdA, axis=1, keepdims=True)
        dLdZ = self.A * (dLdA - dot_product)

        # Reshape back to original dimensions if necessary
        if len(shape) > 2:
            # Restore shapes to original
            dLdZ_moved = dLdZ.reshape(moved_shape)
            dLdZ = np.moveaxis(dLdZ_moved, -1, self.dim)
            self.A = A_original

        return dLdZ
 

    
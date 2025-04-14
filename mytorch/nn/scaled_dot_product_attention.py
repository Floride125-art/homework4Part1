import numpy as np
from .activation import Softmax

class ScaledDotProductAttention:
    """
    Scaled Dot Product Attention
    """ 
    def __init__(self):
        '''
        Initialize the ScaledDotProductAttention class.
        '''
        # Initialize your softmax layer
        # What dimension should you pass to the softmax constructor?
        self.eps = 1e10 # DO NOT MODIFY
        self.softmax = Softmax(dim=-1)
        
    
    def forward(self, Q, K, V, mask=None):
        """
        :param Q: Query matrix of shape (N, ..., H, L, E) where L is target sequence length
        :param K: Key matrix of shape (N, ..., H, S, E) where S is source sequence length
        :param V: Value matrix of shape (N, ..., H, S, Ev) where Ev is value dimension
        :param mask: Boolean mask matrix of shape (N, ..., H, L, S) or broadcastable shape where 1/True indicates a position to ignore
        :return: Output matrix of shape (N, ..., H, L, Ev)
        """
        # TODO: Implement forward pass
        
        # Calculate attention scores: (N, ..., H, L, S)
        # (N, ..., H, L, E) @ (N, ..., H, E, S) -> (N, ..., H, L, S)
        self.Q = Q
        self.K = K
        self.V = V
        d_k = Q.shape[-1]
        K_transposed = np.swapaxes(K, -1, -2)
        dot_product = np.matmul(Q, K_transposed)
        scaled_dot_product = dot_product / np.sqrt(d_k)
     
        # Apply mask before softmax if provided
        # If mask is not None, add -self.eps to the attention scores for positions to ignore
        if mask is not None:
            scaled_dot_product = np.where(mask, -self.eps, scaled_dot_product)

        # Compute attention scores: Apply softmax along S dimension (N, ..., H, L, S)
        self.attention_scores =self.softmax.forward(scaled_dot_product)

        # Calculate output: (N, ..., H, L, Ev)
        # (N, ..., H, L, S) @ (N, ..., H, S, Ev) -> (N, ..., H, L, Ev) 
        output = np.matmul(self.attention_scores, V)

        # Return output
        return output
    
    def backward(self, d_output):
        """
        :param d_output: Gradient of loss wrt output of shape (N, ..., H, L, Ev)
        :return: Gradient of loss wrt input Q, K, V
        """
        # TODO: Implement backward pass

        # Calculate gradients for V: (N, ..., H, S, Ev)
        # (N, ..., H, L, S) @ (N, ..., H, S, Ev) -> (N, ..., H, L, Ev) 
        # Use the transpose of stored softmax output to swap last two dimensions
        attention_scores_transposed = np.swapaxes(self.attention_scores, -1, -2)   
        d_V = np.matmul(attention_scores_transposed, d_output)
        
        # Calculate gradients for attention scores
        # (N, ..., H, L, Ev) @ (N, ..., H, Ev, S) -> (N, ..., H, L, S)
        V_transposed = np.swapaxes(self.V, -1, -2)
        d_attention_scores = np.matmul(d_output, V_transposed)
        d_scaled_dot_product = self.softmax.backward(d_attention_scores)
        
        # Scale gradients by sqrt(d_k)
        d_k = self.Q.shape[-1]
        
        # Calculate gradients for Q and K
        # (N, ..., H, L, S) @ (N, ..., H, S, E) -> (N, ..., H, L, E)   
        d_Q = np.matmul(d_scaled_dot_product, self.K) / np.sqrt(d_k)
        # (N, ..., H, L, S) @ (N, ..., H, L, E) -> (N, ..., H, S, E)
        d_scaled_dot_product_transposed = np.swapaxes(d_scaled_dot_product, -1, -2)
        d_K = np.matmul(d_scaled_dot_product_transposed, self.Q) / np.sqrt(d_k)
        
        # Return gradients for Q, K, V
        return d_Q, d_K, d_V


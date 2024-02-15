import numpy as np


def correlation(x, y, should_flip=True) -> np.ndarray:
    """
    Gets the correlation between two matrices, allowing for flipping between similarity/dissimilarity. Expects both
    matrices to be numpy array containing real number values from 0 to 1, and also to have the same shape.
    Args:
        x: First matrix to compare.
        y: Second matrix to compare.
        should_flip: If True, the x matrix is inverted by subtracting each value from 1 for each cell in x. This means
                     that one of the matrices is a similarity matrix and the other is a distance matrix.

    Returns:
        Correlation between the two matrices.
    """
    if should_flip:
        one_matrix = np.array(list([1.0 for _ in range(np.shape(x)[0])] for _ in range(np.shape(x)[0])))
        x = np.subtract(one_matrix, x)
    return np.corrcoef(x.flatten(), y.flatten())


def get_output_matrix_from_embedding(embedding, matrix_len):
    """
    Args:
        embedding: Embeddings to calculate distances from.
        matrix_len: Row and col length of the square output matrix.

    Returns:
        Distance (dissimilarity) matrix based on embeddings, which is always square.
    """
    output_matrix = np.array([[0.0 for _ in range(matrix_len)] for _ in range(matrix_len)])
    for row_num, row in enumerate(embedding):
        for row_num_2 in range(len(embedding)):
            dist = np.linalg.norm(embedding[row_num_2] - row)
            output_matrix[row_num][row_num_2] = dist
    return output_matrix

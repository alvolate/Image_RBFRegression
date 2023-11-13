import numpy as np

class RBFRegression():
    def __init__(self, centers, widths):
        """ This class represents a radial basis function regression model.

        Args:
        - centers (ndarray (Shape: (K, 2))): A Kx2 matrix corresponding to the 
                                           centers of the 2D radial basis functions.
                                           NOTE: This determines the number of parameters.
        - widths (ndarray (Shape: (K, 1))): A K-column vector corresponding to the
                                            widths of the radial basis functions.
                                            NOTE: We are assuming the function is isotropic.
        """
        assert centers.shape[0] == widths.shape[0], f"The number of centers and widths must match. (Centers: {centers.shape[0]}, Widths: {widths.shape[0]})"
        assert centers.shape[1] == 2, f"Each center should have two components. (Centers: {centers.shape[1]})"
        assert widths.shape[1] == 1, f"Each width should have one component. (Widths: {widths.shape[1]})"
        self.centers = centers
        self.widths = widths
        self.K = centers.shape[0]

        # Remember that we have K weights and 1 bias.
        self.parameters = np.ones((self.K + 1, 1), dtype=np.single)

    def _rbf_2d(self, X, rbf_i):
        """ This private method computes the output of the i'th 2D radial basis function given the inputs.
        Args:
        - X (ndarray (Shape: (N, 2))): A Nx2 matrix consisting N 2D input data.
        - rbf_i (int): The i'th radial basis function. NOTE: 0 <= rbf_i < K

        Output:
        - ndarray (Shape: (N, 1)): A N-column vector consisting N scalar output data.
        """
        assert 0 <= rbf_i < self.K

        # Retrieve the center and the width of the radial basis function
        rbf_center = self.centers[[rbf_i]]
        rbf_width = self.widths[[rbf_i]]

        # Compute the squared difference between X and the RBF center
        X_diff = np.sum(np.square(X - rbf_center), axis=1, keepdims=True)

        # Compute the RBF output
        z = np.exp(-X_diff / (2 * np.square(rbf_width)))
        return z

    def predict(self, X):
        """ This method predicts the output of the given input data using the model parameters.
        Given a single scalar input x,
        f(x) = w_0 + w_1 * b_0(x) + w_2 * b_1(x) + ... + w_K * b_K(x), 
        where b_i is the i'th radial basis function.

        Args:
        - X (ndarray (Shape: (N, 2))): A Nx2 matrix consisting N 2D input data.

        Output:
        - ndarray (shape: (N, 1)): A N-column vector consisting N scalar output data.

        ASIDE: Do you see a way to do this without any loop at all?
        """
        assert X.shape[1] == 2, f"Each input should contain two components. Got: {X.shape[1]}"


        X_radial = np.ones((X.shape[0],1)) #first entry is the basis -> np.ones
        for i in range (self.K):
            X_radial=np.append(X_radial,self._rbf_2d(X,i),axis=1) #get i+1 basis functionx
        return X_radial@self.parameters
        

    
    def fit_with_l2_regularization(self, train_X, train_Y, l2_coef):
        """ This method fits the model parameters, given the training inputs and outputs.


        Args:
        - train_X (ndarray (shape: (N, 2))): A Nx2 matrix consisting N 2D training inputs.
        - train_Y (ndarray (shape: (N, 1))): A N-column vector consisting N scalar training outputs.
        - l2_coef (float): The lambda term that decides how much regularization we want.
        """
        assert train_X.shape[0] == train_Y.shape[0], f"Number of inputs and outputs are different. (train_X: {train_X.shape[0]}, train_Y: {train_Y.shape[0]})"
        assert train_X.shape[1] == 2, f"Each input should contain two components. Got: {train_X.shape[1]}"
        assert train_Y.shape[1] == 1, f"Each output should contain 1 component. Got: {train_Y.shape[1]}"

        X = np.ones((train_X.shape[0],1)) #first entry is the basis -> np.ones
        for i in range (self.K):
            X=np.append(X,self._rbf_2d(train_X,i),axis=1) #get i+1 basis function

        X_t=np.transpose(X)# X^{T}
        I = np.identity(X_t.shape[0])# I matrix
        inverse=np.linalg.inv((X_t@ X) +(l2_coef*I))#inverse matrix


        self.parameters = inverse @ X_t @ train_Y

        assert self.parameters.shape == (self.K + 1, 1)


if __name__ == "__main__":
    # NOTE: This is just a quick check 
    centers = np.tile(np.expand_dims(np.arange(2), axis=1), reps=(1, 2))
    widths = np.ones((2, 1))
    model = RBFRegression(centers, widths)

    train_X = np.tile(np.expand_dims(np.arange(3), 1), reps=(1, 2))
    train_Y = np.array([[4.10363832], [4.73575888], [2.1402696]])

    optimal_parameters = np.array([[1], [2], [3]])
    model.fit_with_l2_regularization(train_X, train_Y, l2_coef=0)
    print("Correct optimal weights: {}".format(np.allclose(model.parameters, optimal_parameters)))

    pred_Y = model.predict(train_X)
    print("Correct predictions: {}".format(np.allclose(pred_Y, train_Y)))

    # Regularization pulls the weights closer to 0.
    optimal_parameters = np.array([[1.78497818], [1.32962937], [1.66446937]])
    model.fit_with_l2_regularization(train_X, train_Y, l2_coef=0.5)
    print("Correct optimal weights: {}".format(np.allclose(model.parameters, optimal_parameters)))

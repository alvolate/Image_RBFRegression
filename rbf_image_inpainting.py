
import matplotlib.image as mpimg
import numpy as np
import os

from rbf_regression import RBFRegression

def RBF_image_inpainting(image_name, fill_rgb, spacing, width, l2_coef, patch_size=25, tolerance=0.1):
    """ This performs image inpainting using RBF regression.
    Args:
    - image_name (str): The name of the image to be inpainted.
    - fill_rgb (list of 3 ints): The colour to be filled in RGB. NOTE: len(fill_rgb) == 3.
    - spacing (int): The spacing between each radial basis function center. NOTE: 1 <= spacing <= 9.
    - width (float): The width of the radial basis functions. NOTE: 1 <= width <= 2 * spacing.
    - l2_coef (float): The lambda term controlling the amount of regularization.
    - patch_size (int): The size of each image patch. NOTE: 1 <= patch_size
    - tolerance (float): The tolerance of treating the specified colour and fill in colour as similar.
    """
    assert os.path.isfile(image_name), f"The image file {image_name} does not exist."
    assert len(fill_rgb) == 3 and all([0 <= element <= 255 for element in fill_rgb]), f"fill_rgb must be a list of 3 integers. Got: {fill_rgb}"
    assert 1 <= spacing <= 9, f"spacing must be between 1 and 9. Got: {spacing}"
    assert 1 <= width <= 2 * spacing, f"width must be between 1 and {2 * spacing}. Got: {width}"
    assert 1 <= patch_size, f"patch_size must be at least 1. Got: {patch_size}"
    
    CENTER_SPACING = spacing
    PATCH_SIZE = patch_size
    TOL = tolerance
    
    fill_rgb = fill_rgb.astype(np.single) / 255

    # Read Images 
    im = mpimg.imread(image_name)
    im = im.astype(np.single) / 255
    im_rec = im
    
    # Iterate through image patches
    # i corresponds to left-to-right
    # j corresponds to up-to-down
    for i in range(CENTER_SPACING+1,im.shape[1]-(PATCH_SIZE+CENTER_SPACING)+1,PATCH_SIZE):
        for j in range(CENTER_SPACING+1,im.shape[0]-(PATCH_SIZE+CENTER_SPACING)+1,PATCH_SIZE):
            # Splat RBFs over this image patch
            [XX,YY]=np.meshgrid(list(range(i-CENTER_SPACING,i+PATCH_SIZE+CENTER_SPACING+1,CENTER_SPACING)),
                                list(range(j-CENTER_SPACING,j+PATCH_SIZE+CENTER_SPACING+1,CENTER_SPACING)))

            # Construct the centers and the widths of RBFs
            # NOTE: We assume all centers are spreadout evenly and all widths to be the same
            centers = np.array((XX.flatten(), YY.flatten()), dtype=np.single).T
            num_centers = centers.shape[0]
            widths = np.ones(shape=(num_centers, 1), dtype=np.single) * width

            # Construct one model for each colour channel
            # Training is done below.
            red_model = RBFRegression(centers=centers, widths=widths)
            green_model = RBFRegression(centers=centers, widths=widths)
            blue_model = RBFRegression(centers=centers, widths=widths)
            
            # Grid of pixel coordinates helps to find the coordinates of pixels that we will fill in
            [XX,YY] = np.meshgrid(list(range(i,i+PATCH_SIZE+1)),list(range(j,j+PATCH_SIZE+1)))
            Pfill = np.array([XX.reshape(-1,order='F'), YY.reshape(-1,order='F')])
            patch_fill=im[j-1:j+PATCH_SIZE, i-1:i+PATCH_SIZE]
            
            # Uses squared distance to find indcies to be filled
            ref = patch_fill - fill_rgb
            ref = np.power(ref,2)
            ref = np.sum(ref,2)
            index_fill = np.argwhere(ref<=TOL)
            idx_fill = np.sort(index_fill[:,1]*ref.shape[0]+index_fill[:,0])
            
            # Grid of pixel coordinates helps to find the coordinates of pixels that we will use to train the RBF models
            [XX,YY] = np.meshgrid(list(range(i-CENTER_SPACING,i+PATCH_SIZE+CENTER_SPACING+1)),
                                  list(range(j-CENTER_SPACING,j+PATCH_SIZE+CENTER_SPACING+1)))
            P = np.array([XX.reshape(-1,order='F'), YY.reshape(-1,order='F')])
            
            patch=im[j-CENTER_SPACING-1:j+PATCH_SIZE+CENTER_SPACING, i-CENTER_SPACING-1:i+PATCH_SIZE+CENTER_SPACING]

            # Uses squared distance to find training data indicies
            ref = patch - fill_rgb
            ref = np.power(ref,2)
            ref = np.sum(ref,2)
            index_data = np.argwhere(ref>TOL)
            idx_data = np.sort(index_data[:,1]*ref.shape[0]+index_data[:,0])
            
            # If there are pixels that need to be filled, then we try to train the models and fill.
            # Otherwise, we use the original patch
            if (idx_fill.size>0):
                print('Reconstructing patch at selected color')
                if(idx_data.size <= num_centers):
                    print('Not enough pixels to estimate RBF model! copying patch\n')
                    patch_rec = patch_fill
                else:
                    # Valid locations for sampling pixels
                    PP = P[:,idx_data]

                    # Reconstruct each colour layer using a separate RBF model
                    # Red channel
                    patch_R=patch[:,:,0]
                    z_R = patch_R.reshape(patch_R.size,1, order='F')
                    z_R = z_R[idx_data]
                    red_model.fit_with_l2_regularization(PP.T, z_R, l2_coef)
                    
                    #Green channel
                    patch_G=patch[:,:,1]
                    z_G = patch_G.reshape(patch_G.size,1, order='F')
                    z_G = z_G[idx_data]
                    green_model.fit_with_l2_regularization(PP.T, z_G, l2_coef)
                    
                    #Blue channel
                    patch_B=patch[:,:,2]
                    z_B = patch_B.reshape(patch_B.size,1, order='F')
                    z_B = z_B[idx_data]
                    blue_model.fit_with_l2_regularization(PP.T, z_B, l2_coef)
                    
                    # Reconstruct pixel values at fill-in locations
                    PP = Pfill[:,idx_fill].T
                    fill_R = red_model.predict(PP)
                    fill_G = green_model.predict(PP)
                    fill_B = blue_model.predict(PP)
                    
                    # Assemble reconstructed patch
                    patch_rec=patch_fill
                    pr_R=patch_rec[:,:,0]
                    pr_G=patch_rec[:,:,1]
                    pr_B=patch_rec[:,:,2]
                    pr_R[index_fill[:,0],index_fill[:,1]]=np.squeeze(np.asarray(fill_R))
                    pr_G[index_fill[:,0],index_fill[:,1]]=np.squeeze(np.asarray(fill_G))
                    pr_B[index_fill[:,0],index_fill[:,1]]=np.squeeze(np.asarray(fill_B))
                    patch_rec[:,:,0]=pr_R
                    patch_rec[:,:,1]=pr_G
                    patch_rec[:,:,2]=pr_B
            else:
                print('Copying patch at %d--%d\n'%(i,j))
                patch_rec=patch_fill
            im_rec[j-1:j+PATCH_SIZE,i-1:i+PATCH_SIZE]=patch_rec
        
    return np.round(im_rec,4)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    image_name = 'Amazing_jellyfish_corrupted_by_text.tif'

    # The color need to be filled, you could modify it to suit your own test cases
    # The array specifies the RGB, integers ranging from 0 to 255.
    fill_rgb = np.array([255, 0, 0]) #Only changes color red, other rgb value will not change the corrupted 
    spacing = 6
    width = 7
    l2_coef = 1.5
    
    im_rec = RBF_image_inpainting(image_name, fill_rgb, spacing, width, l2_coef)

    plt.imshow(im_rec)
    plt.show()

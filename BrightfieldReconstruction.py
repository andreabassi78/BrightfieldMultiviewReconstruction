"""

Example code for Brightfield Multiview Reconstruction as described in Calisesi et al. ----- (2020)

The code reconstructs a single section (y) of a sample acquired in brightfield mode from multiple angles.

The data should be ordered (resliced after multi-view acquisition) in the following order:
                                [angles, z, x]
                                
Please read Calisesi et al. for details on acquisition and on the coordinates definition.  

In order to reconstruct a full 3D sample, the code should be run section by section 
or could be modified to take into account 3D multiview stacks. Please contact
the corresponding author if you need help in performing this step. 

"""

from tifffile import imread
import matplotlib.pyplot as plt
from skimage.transform import rotate
import numpy as np
from skimage.restoration import richardson_lucy as deconv
#from skimage.restoration import wiener as deconv
from tqdm import tqdm
from numpy.fft import fft2


def show_some_views(sample, num_views_to_show):
    """ 
    Shows some of the data to be processed. The sample data consist of a stack of xz
    images at multiple angles. These can be obtained by reslicing brightfield stacks 
    acquired at different view angles
    """     
    for angle_idx in range(0,nangles, int(nangles/num_views_to_show)):
        
        plt.gray()
        plt.imshow(sample[angle_idx,:,:])
        plt.colorbar()
        title = ('Original section, view {:2.1f} deg'.format(angle_idx*rotation_angle))
        plt.title(title)
        plt.show()
    

def find_z(sample, cx , distance = 10, half_width = 120, plot_all = False):
    """
    Finds the z coordinate of the rotation axis.
    Parameters:
        sample :    data (ordered as angle, z, x)
        cx:         x coordinate of the rotation axis
        distance:   range of pixels to be included in the search of the rotation axis, 
                    around the image center
        half_width: semize of the reconstructed rectangle 
        plot_all :  if True shows reconstructed image during the center search
    Return:
        best_cz:    the found z coordinate 
    """ 
    contrasts = []
    centers = [] 
    
    czs = range(int(nz/2-distance), int(nz/2+distance)) 
    area_width = int(half_width/2)
    
    for cz in tqdm(czs):
        
        # reconstruct the image at different possible z cohordinates of the center of rotation 
        reconstructed = reconstruct(sample, cx, cz, half_width = half_width, deconvolve = False) 
        
        # calculate the contrast only in the central part of the sample, to avoid border artifacts
        contrast_area = reconstructed[half_width-area_width: half_width+area_width,
                                      half_width-area_width: half_width+area_width]
        
        # calculate the 2D fft of the reconstructed image, excluding the CW component
        contrast_area_fft = np.abs((fft2(contrast_area)))[1:,1:]  
        # the contrast is given by the standard deviation of the fft
        # this is an estimate of the bandwidth of the reconstructed image 
        contrast = np.std(contrast_area_fft)
        
        if plot_all:
            plt.gray()
            plt.imshow(contrast_area)
            plt.colorbar()
            
            title = ('Reconstructed area at cz = {:2.0f} \n'.format(cz) +
                     'Contrast = {:2.3f} \n'.format(contrast) )
            plt.title(title)
            plt.show()
        
        centers.append(cz)
        contrasts.append(contrast)
    
    idx = contrasts.index(max(contrasts))
    best_cz = centers[idx]
    
    print ('Best contrast at cz = ',best_cz)
    plt.plot(centers,contrasts)
    plt.title('Contrast')
    plt.xlabel('cz: axial position of the rotation axis (px)')
    
    return best_cz


def reconstruct(sample, Cx, Cz, half_width = 140, deconvolve = False):
    """
    Reconstruct the multiview data.
    Parameters:
        sample :    data (ordered as angle, z, x)
        cx:         x coordinate of the rotation axis
        cz:         z coordinate of the rotation axis
        half_width: semize of the reconstructed rectangle 
        deconvolve: if True apply xz devonvolution on each view before reconstruction
    Return:
        reconstructed:     reconstructed section 
    """ 
    sample_selection = sample[:,Cz-half_width:Cz+half_width,Cx-half_width:Cx+half_width]
    
    sum_im = np.zeros((2*half_width,2*half_width))
    
    for angle_index, angle in enumerate(np.arange(0,360, rotation_angle)):
        
        if deconvolve:
            view = deconv(sample_selection[angle_index,:,:],
                          psf,
                          iterations=20)
        else:
            view = sample_selection[angle_index,:,:]    
            
        rotated = rotate(view,
                         angle,
                         preserve_range=True,
                         mode='reflect')
        sum_im += rotated
    reconstructed = sum_im/nangles
    
    return(reconstructed)


"""
Example code for reconstruction of a zebrafish embryo (3.5dpf) section, 
acquired with a 10X , NA 0.3 multiview brightfield microscope with angular step of 10°
shown in Calisesi et al., Supplementary Figure 2. Pixelsize is 1.625um.

Please note that the resliced data have been previously interpolated to provide isotropic 
sampling in x and z and zero padded to obtain a squared image. 
This step could be avoided to significantly increase reconstruction speed.

Along the x direction the data have been centered following the method described in Calisesi et al. (Fig 3).  

The region reconstructed around the sample where the views only partially overlap is left on porpuse.

"""

psf_path = 'psf.tif'
psf = imread(psf_path).astype('float')
psf = psf/np.sum(psf)

path = 'zebrafish.tif'

sample = imread(path).astype('float')
sample = sample/np.amax(sample)

nangles,nz,nx  = sample.shape
rotation_angle = 360/(nangles)  

#show_some_views(sample, num_views_to_show=4)


Cx = int(nx/2) # the x component of the rotation axis is in the center of sample along x

Cz = find_z(sample, Cx, distance = 15, plot_all=False)

halfwidth =  int(min ((Cz,nz-Cz,nz/2,nx/2))) 
reconstructed = reconstruct(sample,Cx,Cz,halfwidth, deconvolve = False)

plt.figure()
plt.gray()
plt.imshow(reconstructed)
plt.colorbar()
plt.title('Reconstructed section')
plt.show()


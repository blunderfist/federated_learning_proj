import random 
import torch
import torchvision.transforms as T




def augment(image,x):
    
    # for image in images:
    
    '''
    function to add
    center_crop
    gaussian_blurr
    grayscale
    random_affine
    
    
    '''
    

    if x == 1:
        image = T.ColorJitter(brightness=.5, hue=.3)(image)



    elif x == 2:
        image = T.RandomPerspective(distortion_scale=0.6, p=1.0)(image)


    elif x == 3:
        image = T.RandomRotation(degrees=(0, 180))(image)


    elif x == 4:
        image = T.RandomInvert()(image)


    elif x == 5:
        image = T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))(image)
        
    elif x == 6:
        
        image = T.RandomAdjustSharpness(sharpness_factor=2)(image)
        
    elif x == 7:
        image = T.RandomAutocontrast()(image)
        
    elif x == 8:

        
        image = image
    
    
    

    return T.ToTensor()(image)
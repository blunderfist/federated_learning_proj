

# somewhere around line 280 in the s2kew_fl_kfold2.py you'll see this section where the models are loaded
# if you are adding it to a different script just find the same section

# ======================= Model | Loss Function | Optimizer ======================= # 

    if args.model == 'efficientnet':
        
        model = models.efficientnet_b0(pretrained=True)
        model.classifier[1].out_features = 9
        
    elif args.model == 'resnet':
        model = ResNet(args)
    
    elif args.model == 'vgg19':
        model = VGG(args)
        
    elif args.model == 'custom_EN_b0':
        model = custom_EN_b0(args)
        # print(type(model))
##################################################
    # this part needs to be added
##################################################
    c = 0
    for name, param in model.named_parameters():
        if c < 208: # this is where the layers changed, all new layers should be set to True
            param.requires_grad = False
        # print(name, ':', param.requires_grad) # if you want to print and check they are froze uncomment these
        # print("PARAM # ", c)
        c += 1

# the rest of the script continues
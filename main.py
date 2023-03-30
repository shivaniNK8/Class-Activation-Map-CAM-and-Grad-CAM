from CAM import *


def main(train=True):
    data_path = '/Users/shivaninaik/Documents/MSDAE/Computer Vision/Projects/Final Project /data/'
    dataloaders, class_names, dataset_sizes = get_dataloaders(data_path)
    print(class_names)
    # plot_samples(dataloaders['train'], class_names)
    model = CAM_model()
    print(model)
    model.to(device)
    trainable_parameters = []
    for name, p in model.named_parameters():
        if "fc" in name:
            trainable_parameters.append(p)

    loss_fn = torch.nn.CrossEntropyLoss()

    # all parameters are being optimized
    optimizer = torch.optim.SGD(trainable_parameters, lr=0.001, momentum=0.9)

    # decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    if train:
        model = train_model(model, dataloaders, dataset_sizes, loss_fn, optimizer, exp_lr_scheduler, num_epochs=1)
        torch.save(model.state_dict(), 'model_vggcam.pth')
    else:
        # Load weights
        model_dict = torch.load('model_vggcam.pth', map_location=device)
        model.load_state_dict(model_dict)
    model.eval()
    # visualize_model(model, dataloaders, class_names)
    grid_1, grid_2 = get_grid_plots(dataloaders, model, class_names)
    # train false to load already trained model
    # grad_cam = grad_cam_pipeline(dataloaders, class_names, dataset_sizes, train=False)
    # grid_1 = get_grid_plots_comp(dataloaders, model, model_alex, grad_cam, class_names)



def grad_cam_pipeline(dataloaders, class_names, dataset_sizes, train = True):

    # get and change last layer of pretrained vgg19
    vgg19 = torchvision.models.vgg19(pretrained=True)
    # replace last layer with 7 nodes
    vgg19.classifier[6] = nn.Linear(4096, 7)
    vgg19.to(device)
    # set trainable parameters to only last layer
    trainable_parameters = []
    for name, p in vgg19.named_parameters():
        if "classifier.6" in name:
            trainable_parameters.append(p)
    print(trainable_parameters)
    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(trainable_parameters, lr=0.001, momentum=0.9)

    # decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    if train:
        vgg19 = train_model(vgg19, dataloaders, dataset_sizes, loss_fn, optimizer, exp_lr_scheduler, num_epochs=1)
        torch.save(vgg19.state_dict(), 'model_vgg19.pth')
    else:
        vgg19.to(device)
        model_dict = torch.load('model_vgg19.pth')
        vgg19.load_state_dict(model_dict)
    vgg19.eval()
    # make grad cam model from trained VGG19
    grad_cam = Grad_Cam_VGG19(vgg19)
    grad_cam.to(device)
    # set the evaluation mode
    grad_cam.eval()
    return grad_cam


if __name__ == '__main__':
    main(train=False)




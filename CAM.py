"""
CS 5330 - Final Project
Class Activation Map by fine tuning pre-trained networks
Authored by Shivani Naik and Pulkit Saharan
data - 12/2/2022
"""

# Import statements
import os
import cv2
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision
import torchvision.transforms as trans
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets
from mpl_toolkits.axes_grid1 import ImageGrid
from collections import defaultdict

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


class CAM_model(torch.nn.Module):
    def __init__(self):
        super(CAM_model, self).__init__()
        vgg = torchvision.models.vgg16(pretrained=True)
        self.conv = nn.Sequential(
            vgg.features,
            vgg.avgpool
        )
        self.fc = nn.Linear(512, 7)

    def forward(self, x):
        #  forward pass through conv stack and average pool of vgg
        x = self.conv(x)
        x = x.view(512, -1).mean(1).view(1, -1)  # check why
        x = self.fc(x)
        return x


class Grad_Cam_VGG19(nn.Module):
    def __init__(self, vgg19):
        super(Grad_Cam_VGG19, self).__init__()

        # get the pretrained VGG19 network
        self.vgg = vgg19
        # self.vgg.to

        # disect the network to access its last convolutional layer
        self.features_conv = self.vgg.features[:36]

        # get the max pool of the features stem
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        # get the classifier of the vgg19
        # self.classifier = nn.Linear(512,7)
        self.classifier = self.vgg.classifier
        # placeholder for the gradients
        self.gradients = None

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.features_conv(x)

        # register the hook
        h = x.register_hook(self.activations_hook)

        # apply the remaining pooling
        x = self.max_pool(x)
        # x = x.view(512,7*7).mean(1).view(1,-1)
        x = x.view((1, -1))
        x = self.classifier(x)
        return x

    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients

    # method for the activation exctraction
    def get_activations(self, x):
        return self.features_conv(x)


# Function to load data and return data loaders
def get_dataloaders(data_path, batch_size_train=1, batch_size_test=1, ):
    # Load train data
    train_data = torchvision.datasets.ImageFolder(
        root=data_path+'data/train',
        transform=trans.Compose([
            trans.Resize((224, 224)),
            trans.ToTensor(),
            trans.Normalize([0.485, 0.456, 0.406],  # need to calculate mean of RGB
                            [0.229, 0.224, 0.225])
        ])
    )
    # Load test data
    test_data = torchvision.datasets.ImageFolder(
        root=data_path+'data/test',
        transform=trans.Compose([
            trans.Resize(256),
            trans.CenterCrop(224),
            trans.ToTensor(),
            trans.Normalize([0.485, 0.456, 0.406],  # mean of RGB
                            [0.229, 0.224, 0.225])
        ])
    )

    display_test_data = ImageFolderWithPaths(
        root=data_path+'data/test',
        transform=trans.Compose([
            trans.Resize(256),
            trans.CenterCrop(224),
            trans.ToTensor(),
            trans.Normalize([0.485, 0.456, 0.406],  # mean of RGB
                            [0.229, 0.224, 0.225])
        ])
    )

    # Create a train data loader
    dataloaders = {
        'train': Data.DataLoader(
            dataset=train_data,
            shuffle=True,
            batch_size=1
        ),
        'test': Data.DataLoader(
            dataset=test_data,
            shuffle=True,
            batch_size=1
        ),
        'display_test': Data.DataLoader(
            dataset=display_test_data,
            shuffle=True,
            batch_size=1,

        )
    }
    dataset_sizes = {'train': len(train_data), 'test': len(test_data)}
    return dataloaders, train_data.classes,dataset_sizes

# generic function to display predictions for a few images
def visualize_model(model, dataloaders, class_names, num_images=6):
    was_training = model.training  # if true, the model is in training mode otherwise in evaluate mode
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    for step, (images, labels) in enumerate(dataloaders['test']):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = torch.max(outputs, 1)[1]

        for i in range(images.size(0)):
            images_so_far += 1
            ax = plt.subplot(num_images // 2, 2, images_so_far)
            ax.axis('off')
            ax.set_title('predicted: {}'.format(class_names[preds[i]]))
            imshow(images.cpu().data[i])

            if images_so_far == num_images:
                model.train(mode=was_training)
                return

    model.train(mode=was_training)

    plt.show()

def train_model(model, dataloaders, dataset_sizes, loss_fn, optimizer, scheduler, num_epochs=25):
    """
    net: the model to be trained
    loss_fn: loss function
    scheduler: torch.optim.lr_scheduler
    """

    best_model_weights = copy.deepcopy(model.state_dict())
    best_accuracy = 0.0
    log_interval = 100
    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        step_loss = 0.0
        epoch_accuracy = 0.0

        # each epoch has a training and test phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # set to training mode
            else:
                model.eval()  # set to evaluate mode

            step_loss = 0.0
            step_corrects = 0

            for batch_idx, (images, labels) in enumerate(dataloaders[phase]):
                images, labels = images.to(device), labels.to(device)
                # forward pass, compute loss and make predictions
                outputs = model(images)
                preds = torch.max(outputs, 1)[1]
                loss = loss_fn(outputs, labels)

                # backward pass and update weights if in training phase
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # compute step loss and step corrects
                step_loss += loss.item() * images.size(0)  # loss.item() extracts the loss's value
                step_corrects += torch.sum(preds == labels.data)

                if phase == 'train' and batch_idx % log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}'.format(
                        epoch, batch_idx * len(images), len(dataloaders[phase].dataset),
                               100. * batch_idx / len(dataloaders[phase]), loss.item()))

            if phase == 'train':
                scheduler.step()

            epoch_loss = step_loss / dataset_sizes[phase]
            epoch_accuracy = step_corrects.double() / dataset_sizes[phase]
            print('{} Loss: {:.4f} Accuracy: {:.4f}'.format(
                phase, epoch_loss, epoch_accuracy))

            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_accuracy.item())
            else:
                test_loss.append(epoch_loss)
                test_acc.append(epoch_accuracy.item())

            # deep copy the model
            if phase == 'test' and epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                best_model_weights = copy.deepcopy(model.state_dict())
                # best_model_weights = best_model_weights.to(device)

        print()

    print('Best Test Accuracy: {:4f}'.format(best_accuracy))

    print(type(train_loss), type(train_loss[0]), type(train_acc), type(train_acc[0]))

    # draw the loss history and accuracy history
    fig = plt.figure(figsize=(15, 8))
    x = np.arange(num_epochs)
    plt.subplot(221)
    plt.plot(x, train_loss, c='orange', label='train loss')
    plt.plot(x, test_loss, c='purple', label='test loss')
    plt.legend(loc='best')

    plt.subplot(222)
    # train_acc = train_acc.to("cpu")
    # train_acc = train_acc.to("cpu")
    plt.plot(x, train_acc, c='orange', label='train acc')
    plt.plot(x, test_acc, c='purple', label='test acc')
    plt.legend(loc='best')

    plt.show()

    # load best model weights
    model.load_state_dict(best_model_weights)
    return model


def generate_CAM(conv_feature, weights, class_id):
  img_size = (224,224)
  b, n_channel, height, width = conv_feature.shape
  cam_out = []
  for id in class_id:
    #flatten to (n_channels, height*width)
    flat_feature = conv_feature.reshape(n_channel, height * width)
    # multiply by weights of each class to get cam
    cam = np.matmul(weights[id], flat_feature) # 1xn_channel into n_channelx height*width -> 1xheight*width
    cam = cam.reshape(height, width) # make it height x width
    # normalize
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    # convert to 0 to 255 range
    cam_img = np.uint8(255 * cam_img)
    cam_out.append(cv2.resize(cam_img, img_size))
  return cam_out


def get_prediction(model, image, class_names, label, isPrint = True):
  model.eval()
  image = image.to(device)
  scores = model(image)
  probs = F.softmax(scores, dim=1).data.squeeze()
  probs, idx = probs.sort(0, True)

  if isPrint:
    print('true class: ', class_names[label])
    print('predicated class: ', class_names[idx[0].cpu().numpy()])

  return probs.cpu(), idx.cpu()


# Function predicts the class for an image and generates CAM and overlays it onto image
def predict_generate_CAM(model, class_names, image, label, path, isPrint = True):
  # get predicted class id for image using model
  probs, pred_idx = get_prediction(model, image, class_names,label, isPrint=isPrint)

  # Get weights of last layer
  parameters = list(model.fc.parameters())
  fc_weights_np = parameters[0].cpu().data.numpy()

  # get convolutional feature map from model
  feature_maps = model.conv(image).cpu()

  # generate CAM using feature maps and weights
  CAMs = generate_CAM(feature_maps.detach().numpy(), fc_weights_np, pred_idx.numpy())

  # Get cam heatmap and blended image
  cv_image, cam_heatmap, added_img = blend_image_cam(CAMs, path)

  return cv_image, cam_heatmap, added_img

# Function plots first 6 digits from data loader
def plot_samples(data_loader, class_names, show_title=True, fig_title='Figure'):
    examples = enumerate(data_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    # print(example_data.shape)

    fig = plt.figure(fig_title)
    for i in range(6):
        image = example_data[i]
        # print(image.shape)
        image = image.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image * std + mean
        image = np.clip(image, 0, 1)
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(image)
        if show_title:
            plt.title("Label: {}".format(class_names[example_targets[i]]))
        plt.axis("off")
    plt.show()


# denormalize and show an image
def imshow(image, title=None):
    image = image.numpy().transpose((1, 2, 0))
    std, mean = 1, 0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image * std + mean
    image = np.clip(image, 0, 1)
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.show()


def get_prediction_gradcam(model, image, class_names, isPrint=True):
    model.eval()
    # get the most likely prediction of the model
    pred = model(image)  # .argmax(dim=1)
    pred_max_ind = pred.argmax(dim=1)
    # class_names[pred_max_ind], pred_max_ind
    # get the gradient of the output with respect to the parameters of the model
    pred[:, pred_max_ind].backward()

    # pull the gradients out of the model
    gradients = model.get_activations_gradient()

    # pool the gradients across the channels
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    # get the activations of the last convolutional layer
    activations = model.get_activations(image).detach()

    # weight the channels by corresponding gradients
    for i in range(512):
        activations[:, i, :, :] *= pooled_gradients[i]

    # average the channels of the activations
    heatmap = torch.mean(activations, dim=1).squeeze().cpu()

    # relu on top of the heatmap
    heatmap = np.maximum(heatmap, 0)

    # normalize the heatmap
    heatmap /= torch.max(heatmap)
    heatmap = heatmap.numpy()
    heatmap = np.uint8(255 * heatmap)

    return heatmap


def get_grid_plots(dataloaders, model, class_names):
    images = defaultdict(list)
    heatmaps = defaultdict(list)
    blended_images = defaultdict(list)

    for step, (image, label, path) in enumerate(dataloaders['display_test']):
        class_label = class_names[label]

        image, label = image.to(device), label.to(device)
        # hh = get_prediction_gradcam(grad_cam, image, class_names, isPrint=True)
        # cv_image, cam_heatmap, added_img = blend_image_cam(hh, path, is_Multiple=False)
        cv_image, cam_heatmap, added_img = predict_generate_CAM(model, class_names, image, label, path, isPrint = False)
        images[class_label].append(cv_image)
        heatmaps[class_label].append(cam_heatmap)
        blended_images[class_label].append(added_img)
    fig = plt.figure(figsize=(20., 20.))
    grid_1 = ImageGrid(fig, 111,
                       nrows_ncols=(3, 7),  # creates 1x3 grid of axes
                       axes_pad=0.1,  # pad between axes
                       )
    for i in range(7):
        label = class_names[i]
        grid_1[i].imshow(images[label][2])
        grid_1[7 * 1 + i].imshow(heatmaps[label][2])
        grid_1[7 * 2 + i].imshow(blended_images[label][2])

    plt.show()

    fig = plt.figure(figsize=(20., 20.))
    grid_2 = ImageGrid(fig, 111,
                       nrows_ncols=(3, 7),  # creates 1x3 grid of axes
                       axes_pad=0.1,  # pad between axes
                       )
    for i in range(7):
        label = class_names[i]
        grid_2[i].imshow(blended_images[label][2])
        grid_2[7 * 1 + i].imshow(blended_images[label][4])
        grid_2[7 * 2 + i].imshow(blended_images[label][6])

    plt.show()

    return grid_1, grid_2


def get_grid_plots_comp(dataloaders, vgg_cam_model, alex_cam_model, gradcam_model, class_names):
    images = defaultdict(list)
    blended_images_cam = defaultdict(list)
    blended_images_cam_alex = defaultdict(list)
    blended_images_gradcam = defaultdict(list)

    for step, (image, label, path) in enumerate(dataloaders['display_test']):
        class_label = class_names[label]

        image, label = image.to(device), label.to(device)
        hm = get_prediction_gradcam(gradcam_model, image, class_names, isPrint=True)
        cv_image, cam_heatmap, added_img_gradcam = blend_image_cam(hm, path, is_Multiple=False)
        cv_image, cam_heatmap, added_img_cam = predict_generate_CAM(vgg_cam_model, class_names, image, label, path,
                                                                    isPrint=False)
        cv_image, cam_heatmap, added_img_cam_alex = predict_generate_CAM(alex_cam_model, class_names, image, label,
                                                                         path, isPrint=False)

        images[class_label].append(cv_image)
        blended_images_cam[class_label].append(added_img_cam)
        blended_images_cam_alex[class_label].append(added_img_cam_alex)
        blended_images_gradcam[class_label].append(added_img_gradcam)

    fig = plt.figure(figsize=(20., 20.))
    grid_1 = ImageGrid(fig, 111,
                       nrows_ncols=(4, 7),  # creates 1x3 grid of axes
                       axes_pad=0.1,  # pad between axes
                       )
    for i in range(7):
        label = class_names[i]
        grid_1[i].imshow(images[label][2])
        grid_1[7 * 1 + i].imshow(blended_images_cam[label][2])
        grid_1[7 * 2 + i].imshow(blended_images_cam_alex[label][2])
        grid_1[7 * 3 + i].imshow(blended_images_gradcam[label][2])

    return grid_1


# Function overlays CAM on image
def blend_image_cam(CAMs, path, is_Multiple=True):
    if is_Multiple:
        # Get CAM of prediction
        cam = CAMs[0]
    else:
        cam = CAMs
    # read original image
    img_size = (224, 224)
    cv_img = cv2.resize(cv2.cvtColor(cv2.imread(path[0]), cv2.COLOR_BGR2RGB), img_size)

    # Create CAM heatmap of original image size
    img_h, img_w, _ = cv_img.shape
    cam_heatmap = cv2.cvtColor(cv2.applyColorMap(
        cv2.resize(cam, (img_w, img_h)), cv2.COLORMAP_JET),
        cv2.COLOR_BGR2RGB)

    # OVerlay heatmap onto original image
    added_image = cv2.addWeighted(cv_img, 0.5, cam_heatmap, 0.5, 0)
    return cv_img, cam_heatmap, added_image

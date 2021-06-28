import asyncio
import io

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image

import torchvision.transforms as transforms
from torchvision.utils import save_image


class ContentLoss(nn.Module):
    def __init__(self, target, ):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = self.gram_matrix(target_feature).detach()

    def forward(self, input):
        g = self.gram_matrix(input)
        self.loss = F.mse_loss(g, self.target)
        return input

    def gram_matrix(self, input):
        a, b, c, d = input.size()  # a=batch size(=1)
        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL
        G = torch.mm(features, features.t())  # compute the gram product
        return G.div(a * b * c * d)


class Normalization(nn.Module):
    def __init__(self):
        super(Normalization, self).__init__()
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


class StyleTransfer:
    def __init__(self, style_img, content_img, imsize=256, epochs=200, style_weight=100000, content_weight=1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.imsize = imsize
        self.style_img = self.image_loader(style_img)
        self.content_img = self.image_loader(content_img)
        self.input_img = self.content_img.clone()
        # if you want to use white noise instead uncomment the below line:
        # input_img = torch.randn(content_img.data.size(), device=device)
        self.epochs = epochs
        self.style_weight = style_weight
        self.content_weight = content_weight

    def image_loader(self, image_name):
        loader = transforms.Compose([
            transforms.Resize(self.imsize),  # scale imported image
            transforms.CenterCrop(self.imsize),
            transforms.ToTensor()])  # transform it into a torch tensor
        image = Image.open(image_name)
        image = loader(image).unsqueeze(0)
        return image.to(self.device, torch.float)

    def get_input_optimizer(self, input_img):
        # this line to show that input is a parameter that requires a gradient
        optimizer = optim.LBFGS([input_img.requires_grad_()])
        return optimizer

    def run_style_transfer(self):
        """Run the style transfer."""
        print('Building the style transfer model..')

        optimizer = self.get_input_optimizer(self.input_img)

        model = torch.load('model.pth', map_location=self.device).eval()

        for i in range(len(model)):
            if str(model[i]) == 'StyleLoss()':
                f = torch.nn.Sequential(*list(model.children())[:i])
                features = f(self.style_img).detach()
                model[i] = StyleLoss(features)
            elif str(model[i]) == 'ContentLoss()':
                f = torch.nn.Sequential(*list(model.children())[:i])
                features = f(self.content_img).detach()
                model[i] = ContentLoss(features)

        style_losses = [model.style_loss_1, model.style_loss_2, model.style_loss_3, model.style_loss_4,
                        model.style_loss_5]
        content_losses = [model.content_loss_4]

        print('Optimizing..')
        run = [0]
        while run[0] <= self.epochs:
            # await asyncio.sleep(0)

            def closure():
                # correct the values of updated input image
                self.input_img.data.clamp_(0, 1)

                optimizer.zero_grad()
                model(self.input_img)
                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss

                style_score *= self.style_weight
                content_score *= self.content_weight

                loss = style_score + content_score
                loss.backward()

                run[0] += 1
                if run[0] % 50 == 0:
                    print("run {}:".format(run))
                    print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                        style_score.item(), content_score.item()))
                    print()

                return style_score + content_score

            optimizer.step(closure)

        # a last correction...
        self.input_img.data.clamp_(0, 1)

        return self.input_img


# output = run_style_transfer(content_img, style_img, input_img)

# save_image(output, 'test2.jpg')

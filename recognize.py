import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
from PIL import Image


RELEVANT_PLACES = ['construction_site', 'street', 'desert_road', 'field_road', 'industrial_area', 'highway']
MINIMUM_PROBABILITY = 0.01


class Recognizer:
    def __init__(self, classes_white_list, minimum_probability=MINIMUM_PROBABILITY):
        self.classes_white_list = classes_white_list
        self.minimum_probability = minimum_probability

        # th architecture to use
        self.arch = 'resnet18'

        # load the pre-trained weights
        model_file = '%s_places365.pth.tar' % self.arch
        if not os.access(model_file, os.W_OK):
            weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
            os.system('wget ' + weight_url)

        self.model = models.__dict__[self.arch](num_classes=365)
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # load the image transformer
        self.centre_crop = trn.Compose([
                trn.Resize((256,256)),
                trn.CenterCrop(224),
                trn.ToTensor(),
                trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # load the class label
        file_name = 'categories_places365.txt'
        if not os.access(file_name, os.W_OK):
            synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
            os.system('wget ' + synset_url)
        classes = list()
        with open(file_name) as class_file:
            for line in class_file:
                classes.append(line.strip().split(' ')[0][3:])
        self.classes = tuple(classes)

    def recognize_image(self, image_path):
        img = Image.open(image_path)
        input_img = V(self.centre_crop(img).unsqueeze(0))

        # forward pass
        logit = self.model.forward(input_img)
        h_x = F.softmax(logit, 1).data.squeeze()
        probs, idx = h_x.sort(0, True)

        print('{} prediction on {}'.format(self.arch, image_path))
        # output the prediction
        result = {}
        for i in range(0, 5):
            print('{:.3f} -> {}'.format(probs[i], self.classes[idx[i]]))
            result[self.classes[idx[i]]] = probs[i]

        return result

    def is_image_relevant(self, image_path):
        result = self.recognize_image(image_path)
        for key in result:
            if key in self.classes_white_list and result[key] > self.minimum_probability:
                return True

    def get_relevant_images(self, images_folder_path):
        relevant_images = []
        not_relevant_images = []
        for image_name in os.listdir(images_folder_path):
            image_path = os.path.join(images_folder_path, image_name)
            if self.is_image_relevant(image_path):
                relevant_images.append(image_path)
            else:
                not_relevant_images.append(image_path)
        return relevant_images, not_relevant_images


def main():
    recognizer = Recognizer(RELEVANT_PLACES, MINIMUM_PROBABILITY)
    relevant_images, not_relevant_images = recognizer.get_relevant_images('images')
    print('Relevant images:')
    print(relevant_images)
    print('Not relevant images:')
    print(not_relevant_images)


if __name__ == '__main__':
    main()

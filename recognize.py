import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
import cv2
import tempfile
from PIL import Image, UnidentifiedImageError


RELEVANT_PLACES = ['construction_site', 'street', 'desert_road', 'field_road', 'industrial_area', 'highway']
MINIMUM_PROBABILITY = 0.01
MAX_FRAMES_TO_CHECK = 100
JUMP_FRAMES = 10


class Recognizer:
    def __init__(self, classes_white_list,
                 minimum_probability=MINIMUM_PROBABILITY,
                 max_frames_to_check=MAX_FRAMES_TO_CHECK,
                 jump_frames=JUMP_FRAMES):
        self.classes_white_list = classes_white_list
        self.minimum_probability = minimum_probability
        self.max_frames_to_check = max_frames_to_check
        self.jump_frames = jump_frames

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

    @staticmethod
    def is_video(video_path):
        try:
            video = cv2.VideoCapture(video_path)
            success, image = video.read()
            return success
        except cv2.error:
            return False

    def is_video_relevant(self, video_path):
        try:
            video = cv2.VideoCapture(video_path)
            success, image = video.read()
            if not success:
                return {}
        except cv2.error:
            return {}

        count = 0
        while success and count < self.max_frames_to_check:
            with tempfile.NamedTemporaryFile(suffix='.jpeg') as tmp:
                cv2.imwrite(tmp.name, image)
                if self.is_image_relevant(tmp.name):
                    return True

            for i in range(self.jump_frames):
                success, image = video.read()
                print('Read a new frame: ', success)
                if not success:
                    break
            count += 1

    @staticmethod
    def is_image(image_path):
        try:
            Image.open(image_path)
        except UnidentifiedImageError:
            return False
        return True

    def recognize_image(self, image_path):
        try:
            img = Image.open(image_path)
        except UnidentifiedImageError:
            print('UnidentifiedImageError: {}'.format(image_path))
            return {}

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
        if len(result) == 0:
            return True

        for key in result:
            if key in self.classes_white_list and result[key] > self.minimum_probability:
                return True
        return False

    def get_relevant_files(self, folder_path):
        relevant_files = []
        not_relevant_files = []
        for path in os.listdir(folder_path):
            file_path = os.path.join(folder_path, path)
            if (self.is_image(file_path) and self.is_image_relevant(file_path)) or \
                    (self.is_video(file_path) and self.is_video_relevant(file_path)):
                relevant_files.append(file_path)
            else:
                not_relevant_files.append(file_path)
        return relevant_files, not_relevant_files


def main():
    recognizer = Recognizer(RELEVANT_PLACES, MINIMUM_PROBABILITY, MAX_FRAMES_TO_CHECK, JUMP_FRAMES)
    relevant_files, not_relevant_files = recognizer.get_relevant_files('inputs')
    print('Relevant files:')
    print(relevant_files)
    print('Not relevant files:')
    print(not_relevant_files)


if __name__ == '__main__':
    main()

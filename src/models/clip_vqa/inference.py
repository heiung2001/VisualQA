import torch
from transformers import ViltProcessor, ViltForQuestionAnswering
from model import VQAModel
import pickle


class ClipBasedInference:
    def __init__(self, num_classes):
        weight = "clip_vqa/epoch_45.pth"
        print("Loading: {}".format(weight))
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VQAModel(num_classes=num_classes, device=DEVICE, hidden_size=512, model_name="ViT-L/14@336px").to(DEVICE)
        self.model.load_model(weight)


    def __call__(self, image, text):
        with open('answer_onehotencoder.pkl', 'rb') as f:
            ANSWER_ONEHOTENCODER = pickle.load(f)
        with open('answer_type_onehotencoder.pkl', 'rb') as f:
            ANSWER_TYPE_ONEHOTENCODER = pickle.load(f)

        predicted_answer, predicted_answer_type, answerability = self.model.test_model(image_path=image, question=text)
        answer = ANSWER_ONEHOTENCODER.inverse_transform(predicted_answer.cpu().detach().numpy())
        answer_type = ANSWER_TYPE_ONEHOTENCODER.inverse_transform(predicted_answer_type.cpu().detach().numpy())